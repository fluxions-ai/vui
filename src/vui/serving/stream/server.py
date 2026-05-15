"""Main process: aiohttp server, WebRTC, WebSocket, process management.

No GPU, no CUDA - pure async I/O. Starts worker processes and routes
messages between them.
"""

import asyncio
import json
import os
import platform
import sys
import time
from multiprocessing import Process
from pathlib import Path

import httpx
import numpy as np
from aiohttp import web
from aiortc.mediastreams import MediaStreamTrack
from av.audio.resampler import AudioResampler
from torch.multiprocessing import Queue

# --- Bind extracted modules ---
from vui.serving.stream import (  # noqa: E402
    connection,
    drains,
    memories,
    model_routes,
    prompt_routes,
    tasks,
    voice_turn,
)
from vui.serving.stream._log import _slog, _spawn  # noqa: E402
from vui.serving.stream.asr_worker import DEFAULT_ASR_MODEL, asr_process

# LLM config + helpers live in `llm.py`; re-exported here for callers that
# imported them from `server` historically.
from vui.serving.stream.llm import (  # noqa: E402, F401
    DEFAULT_OLLAMA_MODEL,
    GGUF_MODEL_NAME,
    LLM_CHUNK_TOKENS,
    MLX_MODEL_DIR,
    MLX_MODEL_HF,
    MLX_MODEL_NAME,
    OLLAMA_URL,
    ensure_mlx_model,
    llm_next_chunk,
    llm_prefill_system,
    llm_prefill_user,
    llm_stream_chunks,
)
from vui.serving.stream.playback import TTSPlaybackTrack
from vui.serving.stream.prompts import (  # noqa: E402
    SOUL,
    TASK_SERVER_URL,
    build_soul,
    probe_task_server,
)
from vui.serving.stream.text_utils import strip_emoji  # noqa: E402, F401

IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

if IS_APPLE_SILICON:
    from vui.serving.stream.tts_worker_mlx import tts_process_mlx as tts_process
else:
    from vui.serving.stream.tts_worker import tts_process

QWEN_SR = 24000
PROMPTS_DIR = Path("prompts")
STATE_FILE = PROMPTS_DIR / ".last_prompt"


class AudioRecordingSink(MediaStreamTrack):
    """Accumulates incoming WebRTC audio, forwards to TTS (codec encode) + ASR processes."""

    kind = "audio"

    def __init__(
        self,
        track: MediaStreamTrack,
        tts_queue: Queue,
        asr_queue: Queue,
        vad_queue: Queue,
        asr_result_queue: "Queue | None" = None,
    ):
        super().__init__()
        self.track = track
        self.recording = False
        self.vad_enabled = False
        self._speaking = False
        self._samples_recorded = 0
        self._tts_queue = tts_queue
        self._asr_queue = asr_queue
        self._vad_queue = vad_queue
        # Side-channel into the ASR drain so the sink can emit
        # `vad_audio_ready` on the first frame WITHOUT waiting for the ASR
        # worker to finish loading. Keeps the mic-live indicator fast on
        # cold start.
        self._asr_result_queue = asr_result_queue
        self._resamp_codec = AudioResampler("s16", "mono", QWEN_SR)
        self._resamp_asr = AudioResampler("s16", "mono", 16000)
        self._pre_buffer = np.array([], dtype=np.float32)
        self._pre_buffer_24k = np.array([], dtype=np.float32)
        self._audio_16k_chunks: list[np.ndarray] = []
        self.last_audio_t: float = 0.0

    _vad_audio_sent = False
    _PRE_BUFFER_SAMPLES = 16000
    _PRE_BUFFER_SAMPLES_24K = 24000

    def start_recording(self, *, continuation: bool = False):
        """Start per-turn capture: codec stream + audio_16k buffer.

        ASR session is independent and persistent (managed at warmup).
        Caller should set `speaking = True` separately so the 16k pre-buffer
        is flushed to ASR + audio_16k_chunks via the property setter.

        With `continuation=True`, the TTS worker keeps its accumulated
        incremental-prefill state (so a brief pause-resume in the middle of
        a user turn doesn't throw away `[user] "<chunk>"` blocks already
        written to KV by `prefill_user_chunk`).
        """
        self.recording = True
        self._samples_recorded = 0
        self._audio_16k_chunks = []
        self._tts_queue.put({"cmd": "stream_start", "continuation": continuation})
        if self._pre_buffer_24k.size > 0:
            self._tts_queue.put(
                {"cmd": "stream_feed", "audio": self._pre_buffer_24k.copy()}
            )
            self._pre_buffer_24k = np.array([], dtype=np.float32)

    @property
    def rec_duration(self) -> float:
        return self._samples_recorded / 16000

    def stop_recording(self) -> float:
        """Stop per-turn codec stream. ASR session keeps running."""
        self.recording = False
        self._tts_queue.put({"cmd": "stream_stop"})
        return self._samples_recorded / 16000

    @property
    def speaking(self) -> bool:
        return self._speaking

    @speaking.setter
    def speaking(self, value: bool):
        if value and not self._speaking and self._pre_buffer.size > 0:
            # Flush the pre-vad-start buffer (~1s rolling) to:
            #   1. The persistent ASR session — Silero needs ~0.2s of speech
            #      before it fires vad_start, so the user's first word(s)
            #      are in this buffer, not in any subsequent feed. Without
            #      this flush, fwhisper's first partial is e.g. "out and..."
            #      when the user actually said "I've been doing stuff out
            #      and...". vad_filter=True on fwhisper's transcribe call
            #      drops the silent prefix of the pre-buffer, so this is
            #      safe — no more "thank you" hallucinations.
            #   2. audio_16k_chunks — for the user-audio echo-back used in
            #      the TTS user prompt.
            pre = self._pre_buffer.copy()
            self._asr_queue.put({"cmd": "feed", "audio": pre, "sample_rate": 16000})
            if self.recording:
                self._audio_16k_chunks.append(pre)
                self._samples_recorded += pre.size
            self._pre_buffer = np.array([], dtype=np.float32)
        self._speaking = value

    def get_audio_16k(self) -> np.ndarray | None:
        if not self._audio_16k_chunks:
            return None
        return np.concatenate(self._audio_16k_chunks)

    async def run(self):
        while True:
            try:
                frame = await self.track.recv()
            except Exception:
                break

            self.last_audio_t = time.monotonic()

            if not self.recording and not self.vad_enabled:
                continue

            for rf in self._resamp_asr.resample(frame):
                a = rf.to_ndarray().astype(np.float32) / 32768.0
                audio_16k = a.flatten()
                if self.vad_enabled:
                    if not self._vad_audio_sent:
                        print(f"[Sink] First VAD audio chunk, len={len(audio_16k)}")
                        self._vad_audio_sent = True
                        # Eagerly tell main "mic is live" — don't wait for
                        # the ASR worker to finish loading the model.
                        if self._asr_result_queue is not None:
                            try:
                                self._asr_result_queue.put_nowait(
                                    {"type": "vad_audio_ready"}
                                )
                            except Exception:
                                pass
                    self._vad_queue.put(audio_16k)
                    # Always feed ASR while VAD is enabled — independent of
                    # speaking state. Mid-sentence pauses don't truncate ASR
                    # text; the persistent session keeps transcribing through
                    # silence into the next utterance.
                    self._asr_queue.put(
                        {"cmd": "feed", "audio": audio_16k, "sample_rate": 16000}
                    )
                    if not self.speaking:
                        self._pre_buffer = np.concatenate(
                            (self._pre_buffer, audio_16k)
                        )[-self._PRE_BUFFER_SAMPLES :]
                if self.recording:
                    self._samples_recorded += audio_16k.size
                    self._audio_16k_chunks.append(audio_16k.copy())

            for rf in self._resamp_codec.resample(frame):
                a = rf.to_ndarray().astype(np.float32) / 32768.0
                audio_24k = a.flatten()
                if self.recording:
                    self._tts_queue.put({"cmd": "stream_feed", "audio": audio_24k})
                elif self.vad_enabled:
                    self._pre_buffer_24k = np.concatenate(
                        (self._pre_buffer_24k, audio_24k)
                    )[-self._PRE_BUFFER_SAMPLES_24K :]

    async def recv(self):
        return await self.track.recv()


DEFAULT_SETTINGS = {
    "temperature": 0.7,
    "top_k": 50,
    "wps_score": 0,
    "rep_penalty": 1.1,
    "rep_window": 24,
    "max_duration": 120,
    "sq_scores": [0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
    "chunk_words": 60,
    "user_audio": True,
    "n_codebooks": 16,
    "eos_threshold": 0.2,
    "keep_context": False,
    # Cap LLM context (soul + recent conversation) to this many
    # minutes of wall-time history. 0 = unbounded. Trimming is applied
    # right before each LLM call — older turns drop off the front,
    # the soul always stays.
    "context_minutes": 3.0,
    "tool_check": True,
    # Hallucination gate (engine.GenConfig). 0 = off.
    "gate_frames": 0,
    "gate_entropy_max": 1.9,
    "gate_retries": 2,
    # Endpointing / response-latency knobs (live-tunable via /test/settings).
    "vad_stop_secs": 0.3,  # silero silence-to-stop. Lower = snappier.
    "asr_settle_s": 0.12,  # post vad_stop wait for last fwhisper interim.
    "trailing_off_delay": 0.0,  # extra wait when ASR text doesn't end in .?!
}


class SessionState:
    def __init__(self):
        from aiortc import RTCPeerConnection

        self.pc: RTCPeerConnection | None = None
        self.ws: web.WebSocketResponse | None = None
        self.client_id: str | None = None
        self.recording_sink: AudioRecordingSink | None = None
        self.playback_track: TTSPlaybackTrack | None = None
        self.settings = dict(DEFAULT_SETTINGS)
        self.conversation: list[dict] = []
        self.assistant_name = _load_assistant_name() or "Vui"
        # User-edited soul overrides; otherwise rebuild with the
        # configured name baked into the {name} placeholder.
        self.soul = _load_soul() or build_soul(
            with_claude=True, name=self.assistant_name
        )
        self.cancel_generation = False
        self.ready = False
        # Set on initial (non-reconnect) WS connect; cleared after the
        # greeting fires. Deferred when workers aren't yet ready, then
        # picked up by _signal_fully_ready().
        self.greeting_pending = False


class StreamServer:
    def __init__(
        self, checkpoint_path: str, n_quantizers: int = 16, asr_model_arch: int = 5
    ):
        self.checkpoint_path = checkpoint_path
        self.n_quantizers = n_quantizers
        self.asr_model_arch = asr_model_arch
        self.session = SessionState()

        self.tts_cmd_queue = Queue()
        self.tts_audio_queue = Queue()
        self.asr_cmd_queue = Queue()
        self.asr_result_queue = Queue()
        self.vad_queue = Queue()

        self.tts_proc: Process | None = None
        self.asr_proc: Process | None = None

        self.tts_ready = False
        self.asr_ready = False

        self.tts_T = 0
        self.tts_max_T = 0
        self.conv_ctx = 0
        self.conv_ctx_max = 8192
        self.thoughts_ctx = 0
        self.thoughts_ctx_max = 8192

        self._tts_response_queue: asyncio.Queue | None = None
        self._drain_task: asyncio.Task | None = None
        self._conv_log_path = Path("debug_dump") / "conversation_log.jsonl"
        self._conv_log_path.parent.mkdir(exist_ok=True)

        self.asr_model = DEFAULT_ASR_MODEL
        self._asr_backend_set_event: asyncio.Event | None = None
        self._asr_backend_set_result: dict | None = None

        self.ollama_model = DEFAULT_OLLAMA_MODEL
        self._warmup_done = False
        self._server_vad = True

        # "Hold the floor" — UI press-and-hold defers VAD endpointing so a
        # thinking pause doesn't auto-commit. If VAD stops while held, we
        # remember the rec_dur and fire tiered_commit on release.
        self._floor_held = False
        self._floor_pending_dur: float | None = None

        # "Listen mode" — assistant keeps listening but doesn't respond
        # unless addressed by name (assistant_name + Whisper variants).
        # Toggled by the UI button or by deterministic voice phrases —
        # see wake_word.py (`match` for wake, `shut_up_match` for sleep).
        self._listen_mode = False

        # Initial ready-blockers. Cleared as each precondition resolves; the
        # frontend's `workers_ready` event fires only when the set is empty.
        # - "warmup": Ollama prefill, prompt load, etc. — cleared by warmup().
        # - "mic":    first VAD audio chunk has arrived — proves the WebRTC
        #             audio pipeline is live, not just connected.
        self._ready_blockers: set[str] = {"warmup", "mic"}

        self._prompt_audio_16k: np.ndarray | None = None
        self._prompt_audio_24k: np.ndarray | None = None

        self._last_asr_text: str | None = None
        self._last_prefill_text: str = ""
        self._last_prefill_t: float = 0.0
        # Single in-flight gate. Without this we fire one task per ASR
        # partial; with slow models (1+ second per call) they stack up,
        # serialise on Ollama's KV cache, and waste compute warming a
        # transcript the user has already revised past.
        self._prefill_inflight: bool = False
        self._last_transcription: str | None = None
        self._last_codes_final_n = 0
        self._codes_final_event = asyncio.Event()

        self._user_chunks_prefilled: int = 0
        # Tracks the last-incremental-chunk text/time so the stable_prefix
        # handler only fires `prefill_user_chunk` when meaningful new text
        # has been locked in. Without throttling, fwhisper's interim
        # commits fire every ~500ms with the same growing text, which
        # causes the model to parrot user-audio fragments back.
        self._last_user_chunk_text: str = ""
        self._last_user_chunk_t: float = 0.0

        from vui.serving.stream.tasks import LocalTask, load_tasks

        self._tasks: dict[str, dict] = load_tasks()
        # In-process cancel-callback registry for tasks tools surfaced via
        # their `TASK` block (e.g. set_timer). Not persisted — running rows
        # get reclassified to "cancelled" on next boot by `load_tasks`.
        self._local_tasks: dict[str, LocalTask] = {}
        self._pending_task_id: str | None = None
        self._thoughts_stop_llm = False
        self._pending_task_results: list[dict] = []

        self._endpointing_task: asyncio.Task | None = None
        self._latest_partial = ""
        self._turn_best_asr = ""

        # Persistent ASR transcript accumulator (across pauses + commits within a phase).
        # _phase_transcript is the running text. _committed_len marks how many chars
        # have been committed to LLM. _carry holds finalised text from prior ASR
        # sessions during 30s rotation.
        self._phase_transcript: str = ""
        # Position-based committed marker. Kept for backward-compat /
        # logging. The authoritative one is _committed_text — see below.
        self._committed_len: int = 0
        # Set by delayed_turn_end to the snapshot boundary; consumed by
        # commit_turn to advance _committed_len.
        self._next_committed_len: int | None = None
        # Content-based committed marker: the actual text we committed at
        # the previous turn. Used to find the new portion at next commit
        # via longest-common-prefix. Robust to fwhisper revising the
        # earlier-committed tail (which breaks position-based slicing).
        self._committed_text: str = ""
        self._next_committed_text: str | None = None
        self._carry: str = ""
        self._current_partial: str = ""
        # Text the ASR backend has finalised (line_completed events). Stable —
        # won't be revised. Frontend gets this stream separately from
        # `partial_asr` so it can show a settled transcript.
        self._committed_asr: str = ""
        self._asr_session_started_t: float = 0.0
        self._asr_speaking_secs: float = 0.0
        self._asr_speech_start_t: float = 0.0
        self._asr_rotating: bool = False
        self._asr_rotation_done: asyncio.Event | None = None
        # Number of pending `session_ready` acks before un-quenching ASR
        # output. Bumped by hard-reset paths so ALL msgs from the dying
        # session (partials, stable_prefix, line_completed, final) are
        # discarded until the new session declares itself ready. Without
        # this, fwhisper's pending interim threads emit partials with the
        # OLD `_committed_words` prefix as `stop()` joins them, and those
        # messages overwrite `_phase_transcript` with the prior turn's
        # text right before tiered_commit reads it.
        self._asr_quench_pending: int = 0
        self._phase_end_task: asyncio.Task | None = None
        # In-flight tasks that stream `reply` chunks to the frontend WS.
        # `_reset_session_state` cancels + awaits all of these BEFORE
        # clearing conversation + sending `context_cleared`, so no late
        # reply lands after the clear and re-populates the chat. Members
        # are added by `_spawn_response` and removed on task completion.
        self._response_tasks: set[asyncio.Task] = set()

        self._generates_sent = 0
        self._generates_done = 0
        self._llm_streaming = False
        self._turn_chunk_idx = 0
        self._turn_vad_stop_ms: float = 0
        self._turn_respond_start_ms: float = 0
        self._pending_done: dict | None = None
        self._turn_t: dict[str, float] = {}

        self._task_server_available: bool = True
        self._task_server_capabilities: list[str] = []
        self._llm_available: bool = True

        self._test_capture_sink: list | None = None
        self._test_done_event: asyncio.Event | None = None
        self._test_first_frame_ms: float | None = None
        self._test_gen_info: dict | None = None

        self._memories_store: list[dict] = memories.load_memories()
        memories.seed_geo_memory(self._memories_store)
        self._memories: list[str] = memories.memories_to_strings(self._memories_store)

        from vui.serving.stream.thoughts import ThoughtsStream

        self._thoughts = ThoughtsStream(self)

    def _log_conv(self, event: str, **data):
        import time as _time

        entry = {"t": _time.time(), "event": event, **data}
        try:
            with open(self._conv_log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    def start_workers(self):
        import multiprocessing

        self.tts_cancel_event = multiprocessing.Event()
        print("Starting worker processes...")
        self.tts_proc = Process(
            target=tts_process,
            args=(
                self.tts_cmd_queue,
                self.tts_audio_queue,
                self.checkpoint_path,
                self.tts_cancel_event,
            ),
            daemon=False,
        )
        self.asr_proc = Process(
            target=asr_process,
            args=(
                self.asr_cmd_queue,
                self.asr_result_queue,
                self.vad_queue,
                self.asr_model_arch,
            ),
            daemon=True,
        )
        self.tts_proc.start()
        self.asr_proc.start()
        print(
            f"Workers started: TTS(pid={self.tts_proc.pid}), ASR(pid={self.asr_proc.pid})"
        )

    def stop_workers(self):
        for q in [self.tts_cmd_queue, self.asr_cmd_queue]:
            try:
                q.put({"cmd": "shutdown"})
            except Exception:
                pass
        for p in [self.tts_proc, self.asr_proc]:
            if p is not None and p.is_alive():
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

    def _derive_worker_states(self) -> dict[str, str]:
        def proc_state(proc: Process | None, ready: bool) -> str:
            if proc is None or not proc.is_alive():
                return "down"
            return "ready" if ready else "loading"

        sink = self.session.recording_sink
        if sink is None:
            mic = "down"
        elif sink.last_audio_t == 0.0:
            mic = "loading"
        elif time.monotonic() - sink.last_audio_t > 2.0:
            mic = "down"
        else:
            mic = "ready"

        return {
            "tts": proc_state(self.tts_proc, self.tts_ready),
            "asr": proc_state(self.asr_proc, self.asr_ready),
            "mic": mic,
            "warmup": "ready" if self._warmup_done else "loading",
        }

    async def _send_worker_status(self):
        states = self._derive_worker_states()
        ws = self.session.ws
        if ws and not ws.closed:
            try:
                await ws.send_json({"type": "worker_status", "states": states})
            except Exception:
                pass

    async def _worker_status_loop(self):
        while True:
            await asyncio.sleep(1.0)
            try:
                await self._send_worker_status()
            except Exception:
                pass

    async def _send_llm_status(self):
        ws = self.session.ws
        if ws and not ws.closed:
            try:
                await ws.send_json(
                    {"type": "llm_status", "available": self._llm_available}
                )
            except Exception:
                pass

    async def _set_llm_available(self, available: bool):
        if available == self._llm_available:
            return
        self._llm_available = available
        print(f"[main] LLM transitioned: {'DOWN → up' if available else 'up → DOWN'}")
        await self._send_llm_status()

    async def _signal_fully_ready(self):
        if self._server_vad:
            self.asr_cmd_queue.put({"cmd": "vad_enable"})
        ws = self.session.ws
        if ws and not ws.closed:
            try:
                await ws.send_json({"type": "workers_ready"})
            except Exception:
                pass
        from vui.serving.stream.connection import try_fire_greeting

        try_fire_greeting(self)

    async def _block_ready(self, reason: str):
        was_ready = not self._ready_blockers
        self._ready_blockers.add(reason)
        self.session.ready = False
        if was_ready:
            if self._server_vad and self.session.recording_sink:
                self.session.recording_sink.vad_enabled = False
            self.asr_cmd_queue.put({"cmd": "vad_disable"})
            ws = self.session.ws
            if ws and not ws.closed:
                try:
                    await ws.send_json({"type": "busy", "reason": reason})
                except Exception:
                    pass

    async def _unblock_ready(self, reason: str):
        self._ready_blockers.discard(reason)
        if not self._ready_blockers:
            self.session.ready = not self._ready_blockers
            if self._server_vad:
                if self.session.recording_sink:
                    self.session.recording_sink.vad_enabled = True
                self.asr_cmd_queue.put({"cmd": "vad_enable"})
            ws = self.session.ws
            if ws and not ws.closed:
                try:
                    await ws.send_json({"type": "workers_ready"})
                except Exception:
                    pass
            from vui.serving.stream.connection import try_fire_greeting

            try_fire_greeting(self)

    async def _log(self, text: str, level: str = "info"):
        print(f"[{level}] {text}")
        ws = self.session.ws
        if ws and not ws.closed:
            try:
                await ws.send_json({"type": "log", "text": text, "level": level})
            except Exception:
                pass

    def _log_sync(self, text: str, level: str = "info"):
        print(f"[{level}] {text}")

    async def _reset_session_state(self):
        self.session.cancel_generation = True
        self.tts_cancel_event.set()
        self.tts_cmd_queue.put({"cmd": "cancel"})
        if self.session.playback_track:
            self.session.playback_track.flush()
        if self._endpointing_task and not self._endpointing_task.done():
            self._endpointing_task.cancel()
            self._endpointing_task = None
        if self._phase_end_task and not self._phase_end_task.done():
            self._phase_end_task.cancel()
            self._phase_end_task = None
        # Cancel + await every in-flight task that streams `reply`
        # chunks (voice_respond, deliver_task_results) BEFORE we clear
        # conversation + send context_cleared. Without this, late chunks
        # arrive on the WS after the frontend has cleared and re-populate
        # the chat the user just reset.
        live = [t for t in list(self._response_tasks) if not t.done()]
        for t in live:
            t.cancel()
        for t in live:
            try:
                await asyncio.wait_for(t, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:
                pass
        self._response_tasks.clear()
        if self.session.recording_sink and self.session.recording_sink.recording:
            self.session.recording_sink.stop_recording()
        if self.session.recording_sink:
            self.session.recording_sink.speaking = False
        # Hard-reset the persistent ASR session so its internal accumulator
        # doesn't carry prior conversation text into the next turn.
        from vui.serving.stream.drains import _hard_reset_asr

        _hard_reset_asr(self)
        self._last_asr_text = None
        self.session.conversation.clear()
        self.conv_ctx = 0
        self.thoughts_ctx = 0
        self._llm_streaming = False
        self._generates_sent = 0
        self._generates_done = 0
        self._last_codes_final_n = 0
        self._codes_final_event.clear()
        self._last_prefill_text = ""
        self._user_chunks_prefilled = 0
        self._last_user_chunk_text = ""
        self._last_user_chunk_t = 0.0
        self._pending_task_results.clear()
        self._pending_task_id = None
        self._thoughts_stop_llm = False
        if self._thoughts._task and not self._thoughts._task.done():
            self._thoughts._task.cancel()
            self._thoughts._task = None
        for task_id in list(self._tasks):
            try:
                await self._cancel_task(task_id)
            except Exception:
                pass
        self._tasks.clear()
        self._local_tasks.clear()
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                await client.post(f"{TASK_SERVER_URL}/session/clear")
        except Exception:
            pass
        ws = self.session.ws
        if ws and not ws.closed:
            try:
                await ws.send_json({"type": "context_cleared"})
                await ws.send_json(
                    {
                        "type": "ctx_status",
                        "conv_ctx": self.conv_ctx,
                        "conv_ctx_max": self.conv_ctx_max,
                        "thoughts_ctx": self.thoughts_ctx,
                        "thoughts_ctx_max": self.thoughts_ctx_max,
                    }
                )
            except Exception:
                pass

    async def handle_reset(self, request):
        _slog("=" * 60)
        _slog("RESET — clearing conversation, rewinding to prompt")
        _slog("=" * 60)
        await self._block_ready("prompt")
        await self._reset_session_state()

        self.tts_cmd_queue.put({"cmd": "rewind"})
        # Don't reset cancel_generation here — `_reset_session_state` has
        # already awaited the in-flight voice_respond, so there's no loop
        # to gate. The next turn's voice_respond body sets it False at
        # entry. Resetting it here previously raced with any LLM stream
        # iteration that was about to break on cancel, letting late
        # `reply` chunks slip through and re-populate the just-cleared
        # frontend chat.
        await self._unblock_ready("prompt")
        return web.json_response({"ok": True})

    async def handle_cancel(self, request):
        self.session.cancel_generation = True
        self.tts_cancel_event.set()
        self.tts_cmd_queue.put({"cmd": "cancel"})
        if self.session.playback_track:
            self.session.playback_track.flush()
        await self._log("Generation cancelled", "warn")
        return web.json_response({"ok": True})

    async def handle_state(self, request):
        last = _get_last_prompt()
        return web.json_response(
            {
                "last_prompt": last,
                "soul": self.session.soul,
                "tts_ready": self.tts_ready,
                "asr_ready": self.asr_ready,
                "session_ready": self.session.ready,
                "tts_T": self.tts_T,
                "tts_max_T": self.tts_max_T,
                "conv_ctx": self.conv_ctx,
                "conv_ctx_max": self.conv_ctx_max,
                "thoughts_ctx": self.thoughts_ctx,
                "thoughts_ctx_max": self.thoughts_ctx_max,
                "prompt_text": self.session.conversation,
            }
        )

    async def handle_tools_reload(self, request):
        """Re-walk src/vui/serving/stream/tools/ and rebuild the registry.

        Lets newly written tool files (hand-authored or codegen) go live
        without restarting the server. The next thoughts evaluation sees
        the updated tool list.
        """
        from vui.serving.stream import tools as tools_registry

        count = tools_registry.load_tools()
        return web.json_response(
            {"ok": True, "count": count, "tools": sorted(tools_registry._HANDLES)}
        )

    async def handle_tools_list(self, request):
        from vui.serving.stream import tools as tools_registry

        return web.json_response(
            {"count": len(tools_registry._HANDLES), "tools": tools_registry.tools_list()}
        )

    _RESPONSE_TYPES = {
        "prompt_loaded",
        "kv_saved",
        "kv_loaded",
        "state",
        "encoded",
        "prompt_audio",
        "user_prefilled",
        "text_prefilled",
    }

    _VAD_STOP_SECS = 0.1

    async def handle_visualizer_js(self, request):
        from vui.serving.stream.frontend import get_visualizer_js

        return web.Response(
            text=get_visualizer_js(),
            content_type="application/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    async def handle_visualizer_demo(self, request):
        from vui.serving.stream.frontend import get_visualizer_demo_html

        return web.Response(
            text=get_visualizer_demo_html(), content_type="text/html"
        )

    async def run(self):
        app = web.Application()
        app.router.add_get("/", self.handle_index)
        app.router.add_get("/visualizer.js", self.handle_visualizer_js)
        app.router.add_get("/visualizer", self.handle_visualizer_demo)
        app.router.add_post("/offer", self.handle_offer)
        app.router.add_post("/upload-prompt", self.handle_upload_prompt)
        app.router.add_post("/update-prompt-text", self.handle_update_prompt_text)
        app.router.add_post("/save-prompt", self.handle_save_prompt)
        app.router.add_get("/prompts", self.handle_list_prompts)
        app.router.add_post("/load-prompt", self.handle_load_prompt)
        app.router.add_post("/reset", self.handle_reset)
        app.router.add_post("/cancel", self.handle_cancel)
        app.router.add_get("/ollama/models", self.handle_ollama_models)
        app.router.add_post("/ollama/model", self.handle_ollama_set_model)
        app.router.add_post("/ollama/pull", self.handle_ollama_pull)
        app.router.add_get("/asr/models", self.handle_asr_models)
        app.router.add_post("/asr/model", self.handle_asr_set_model)
        app.router.add_get("/prompt-audio", self.handle_prompt_audio)
        app.router.add_get("/state", self.handle_state)
        app.router.add_post("/task_done", self.handle_task_done)
        app.router.add_post("/tasks/delete", self.handle_delete_task_http)
        app.router.add_post("/tasks/clear", self.handle_clear_tasks_http)
        from vui.serving.stream.test_routes import register_test_routes

        register_test_routes(app, self)
        app.router.add_get("/memories", self.handle_list_memories)
        app.router.add_post("/memories", self.handle_add_memory)
        app.router.add_post("/memories/remove", self.handle_remove_memory)
        app.router.add_post("/memories/delete", self.handle_delete_memory_by_index)
        app.router.add_post("/memories/clear", self.handle_clear_memories)
        app.router.add_post("/tools/reload", self.handle_tools_reload)
        app.router.add_get("/tools", self.handle_tools_list)
        app.router.add_get("/ws", self.handle_ws)

        from vui.serving.stream.realtime import handle_realtime_ws

        app.router.add_get(
            "/v1/realtime", lambda request: handle_realtime_ws(self, request)
        )

        from vui.serving.stream.voice_note_routes import register_voice_note_routes

        register_voice_note_routes(app, self)

        runner = web.AppRunner(app)
        await runner.setup()
        http_port = int(os.environ.get("VUI_HTTP_PORT", "8080"))
        site = web.TCPSite(runner, "0.0.0.0", http_port)
        await site.start()
        print(f"Server running at http://localhost:{http_port}")

        # HTTPS on a self-signed cert — `getUserMedia` won't grant the mic on
        # a non-secure origin, so phones on the LAN can only use the app via
        # the https URL. Disable with VUI_TLS=0.
        if os.environ.get("VUI_TLS", "1") != "0":
            try:
                from vui.serving.stream.tls import get_ssl_context, lan_urls

                https_port = int(os.environ.get("VUI_HTTPS_PORT", "8443"))
                ssl_ctx = get_ssl_context()
                tls_site = web.TCPSite(
                    runner, "0.0.0.0", https_port, ssl_context=ssl_ctx
                )
                await tls_site.start()
                print(f"HTTPS at https://localhost:{https_port}")
                for url in lan_urls(https_port)[1:]:
                    print(f"           {url}")
            except Exception as e:
                print(f"[tls] TLS setup failed ({e}) — HTTPS disabled")

        self._drain_task = _spawn(self.drain_tts_audio(), "drain_tts_audio")
        self._asr_drain_task = _spawn(self.drain_asr_results(), "drain_asr_results")
        _spawn(self._warmup(), "warmup")
        _spawn(self._worker_status_loop(), "worker_status_loop")
        from vui.serving.stream.version_check import check_for_updates

        _spawn(check_for_updates(), "update_check")

        await asyncio.Event().wait()


connection.bind(StreamServer)
drains.bind(StreamServer)
prompt_routes.bind(StreamServer)
model_routes.bind(StreamServer)
voice_turn.bind(StreamServer)
tasks.bind(StreamServer)
memories.bind(StreamServer)


# --- Module-level helpers ---


DEFAULT_VOICE = os.environ.get("VUI_DEFAULT_VOICE", "maeve")


def _save_last_prompt(name: str):
    PROMPTS_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(name)


def _get_last_prompt() -> str | None:
    if STATE_FILE.exists():
        saved = STATE_FILE.read_text().strip()
        if saved:
            return saved
    return DEFAULT_VOICE


SOUL_FILE = PROMPTS_DIR / ".soul"
ASSISTANT_NAME_FILE = PROMPTS_DIR / ".assistant_name"


def _save_soul(text: str):
    PROMPTS_DIR.mkdir(exist_ok=True)
    SOUL_FILE.write_text(text)


def _load_soul() -> str | None:
    if SOUL_FILE.exists():
        return SOUL_FILE.read_text().strip() or None
    return None


def _save_assistant_name(name: str):
    PROMPTS_DIR.mkdir(exist_ok=True)
    ASSISTANT_NAME_FILE.write_text(name)


def _load_assistant_name() -> str | None:
    if ASSISTANT_NAME_FILE.exists():
        return ASSISTANT_NAME_FILE.read_text().strip() or None
    return None


def main():
    import multiprocessing

    import torch

    from vui.telemetry import enabled as _telemetry_enabled

    multiprocessing.set_start_method("spawn", force=True)

    if _telemetry_enabled():
        print(
            "[telemetry] anonymous events enabled — only {voice, seconds} per turn. "
            "Disable: VUI_TELEMETRY=0 (see README §Telemetry).",
            file=sys.stderr,
        )
    else:
        print("[telemetry] disabled (VUI_TELEMETRY=0)", file=sys.stderr)

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "vui-nano.safetensors"
    from vui.hf import download

    checkpoint_path = download(checkpoint_path)

    from vui.config import Config

    if checkpoint_path.endswith(".safetensors"):
        import json

        from safetensors import safe_open

        with safe_open(checkpoint_path, framework="pt") as f:
            cfg_dict = json.loads(f.metadata()["config"])
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt["config"]
        del ckpt
    cfg = Config(**cfg_dict)
    n_quantizers = cfg.model.n_quantizers
    print(f"Model n_quantizers={n_quantizers}")

    asr_arch = 5
    if "--asr-medium" in sys.argv:
        asr_arch = 5
    elif "--asr-small" in sys.argv:
        asr_arch = 4
    elif "--asr-base" in sys.argv:
        asr_arch = 3
    server = StreamServer(
        checkpoint_path, n_quantizers=n_quantizers, asr_model_arch=asr_arch
    )
    server.start_workers()

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop_workers()


if __name__ == "__main__":
    main()
