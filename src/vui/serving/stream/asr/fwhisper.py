"""faster-whisper streaming ASR backend (local GPU).

faster-whisper isn't a true streaming model — we simulate streaming by
buffering incoming audio and re-transcribing the last N seconds every
~500ms for "interim" updates, then a full-accuracy pass on stop.

Models: `distil-small.en` (best latency/quality tradeoff, ~250MB fp16),
`distil-large-v3`, `turbo`, etc. Benchmarks (vui/user_recordings):

  | model               | final_after_stop | WER  |
  |---------------------|------------------|------|
  | distil-small.en     |              7ms | 0.03 |
  | turbo               |             56ms | 0.04 |
  | distil-large-v3     |             52ms | 0.05 |

See `tools/asr_benchmark/bench.py`.
"""

from __future__ import annotations

import threading
import time
from multiprocessing import Queue

import numpy as np

from vui.serving.stream.asr.base import ASRBackend, ASRSession


def _resample_if_needed(
    audio: np.ndarray, src_sr: int, tgt_sr: int = 16000
) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio.astype(np.float32, copy=False)
    import torch
    from julius.resample import resample_frac

    t = torch.from_numpy(audio).float()
    t = resample_frac(t, src_sr, tgt_sr)
    return t.numpy().astype(np.float32)


class FWhisperSession(ASRSession):
    def __init__(
        self,
        model,
        result_queue: Queue,
        interim_every_s: float = 0.5,
        interim_window_s: float = 4.0,
        beam_final: int = 5,
        # beam=1 (greedy) is too flaky on the first ~300-500ms of an
        # utterance — it commits "Peace" / "Oh, gee." / "going." for
        # what's actually a longer sentence. beam=3 is ~2-3x slower per
        # interim (~15ms vs ~5ms on a 4s window) but vastly more
        # accurate, and the persistent ASR session never gets the
        # beam=5 _transcribe pass (only fires on session stop, which
        # only happens on hard-reset / rotation).
        beam_interim: int = 3,
        stable_iters: int = 2,
        stable_min_new_words: int = 4,
    ):
        super().__init__(result_queue)
        self._model = model
        self._buffer = np.zeros(0, dtype=np.float32)
        self._interim_every_n = int(interim_every_s * 16000)
        self._interim_window_n = int(interim_window_s * 16000)
        self._last_interim_i = 0
        self._last_text = ""
        self._pending: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._beam_interim = beam_interim
        self._beam_final = beam_final
        self._feed_count = 0
        self._started_at = 0.0
        self._stable_iters = stable_iters
        self._stable_min_new = stable_min_new_words
        self._last_words: list[str] = []
        self._last_word_ends: list[float] = []
        self._stable_counts: list[int] = []
        self._emitted_stable_n: int = 0
        self._committed_words: list[str] = []
        self._committed_end_s: float = 0.0

    def start(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_interim_i = 0
        self._last_text = ""
        self._pending.clear()
        self._feed_count = 0
        self._started_at = time.perf_counter()
        self._last_words = []
        self._last_word_ends = []
        self._stable_counts = []
        self._emitted_stable_n = 0
        self._committed_words = []
        self._committed_end_s = 0.0
        print("[ASR] Stream started (fwhisper)")

    # Tuned VAD parameters for the streaming case:
    #   - speech_pad_ms=800: pad each VAD-detected speech region by 0.8s on
    #     each side, well beyond the default 400ms. Prevents Silero from
    #     clipping unvoiced word onsets ("th-" in "think", "h-" in "how's"),
    #     which was producing leading-text truncations like "nk of ways" and
    #     "ot doing that anymore".
    #   - min_silence_duration_ms=2000: don't treat short pauses as segment
    #     boundaries; keeps mid-utterance pauses inside one transcribe call.
    _VAD_PARAMS = {
        "speech_pad_ms": 800,
        "min_silence_duration_ms": 2000,
    }

    # Bias whisper toward keeping disfluencies in transcripts. By default
    # Whisper's training-time normaliser strips "um"/"uh" as formatting
    # noise; seeding with a filler-rich prompt has been shown (arxiv
    # 2503.06924) to lift the filler-inclusion rate ~15x. The downstream
    # LLM uses these as emotional / pacing cues — a "hmm" or "um" carries
    # meaning the assistant should react to.
    _FILLER_PROMPT = (
        "Umm, like, you know, uh, hmm, yeah, so, well, I mean, right, ah."
    )

    def _transcribe(self, audio: np.ndarray, beam: int) -> str:
        # vad_filter=True drops silent regions before whisper sees them, which
        # prevents the well-known "Thank you for watching" hallucination on
        # silence. The tuned vad_parameters above keep us from clipping
        # unvoiced word onsets while we're at it.
        segs, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=beam,
            vad_filter=True,
            vad_parameters=self._VAD_PARAMS,
            condition_on_previous_text=False,
            initial_prompt=self._FILLER_PROMPT,
        )
        return " ".join(s.text.strip() for s in segs).strip()

    def _transcribe_words(
        self, audio: np.ndarray, beam: int
    ) -> list[tuple[str, float]]:
        """Transcribe with word timestamps. Returns [(word, end_time_s), ...]."""
        segs, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=beam,
            vad_filter=True,
            vad_parameters=self._VAD_PARAMS,
            condition_on_previous_text=False,
            word_timestamps=True,
            initial_prompt=self._FILLER_PROMPT,
        )
        result = []
        for s in segs:
            if s.words:
                for w in s.words:
                    word = w.word.strip()
                    if word:
                        result.append((word, w.end))
        return result

    def _run_interim(self):
        with self._lock:
            committed_n = int(self._committed_end_s * 16000)
            start = max(committed_n, len(self._buffer) - self._interim_window_n)
            buf = self._buffer[start:].copy()
        if len(buf) < 16000 * 0.2:
            return
        buf_offset_s = start / 16000.0
        try:
            word_times = self._transcribe_words(buf, self._beam_interim)
        except Exception as e:
            print(f"[ASR] fwhisper interim error: {e}")
            return
        if not word_times:
            return

        words = [w for w, _ in word_times]
        full_words = list(self._committed_words) + words
        text = " ".join(full_words)

        if text != self._last_text:
            self._last_text = text
            self.result_queue.put({"type": "partial", "text": text})

        new_counts: list[int] = []
        for i, w in enumerate(words):
            if i < len(self._last_words) and w == self._last_words[i]:
                prev = self._stable_counts[i] if i < len(self._stable_counts) else 0
                new_counts.append(prev + 1)
            else:
                break
        new_counts.extend([0] * (len(words) - len(new_counts)))
        self._stable_counts = new_counts
        self._last_words = words
        self._last_word_ends = [buf_offset_s + t for _, t in word_times]

        stable_n = 0
        for i, c in enumerate(new_counts):
            if c >= self._stable_iters:
                stable_n = i + 1
            else:
                break

        # Commit policy — we have to force commits or the rolling 4s window
        # eats early words on long monologues. Three rules, applied in order:
        #
        #   1. Time-based: any word ending > SCROLL_FORCE_S into the window
        #      is about to scroll off, force-commit it (even if it never
        #      reached stable_iters confirmations — which can happen when
        #      mid-sentence revisions keep stable_n low).
        #   2. Stability-based: stable words past the last TAIL_WORDS also
        #      commit, so a long stable run is locked in.
        #   3. Sentence-boundary upgrade: within the kept revisable tail,
        #      prefer to extend the commit to a `.`/`!`/`?` if one appears.
        buf_len_s = len(self._buffer) / 16000
        interim_window_s = self._interim_window_n / 16000
        commit_up_to = 0
        # Only force-commit when the buffer has filled the window — i.e. the
        # rolling window is actually scrolling and earlier words would
        # otherwise be lost. Before then, beam=1 interim transcription is
        # still flaky on short audio; let stability checks handle commits.
        if buf_len_s > interim_window_s:
            window_start_s = buf_len_s - interim_window_s
            SCROLL_FORCE_S = 1.0
            force_commit_threshold_s = window_start_s + SCROLL_FORCE_S
            for i, end_s in enumerate(self._last_word_ends):
                if end_s < force_commit_threshold_s:
                    commit_up_to = i + 1
                else:
                    break

        TAIL_WORDS = 12
        commit_up_to = max(commit_up_to, stable_n - TAIL_WORDS)
        commit_up_to = max(0, commit_up_to)

        for i in range(commit_up_to, stable_n):
            w = words[i].rstrip("\",')")
            if w.endswith((".", "!", "?")):
                commit_up_to = i + 1

        if commit_up_to > 0:
            new_words = words[:commit_up_to]
            end_s = self._last_word_ends[commit_up_to - 1]
            chunk = " ".join(new_words)
            n_total = len(full_words)

            self._committed_words.extend(new_words)
            full_text = " ".join(self._committed_words)
            start_s = self._committed_end_s
            self._committed_end_s = end_s

            self.result_queue.put(
                {
                    "type": "stable_prefix",
                    "text": full_text,
                    "stable_words": len(self._committed_words),
                    "total_words": n_total,
                    "stable_end_time": end_s,
                }
            )
            self.result_queue.put(
                {
                    "type": "line_completed",
                    "text": full_text,
                    "sentence": chunk,
                    "start_time": start_s,
                    "duration": end_s - start_s,
                }
            )

            words = words[commit_up_to:]
            self._last_words = words
            self._last_word_ends = self._last_word_ends[commit_up_to:]
            self._stable_counts = new_counts[commit_up_to:]
            self._emitted_stable_n = 0

    def feed(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        audio = _resample_if_needed(audio, sample_rate, 16000)
        with self._lock:
            self._buffer = np.concatenate([self._buffer, audio])
            need_interim = (
                len(self._buffer) - self._last_interim_i
            ) >= self._interim_every_n
            if need_interim:
                self._last_interim_i = len(self._buffer)
        self._feed_count += 1
        if self._feed_count == 1:
            print(
                f"[ASR] First feed: shape={audio.shape}, dtype={audio.dtype}, "
                f"range=[{audio.min():.4f}, {audio.max():.4f}]"
            )
        if need_interim:
            t = threading.Thread(target=self._run_interim, daemon=True)
            t.start()
            self._pending.append(t)

    def stop(self) -> None:
        # Wait briefly for any in-flight interim threads
        for th in list(self._pending):
            th.join(timeout=2.0)
        with self._lock:
            buf = self._buffer.copy()
        dur = len(buf) / 16000
        t0 = time.perf_counter()
        try:
            text = self._transcribe(buf, self._beam_final) if len(buf) > 0 else ""
        except Exception as e:
            print(f"[ASR] fwhisper final error: {e}")
            text = self._last_text  # fall back to last interim if final fails
        ms = (time.perf_counter() - t0) * 1000
        print(
            f"[ASR] Final (fwhisper): '{text}' (buf={dur:.2f}s, took={ms:.0f}ms, "
            f"feeds={self._feed_count})"
        )
        if text:
            self.result_queue.put(
                {
                    "type": "line_completed",
                    "text": text,
                    "start_time": 0.0,
                    "duration": dur,
                }
            )
        self.result_queue.put({"type": "final", "text": text})


class FWhisperBackend(ASRBackend):
    def __init__(
        self,
        model: str = "distil-small.en",
        device: str = "cuda",
        compute_type: str | None = None,
        interim_every_s: float = 0.5,
        interim_window_s: float = 4.0,
        **_,
    ):
        from faster_whisper import WhisperModel

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        t0 = time.perf_counter()
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self.name = f"fwhisper:{model}:{device}:{compute_type}"
        self._interim_every_s = interim_every_s
        self._interim_window_s = interim_window_s
        print(f"[ASR] Loaded in {(time.perf_counter()-t0)*1000:.0f}ms ({self.name})")
        # Warm
        _ = list(
            self._model.transcribe(
                np.zeros(16000, dtype=np.float32),
                language="en",
                beam_size=1,
                vad_filter=False,
            )[0]
        )

    def make_session(self, result_queue: Queue) -> ASRSession:
        return FWhisperSession(
            self._model,
            result_queue,
            interim_every_s=self._interim_every_s,
            interim_window_s=self._interim_window_s,
        )

    def transcribe_once(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        audio = _resample_if_needed(audio, sample_rate, 16000)
        segs, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=False,
            initial_prompt=FWhisperSession._FILLER_PROMPT,
        )
        return " ".join(s.text.strip() for s in segs).strip()
