"""TTS + ASR drain loops and response routing."""

from __future__ import annotations

import asyncio
import time
from queue import Empty
from typing import TYPE_CHECKING

from vui.serving.stream._log import _slog, _spawn
from vui.telemetry import record_render

if TYPE_CHECKING:
    from vui.serving.stream.server import StreamServer


def _join_carry(carry: str, current: str) -> str:
    """Join carried (rotated) text with the current session's partial."""
    carry = (carry or "").strip()
    current = (current or "").strip()
    if carry and current:
        return f"{carry} {current}"
    return carry or current


def _hard_reset_asr(srv: StreamServer):
    """Synchronously rotate the persistent ASR session for a phase boundary.

    Sends stop+start without awaiting and clears transcript state. All
    output from the dying session (partials, stable_prefix, line_completed,
    final) is discarded via the quench counter until the worker emits
    `session_ready` for the new session — fwhisper's `stop()` joins
    in-flight interim threads, which keep enqueueing partials carrying the
    OLD `_committed_words` prefix as they wind down. Without quench, those
    msgs overwrite `_phase_transcript` and bleed prior-turn text into the
    next commit.

    Used when (a) a fresh phase starts after the prior turn fully completed,
    or (b) phase_end_task fires after grace. Within a phase, pauses and
    barge-ins do NOT call this — the persistent session keeps its context so
    partials extend continuously.
    """
    srv._asr_quench_pending += 1
    srv.asr_cmd_queue.put({"cmd": "stop"})
    srv.asr_cmd_queue.put({"cmd": "start"})
    srv._asr_session_started_t = time.monotonic()
    srv._asr_speaking_secs = 0.0
    srv._phase_transcript = ""
    srv._committed_len = 0
    srv._committed_text = ""
    srv._carry = ""
    srv._current_partial = ""
    srv._latest_partial = ""
    srv._turn_best_asr = ""
    srv._committed_asr = ""
    _slog("[main.asr] hard reset (phase boundary)")


async def _rotate_asr_session(srv: StreamServer):
    """Rotate the persistent ASR session: stop -> wait final -> start.

    Called when accumulated speaking time crosses 30s. Caller must ensure
    no audio is currently being fed (i.e. user is silent). Shielded from
    cancellation so the new session is always restarted.
    """
    if srv._asr_rotating:
        return

    async def _do():
        srv._asr_rotating = True
        srv._asr_rotation_done = asyncio.Event()
        try:
            _slog(
                f"[main.asr] rotating session ({srv._asr_speaking_secs:.1f}s "
                f"speaking, transcript={len(srv._phase_transcript)}c)"
            )
            srv.asr_cmd_queue.put({"cmd": "stop"})
            try:
                await asyncio.wait_for(srv._asr_rotation_done.wait(), timeout=5)
            except asyncio.TimeoutError:
                _slog("[main.asr] rotation: stop final timed out")
        finally:
            srv.asr_cmd_queue.put({"cmd": "start"})
            srv._asr_session_started_t = time.monotonic()
            srv._asr_speaking_secs = 0.0
            srv._current_partial = ""
            srv._asr_rotating = False
            srv._asr_rotation_done = None
            _slog(f"[main.asr] rotation done, carry='{srv._carry[:60]}'")

    await asyncio.shield(_do())


def update_state_from_msg(srv: StreamServer, msg):
    msg_type = msg.get("type")
    if msg_type == "prompt_loaded":
        srv.tts_T = msg["T"]
        rms = msg.get("prompt_rms", 0)
        if rms > 0 and srv.session.playback_track:
            srv.session.playback_track.set_target_rms(rms)
    elif msg_type == "kv_loaded":
        srv.tts_T = msg.get("T", 0)
    elif msg_type == "state":
        srv.tts_T = msg.get("T", 0)
        srv.tts_max_T = msg.get("max_T", 0)


async def wait_tts_response(
    srv: StreamServer, expected_type: str, timeout: float = 30
) -> dict | None:
    q: asyncio.Queue = asyncio.Queue()
    srv._tts_response_queue = q
    try:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(q.get(), timeout=remaining)
                if msg.get("type") == expected_type:
                    return msg
                await notify_ws_from_response(srv, msg)
            except asyncio.TimeoutError:
                return None
    finally:
        srv._tts_response_queue = None


async def notify_ws_from_response(srv: StreamServer, msg):
    ws = srv.session.ws
    if not ws or ws.closed:
        return
    msg_type = msg.get("type")
    if msg_type == "prompt_loaded":
        await ws.send_json(
            {
                "type": "prompt_loaded",
                "T": msg["T"],
                "text": msg.get("text", ""),
                "max_T": srv.tts_max_T,
            }
        )
    elif msg_type == "kv_loaded" and msg.get("ok"):
        await ws.send_json(
            {
                "type": "prompt_loaded",
                "T": msg["T"],
                "text": msg.get("text", ""),
                "max_T": srv.tts_max_T,
            }
        )
    elif msg_type == "kv_saved" and msg.get("ok"):
        await ws.send_json({"type": "status", "text": f"Saved: {msg['name']}"})


async def send_turn_done(srv: StreamServer, msg: dict):
    srv._pending_done = None
    turn_secs = getattr(srv, "_turn_render_secs", 0.0)
    if turn_secs > 0:
        record_render(turn_secs)
    srv._turn_render_secs = 0.0
    ws = srv.session.ws
    if ws and not ws.closed:
        total_secs = msg.get("total_secs", 0)
        total_frames = msg.get("total_frames", 0)
        total_gen = msg.get("total_gen_time", 0)
        rtf = (total_gen / total_secs) if total_secs > 0 else 0
        await ws.send_json(
            {
                "type": "turn_done",
                "T": msg.get("T", srv.tts_T),
                "max_T": srv.tts_max_T,
                "total_secs": round(total_secs, 1),
                "total_frames": total_frames,
                "rtf": round(1 / rtf, 1) if rtf > 0 else 0,
                "conv_ctx": srv.conv_ctx,
                "conv_ctx_max": srv.conv_ctx_max,
                "thoughts_ctx": srv.thoughts_ctx,
                "thoughts_ctx_max": srv.thoughts_ctx_max,
            }
        )


async def _wait_and_send_turn_done(srv: StreamServer, msg: dict):
    if srv.session.playback_track:
        await srv.session.playback_track.wait_drained()
    await send_turn_done(srv, msg)


async def flush_pending_done(srv: StreamServer):
    if (
        srv._pending_done
        and not srv._llm_streaming
        and srv._generates_done >= srv._generates_sent
    ):
        msg = srv._pending_done
        srv._pending_done = None
        _spawn(_wait_and_send_turn_done(srv, msg), "wait_and_send_turn_done")


async def drain_tts_audio(srv: StreamServer):
    loop = asyncio.get_event_loop()
    while True:
        try:
            msg = await loop.run_in_executor(
                None, lambda: srv.tts_audio_queue.get(timeout=0.1)
            )
        except Empty:
            await asyncio.sleep(0.01)
            continue

        msg_type = msg.get("type")

        if msg_type == "ready":
            srv.tts_ready = True
            print("[main] TTS worker ready")
            await srv._send_worker_status()
            continue

        if msg_type in srv._RESPONSE_TYPES:
            update_state_from_msg(srv, msg)
            if srv._tts_response_queue is not None:
                await srv._tts_response_queue.put(msg)
            continue

        if msg_type == "audio":
            if srv.session.playback_track:
                srv.session.playback_track.enqueue_audio(msg["data"])
            if srv._test_capture_sink is not None:
                srv._test_capture_sink.append(msg["data"])
            srv.tts_T = msg.get("T", srv.tts_T)

        elif msg_type == "timing":
            ft = msg
            if srv._test_capture_sink is not None:
                srv._test_first_frame_ms = ft.get("first_frame_ms", 0)
            now = time.monotonic()
            now_ms = now * 1000
            t_origin = srv._turn_t.get("vad_start", 0)
            if "tts_first_audio" not in srv._turn_t:
                srv._turn_t["tts_first_audio"] = now
            e2e_ms = (now - t_origin) * 1000 if t_origin else 0
            respond_ms = (
                now_ms - srv._turn_respond_start_ms
                if getattr(srv, "_turn_respond_start_ms", 0)
                else 0
            )
            tt = srv._turn_t
            stages = []
            for label, key in [
                ("asr_1st", "asr_first_partial"),
                ("asr_final", "asr_final"),
                ("asr_settled", "asr_settled"),
                ("eou", "eou_done"),
                ("ep_done", "endpointing_done"),
                ("respond", "respond_start"),
                ("prefill", "user_prefill_done"),
                ("llm_1st", "llm_first_chunk"),
                ("tts_audio", "tts_first_audio"),
            ]:
                if key in tt:
                    stages.append(f"{label}={( tt[key]-t_origin)*1000:.0f}")
            pipeline = " | ".join(stages)
            _slog(
                f"[main.pipeline] e2e={e2e_ms:.0f}ms respond_to_audio={respond_ms:.0f}ms "
                f"tts_first_frame={ft.get('first_frame_ms', 0):.0f}ms\n"
                f"  stages(ms): {pipeline}"
            )
            await srv._log(
                f"e2e={e2e_ms:.0f}ms | respond_to_audio={respond_ms:.0f}ms | "
                f"tts_first_frame={ft.get('first_frame_ms', 0):.0f}ms",
                "timing",
            )

        elif msg_type == "chunk_done":
            srv._turn_chunk_idx += 1
            # Track per-chunk boundary so barge-in can rewind to the end
            # of the last fully-heard chunk. Cumulative audio_secs is
            # measured against playback._frame_count to decide which
            # chunks the user actually heard.
            chunk_T = msg.get("T", srv.tts_T)
            chunk_secs = msg["secs"]
            srv._turn_render_secs = getattr(srv, "_turn_render_secs", 0.0) + chunk_secs
            prior = (
                srv._chunk_boundaries[-1][1]
                if getattr(srv, "_chunk_boundaries", None)
                else 0.0
            )
            cum_secs = prior + chunk_secs
            if not hasattr(srv, "_chunk_boundaries"):
                srv._chunk_boundaries = []
            srv._chunk_boundaries.append((chunk_T, cum_secs, msg["text"]))
            ws = srv.session.ws
            if ws and not ws.closed:
                await ws.send_json(
                    {
                        "type": "generating",
                        "chunk": srv._turn_chunk_idx,
                        "text": msg["text"],
                        "secs": round(msg["secs"], 1),
                    }
                )
            secs = msg["secs"]
            frames = msg["frames"]
            gen_time = msg["gen_time"]
            ms_per_frame = (gen_time / frames * 1000) if frames > 0 else 0
            rtf = gen_time / secs if secs > 0 else 0
            if rtf > 0:
                text = (
                    f"Chunk {srv._turn_chunk_idx}: {secs:.1f}s audio | {frames} frames | "
                    f"{ms_per_frame:.1f}ms/frame | {1/rtf:.1f}x RT"
                )
                await srv._log(text, "timing")

        elif msg_type in ("done", "cancelled"):
            srv.tts_T = msg.get("T", srv.tts_T)

            if srv._test_capture_sink is not None and srv._test_done_event is not None:
                srv._test_gen_info = {
                    "type": msg_type,
                    "T": msg.get("T", srv.tts_T),
                    "total_secs": msg.get("total_secs", 0),
                    "total_frames": msg.get("total_frames", 0),
                    "total_gen_time": msg.get("total_gen_time", 0),
                }
                srv._test_done_event.set()

            if srv._generates_sent == 0:
                continue

            srv._generates_done += 1

            if msg_type == "cancelled":
                if srv.session.playback_track:
                    srv.session.playback_track.flush()
                await send_turn_done(srv, msg)
            elif not srv._llm_streaming and srv._generates_done >= srv._generates_sent:
                _spawn(_wait_and_send_turn_done(srv, msg), "wait_and_send_turn_done")
            else:
                srv._pending_done = msg

        elif msg_type == "user_chunk_prefilled":
            n = msg.get("frames", 0)
            is_final = msg.get("final", False)
            print(f"[main] User chunk prefilled: {n} frames, final={is_final}")

        elif msg_type == "codes_final":
            srv._last_codes_final_n = msg.get("n_frames", 0)
            srv._last_codes_final_t = time.monotonic()
            srv._codes_final_event.set()
            _slog(
                f"[main.codes] codes_final received: {srv._last_codes_final_n} frames"
            )

        elif msg_type == "sc_prob":
            srv.tts_T = msg.get("T", srv.tts_T)

        elif msg_type == "context_reset":
            srv.tts_T = msg.get("T", 0)
            await srv._log(f"Context reset, T={srv.tts_T}", "warn")

        elif msg_type == "error":
            await srv._log(f"TTS: {msg.get('msg')}", "error")


async def drain_asr_results(srv: StreamServer):
    loop = asyncio.get_event_loop()
    while True:
        try:
            msg = await loop.run_in_executor(
                None, lambda: srv.asr_result_queue.get(timeout=0.1)
            )
        except Empty:
            await asyncio.sleep(0.01)
            continue

        msg_type = msg.get("type")

        if msg_type == "ready":
            srv.asr_ready = True
            print("[main] ASR worker ready")
            if srv._server_vad:
                srv.asr_cmd_queue.put({"cmd": "vad_enable"})
            await srv._send_worker_status()
            continue

        if msg_type == "backend_set":
            srv._asr_backend_set_result = msg
            if srv._asr_backend_set_event:
                srv._asr_backend_set_event.set()
            continue

        if msg_type == "session_ready":
            if srv._asr_quench_pending > 0:
                srv._asr_quench_pending -= 1
                _slog(
                    f"[main.asr] session_ready (quench_pending="
                    f"{srv._asr_quench_pending})"
                )
            continue

        # Discard any session-emitted text from a torn-down session.
        # Cleared by `session_ready` once the new session is live.
        if (
            srv._asr_quench_pending > 0
            and msg_type in ("partial", "stable_prefix", "line_completed", "final")
        ):
            _slog(f"[main.asr] dropped stale {msg_type} (quenched)")
            continue

        ws = srv.session.ws

        if msg_type == "vad_audio_ready":
            _slog("[main.vad] mic audio flowing")
            if ws and not ws.closed:
                try:
                    await ws.send_json({"type": "mic_live"})
                except Exception:
                    pass
            # Gate session-ready on the audio pipeline actually flowing,
            # not just on warmup. Without this `workers_ready` can fire
            # before the first VAD chunk lands and the user can press
            # Talk before the server can hear them.
            await srv._unblock_ready("mic")
            continue

        if msg_type == "vad_start":
            if ws and not ws.closed:
                await ws.send_json({"type": "vad_start"})
            # User resumed speaking — clear any deferred commit so it doesn't
            # fire on the next floor release.
            srv._floor_pending_dur = None
            if not srv._server_vad or not srv.session.recording_sink:
                continue

            tts_playing = bool(
                srv.session.playback_track and srv.session.playback_track.can_pause
            )
            barging_in = (
                srv._llm_streaming
                or srv._generates_done < srv._generates_sent
                or tts_playing
            )
            commit_pending = bool(
                srv._endpointing_task and not srv._endpointing_task.done()
            )

            if not srv.session.ready and not barging_in:
                _slog(
                    f"[main] VAD: speech detected but not ready "
                    f"(ready={srv.session.ready} barging_in={barging_in})"
                )
                continue

            # Always cancel any pending tiered_commit and pending phase-end
            # reset — user is continuing or interrupting, either way we
            # don't want a stale commit/reset to fire.
            if srv._endpointing_task and not srv._endpointing_task.done():
                srv._endpointing_task.cancel()
                srv._endpointing_task = None
            if srv._phase_end_task and not srv._phase_end_task.done():
                srv._phase_end_task.cancel()
                srv._phase_end_task = None

            srv._asr_speech_start_t = time.monotonic()

            # Three start scenarios:
            #   A) commit_pending:  user resumed during tiered_commit's
            #      settle / trailing-off wait. Keep _phase_transcript +
            #      worker's incremental KV state intact. If codec was
            #      stopped (post-stop_recording), restart with
            #      continuation=True so prior chunks stay valid.
            #   B) barging_in:      assistant is replying. Cancel TTS,
            #      revert the assistant turn, hard-reset ASR + restart
            #      codec fresh — the interrupting speech is a NEW turn,
            #      so any prior transcript (committed user text + any
            #      assistant TTS bleed picked up by the mic) must not
            #      leak into the next commit.
            #   C) fresh:           idle → speaking. Hard-reset ASR + KV
            #      tracking, start a new turn.
            if commit_pending:
                if not srv.session.recording_sink.recording:
                    srv._codes_final_event.clear()
                    srv.session.recording_sink.start_recording(continuation=True)
                pause_ms = (
                    time.monotonic() - srv._turn_t.get("vad_stop", time.monotonic())
                ) * 1000
                _slog(
                    f"[main.vad] resumed during commit_wait "
                    f"+{pause_ms:.0f}ms transcript='{srv._phase_transcript[:60]}'"
                )
            elif barging_in:
                srv._last_codes_final_n = 0
                srv._codes_final_event.clear()
                srv._user_chunks_prefilled = 0
                _hard_reset_asr(srv)
                # Force codec reset: stop if still recording, then start
                # fresh. `start_recording` issues `stream_start` with
                # continuation=False, which resets the streaming codec
                # encoder and `_user_codes_parts` / `_user_text_prefilled`
                # in the TTS worker — so no prior-turn audio leaks into
                # the new user prefill.
                if srv.session.recording_sink.recording:
                    srv.session.recording_sink.stop_recording()
                srv.session.recording_sink.start_recording()
                srv._turn_t = {"vad_start": time.monotonic()}
            elif not srv.session.recording_sink.recording:
                srv._last_codes_final_n = 0
                srv._codes_final_event.clear()
                srv._user_chunks_prefilled = 0
                _hard_reset_asr(srv)
                srv.session.recording_sink.start_recording()
                srv._turn_t = {"vad_start": time.monotonic()}
            else:
                continue

            srv.session.recording_sink.speaking = True
            _slog(
                f"[main.vad] speech start "
                f"(commit_pending={commit_pending} barging_in={barging_in} "
                f"tts_playing={tts_playing})"
            )
            await srv._log("VAD: speech detected", "info")
            if srv.session.playback_track:
                srv.session.playback_track.flush()

            # Barge-in: always revert. If the LLM was logically done and
            # TTS was just draining, the user is unlikely to have heard
            # enough to be responding to it (and even if they were, with
            # the new tiered_commit threshold the next commit will pick
            # up a clean turn). Simpler than the old _llm_streaming
            # heuristic.
            if barging_in and srv.session.cancel_generation is False:
                srv.session.cancel_generation = True
                srv.tts_cancel_event.set()
                # On barge-in we revert past the current turn entirely so
                # half-spoken assistant chunks don't poison the next turn's
                # KV context. Two flavors, picked by `keep_context`:
                #   - keep_context=True: rewind to `_pre_turn_T` — the T
                #     captured at the start of voice_respond, i.e. the END
                #     of the last completed assistant response. Prior
                #     conversation stays in KV.
                #   - keep_context=False: full `rewind` to the loaded
                #     prompt. Cleaner slate. (`_pre_turn_T` is invalid
                #     here because voice_respond rewinds to prompt at the
                #     start of the turn, leaving _pre_turn_T pointing
                #     past current T into wiped KV.)
                #
                # We do NOT touch session.conversation — voice_respond's
                # cleanup path appends `full_reply` (everything the LLM
                # emitted) so the chat history stays consistent with what
                # the LLM actually said.
                keep_context = srv.session.settings.get("keep_context", False)
                pre_turn_T = getattr(srv, "_pre_turn_T", 0)

                if keep_context and pre_turn_T > 0:
                    srv.tts_cmd_queue.put(
                        {"cmd": "cancel", "rewind_to": pre_turn_T}
                    )
                    _slog(
                        f"[main.vad] cancel + rewind to pre_turn_T={pre_turn_T} "
                        f"(keep_context=True)"
                    )
                else:
                    srv.tts_cmd_queue.put({"cmd": "cancel"})
                    srv.tts_cmd_queue.put({"cmd": "rewind"})
                    _slog("[main.vad] cancel + full rewind to prompt")
                srv.session.ready = not srv._ready_blockers
            continue

        if msg_type == "vad_stop":
            if ws and not ws.closed:
                await ws.send_json({"type": "vad_stop"})
            if (
                srv._server_vad
                and srv.session.recording_sink
                and srv.session.recording_sink.recording
            ):
                if srv.session.recording_sink.speaking:
                    srv._asr_speaking_secs += time.monotonic() - srv._asr_speech_start_t
                srv.session.recording_sink.speaking = False
                rec_dur = srv.session.recording_sink.rec_duration
                t_now = time.monotonic()
                srv._turn_vad_stop_ms = t_now * 1000
                srv._turn_t["vad_stop"] = t_now
                await srv._log(f"VAD: silence ({rec_dur:.2f}s)", "info")
                if srv._asr_speaking_secs >= 30.0 and not srv._asr_rotating:
                    _spawn(_rotate_asr_session(srv), "asr_rotate")
                if srv._floor_held:
                    # User is holding the floor — defer the commit until
                    # they release. Audio keeps flowing (recording_sink.
                    # recording is still True), so when they resume the
                    # transcript just extends. On release, the connection
                    # handler fires _tiered_commit with this rec_dur.
                    srv._floor_pending_dur = rec_dur
                    _slog(
                        f"[main] VAD: silence ({rec_dur:.2f}s) — floor held, "
                        f"deferring commit"
                    )
                elif rec_dur >= 0.3:
                    _slog(f"[main] VAD: silence ({rec_dur:.2f}s), tiered commit")
                    # Tiered: punctuation → commit fast; trailing-off →
                    # wait an extra 1s in case user resumes. Cancellable
                    # via _endpointing_task on next vad_start.
                    srv._endpointing_task = _spawn(
                        srv._tiered_commit(rec_dur), "tiered_commit"
                    )
                else:
                    if ws and not ws.closed:
                        await ws.send_json(
                            {"type": "transcription", "text": "(too short)"}
                        )
            continue

        if msg_type == "partial":
            partial_text = msg["text"]
            srv._current_partial = partial_text
            srv._phase_transcript = _join_carry(srv._carry, partial_text)
            srv._latest_partial = srv._phase_transcript
            if len(srv._phase_transcript) > len(srv._turn_best_asr):
                srv._turn_best_asr = srv._phase_transcript
            if "asr_first_partial" not in srv._turn_t and "vad_start" in srv._turn_t:
                srv._turn_t["asr_first_partial"] = time.monotonic()
                t0 = srv._turn_t["vad_start"]
                _slog(
                    f"[main.asr] first partial: '{partial_text[:60]}' +{(time.monotonic()-t0)*1000:.0f}ms"
                )
            else:
                _slog(f"[asr] {partial_text}")
            srv._turn_t["asr_last_partial"] = time.monotonic()
            if ws and not ws.closed:
                await ws.send_json(
                    {"type": "partial_asr", "text": srv._phase_transcript}
                )
            # Speculative prefill is fired on `line_completed` (one call
            # per stable sentence), not on every partial. With the new
            # tiered_commit threshold there's enough lead time for the
            # warmup to land before commit_turn fires.

        elif msg_type == "line_completed":
            line_text = msg.get("text", "")
            srv._turn_t["asr_line_completed"] = time.monotonic()
            if "vad_start" in srv._turn_t:
                _slog(
                    f"[main.asr] line completed: '{line_text[:60]}' +{(time.monotonic()-srv._turn_t['vad_start'])*1000:.0f}ms"
                )
            else:
                _slog(f"[main.asr] line completed: '{line_text[:60]}'")
            # The ASR backend's partial already includes line_completed text,
            # so we don't append separately. Just update the speculative prefill.
            merged = _join_carry(srv._carry, line_text)
            srv._last_asr_text = merged
            if len(merged) >= len(srv._turn_best_asr):
                srv._turn_best_asr = merged
            # Speculative LLM/thoughts prefill: fire on line_completed (one
            # call per stable sentence) so Ollama's KV is warm by the time
            # tiered_commit lands. Throttle by char-growth + min interval to
            # avoid stacking redundant ~500ms prefills.
            now = time.monotonic()
            grew = len(merged) >= len(srv._last_prefill_text or "") + 20
            if (
                len(merged.split()) >= 3
                and grew
                and (now - srv._last_prefill_t) > 1.5
                and merged != srv._last_prefill_text
            ):
                srv._last_prefill_text = merged
                srv._last_prefill_t = now
                _spawn(srv._llm_speculative_prefill(merged), "llm_spec_prefill")
                _spawn(
                    srv._thoughts.speculative_prefill(merged), "thoughts_spec_prefill"
                )
            if len(merged) > len(srv._committed_asr):
                srv._committed_asr = merged
                if ws and not ws.closed:
                    await ws.send_json({"type": "committed_asr", "text": merged})

        elif msg_type == "stable_prefix":
            stable_text = msg.get("text", "")
            stable_words = msg.get("stable_words", 0)
            total_words = msg.get("total_words", 0)
            stable_end_time = msg.get("stable_end_time", 0.0)
            _slog(
                f"[main.asr] stable_prefix: {stable_words}/{total_words}w "
                f"end_t={stable_end_time:.3f}s '{stable_text[:50]}'"
            )
            # Speculative LLM prefill: fire on stable_prefix too so we get
            # real lead time. line_completed only fires when fwhisper sees
            # `.!?` in interim text, which often lands at end-of-speech —
            # too late for speculation. Shares throttle state with the
            # line_completed branch so we don't double-fire. In
            # speculative_reply mode the spec task in llm_speculative_prefill
            # cancels + replaces on each new trigger.
            now = time.monotonic()
            grew = len(stable_text) >= len(srv._last_prefill_text or "") + 20
            if (
                stable_text
                and len(stable_text.split()) >= 4
                and grew
                and (now - srv._last_prefill_t) > 1.5
                and stable_text != srv._last_prefill_text
            ):
                srv._last_prefill_text = stable_text
                srv._last_prefill_t = now
                _spawn(
                    srv._llm_speculative_prefill(stable_text),
                    "llm_spec_prefill_stable",
                )
            # Incremental TTS-KV prefill: when the stable prefix has grown
            # by ≥6 words AND ≥600ms since the last chunk, ship it as
            # `prefill_user_chunk` so the TTS worker can write a partial
            # `[user] "<chunk>" [<n>f]` to KV and the final commit only
            # has to handle the remainder. Throttle is critical — without
            # it fwhisper's force-commit fires every ~500ms with the same
            # growing text and the model parrots user-audio fragments back.
            if (
                srv.session.recording_sink
                and srv.session.recording_sink.recording
                and stable_text
                and stable_end_time > 0
            ):
                cur_words = stable_text.strip().split()
                last_words = (srv._last_user_chunk_text or "").strip().split()
                grew_by = len(cur_words) - len(last_words)
                now = time.monotonic()
                elapsed = now - srv._last_user_chunk_t
                if grew_by >= 6 and (srv._last_user_chunk_t == 0.0 or elapsed >= 0.6):
                    is_first = srv._user_chunks_prefilled == 0
                    srv._last_user_chunk_text = stable_text
                    srv._last_user_chunk_t = now
                    srv._user_chunks_prefilled += 1
                    chunk_msg = {
                        "cmd": "prefill_user_chunk",
                        "text": stable_text,
                        "stable_end_time": stable_end_time,
                        "final": False,
                    }
                    # Spk token is one-shot inside the worker (idempotent
                    # `_ensure_user_spk_token`). Only ship audio_16k on the
                    # first incremental — afterwards the worker already
                    # has the projected token cached.
                    if is_first:
                        audio_16k = srv.session.recording_sink.get_audio_16k()
                        if audio_16k is not None and audio_16k.size >= 16000:
                            chunk_msg["audio_16k"] = audio_16k
                    srv.tts_cmd_queue.put(chunk_msg)
                    _slog(
                        f"[main.chunk] -> incr #{srv._user_chunks_prefilled} "
                        f"+{grew_by}w end_t={stable_end_time:.2f}s "
                        f"'{stable_text[-60:]}'"
                    )

        elif msg_type == "final":
            # Persistent ASR: "final" fires only on rotation or hard-reset.
            # Hard-reset finals are dropped by the quench gate above; only
            # rotation finals (carry-promotion) and idle finals reach here.
            text = msg.get("text", "")
            srv._turn_t["asr_final"] = time.monotonic()
            _slog(f"[main.asr] final: '{text[:60]}'")
            if srv._asr_rotating:
                # 30s rotation: promote text into carry so next partials extend.
                if text.strip():
                    srv._carry = _join_carry(srv._carry, text)
                srv._current_partial = ""
                srv._phase_transcript = srv._carry
                if srv._asr_rotation_done is not None:
                    srv._asr_rotation_done.set()
            else:
                merged = _join_carry(srv._carry, text)
                srv._last_asr_text = merged
                if len(merged) >= len(srv._turn_best_asr):
                    srv._turn_best_asr = merged

        elif msg_type == "transcribed":
            srv._last_transcription = msg.get("text", "")


def bind(cls):
    cls._update_state_from_msg = lambda self, msg: update_state_from_msg(self, msg)
    cls._wait_tts_response = lambda self, *a, **kw: wait_tts_response(self, *a, **kw)
    cls._notify_ws_from_response = lambda self, *a, **kw: notify_ws_from_response(
        self, *a, **kw
    )
    cls._send_turn_done = lambda self, *a, **kw: send_turn_done(self, *a, **kw)
    cls._flush_pending_done = lambda self: flush_pending_done(self)
    cls.drain_tts_audio = lambda self: drain_tts_audio(self)
    cls.drain_asr_results = lambda self: drain_asr_results(self)
