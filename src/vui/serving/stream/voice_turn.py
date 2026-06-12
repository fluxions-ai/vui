"""Voice turn pipeline: ASR -> LLM -> TTS streaming + endpointing."""

from __future__ import annotations

import asyncio
import base64
import re
import struct
import time
import traceback
from typing import TYPE_CHECKING

import numpy as np

from vui.serving.stream._log import _slog, _spawn, _spawn_response
from vui.serving.stream.llm import llm_stream_chunks
from vui.serving.stream.text_utils import strip_emoji

if TYPE_CHECKING:
    from vui.serving.stream.server import StreamServer


# Narrow filler set: only "uh" and "um" (with repeated-letter variants).
# Explicitly NOT "oh" / "huh" / "ahh" — those are real reactions.
_FILLER_RE = re.compile(r"^(?:u+h+m*|u+m+)$")


def _ends_with_filler(text: str) -> bool:
    """True iff the LAST word of `text` is 'uh' or 'um' (or a stretched
    variant like 'uhhh', 'umm'). Trailing punctuation is ignored.
    """
    words = re.findall(r"[A-Za-z']+", text or "")
    if not words:
        return False
    return bool(_FILLER_RE.match(words[-1].lower()))


def append_turn(srv: StreamServer, role: str, content: str) -> None:
    """Append to session.conversation with a wall-time `ts`. The `ts`
    field is stripped by `recent_conversation` before messages go to the
    LLM, so it doesn't pollute the wire format.
    """
    srv.session.conversation.append({"role": role, "content": content, "ts": time.time()})


def recent_conversation(srv: StreamServer) -> list[dict]:
    """Trim conversation to the last `context_minutes` of wall-time
    history and strip non-LLM fields (`ts`). Returns a NEW list of
    {role, content} dicts ready to send to Ollama.

    `context_minutes <= 0` → no trim, return the full conversation.
    Older turns missing `ts` (legacy entries) are kept.
    """
    convo = srv.session.conversation
    minutes = float(srv.session.settings.get("context_minutes", 0) or 0)
    if minutes <= 0:
        return [{"role": m["role"], "content": m["content"]} for m in convo]
    cutoff = time.time() - minutes * 60.0
    out: list[dict] = []
    for m in convo:
        ts = m.get("ts")
        if ts is not None and ts < cutoff:
            continue
        out.append({"role": m["role"], "content": m["content"]})
    return out


async def wait_for_asr(srv: StreamServer, timeout_iters: int = 100) -> str:
    for i in range(timeout_iters):
        if srv._last_asr_text is not None:
            text = srv._last_asr_text
            srv._last_asr_text = None
            print(f"[main] ASR ready after {i*100}ms: '{text}'")
            return text
        await asyncio.sleep(0.1)
    print("[main] ASR timeout")
    return ""


def clean_llm_text(srv: StreamServer, text: str) -> str:
    text = strip_emoji(text)
    text = re.sub(r"\b[Hh]a(?:[\s,]*[Hh]a)+\b", "[laugh]", text)
    text = re.sub(r"\bHa\b", "[laugh]", text)
    text = text.replace('"', "").replace("`", "")
    text = re.sub(
        r"\[(?!breath\]|hesitate\]|laugh\]|sigh\]|gasp\]|cough\])[^\]]*\]", "", text
    )
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(
        r"(\[(?:breath|hesitate|laugh|sigh|gasp|cough)\])[.,!?;:]+", r"\1", text
    )
    text = re.sub(r"\b(\w+)in'", r"\1ing", text)
    text = re.sub(r"[.!?]+", lambda m: m.group()[-1], text)
    text = re.sub(r"\s+([?.,!;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_orphan_brackets(srv: StreamServer, text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "[":
            close = text.find("]", i + 1)
            next_open = text.find("[", i + 1)
            if close == -1 or (next_open != -1 and next_open < close):
                j = i + 1
                while j < n and text[j].isalpha() and j - i < 16:
                    j += 1
                i = j
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _norm_for_spec_match(s: str) -> str:
    """Strict-prefix match key for speculative reply lookup. ASR partials
    rarely have terminal punctuation; the committed text often does. Strip
    trailing punct + whitespace, lowercase. Anything stricter (token-level)
    is overkill for a heuristic gate."""
    return (s or "").rstrip(" .?!,").lower()


async def _spec_stream_reply(srv: StreamServer, partial_text: str):
    """Stream a full LLM reply for `partial_text` into srv._spec_reply.
    Runs as a background task; voice_respond awaits it on commit if the
    final transcript matches.
    """
    from vui.serving.stream.llm_backend import get_backend
    from vui.serving.stream.prompts import build_memory_context

    t0 = time.monotonic()
    sys_prompt = srv.session.soul
    mem_ctx = build_memory_context(srv)
    if mem_ctx:
        sys_prompt = sys_prompt + "\n\n" + mem_ctx
    messages = (
        [{"role": "system", "content": sys_prompt}]
        + recent_conversation(srv)
        + [{"role": "user", "content": partial_text}]
    )
    buf: list[str] = []
    try:
        async for tok in get_backend().stream(messages, max_tokens=2048):
            buf.append(tok)
            srv._spec_reply = "".join(buf)
        srv._spec_reply_done = True
        _slog(
            f"[main.llm] spec reply done {(time.monotonic()-t0)*1000:.0f}ms "
            f"n={len(srv._spec_reply)}c text='{partial_text[:40]}'"
        )
    except asyncio.CancelledError:
        _slog(
            f"[main.llm] spec reply cancelled +{(time.monotonic()-t0)*1000:.0f}ms "
            f"text='{partial_text[:40]}'"
        )
        raise
    except Exception as e:
        _slog(f"[main.llm] spec reply error: {e}")


async def llm_speculative_prefill(srv: StreamServer, partial_text: str):
    if bool(srv.session.settings.get("speculative_reply", False)):
        # Full-reply speculation: cancel any in-flight spec for older
        # partial text and start fresh. Unlike the KV-warm path (1-token
        # completion), we can't just skip on collision — a stale stream
        # would keep growing into the buffer for text the user has
        # already revised past, then get matched by the strict-equality
        # check below in voice_respond.
        if srv._spec_task and not srv._spec_task.done():
            srv._spec_task.cancel()
            try:
                await srv._spec_task
            except (asyncio.CancelledError, Exception):
                pass
        srv._spec_reply_for_text = partial_text
        srv._spec_reply = ""
        srv._spec_reply_done = False
        srv._spec_task = asyncio.create_task(_spec_stream_reply(srv, partial_text))
        return

    # Single in-flight gate — see StreamServer._prefill_inflight. Without
    # this, slow LLMs stack up multiple prefills per turn, all useless.
    if srv._prefill_inflight:
        return
    srv._prefill_inflight = True
    t0 = time.monotonic()
    try:
        # Soul MUST match the actual chat call exactly so the backend's
        # KV cache for the system prefix doesn't miss on the real call
        # and re-prefill the entire conversation from scratch.
        from vui.serving.stream.llm_backend import get_backend
        from vui.serving.stream.prompts import build_memory_context

        sys_prompt = srv.session.soul
        mem_ctx = build_memory_context(srv)
        if mem_ctx:
            sys_prompt = sys_prompt + "\n\n" + mem_ctx
        messages = (
            [{"role": "system", "content": sys_prompt}]
            + recent_conversation(srv)
            + [{"role": "user", "content": partial_text}]
        )
        await get_backend().prefill(messages)
        _slog(
            f"[main.llm] speculative prefill done {(time.monotonic()-t0)*1000:.0f}ms text='{partial_text[:40]}'"
        )
    except Exception:
        pass
    finally:
        srv._prefill_inflight = False


async def _chunks_from_buffer(text: str, max_words: int):
    """Yield (chunk, is_final) over a complete buffered LLM reply, using
    the same sentence-end / word-count splitter as llm_stream_chunks.
    Lets voice_respond reuse its chunk loop unchanged when consuming a
    speculated reply.
    """
    from vui.serving.stream.llm import _SENT_END_RE

    buf = text
    pending: str | None = None
    while True:
        m = _SENT_END_RE.search(buf)
        if m is None:
            break
        chunk = buf[: m.end()].strip()
        buf = buf[m.end() :].lstrip()
        if not chunk:
            continue
        if pending is not None:
            yield pending, False
        pending = chunk
    while len(buf.split()) >= max_words:
        words = buf.split()
        chunk = " ".join(words[:max_words]).rstrip(",;:")
        buf = " ".join(words[max_words:])
        if buf:
            buf = " " + buf
        if chunk:
            if pending is not None:
                yield pending, False
            pending = chunk
    trailing = buf.strip()
    while trailing and trailing[-1] in ",;:- ":
        trailing = trailing[:-1].rstrip()
    if trailing and trailing[-1] not in ".!?\"')]":
        trailing = trailing + "."
    if pending is not None and trailing:
        yield pending, False
        yield trailing, True
    elif pending is not None:
        yield pending, True
    elif trailing:
        yield trailing, True


async def voice_respond(srv: StreamServer, asr_text: str, user_codes, audio_16k=None):
    if not srv.session.ready:
        return
    srv.session.ready = False
    try:
        await _voice_respond_body(srv, asr_text, user_codes, audio_16k)
    except asyncio.CancelledError:
        _slog("[main.turn] voice_respond cancelled — releasing ready")
        raise
    finally:
        # Belt-and-braces: any exit path (normal, exception, cancellation)
        # must restore ready=True, otherwise commit_turn → voice_respond
        # silently no-ops on subsequent turns and the assistant goes mute.
        srv._llm_streaming = False
        srv.session.ready = not srv._ready_blockers


async def _voice_respond_body(
    srv: StreamServer, asr_text: str, user_codes, audio_16k=None
):
    # Filler trailing-off gate. If the user's turn ends in "uh" / "um"
    # (or a stretched variant) AND was short (<3s of audio), they're
    # almost certainly mid-thought — drop the reply, let the next turn
    # pick up the continuation. The user message stays in conversation
    # history so the LLM has context on the next real turn. Long turns
    # (>=3s) fall through regardless.
    if _ends_with_filler(asr_text):
        audio_secs = (audio_16k.shape[-1] / 16000) if audio_16k is not None else 0
        if audio_secs < 3.0:
            _slog(
                f"[filler] dropped reply ({audio_secs:.1f}s): {asr_text!r}"
            )
            return

    # Listen-mode gate. Deterministic keyword match — the assistant's name
    # (fuzzy / phonetic variants, anywhere in the transcript) or an
    # explicit wake phrase ("wake up", "come back", "you can talk again")
    # flips listen mode off and lets this turn reply. Otherwise drop the
    # reply (ASR + VAD + prefill keep running upstream so the user can
    # still wake it on the next turn).
    from vui.serving.stream import wake_word

    if getattr(srv, "_listen_mode", False):
        matched, stripped = wake_word.match(asr_text, srv.session.assistant_name)
        if not matched:
            _slog(f"[listen] no wake word, dropping reply: {asr_text!r}")
            return
        srv._listen_mode = False
        _slog(f"[listen] disabled (wake match: {asr_text!r})")
        ws_for_lm = srv.session.ws
        if ws_for_lm and not ws_for_lm.closed:
            try:
                await ws_for_lm.send_json({"type": "listen_mode", "enabled": False})
            except Exception:
                pass
        # Strip the wake word from the conversation entry so the LLM
        # doesn't see "Vui, what time?" — just "what time?".
        if stripped:
            if srv.session.conversation and (
                srv.session.conversation[-1].get("role") == "user"
            ):
                srv.session.conversation[-1]["content"] = stripped
            asr_text = stripped
    elif wake_word.shut_up_match(asr_text):
        # Inverse of the wake gate: deterministic shut-up phrases engage
        # listen mode every time. Skip the LLM and TTS a single canned
        # acknowledgement so the assistant doesn't ramble before going quiet.
        srv._listen_mode = True
        _slog(f"[listen] enabled (shut-up match: {asr_text!r})")
        ws_for_lm = srv.session.ws
        if ws_for_lm and not ws_for_lm.closed:
            try:
                await ws_for_lm.send_json({"type": "listen_mode", "enabled": True})
            except Exception:
                pass
        ack = "Okay."
        srv.session.conversation.append({"role": "assistant", "content": ack})
        if ws_for_lm and not ws_for_lm.closed:
            try:
                await ws_for_lm.send_json({"type": "reply", "text": ack})
            except Exception:
                pass
        srv.tts_cmd_queue.put(
            {
                "cmd": "generate",
                "text": ack,
                "is_voice": True,
                "new_turn": True,
                "is_final": True,
                "settings": srv.session.settings,
                "context": "",
            }
        )
        return

    srv._pre_turn_T = srv.tts_T
    # Cleared at start of turn so a barge-in fired BEFORE the user prefill
    # completes can't read a stale _post_user_T from a prior turn.
    srv._post_user_T = None
    srv._chunk_boundaries = []
    ws = srv.session.ws

    last_assistant = ""
    for m in reversed(srv.session.conversation):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break

    t0 = time.monotonic()
    t_vad_stop = srv._turn_vad_stop_ms / 1000 if srv._turn_vad_stop_ms else t0
    srv._turn_respond_start_ms = t0 * 1000
    srv._turn_t["respond_start"] = t0
    t_origin = srv._turn_t.get("vad_start", t0)
    _slog(
        f"[main.turn] respond start +{(t0-t_origin)*1000:.0f}ms "
        f"(vad_stop+{(t0-t_vad_stop)*1000:.0f}ms)"
    )
    srv._log_conv(
        "user_turn", text=asr_text, conversation=srv.session.conversation.copy()
    )
    await ws.send_json({"type": "status", "text": "Thinking..."})
    keep_context = srv.session.settings.get("keep_context", False)
    user_audio_flag = srv.session.settings.get("user_audio")
    s = srv.session.settings
    _slog(
        f"[main.resp] turn={len(srv.session.conversation)} "
        f"keep_context={keep_context} user_audio={user_audio_flag} "
        f"user_chunks_prefilled={srv._user_chunks_prefilled} "
        f"asr_text='{asr_text[:60]}' "
        f"settings(temp={s.get('temperature')} top_k={s.get('top_k')} "
        f"rep={s.get('rep_penalty')}/w{s.get('rep_window')} "
        f"eos={s.get('eos_threshold')} nq={s.get('n_codebooks')} "
        f"chunk_w={s.get('chunk_words')} max_dur={s.get('max_duration')})"
    )

    # Single commit path: prefill_user_chunk(final=True) for every turn.
    # Worker dedupes against its own _user_text_prefilled so passing the
    # full asr_text is safe whether or not incremental chunks fired during
    # speech. Always ship audio_16k on the final — worker's
    # `_ensure_user_spk_token` is idempotent (skips if already cached).
    # Without this, a first-incremental that didn't qualify (audio < 1s,
    # or embedding failed) leaves _user_spk_token=None for the whole turn,
    # silently degrading WER since the [user_spk] prefix is missing.
    if not keep_context:
        srv.tts_cmd_queue.put({"cmd": "rewind"})
    prefill_msg = {
        "cmd": "prefill_user_chunk",
        "text": asr_text,
        "final": True,
    }
    if audio_16k is not None:
        prefill_msg["audio_16k"] = audio_16k
    _slog(
        f"[main.prefill] -> prefill_user_chunk(final=True): "
        f"text={len(asr_text.split())}w '{asr_text[:60]}', "
        f"prior_chunks={srv._user_chunks_prefilled}"
    )
    srv.tts_cmd_queue.put(prefill_msg)
    t_prefill_wait = time.monotonic()
    await srv._wait_tts_response("user_prefilled", timeout=10)
    t_prefill_done = time.monotonic()

    srv._turn_t["user_prefill_done"] = t_prefill_done
    t_origin = srv._turn_t.get("vad_start", t0)
    _slog(
        f"[main.turn] user prefill done +{(t_prefill_done-t_origin)*1000:.0f}ms "
        f"(prefill_wait={( t_prefill_done - t_prefill_wait)*1000:.0f}ms)"
    )

    # Snapshot T after the user prefill — barge-in rewinds to here when
    # no agent chunks were heard yet (so the user's text stays in KV).
    srv._post_user_T = srv.tts_T
    # Per-chunk boundaries: filled by drains.py chunk_done. Each entry is
    # (T_after_chunk, cumulative_audio_secs, chunk_text). On barge-in we
    # rewind to the boundary of the LAST fully-heard chunk, preserving
    # the audibly-delivered portion of the assistant turn.
    srv._chunk_boundaries = []

    srv.session.cancel_generation = False
    srv._thoughts_stop_llm = False
    srv._generates_sent = 0
    srv._generates_done = 0
    srv._turn_chunk_idx = 0
    srv._llm_streaming = True
    full_reply = []

    try:
        chunk_carry = ""
        t_llm_start = time.monotonic()
        chunk_w = int(srv.session.settings.get("chunk_words", 20))
        from vui.serving.stream.prompts import build_memory_context

        mem_ctx = build_memory_context(srv)
        sys_prompt = srv.session.soul
        if mem_ctx:
            sys_prompt = sys_prompt + "\n\n" + mem_ctx
        conv = recent_conversation(srv)
        llm_stats: dict = {}

        # Speculative-reply consumer. If a background spec stream was
        # fired for a partial transcript that's a bounded prefix of the
        # committed asr_text, await it and pump its buffered text through
        # the same chunker instead of starting a fresh LLM stream. Prefix
        # (not equality) because the last ASR stable_prefix almost never
        # includes the trailing few words — by commit time the user has
        # added "you know" / "today" / etc. We tolerate up to ~30 chars
        # of trailing content (≈5 words); beyond that the user likely
        # added new semantic content the LLM didn't see, so fall back.
        spec_text = ""
        if bool(srv.session.settings.get("speculative_reply", False)):
            spec_for = _norm_for_spec_match(srv._spec_reply_for_text)
            real_for = _norm_for_spec_match(asr_text)
            extra = len(real_for) - len(spec_for)
            prefix_ok = (
                bool(spec_for)
                and real_for.startswith(spec_for)
                and 0 <= extra <= 30
            )
            if prefix_ok and srv._spec_task is not None:
                _slog(
                    f"[main.spec] match — awaiting spec task "
                    f"(done={srv._spec_reply_done}, n={len(srv._spec_reply)}c)"
                )
                try:
                    await srv._spec_task
                except (asyncio.CancelledError, Exception):
                    pass
                if srv._spec_reply_done and srv._spec_reply.strip():
                    spec_text = srv._spec_reply
                    _slog(
                        f"[main.spec] using buffered reply ({len(spec_text)}c) "
                        f"+{(time.monotonic()-t_llm_start)*1000:.0f}ms"
                    )
            else:
                _slog(
                    f"[main.spec] miss (spec_for='{spec_for[:40]}' "
                    f"real_for='{real_for[:40]}')"
                )
                # Miss: an in-flight spec for a different partial is now
                # useless and would compete with the real LLM stream for
                # backend bandwidth. Cancel before falling through.
                if srv._spec_task and not srv._spec_task.done():
                    srv._spec_task.cancel()
                    try:
                        await srv._spec_task
                    except (asyncio.CancelledError, Exception):
                        pass
            # One-shot: clear the buffer either way so the next turn
            # starts clean. Stale buffers from a prior turn must never
            # be reused on a future false match.
            srv._spec_reply_for_text = ""
            srv._spec_reply = ""
            srv._spec_reply_done = False
            srv._spec_task = None

        stream_source = (
            _chunks_from_buffer(spec_text, chunk_w)
            if spec_text
            else llm_stream_chunks(
                conv,
                sys_prompt,
                srv.ollama_model,
                max_words=chunk_w,
                stats=llm_stats,
            )
        )
        async for chunk_text, is_done in stream_source:
            if srv.session.cancel_generation or srv._thoughts_stop_llm:
                break
            t_llm_done = time.monotonic()
            chunk_text = clean_llm_text(srv, chunk_text)

            combined = chunk_carry + chunk_text
            last_open = combined.rfind("[")
            last_close = combined.rfind("]")
            if last_open > last_close and not is_done:
                chunk_carry = combined[last_open:]
                tts_chunk = combined[:last_open].rstrip()
            else:
                chunk_carry = ""
                tts_chunk = combined
            tts_chunk = strip_orphan_brackets(srv, tts_chunk).strip()
            if not tts_chunk:
                if is_done:
                    break
                continue

            if not re.search(r"[A-Za-z0-9]", re.sub(r"\[[a-z]+\]", "", tts_chunk)):
                if is_done:
                    print(f"[main] Dropping trailing non-speech chunk: {tts_chunk!r}")
                    break
                chunk_carry = tts_chunk + " "
                continue

            chunk_text = tts_chunk
            full_reply.append(chunk_text)
            await ws.send_json({"type": "reply", "text": chunk_text})
            srv._generates_sent += 1
            t_origin = srv._turn_t.get("vad_start", t_llm_start)
            if srv._generates_sent == 1:
                srv._turn_t["llm_first_chunk"] = t_llm_done
            _slog(
                f"[main.tts#{srv._generates_sent}] new_turn={srv._generates_sent == 1} "
                f"is_final={is_done} llm_ms={(t_llm_done-t_llm_start)*1000:.0f} "
                f"+{(t_llm_done-t_origin)*1000:.0f}ms "
                f"text='{chunk_text[:60]}'"
            )
            srv.tts_cmd_queue.put(
                {
                    "cmd": "generate",
                    "text": chunk_text,
                    "is_voice": True,
                    "new_turn": srv._generates_sent == 1,
                    "is_final": is_done,
                    "settings": srv.session.settings,
                    "context": last_assistant,
                }
            )
            t_llm_start = time.monotonic()

            if is_done or srv._thoughts_stop_llm:
                break

            while (
                srv._generates_done < srv._generates_sent
                and not srv.session.cancel_generation
                and not srv._thoughts_stop_llm
            ):
                await asyncio.sleep(0.005)

    except Exception as e:
        await srv._log(f"LLM error: {e}", "error")
        traceback.print_exc()

    srv._llm_streaming = False
    if "ctx_used" in llm_stats:
        srv.conv_ctx = llm_stats["ctx_used"]
        srv.conv_ctx_max = llm_stats["ctx_max"]

    if srv.session.cancel_generation or srv._thoughts_stop_llm:
        _slog(
            f"[main.turn] voice respond stopped "
            f"(cancel={srv.session.cancel_generation} "
            f"thoughts={srv._thoughts_stop_llm})"
        )
        # Simplification: assume everything the LLM streamed has been
        # heard. If full_reply is non-empty, append it as the assistant
        # turn so subsequent turns know what was said. drains.py barge-in
        # mirrors this with _generates_sent > 0 to decide whether to roll
        # the user turn back (no LLM chunks → merge) or keep it (chunks
        # generated → preserved as a normal turn).
        if full_reply:
            heard_text = " ".join(full_reply).strip()
            if heard_text:
                append_turn(srv, "assistant", heard_text)
                srv._log_conv(
                    "llm_reply_truncated",
                    text=heard_text,
                    model=srv.ollama_model,
                )
                _slog(
                    f"[main.turn] kept LLM-streamed reply ({len(heard_text)}c, "
                    f"{len(full_reply)} chunks): '{heard_text[:60]}'"
                )
        if not srv._thoughts_stop_llm:
            srv.session.ready = not srv._ready_blockers
        return

    await srv._flush_pending_done()
    reply_text = " ".join(full_reply)
    append_turn(srv, "assistant", reply_text)
    srv._log_conv("llm_reply", text=reply_text, model=srv.ollama_model)

    srv.session.ready = not srv._ready_blockers

    if srv._phase_end_task and not srv._phase_end_task.done():
        srv._phase_end_task.cancel()
    srv._phase_end_task = _spawn(_phase_end_after_drain(srv), "phase_end")

    await srv._thoughts.on_user_turn(asr_text)

    if srv._pending_task_results:
        _spawn_response(
            srv, srv._deliver_pending_task_results(), "deliver_task_results"
        )
    elif srv._pending_task_id:
        _spawn(srv._thoughts._prefill_conversation_kv(), "thoughts_prefill_kv")


async def _phase_end_after_drain(srv: StreamServer, grace: float = 1.5):
    """After assistant TTS drains + grace period without new speech, hard-reset
    the ASR session and transcript accumulator so the next user turn starts
    from a clean context.

    Cancelled by drains.py vad_start when the user resumes within grace.
    """
    try:
        if srv.session.playback_track:
            await srv.session.playback_track.wait_drained()
        await asyncio.sleep(grace)
    except asyncio.CancelledError:
        return
    from vui.serving.stream.drains import _hard_reset_asr

    _hard_reset_asr(srv)
    _slog("[main.phase] grace expired, ASR hard-reset")


# Tiered commit thresholds. silero stop_secs=0.3 burns the first 300ms
# of silence (= time-to-vad_stop). After vad_stop fires:
#   - ASR settle: ~120ms for fwhisper's last interim to land
#   - Punctuation `.?!` → commit immediately
#   - Otherwise → wait `_TRAILING_OFF_DELAY` extra in case user resumes
# Total commit latency from "user stops speaking":
#   - clean sentence: 0.3 + 0.12 = 0.42s
#   - trailing off:   0.3 + 0.12 + 0.7 = 1.12s
# Barge-in revert covers premature commits: if the user resumes within
# the trailing-off window, tiered_commit is cancelled and codec restarts
# with continuation=True so accumulated state is preserved.
_ASR_SETTLE_S = 0.12
_TRAILING_OFF_DELAY = 0.7
_END_OF_TURN_PUNCT = (".", "?", "!")
# Defaults above are fallbacks; live values come from session.settings
# (see DEFAULT_SETTINGS in server.py: asr_settle_s, trailing_off_delay).


def _ends_a_sentence(text: str) -> bool:
    """Cheap end-of-turn proxy: trailing `.?!`. fwhisper sometimes glues a
    period onto trailing-off speech ("…want to."), but punctuation is still
    a much better signal than fixed-time alone, and the trailing-off branch
    catches the rest."""
    text = text.rstrip()
    return bool(text) and text.endswith(_END_OF_TURN_PUNCT)


async def tiered_commit(srv: StreamServer, rec_dur: float):
    """vad_stop → tiered commit. Replaces the old delayed_turn_end which
    juggled EOU + DynamicEndpointing + asr_settle polling. With silero
    stop_secs=0.5 there's already 500ms of silence elapsed by the time
    we land here; we just need a brief settle + a punctuation check.

    Cancellable: if the user resumes during the trailing-off wait,
    drains.py vad_start cancels this task. The codec is already stopped
    in that case — the resumed_after_pause path restarts it with
    continuation=True so accumulated chunks stay in KV.
    """
    t0 = srv._turn_t.get("vad_start", 0)
    settings = srv.session.settings
    asr_settle = float(settings.get("asr_settle_s", _ASR_SETTLE_S))
    trailing_off = float(settings.get("trailing_off_delay", _TRAILING_OFF_DELAY))
    try:
        # Brief settle for the last fwhisper interim to land.
        await asyncio.sleep(asr_settle)

        if srv.session.recording_sink and srv.session.recording_sink.recording:
            rec_dur = srv.session.recording_sink.stop_recording()
            _slog(f"[main.turn] stopped codec stream ({rec_dur:.2f}s)")

        # Wait for codec_final so the worker has the re-encoded full
        # segment in `_user_codes_parts` before commit_turn fires.
        try:
            await asyncio.wait_for(srv._codes_final_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            _slog("[main.turn] codes_final timeout — committing anyway")

        snapshot = (srv._phase_transcript or "").strip()
        if not snapshot:
            _slog("[main.turn] empty transcript at vad_stop, dropping")
            return

        # Slice off already-committed prefix (commit_turn advances
        # _committed_text on the prior commit). With incremental
        # chunking the worker dedupes via its own _user_text_prefilled,
        # but commit_turn still uses this for the assistant-side
        # conversation history.
        prev_committed = getattr(srv, "_committed_text", "") or ""
        n_match = min(len(prev_committed), len(snapshot))
        common = 0
        while common < n_match and prev_committed[common] == snapshot[common]:
            common += 1
        asr_text = snapshot[common:].strip()
        srv._next_committed_text = snapshot
        srv._next_committed_len = len(snapshot)
        if not asr_text:
            _slog("[main.turn] no new transcript at commit, dropping")
            return

        if not _ends_a_sentence(asr_text):
            # No clear sentence boundary — give the user a window to
            # resume mid-thought before we commit. vad_start during this
            # sleep cancels the task (existing logic in drains.py).
            _slog(
                f"[main.turn] trailing off — wait {trailing_off}s "
                f"(text={asr_text[-40:]!r})"
            )
            await asyncio.sleep(trailing_off)
            # Re-snapshot in case ASR added a few more words during the wait.
            snapshot = (srv._phase_transcript or "").strip()
            asr_text = snapshot[common:].strip() or asr_text
            srv._next_committed_text = snapshot
            srv._next_committed_len = len(snapshot)

        srv._last_asr_text = asr_text
        await commit_turn(srv, rec_dur)
    except asyncio.CancelledError:
        srv._turn_t["endpointing_cancelled"] = time.monotonic()
        _slog(
            f"[main.turn] tiered commit cancelled "
            f"+{(time.monotonic()-t0)*1000:.0f}ms"
        )
        raise


async def send_user_audio(srv: StreamServer, audio_16k: np.ndarray | None):
    if audio_16k is None or len(audio_16k) == 0:
        return
    ws = srv.session.ws
    if not ws or ws.closed:
        return
    samples = (np.clip(audio_16k, -1, 1) * 32767).astype(np.int16)
    raw = samples.tobytes()
    sr = 16000
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(raw),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sr,
        sr * 2,
        2,
        16,
        b"data",
        len(raw),
    )
    b64 = base64.b64encode(header + raw).decode("ascii")
    await ws.send_json({"type": "user_audio", "audio": b64})


async def commit_turn(srv: StreamServer, rec_dur: float):
    ws = srv.session.ws
    if not ws or ws.closed:
        return

    asr_text = srv._last_asr_text or ""
    srv._last_asr_text = None

    # Advance commit pointers (text + length) to the snapshot boundary
    # captured in delayed_turn_end (NOT to len(_phase_transcript) right
    # now — the transcript may have grown since the snapshot and that
    # growth belongs to the next user turn).
    next_text = getattr(srv, "_next_committed_text", None)
    if next_text is None:
        next_text = srv._phase_transcript or ""
    srv._committed_text = next_text
    srv._next_committed_text = None
    next_len = getattr(srv, "_next_committed_len", None)
    if next_len is None:
        next_len = len(srv._phase_transcript or "")
    srv._committed_len = next_len
    srv._next_committed_len = None

    await ws.send_json(
        {"type": "transcription", "text": asr_text or "(no speech detected)"}
    )

    if not asr_text or asr_text.strip() == "":
        return

    user_codes = None
    if srv.session.settings.get("user_audio"):
        t_wait = time.monotonic()
        _slog(
            f"[main.codes] waiting for codes_final (event={srv._codes_final_event.is_set()})..."
        )
        try:
            await asyncio.wait_for(srv._codes_final_event.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        wait_ms = (time.monotonic() - t_wait) * 1000
        n = getattr(srv, "_last_codes_final_n", 0)
        if srv._codes_final_event.is_set():
            _slog(
                f"[main.codes] codes_final OK: {n}f "
                f"({n/12.5:.2f}s) in {wait_ms:.0f}ms"
            )
            user_codes = True
        else:
            _slog(f"[main.codes] codes_final TIMEOUT after {wait_ms:.0f}ms")

    audio_16k = (
        srv.session.recording_sink.get_audio_16k()
        if srv.session.recording_sink
        else None
    )
    _spawn(send_user_audio(srv, audio_16k), "send_user_audio")
    append_turn(srv, "user", asr_text)
    # Detached: a future vad_start that cancels the endpointing task must NOT
    # propagate cancellation into voice_respond. Otherwise voice_respond dies
    # mid-await with session.ready stuck False, and all further turns are
    # silently dropped.
    _spawn_response(
        srv, voice_respond(srv, asr_text, user_codes, audio_16k), "voice_respond"
    )


async def stream_llm_to_tts(
    srv: StreamServer,
    conversation: list[dict],
    system_prompt: str | None = None,
    append_to_history: bool = True,
    user_text_for_tts: str = "",
) -> str:
    ws = srv.session.ws
    if not ws or ws.closed:
        return ""

    srv.session.ready = False
    srv.session.cancel_generation = False
    srv._generates_sent = 0
    srv._generates_done = 0
    srv._turn_chunk_idx = 0
    srv._llm_streaming = True

    last_assistant = ""
    for m in reversed(srv.session.conversation):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break

    keep_context = srv.session.settings.get("keep_context", False)
    if not keep_context:
        srv.tts_cmd_queue.put({"cmd": "rewind"})

    if user_text_for_tts:
        srv.tts_cmd_queue.put(
            {
                "cmd": "prefill_user_turn",
                "text": user_text_for_tts,
                "settings": srv.session.settings,
            }
        )
        await srv._wait_tts_response("user_prefilled", timeout=10)

    full_reply = []
    try:
        from vui.serving.stream.prompts import build_memory_context

        mem_ctx = build_memory_context(srv)
        effective_prompt = system_prompt or srv.session.soul
        if mem_ctx:
            effective_prompt = effective_prompt + "\n\n" + mem_ctx
        conv = list(conversation)
        chunk_w = int(srv.session.settings.get("chunk_words", 20))
        llm_stats: dict = {}
        async for chunk_text, is_done in llm_stream_chunks(
            conv,
            effective_prompt,
            srv.ollama_model,
            max_words=chunk_w,
            stats=llm_stats,
        ):
            if srv.session.cancel_generation:
                break
            chunk_text = clean_llm_text(srv, chunk_text)
            if not chunk_text:
                continue
            full_reply.append(chunk_text)
            srv._generates_sent += 1
            await ws.send_json({"type": "reply", "text": chunk_text})
            srv.tts_cmd_queue.put(
                {
                    "cmd": "generate",
                    "text": chunk_text,
                    "is_voice": True,
                    "new_turn": srv._generates_sent == 1,
                    "is_final": is_done,
                    "settings": srv.session.settings,
                    "context": last_assistant,
                }
            )

            while (
                srv._generates_done < srv._generates_sent
                and not srv.session.cancel_generation
            ):
                await asyncio.sleep(0.005)

    except Exception as e:
        print(f"[main] LLM stream error: {e}")

    srv._llm_streaming = False
    if "ctx_used" in llm_stats:
        srv.conv_ctx = llm_stats["ctx_used"]
        srv.conv_ctx_max = llm_stats["ctx_max"]
    reply_text = " ".join(full_reply)
    if reply_text and append_to_history:
        append_turn(srv, "assistant", reply_text)
    await srv._flush_pending_done()
    srv.session.ready = not srv._ready_blockers
    return reply_text


async def speak_text(srv: StreamServer, text: str):
    ws = srv.session.ws
    if not ws or ws.closed:
        return

    text = text.replace("`", "")
    text = strip_emoji(text)
    srv.session.ready = False
    srv.session.cancel_generation = False
    srv._generates_sent = 1
    srv._generates_done = 0
    srv._turn_chunk_idx = 0

    last_assistant = ""
    for m in reversed(srv.session.conversation):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break

    await ws.send_json({"type": "reply", "text": text})
    srv.tts_cmd_queue.put(
        {
            "cmd": "generate",
            "text": text,
            "is_voice": True,
            "new_turn": True,
            "settings": srv.session.settings,
            "context": last_assistant,
        }
    )

    await srv._flush_pending_done()
    srv.session.ready = not srv._ready_blockers


def bind(cls):
    cls._wait_for_asr = lambda self, *a, **kw: wait_for_asr(self, *a, **kw)
    cls._clean_llm_text = lambda self, *a, **kw: clean_llm_text(self, *a, **kw)
    cls._strip_orphan_brackets = lambda self, *a, **kw: strip_orphan_brackets(
        self, *a, **kw
    )
    cls._llm_speculative_prefill = lambda self, *a, **kw: llm_speculative_prefill(
        self, *a, **kw
    )
    cls._voice_respond = lambda self, *a, **kw: voice_respond(self, *a, **kw)
    cls._tiered_commit = lambda self, *a, **kw: tiered_commit(self, *a, **kw)
    cls._send_user_audio = lambda self, *a, **kw: send_user_audio(self, *a, **kw)
    cls._commit_turn = lambda self, *a, **kw: commit_turn(self, *a, **kw)
    cls._stream_llm_to_tts = lambda self, *a, **kw: stream_llm_to_tts(self, *a, **kw)
    cls._speak_text = lambda self, *a, **kw: speak_text(self, *a, **kw)
