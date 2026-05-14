# The soul

Every chat-model prompt has a "system prompt" — the block of instructions slotted into the LLM's `system` role on every turn that defines tone, rules, and persona. In Vui we call it the **soul**: it's what makes the assistant sound like Vui (and not a chatbot), and it's the single biggest lever you have over how the assistant behaves.

The default soul lives in `src/vui/serving/stream/prompts.py` (`_SOUL_TEMPLATE` → `SOUL`) and bakes in:

- **Speech style** — short sentences, fillers ("um", "yeah", "honestly"), cutoffs, restarts; no markdown, no bullets; numbers/times/units spelled out phonetically so the TTS doesn't read "$50" as "dollar five zero".
- **Conversational rules** — confirm scope before launching into recipes/tutorials, chunk lists in threes and pause, never echo the user back, never fabricate facts, "let me check" only for live data the LLM can't answer itself.
- **Tool-aware behaviour** — when the optional Claude task server is reachable, an extra rules block + few-shot example teaches the model to emit a short filler ("yeah, one sec…") and stop, so the delegated lookup can run and the relay step speaks the result.
- **Identity slot** — the assistant's name is templated in (default `Vui`, editable in the UI), so the model recognises when you address it and uses the right name in its replies.

## Editing it live

The UI has a **Soul** textarea in the side panel — edit, click out (or hit save), and the new soul is:

1. Persisted to `prompts/.soul` so it survives restarts and overrides the template.
2. Pushed back to the LLM via a fresh `system`-role prefill, so the new persona is in the KV cache before your next turn.

Clearing the textarea (empty save) reverts to the bundled template with the current assistant name re-stamped in.

## Why "soul"?

Because "system prompt" is correct but joyless. The soul is what gives the assistant its character — its sense of humour, its voice mannerisms, its conscience about what to say and what to look up. Swap the soul and you swap the personality, no fine-tuning required.

The name is borrowed from [OpenClaw](https://github.com/openclaw/openclaw), where the assistant's persona prompt is also called the *soul*. The concept lines up almost exactly with what we use it for here, so we adopted the term — partly because it's a better word, partly because OpenClaw is one of the clients Vui talks to via the Realtime API (see [`realtime-api.md`](realtime-api.md)) and lining up terminology between the two makes config easier to read.

> **Note on the realtime API.** The OpenAI Realtime protocol calls this `instructions` (sent via `session.update` or `response.create`). Vui maps either field directly onto the same soul slot — so clients written against OpenAI's spec keep working unchanged.
