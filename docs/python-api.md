# Calling the Engine from Python

Most users will reach `vui.engine.Engine` directly — for one-shot rendering, batch jobs, custom pipelines, or anywhere the streaming server's HTTP/WS surface is overkill. This doc covers the public API: loading, prompt encoding (with proper multi-segment chunking for long references), rendering, and streaming.

CUDA-only. For the Apple-Silicon MLX path see [the bottom of this doc](#apple-silicon-mlx).

## Minimal example

If your voice prompt is short (<15s), `Engine()` + a one-segment prefill is all you need:

```python
import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from julius.resample import resample_frac

from vui.engine import Engine, GenConfig, Segment
from vui.qwen_codec import SAMPLE_RATE as SR  # 24 kHz
from vui.qwen_codec import QwenCodecEncoder
from vui.inference import asr

engine = Engine()  # name="vui-nano" by default; pass a name or local path to override

# Encode the voice prompt (audio + transcript -> Segment)
wav_16k = AudioDecoder("prompts/abraham.wav", sample_rate=16000, num_channels=1) \
    .get_all_samples().data.squeeze(0)
wav_24k = resample_frac(wav_16k.unsqueeze(0), 16000, SR)
codec_enc = QwenCodecEncoder.from_pretrained().cuda().float().eval()
with torch.inference_mode():
    codes = codec_enc.encode(wav_24k.float().cuda().unsqueeze(0))
prompt_codes = codes[0, : engine.Q].T.long()  # (T, Q)
prompt_text = asr(wav_16k)                    # ASR transcript

# Render
with engine.new_row() as row:
    row.prefill([Segment(prompt_text, prompt_codes)])
    codes, audio = row.render(
        "So [breath] the thing about this is, it's not what you'd expect.",
        GenConfig(temperature=0.7, max_secs=10),
    )

AudioEncoder(audio.squeeze().cpu().float().unsqueeze(0), sample_rate=SR) \
    .to_file("out.wav")
```

`row.render()` returns `(codes (T, Q), audio (1, 1, S))`. The audio is already decoded through the Qwen codec; if you only need the raw codec codes (e.g. to ship across the wire and decode elsewhere), use `engine._render_row(row, text, cfg)` and skip the vocoder.

## Properly chunked prompts (long voice references)

The minimal example above works for prompts under ~15 seconds. **For longer references — and you want longer, the model improves up to a couple of minutes — you have to chunk.** Stuffing a single 60-second `(text, codes)` segment into prefill destroys the model's per-segment speaker prefix and the output drifts off the speaker.

The right pattern is multi-segment prefill: split the prompt at sentence boundaries every ~10 seconds, encode each chunk's codes, and pass a `list[Segment]` to `row.prefill()`. The engine writes `[spk] text_i codes_i` for each segment — exactly the training format.

`vui.prompt_utils.build_prompt_segments` does all of this for you. You give it audio + two callbacks (encoder, transcriber); it runs ASR on the full clip, force-aligns words to audio with Wav2Vec2, trims trailing partial sentences, splits at sentence terminators near the target segment length, and slices the pre-encoded codes by frame index — no re-encoding per segment.

```python
import torch
from torchcodec.decoders import AudioDecoder

from vui.engine import Engine, GenConfig, Segment
from vui.inference import asr
from vui.prompt_utils import build_prompt_segments
from vui.qwen_codec import SAMPLE_RATE as SR
from vui.qwen_codec import QwenCodecEncoder

engine = Engine()
codec_enc = QwenCodecEncoder.from_pretrained().cuda().float().eval()

# 90s voice prompt -> chunked segments
wav_16k = AudioDecoder("prompts/long_speaker.wav", sample_rate=16000, num_channels=1) \
    .get_all_samples().data.squeeze(0)

def _encode(audio_16k):
    """16 kHz -> codec codes (T, Q)."""
    from julius.resample import resample_frac
    audio_24k = resample_frac(audio_16k.unsqueeze(0), 16000, SR)
    with torch.inference_mode():
        codes = codec_enc.encode(audio_24k.float().cuda().unsqueeze(0))
    return codes[0, : engine.Q].T.long()

# build_prompt_segments -> [(text, codes), (text, codes), ...]
segments = build_prompt_segments(
    wav_16k,
    encode_codes=_encode,
    transcribe=asr,           # any (audio_16k) -> str works (whisper, moonshine, custom)
    align_device="cuda",       # Wav2Vec2 alignment device
    target_seg=10.0,          # target segment length in seconds
)
print(f"{len(segments)} segments, "
      f"{sum(c.shape[0] for _, c in segments)} frames total")

with engine.new_row() as row:
    row.prefill([Segment(t, c) for t, c in segments])
    codes, audio = row.render("Your text here.", GenConfig(temperature=0.7))
```

Tunable: `target_seg=10.0` is the sweet spot for the released checkpoint; pushing it to 15–20s sometimes helps on slower speakers, lower than ~7s loses speaker context. Short audio (`total_secs <= target_seg * 1.5`) short-circuits and returns a single segment — no alignment needed.

**Caching.** `build_prompt_segments` is expensive (ASR + Wav2Vec2 forced alignment, several seconds the first time). For repeated renders against the same speaker, pickle the segments to disk and skip the rebuild — see `demo.py` `_save_prompt_to_disk` / `_load_prompt_from_disk` for the pattern.

### Editing the transcript

The ASR output from `build_prompt_segments` is usually correct but not always — especially for accents, fillers, and proper nouns. The transcript drives speaker conditioning, so a wrong transcript pushes the model toward a wrong pronunciation. Two options:

1. **Fix the segment text in place.** `Segment` is a frozen dataclass but you can rebuild the list: `segments = [Segment(fixed_text, codes) for fixed_text, (_, codes) in zip(my_texts, segments)]`. The codes don't change, only the text the model conditions on.
2. **Trim a hallucinated tail.** If ASR over-transcribed the last segment (silence interpreted as words), use `return_timings=True` to get word boundaries and slice the codes to the end of the last real word. See `demo.py:_apply_text_edits` for the exact frame-math.

## `GenConfig` — the knobs you'll touch

```python
GenConfig(
    temperature=0.9,        # >0.7 lively, <0.7 steady; >1.0 increases drift
    top_k=100,              # restricts each frame's codec-token sampling
    top_p=None,             # nucleus sampling; mutually exclusive with top_k
    rep_penalty=1.1,        # 1.0 = off; useful against filler loops
    rep_window=24,          # frames of cb0 history considered for rep_penalty
    eos_threshold=0.45,     # higher = less likely to cut off mid-sentence
    max_secs=30.0,          # hard cap on output duration
    max_turn_secs=15.0,     # hard cap per turn (single chunk)
    chunk_words=5,          # min words per sub-chunk for streaming
    n_codebooks=0,          # 0 = all 16; drop to ~10 for faster + smaller VRAM
    sentence_only=False,    # True: split chunks only on .!?  (no commas)
)
```

The two you'll reach for first are `temperature` (default 0.9 is lively; drop to 0.7 for steadier delivery) and `max_secs`. For long monologues bump `max_secs` to 120; for one-liners drop to 8 so you fail fast if generation stalls.

`n_codebooks` is the speed/quality lever — the RQ-Transformer head emits 16 quantizer levels per frame by default. Dropping to ~10 gives noticeably faster decode and lower VRAM at the cost of some stability artefacts. Below 8 quality drops sharply.

`sentence_only=True` matches the streaming server's chunking (`llm_stream_chunks`) — only break on `.!?`, never on commas. Useful when you control the input text and want one chunk per sentence; less useful when you're fed LLM tokens incrementally.

## Streaming (per-frame, lowest latency)

For low-latency playback (live chat, voice notes, calls), use `row.stream(...)` instead of `row.render(...)`. It yields decoded audio frames as they're generated — typically ~80ms blocks at 24 kHz — so playback can start before the full reply is done.

```python
import sounddevice as sd, numpy as np

engine = Engine(max_rows=1)  # stream needs max_rows=1

with engine.new_row() as row:
    row.prefill([Segment(prompt_text, prompt_codes)])
    out_stream = sd.OutputStream(samplerate=24000, channels=1, dtype="float32")
    out_stream.start()
    for audio_frame in row.stream("Streaming reply.", GenConfig(temperature=0.7)):
        # audio_frame: (1, 1, DOWNSAMPLE_RATE) float on GPU
        out_stream.write(audio_frame.squeeze().cpu().numpy().astype(np.float32))
    out_stream.stop()
```

Notes:

- `stream()` requires `max_rows=1` — the streaming codec graph isn't batched.
- Pass a `cancel=threading.Event()` (or any `.is_set()`-able object) to abort mid-generation; the loop checks per frame and exits cleanly.
- `reset_rep=False` keeps the repetition-penalty history across multiple `.stream()` calls — useful when you feed LLM chunks one at a time and want rep-penalty to span the full turn.
- `row.rewind()` returns the KV to end-of-prompt without re-prefilling, so you can render another turn with the same speaker prefix already cached.

## Continuous batching (many concurrent renders)

For batch jobs — bulk dataset generation, parallel TTS for multiple users — set `max_rows=N` and use `engine.render_continuous(requests)`:

```python
from vui.engine import Engine, GenConfig, RenderRequest, Segment

engine = Engine(max_rows=8)

requests = [
    RenderRequest(
        segments=[Segment(prompt_text, prompt_codes)],
        text="First line to render.",
    )
    for _ in range(64)
]

# render_continuous returns list[Tensor] of audio (1, 1, S) — one per request,
# in input order. It does NOT return codes; if you need codes, use render_all
# on rows you've prefilled yourself.
audios = engine.render_continuous(requests, cfg=GenConfig(max_secs=15))
for i, audio in enumerate(audios):
    AudioEncoder(audio.squeeze().cpu().float().unsqueeze(0), sample_rate=24000) \
        .to_file(f"out_{i:03d}.wav")
```

The engine maintains an N-row flash KV cache and a continuous-batching loop that swaps finished requests out for queued ones — utilisation stays high even when individual reply lengths vary. `max_rows=8` on a 4090 is a reasonable starting point for the released checkpoint; the bound is VRAM (KV cache) not compute.

For batch dataset generation where you already hold the rows, `engine.render_all(rows, texts, cfg)` takes lists of `Row` and matching `str` and returns one audio tensor per row. Useful when you've prefilled different rows with different speakers and want to render them all in one batched pass. `render_continuous` is the higher-level API — it takes `RenderRequest` objects and owns row lifecycle internally.

## Decoding codes without rendering audio

If you want raw codec codes — to ship over the wire, save for later decode, or feed into a custom vocoder — you can stop after generation and skip the codec decode:

```python
# Engine returns codes + audio by default. To get codes only without spending
# the vocoder cycles, use the internal _render_row:
with engine.new_row() as row:
    row.prefill([Segment(prompt_text, prompt_codes)])
    codes = engine._render_row(row, "Text here.", GenConfig())  # (T, Q) long tensor
```

To decode codec codes back to audio later (anywhere with a `QwenCodecDecoder` available):

```python
from vui.qwen_codec import QwenCodecDecoder
codec_dec = QwenCodecDecoder.from_pretrained().cuda().float().eval()

# codes: (T, Q) long. Shape it back to (1, Q, T) for the decoder.
c = codes.T.unsqueeze(0).cuda()
with torch.inference_mode():
    audio = codec_dec.decode_chunked(c, ctx=6)  # (1, 1, S)
```

The `ctx=6` parameter prepends 6 frames (~500ms) of codec context to smooth chunk boundaries. If you have prompt codes available too, prepend their tail before decoding for a cleaner first frame — see `demo.py:decode_audio` for the exact pattern.

## Lifecycle

`Engine` owns CUDA graphs, a flash KV cache sized at `max_rows + 1`, and a vocoder graph — substantial GPU resources. There's no explicit `teardown()`; let the engine go out of scope and the cached tensors get GC'd (call `torch.cuda.empty_cache()` after if you're tight on VRAM). Most apps keep a single Engine for the process lifetime and reuse rows.

`with engine.new_row() as row:` (context manager) auto-releases the slot back to the engine's free pool on exit. Don't hold rows past their useful life — at `max_rows=N` you only get `N` concurrent slots before `new_row()` blocks.

**Important:** `row.render()` and `row.stream()` require **`max_rows=1`** (single-row engine), because they hit the streaming/B=1 codec path. Multi-row engines are for `render_continuous` / `render_all` only. If you need both single-render and batched paths in one process, build two engines.

## Apple Silicon (MLX)

The MLX path is a separate code surface (`vui.mlx.tts.*`) — same model, different runtime. The end-to-end equivalent of the chunked-prompt flow above is `demo.py:generate_chunked_mlx` (lines ~167–275). Key differences:

- Loaded via `vui.mlx.tts.weights.load_quantized(ckpt_path, "float32"|"int8"|"int4")`.
- No `Engine` class; you call `vui.mlx.tts.generate.generate` per chunk and the prompt is prefilled via `vui.mlx.tts.generate.prefill_prompt` (single segment) or by manually iterating segments with `[spk] [text_i] [audio_i]` (`_prefill_segments_mlx` in demo.py).
- No streaming primitive — chunks are generated one at a time and concatenated.

If you're on M-series and want chunked-prompt MLX rendering, mirror `generate_chunked_mlx`'s `_prefill_segments_mlx` call after building segments with `build_prompt_segments` (the prompt-utils function is backend-agnostic; only the encoder callback needs to be MLX-friendly).

## See also

- [`prompting.md`](prompting.md) — text-side rules (tags, punctuation, phonetic numbers, `|spell|` escape for hard words).
- [`memory-budget.md`](memory-budget.md) — VRAM breakdown per component, `n_codebooks` tradeoffs, `max_rows` budgeting.
- [`configuration.md`](configuration.md) — env-var overrides for the streaming server's defaults.
- `src/vui/demo/cli.py` — full working CLI render with streaming playback, prompt loading, settings, and tab-completion.
- `demo.py` — Gradio app with KV caching, prompt-text editing, MLX/CUDA precision switching, and quantization.
