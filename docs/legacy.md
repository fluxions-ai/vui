# Legacy (pre-1.0) checkpoints

The original vui release shipped 100M-parameter checkpoints on a different
architecture from today's models. They don't load into `Engine` — the loaders
reject them with a pointer here — but they still run via `vui.legacy`.

| Checkpoint | Notes |
|-----------------------|--------------------------------------------|
| `vui-100m-base.pt` | base model |
| `vui-cohost-100m.pt` | two-speaker conversational fine-tune |
| `vui-abraham-100m.pt` | single-voice fine-tune |

All auto-download from [fluxions/vui](https://huggingface.co/fluxions/vui),
as does their codec (`fluac-22hz-22khz.pt`, 22.05 kHz output).

## How they differ from current checkpoints

- Per-quantizer audio embeddings and heads (9 quantizers, codebook 1000)
  decoded with a **delayed codebook pattern** — no RQ-transformer.
- **Fluac** codec at 22.05 kHz instead of the Qwen codec at 24 kHz.
- byt5 text tokenizer, no speaker-embedding prompting, no sq/wps conditioning.

## Usage

```python
from vui.legacy import Vui, render

model = Vui.from_pretrained("vui-cohost-100m.pt").eval()
audio = render(model, "Hello there!")  # (1, 1, S) float at 22050 Hz
```

Runs on **CUDA** or **CPU** (no flash-attn; faster than real-time on an
M-series CPU). `render` accepts `prompt_codes` for continuation,
`temperature`, `top_k`, `top_p`, and `max_secs`; texts over 1000 characters
are chunked line-by-line with rolling code context, as in the original
release.

### Apple Silicon: MLX decoder (~4.5× real-time)

`vui.mlx.legacy` runs the transformer on MLX while embeddings, heads,
sampling, and the Fluac codec stay on (fast, tiny) torch-CPU ops:

```python
from vui.legacy import render
from vui.mlx.legacy import load_legacy_mlx

model, adapter = load_legacy_mlx("vui-cohost-100m.pt")
audio = render(model, "Hello there!", decoder=adapter)
```

Measured on M4: ~4.5× real-time (vs ~1.3× pure CPU), TTFB ~115 ms. The MLX
decoder matches the torch decoder to ~3e-4 max abs difference (fp32).

## Limitations

- **MPS is refused.** `generate` raises on a model moved to MPS: torch-on-Metal
  produces corrupted audio for this architecture and is slower than CPU.
  Use CPU, CUDA, or the MLX adapter.
- **VAD trimming is optional.** The original release trimmed output with
  pyannote VAD, which is no longer a dependency. Without it installed the
  full untrimmed render is returned.
- **Old-model quirks are preserved**, not fixed: occasional early EOS can
  clip trailing words, and the generation loop trims the last 10 frames
  (~0.5 s) as the original code did.
- Batch size 1 only; no streaming API. For streaming, batching, voice
  prompting, and better quality, use the current checkpoints via
  [`Engine`](python-api.md).
