# GPU memory budget

Numbers measured from a running streaming server, plus the formulae behind each
component. Use this to size GPUs and pick ASR / LLM models for your setup.

- [At a glance](#at-a-glance)
- [Per-component breakdown](#per-component-breakdown)
- [ASR backend benchmark](#asr-backend-benchmark)
- [KV cache scaling](#kv-cache-scaling)
- [Hardware fit](#hardware-fit)
- [Tuning levers](#tuning-levers)

---

## At a glance

**Headline: ~12 GB VRAM** for the full stack on a single GPU. Drops to **~8 GB** if you swap the default GPU ASR for a CPU `moonshine.*` backend.

Measured on a running server (`nvidia-smi --query-compute-apps`, Ollama `/api/ps`):

| Component | VRAM | Notes |
|---|---|---|
| Vui TTS + Qwen3-TTS codec process | **~4.6 GiB** | weights + flash-attn workspace + activations + CUDA ctx |
| ASR worker (`fwhisper.distil-small.en`, default) | **~0.85 GiB** | runs in its own process → separate CUDA ctx |
| Ollama `qwen3.5:4b` @ `num_ctx=8192` | **~6.1 GiB** | hybrid SSM+attention 4.7B Q4_K_M |
| **Full stack co-located** | **~12 GiB** | TTS + GPU ASR + Ollama on one card |
| With `moonshine.*` ASR (CPU) | **~8 GiB** | ASR moves off-GPU; TTS + Ollama remain |
| With Ollama on host (recommended compose layout) | ~5.5 GiB | LLM off-GPU — see [readme](../README.md#quick-start-docker-compose-recommended) |

---

## Per-component breakdown

### Vui TTS + codec — measured 4.6 GiB

| Item | Size |
|---|---|
| Vui backbone (305M params, bf16) | ~0.61 GiB |
| Qwen3-TTS codec encoder + decoder + ECAPA speaker encoder (parent 0.6B, bf16) | ~1.2 GiB |
| FlashKVCache pre-allocated by [`Decoder.allocate_flash_kv_cache`](../src/vui/model.py) (22 layers × bf16) | 132 MiB at `max_seqlen=2048`; up to ~1.3 GiB at the trained ceiling |
| flash-attn workspace + CUDA graphs + activations + CUDA ctx | ~1–2 GiB residual |

Backbone arch (from [`readme`](../README.md#vui-nano)): 768 dim, 22 layers, 8 heads → `head_dim = 96`. No GQA — `n_kv_heads = n_heads = 8`.

### ASR worker — measured 0.85 GiB (default)

`distil-small.en` model.bin on disk: 332 MiB.

Cold-process probe (load + 3 s transcribe via `faster-whisper`):

| Stage | VRAM |
|---|---|
| Empty CUDA ctx | 450 MiB |
| After load + warmup | 914 MiB |
| **Delta** | **~464 MiB** for model + CTranslate2 workspace |

Production sees ~850 MiB (extra streaming buffers, longer audio history).

ASR runs in its own [`multiprocessing.Process`](../src/vui/serving/stream/server.py) so it pays a ~450 MiB CUDA-context tax independent of the TTS process.

### Ollama (`qwen3.5:4b`)

Measured at multiple `num_ctx` values via `/api/ps` (live VRAM):

| `num_ctx` | VRAM | Δ KV vs 2048 |
|---|---|---|
| 2048 | 5.89 GB | — |
| 4096 | 5.97 GB | +0.08 |
| 8192 (default) | 6.14 GB | +0.25 |
| 16384 | 6.48 GB | +0.59 |

Per-token KV ≈ **~21 KiB/tok** — much smaller than a vanilla Qwen3-4B because the model is **hybrid SSM + attention** (`/api/show` exposes `qwen35.ssm.v_head_reordered`). Most of the 32 blocks are SSM (no KV cache); only a few carry attention. The model is also vision-capable (4.7B params, Q4_K_M, 3.4 GB on disk) — the 5.8 GB floor is mostly weights + image encoder, not KV.

Default `num_ctx` is set in [`llm_backend.py`](../src/vui/serving/stream/llm_backend.py) (`OllamaBackend.__init__`).

---

## ASR backend benchmark

Same 11 s real-speech clip (`prompts/harry.wav`); transcribe-once + 5 runs of a 4 s streaming window (median). All `fwhisper.*` on CUDA fp16.

| Model | Backend | Device | Load | VRAM/RAM | 4 s window p50 | Full 11 s | Output quality |
|---|---|---|---|---|---|---|---|
| `moonshine.tiny` (arch 0) | ONNX | CPU | 0.2 s | 186 MB RAM | 109 ms | 274 ms | minor errors |
| `moonshine.small` (arch 2) | ONNX | CPU | 0.1 s | 256 MB RAM | 246 ms | 594 ms | minor errors |
| `moonshine.medium` (arch 4, default-CPU) | ONNX | CPU | 0.3 s | 587 MB RAM | ~400 ms¹ | ~1 s¹ | minor errors |
| **`fwhisper.distil-small.en`** (default) | faster-whisper | GPU | 0.6 s | **464 MiB** | **31 ms** | 108 ms | drops punctuation |
| `fwhisper.small.en` | faster-whisper | GPU | 6.3 s | 622 MiB | 43 ms | 169 ms | full punctuation |
| `fwhisper.distil-medium.en` | faster-whisper | GPU | 12.2 s | 1006 MiB | 41 ms | 108 ms | drops punctuation |
| `fwhisper.medium.en` | faster-whisper | GPU | 15.8 s | 1870 MiB | 97 ms | 241 ms | full punctuation |
| `fwhisper.distil-large-v3` | faster-whisper | GPU | 16.4 s | 1998 MiB | 65 ms | 124 ms | full punctuation |
| `fwhisper.turbo` | faster-whisper | GPU | 1.1 s | 2126 MiB | 71 ms | 147 ms | full punctuation |

¹ Moonshine streaming archs (4, 5) are designed for incremental `add_audio` + `update_transcription` use; the figures above are pessimistic non-streaming baselines.

**Streaming budget:** the interim refresh interval is **500 ms** (`interim_every_s` in [`fwhisper.py:FWhisperBackend`](../src/vui/serving/stream/asr/fwhisper.py)). Every model in the table comfortably fits within that — the server stays real-time on any of them.

**Picks:**

- Default for fast English on GPU → `fwhisper.distil-small.en`.
- Best GPU quality without big load-time penalty → `fwhisper.turbo`.
- No GPU available, or want to free VRAM → `moonshine.tiny` (CPU, 109 ms p50, 186 MB RAM, frees the entire ASR-process VRAM).
- Avoid `fwhisper.medium.en` — strictly dominated by `distil-large-v3` and `turbo`.

---

## KV cache scaling

### Vui audio backbone

Formula (verified by allocating `FlashKVCache` and measuring tensor bytes):

```
KV = 2 (k+v) × n_layers × n_kv_heads × head_dim × seq_len × dtype_bytes
   = 2 × 22 × 8 × 96 × seq_len × 2          (bf16)
   = 67,584 B/token  =  66 KiB/token
   = 0.806 MiB/s     (at 12.5 frames/s)
```

At conversational lengths the backbone KV is small:

| Context | Audio frames | Audio KV occupancy |
|---|---|---|
| 30 s | 375 | 24 MiB |
| 60 s | 750 | 48 MiB |
| 2 min | 1500 | 97 MiB |
| 5 min | 3750 | 242 MiB |

The cache is pre-allocated to `max_seqlen` regardless of fill, so the *reserved* VRAM jumps to the trained ceiling (often >1 GiB). Pass `max_seqlen=` to [`allocate_flash_kv_cache`](../src/vui/model.py) to cap it.

The Qwen3-TTS codec produces 16 codebooks per audio frame, but the RQ-Transformer head consumes them depthwise per frame (`n_quantizers=16` in [`config.py:VuiConfig`](../src/vui/config.py)) — it does **not** multiply backbone seq_len by 16.

### Ollama LLM

The `num_ctx` table above shows the only meaningful lever: dropping `num_ctx` from 8192 → 2048 saves ~250 MiB. The hybrid arch means even large context windows don't blow up KV.

---

## Hardware fit

| Setup | TTS+codec | ASR | Ollama | Total |
|---|---|---|---|---|
| Full stack co-located (default) | 4.6 | 0.85 | 6.1 | **~12 GiB** |
| Co-located + `moonshine.*` ASR (CPU) | 4.6 | 0 | 6.1 | **~8 GiB** |
| Ollama on host (recommended compose layout) | 4.6 | 0.85 | — | ~5.5 GiB |
| Ollama on host + `moonshine.*` ASR | 4.6 | 0 | — | ~4.6 GiB |
| LLM on remote server + `moonshine.*` ASR | 4.6 | 0 | — | ~4.6 GiB |
| Apple Silicon (MLX) | unified memory | — | — | varies — see [readme](../README.md#hardware) |

A 12 GB card (e.g. 3060 12GB / 4070) runs the full stack on one GPU. An 8 GB card needs either the moonshine-ASR swap above or the [recommended docker-compose layout](../README.md#quick-start-docker-compose-recommended) that puts Ollama on the host. To go below that — e.g. on a laptop GPU or a small VPS — point the LLM at a remote box ([`configuration.md` → Custom model server](configuration.md#custom-model-server)); both the conversation and thoughts streams will use it.

---

## Tuning levers

Largest VRAM impact first:

1. **Point the LLM at a remote server** — frees the full ~6.1 GiB Ollama footprint. Both the conversation stream and the parallel thoughts stream share one `LLMBackend` singleton ([`llm_backend.py`](../src/vui/serving/stream/llm_backend.py) `get_backend()`), so a single env var change (`VUI_OLLAMA_URL=…` or `VUI_LLM_BACKEND=vllm` + `VUI_VLLM_URL=…`) moves both. Setup details in [`configuration.md` → Custom model server](configuration.md#custom-model-server).
2. **Move Ollama off-GPU to the host** — frees ~6.1 GiB. The default compose layout already does this.
3. **Switch LLM to `qwen3.5:2b`** — 2.7 GB on disk vs 3.4 GB; saves ~1 GiB even on GPU.
4. **`moonshine.tiny` ASR** — frees the full ~0.85 GiB ASR-process VRAM at the cost of ~80 ms extra latency per window. Still well under the 500 ms streaming budget.
5. **`int8_float16` faster-whisper compute_type** — set via `compute_type` arg in [`FWhisperBackend.__init__`](../src/vui/serving/stream/asr/fwhisper.py); saves ~195 MiB on `distil-small.en`. Latency unchanged.
6. **Lower `num_ctx`** in Ollama — only saves ~250 MiB going 8192 → 2048 because of the SSM-heavy arch.
7. **Cap `max_seqlen`** for `allocate_flash_kv_cache` — saves up to ~1 GiB on the TTS side if the checkpoint trained at >>120 s context.

What **doesn't** work for fast streaming:

- **Faster-whisper on CPU** — measured **2299 ms** p50 on a 4 s window vs 31 ms on GPU. ~70× slower per call; would back-pressure the streaming pipeline. RTF on long audio is misleading because Whisper short-circuits silence; per-call latency on real chunks is what matters.
