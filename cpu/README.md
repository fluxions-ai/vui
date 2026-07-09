# CPU Inference for VUI TTS

Standalone C inference engine for VUI TTS. No Python/PyTorch/ONNX at runtime — just a single binary + weight file.

## Quick Start

Build artifacts, weights and prompt caches are written into this `cpu/` dir (all
gitignored). Run these from inside `cpu/`; `.venv/bin/python` refers to the
repo-root venv — either symlink it (`ln -s ../.venv .venv`) or use `../.venv/bin/python`.

```bash
cd cpu

# Export model to single binary (one-time, needs Python).
# `vui-nano.safetensors` is the public release checkpoint — it auto-downloads
# from the fluxions/vui HuggingFace repo on first use. (A local .pt path works too.)
.venv/bin/python export_full.py vui-nano.safetensors vui_full.bin

# Build C inference (binary lands in cpu/)
gcc -O3 -march=native -ffast-math -fopenmp -o vui_tts vui_tts.c -lm -lopenblas

# Run — with no --kv-cache it uses the default (baked-in cond_bias) speaker
OMP_NUM_THREADS=4 ./vui_tts vui_full.bin --text "Hello world." --output out.wav

# Voice cloning: prepare a prompt cache from any wav (one-time; whisper-transcribes + re-encodes)
.venv/bin/python prepare_prompt.py ~/good_prompt.wav prompt_cache.bin
OMP_NUM_THREADS=4 ./vui_tts vui_full.bin --kv-cache prompt_cache.bin --text "Hello world." --output out.wav

# OR build a release-voice cache from OFFICIAL prompt files (exact transcript, pre-encoded
# codes, official spk token — no whisper, no re-encode). This is how prompt_<voice>_official.bin was made:
.venv/bin/python prepare_prompt_official.py prompts/maeve.safetensors prompts/maeve.txt prompt_maeve_official.bin vui-nano.safetensors

# Stream to speaker
OMP_NUM_THREADS=4 ./vui_tts vui_full.bin --kv-cache prompt_cache.bin --text "Hello world." --stream
```

### Warm realtime serving (resident daemon)

```bash
cd cpu
./say.sh "hello there"     # auto-boots the daemon (loads ~3.6GB once), then speaks
./say.sh --server          # foreground daemon; serves until killed
./say.sh --quit            # stop it
# env: VOICE=maeve|abraham TEMP=0.6
```

`say_rt.py` (one-shot C→ONNX render) and `vui_stream.py` (flow-controlled streaming) are the
non-daemon entrypoints; all resolve weights/binary/`.venv` relative to `cpu/`.

## Files

| File | Description |
|------|-------------|
| `vui_tts.c` | Full C inference: backbone, RQ, codec decoder, tokenizer, WAV output |
| `export_full.py` | Export model + codec + tokenizer to single `vui_full.bin` (1.7GB, vui2 format) |
| `prepare_prompt.py` | Create KV cache from any prompt wav (whisper transcribe + encode + prefill) |
| `prepare_prompt_official.py` | Build a release-voice KV cache from official prompt safetensors (exact transcript + pre-encoded codes + official spk token); made the `prompt_<voice>_official.bin` files |
| `cpu_inference.py` | Python+ONNX inference (older, slower, kept for reference) |
| `say.sh` | Warm-serving entrypoint: boots/queries the resident daemon over a unix socket |
| `vui_daemon.py` | Resident daemon — loads model+cache+ONNX once, holds audio device, serves utterances |
| `vui_stream.py` | Flow-controlled streaming TTS (C backbone ⇄ ONNX codec), RTF>1, no-stutter prebuffer |
| `say_rt.py` | One-shot realtime render (C backbone → ONNX codec → play/save) |

## Architecture

- **Backbone**: 22-layer LLaMA-style decoder (d=768, 8 heads, SwiGLU MLP), KV cached
- **RQ Transformer**: 5 layers (d=768, 8 heads), autoregressive Q-1 steps per frame, KV cached
- **Codec Decoder**: SplitResidualVQ (16 codebooks) → Conv1d → 8-layer transformer → 2x upsample → waveform decoder (4 blocks, 480x expansion, 12.5Hz → 24kHz)
- **Tokenizer**: SmolLM2 BPE (49429 vocab), implemented in C
- **Speaker**: QwenSpeakerEncoder (8.9M params, ECAPA-TDNN) → 1024-dim → spk_proj → 1 token. Computed in Python, baked into KV cache.

## Inference Sequence

1. `spk_token` — speaker embedding, no cond_bias
2. `prompt_text + [SC]` — transcript of prompt audio, no cond_bias
3. `prompt_audio_codes` — encoded prompt audio, no cond_bias
4. `generation_text` — WITH cond_bias (`sq_proj([4.0, 4.0, 4.0, 4.5, 4.0, 4.0, 4.5])`)
5. `decode steps` — autoregressive, NO cond_bias

Steps 1-3 are baked into `prompt_cache.bin` by `prepare_prompt.py`.

## Performance (Ryzen 7 7840U, 4 threads, DDR5)

### Token Generation (backbone + RQ, Q=12)

| Implementation | ms/step | RTF |
|---|---|---|
| Hand-rolled matmul (old) | 151ms | 0.53x |
| **cblas_sgemv (current)** | **53ms** | **1.50x** |

RTF = 80ms / step_time (12.5 Hz codec = 80ms per frame)

Generation is **faster than realtime**. The bottleneck is the codec decoder.

### Codec Decoder Breakdown (35 frames → 3s audio)

| Stage | Time | Notes |
|---|---|---|
| Quantizer | 2ms | Codebook lookup + projection |
| Pre-conv | 1ms | Conv1d(512→1024, k=3) |
| Transformer | 19ms | 8 layers, causal attention on T frames |
| Upsample | 183ms | 2x transconv + ConvNeXt blocks |
| Waveform decoder | 1256ms | 4 blocks: transconv + 3 res units each |
| **Total** | **~1500ms** | |

### Waveform Decoder Per-Block

| Block | Channels | Output T | Time |
|---|---|---|---|
| 0 | 1536→768 | 1184 | 149ms |
| 1 | 768→384 | 5920 | 247ms |
| 2 | 384→192 | 23680 | 431ms |
| 3 | 192→96 | 71040 | 546ms |

The waveform decoder is **compute-bound**: ~1-2B FLOPs of convolutions (im2col + GEMM) with SnakeBeta activations (sinf on every element). Not feasible for per-frame streaming on CPU.

### Streaming Mode

`--stream` uses chunked decoding: generate 8 frames, batch codec decode, output new audio, repeat. First audio at ~950ms. Not true per-frame streaming but functional.

## Codec Bottleneck

The waveform decoder expands time 480x (8×5×4×3) through 4 transposed conv blocks. Each block also has 3 residual units with dilated k=7 convolutions. For 35 input frames this means processing 67K+ timesteps through multiple conv layers.

This is fundamentally ~1-2B FLOPs. On 4 CPU cores at ~100 GFLOPS peak, theoretical minimum is ~100ms, but memory bandwidth and overhead push it to 1-2s.

### Possible Solutions

1. **Lighter vocoder**: iSTFT-based decoder instead of transconv stack (requires retraining)
2. **Per-frame streaming with small context**: KV cache codec transformer, run vocoder on ~5-10 context frames only. Borderline realtime (~50-100ms/frame)
3. **GPU codec**: Generate on CPU, decode on GPU (even integrated GPU would be instant)
4. **Accept latency**: Chunked approach gives ~1s to first audio

## Optimizations Applied

- `cblas_sgemv` for all backbone/RQ GEMV (3x speedup over hand-rolled loops)
- `cblas_sgemm` for codec transformer QKV projections (batched)
- `cblas_sgemm` for k=1 convolutions (skip im2col)
- `im2col + cblas_sgemm` for all other convolutions
- OpenMP parallelized SnakeBeta activation
- Fast sin approximation for SnakeBeta
- mmap'd weight file (zero-copy loading)

## Binary Format (vui2)

```
Header (256 bytes):
  magic "vui2", version=2
  backbone config (dim, hidden, layers, heads, kv_heads, max_seq)
  RQ config (dim, hidden, layers, heads, n_q, codebook_size)
  vocab_size, audio_emb_size, rope_theta, eos_bias, sc_token_id

Backbone weights (fp32)
RQ weights (fp32)
Codec decoder weights (fp32)
Tokenizer (vocab bytes + merge scores + special tokens)
```

Total: ~1677MB for the qwen-rq-768 model.

## KV Cache Format

```
int32 pos
float32 key_cache[n_layers × max_seq × kv_dim]
float32 value_cache[n_layers × max_seq × kv_dim]
```

Where kv_dim = n_kv_heads × head_dim. ~1430MB for pos=114 (speaker + prompt).
