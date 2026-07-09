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

# Run — with no --kv-cache it uses the default (baked-in) speaker
OMP_NUM_THREADS=4 ./vui_tts vui_full.bin --text "Hello world." --output out.wav

# Voice cloning: prepare a prompt cache from any wav (whisper-transcribes + re-encodes)
.venv/bin/python prepare_prompt.py ~/good_prompt.wav prompt_cache.bin
OMP_NUM_THREADS=4 ./vui_tts vui_full.bin --kv-cache prompt_cache.bin --text "Hello world." --output out.wav

# Stream to speaker
OMP_NUM_THREADS=4 ./vui_tts vui_full.bin --kv-cache prompt_cache.bin --text "Hello world." --stream
```

## Files

| File | Description |
|------|-------------|
| `vui_tts.c` | Full C inference: backbone, RQ, codec decoder, tokenizer, WAV output |
| `export_full.py` | Export model + codec + tokenizer to a single `vui_full.bin` |
| `prepare_prompt.py` | Create a voice-clone KV cache from any prompt wav (whisper transcribe + encode + prefill) |
| `prepare_prompt_official.py` | Build a release-voice KV cache from official prompt safetensors (exact transcript + pre-encoded codes + official spk token) |
| `cpu_inference.py` | Python + ONNX inference (older, slower, kept for reference) |
| `say.sh` / `vui_daemon.py` | Warm-serving: resident daemon that loads the model once and speaks utterances over a unix socket |
| `vui_stream.py` / `say_rt.py` | Non-daemon streaming / one-shot render (C backbone → ONNX codec) |

## Notes

- Token generation (backbone + RQ) runs faster than realtime on a modern laptop CPU; the codec decoder's waveform stage is the bottleneck (~1–2B FLOPs of convolutions), so first audio lands ~1s in.
- `--stream` uses chunked decoding (generate a few frames, batch-decode, repeat) rather than true per-frame streaming.
- Weights are exported to a self-describing `vui2` binary (256-byte header + fp32 backbone/RQ/codec weights + tokenizer). The KV cache is `int32 pos` + fp32 key/value caches.
