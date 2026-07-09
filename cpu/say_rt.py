"""Realtime vui TTS on CPU: fast C backbone -> ONNX codec -> play.
Called by say.sh. Text = all argv joined.  Env: VOICE, TEMP, NOPLAY=1 (just save)."""

import os
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE = os.environ.get("VOICE", "maeve")
TEMP = os.environ.get("TEMP", "0.6")
TEXT = " ".join(sys.argv[1:]).strip() or "Hello."
MODEL = f"{HERE}/vui_nano_full.bin"
CACHE = f"{HERE}/prompt_{VOICE}_official.bin"
ONNX = f"{HERE}/codec_q12.onnx"
NQ, SR = 12, 24000

for p in (MODEL, CACHE, ONNX):
    if not os.path.exists(p):
        sys.exit(f"missing {p}")

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])

# 1. backbone: stream codes (timing starts at first frame, excludes model load)
proc = subprocess.Popen(
    [
        f"{HERE}/vui_tts",
        MODEL,
        "--kv-cache",
        CACHE,
        "--temperature",
        TEMP,
        "--eos-threshold",
        "0.35",
        "--quantizers",
        str(NQ),
        "--emit-codes",
        "--text",
        TEXT,
    ],
    cwd=HERE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    env={**os.environ, "OMP_NUM_THREADS": "4"},
    text=True,
    bufsize=1,
)
frames, t_first = [], None
for line in proc.stdout:
    line = line.strip()
    if line == "END":
        break
    if line:
        if t_first is None:
            t_first = time.perf_counter()
        frames.append([int(x) for x in line.split()][:NQ])
gen = time.perf_counter() - (t_first or time.perf_counter())
proc.wait()

# 2. ONNX codec decode
t = time.perf_counter()
audio = sess.run(None, {"codes": np.array(frames, dtype=np.int64).T[None]})[0][0, 0]
dec = time.perf_counter() - t
a = len(audio) / SR
print(f'"{TEXT}"')
print(
    f"  {len(frames)} frames, {a:.1f}s audio | backbone {gen:.2f}s ({a/gen:.2f}x) + codec {dec:.2f}s ({a/dec:.2f}x)"
)
print(f"  => {a/(gen+dec):.2f}x realtime ({VOICE}, temp {TEMP})")

sf.write("/tmp/say.wav", audio, SR)
if os.environ.get("NOPLAY") != "1":
    import sounddevice as sd

    sd.play(audio, SR)
    sd.wait()
