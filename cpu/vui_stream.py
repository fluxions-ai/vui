"""RTF>1 streaming vui TTS on CPU.

Flow-controlled chunk pipeline: the backbone generates a chunk and PAUSES; Python decodes it
via ONNX and pushes audio into a buffer that a background callback drains at realtime; then it
signals the backbone to continue. Gen and codec never run concurrently (no memory-bandwidth
contention) so compute throughput stays >1x. A short PREBUFFER is built before playback starts
so chunk-granularity jitter can't underrun the device (no stutter).

Usage: python vui_stream.py "text"   [VOICE=maeve TEMP=0.6]
"""

import os
import subprocess
import sys
import threading
import time

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import soundfile as _sf

# prod text normalization: "9:30"->"nine thirty", "$45.50"->"forty-five dollars and fifty cents", etc.
from vui.inference import simple_clean

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE = os.environ.get("VOICE", "maeve")
TEMP = os.environ.get("TEMP", "0.6")
TEXT = " ".join(sys.argv[1:]).strip() or "Hello, this is a streaming test."
MODEL = f"{HERE}/vui_nano_full.bin"
CACHE = f"{HERE}/prompt_{VOICE}_official.bin"
ONNX = f"{HERE}/codec_q12.onnx"
NQ, SR, DOWN = 12, 24000, 1920
CTX = 6  # lookback frames for clean chunk boundaries
FIRST = 8  # small first chunk -> buffer starts filling fast
CHUNK = 24  # steady chunk; smaller -> shorter per-chunk production -> less jitter
PREBUFFER = (
    1.6  # seconds of audio to build before playback starts (covers one chunk's worth)
)
FADE = int(
    0.05 * SR
)  # crossfade last 50ms to silence (kills end-of-render artifacts, matches prod)

TEXT = simple_clean(TEXT)

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])


def decode(fr):
    return sess.run(None, {"codes": np.array(fr, dtype=np.int64).T[None]})[0][0, 0]


buf = np.zeros(0, dtype=np.float32)
lock = threading.Lock()
gen_done = False
started = False
pre = int(PREBUFFER * SR)


def cb(outdata, n, t, status):
    global buf, started
    with lock:
        if not started:
            if len(buf) >= pre or gen_done:
                started = True  # release playback once primed
            else:
                outdata[:] = 0
                return
        m = min(n, len(buf))
        outdata[:m, 0] = buf[:m]
        outdata[m:, 0] = 0
        buf = buf[m:]


stream = sd.OutputStream(
    samplerate=SR, channels=1, dtype="float32", callback=cb, latency="low"
)
stream.start()

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
        "--chunk-frames",
        str(CHUNK),
        "--first-chunk",
        str(FIRST),
        "--text",
        TEXT,
    ],
    cwd=HERE,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    env={**os.environ, "OMP_NUM_THREADS": "4"},
    text=True,
    bufsize=1,
)

frames, played, t0, _all = [], 0, None, []


def flush(final=False):
    global played, buf
    if played >= len(frames):
        return
    start, stop = played, len(frames)
    cs = min(CTX, start)
    new = decode(frames[start - cs : stop])[cs * DOWN :]
    if final:  # crossfade tail to silence (kills end artifacts)
        k = min(FADE, len(new))
        if k:
            new[-k:] = new[-k:] * np.linspace(1.0, 0.0, k, dtype=np.float32)
    with lock:
        buf = np.concatenate([buf, new])
    _all.append(new)
    played = stop


for line in proc.stdout:
    line = line.strip()
    if line == "END":
        break
    if line == "CHUNK":
        flush()
        try:
            proc.stdin.write("\n")
            proc.stdin.flush()
        except BrokenPipeError:
            break
        continue
    if line:
        if t0 is None:
            t0 = time.perf_counter()
        frames.append([int(x) for x in line.split()][:NQ])
flush(final=True)
gen_done = True
compute = time.perf_counter() - t0
audio_s = played * DOWN / SR
proc.wait()

_sf.write("/tmp/stream_out.wav", np.concatenate(_all), SR)
print(f'"{TEXT}"')
print(
    f"  {audio_s:.1f}s audio, compute {compute:.1f}s => {audio_s/compute:.2f}x realtime  (prebuffer {PREBUFFER}s)"
)
while True:  # drain
    with lock:
        if started and len(buf) == 0:
            break
    time.sleep(0.05)
time.sleep(0.1)
stream.stop()
stream.close()
