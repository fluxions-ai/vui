"""Resident vui TTS daemon: loads model+cache+ONNX+normalizer ONCE, holds the audio device,
and serves utterances over a unix socket so each `say` is warm (no reload). Streaming, RTF>1,
prebuffered (no stutter), prod text-normalized, with a 50ms end-crossfade.

Run:   .venv/bin/python vui_daemon.py        (blocks; socket appears once loaded)
Speak: connect /tmp/vui.sock, send text, get back "ok <rtf>".  (say.sh does this)
"""

import os
import socket
import subprocess
import threading
import time

import numpy as np
import onnxruntime as ort

# audio device, held by the daemon for its whole life
import sounddevice as sd

from vui.inference import simple_clean

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE = os.environ.get("VOICE", "maeve")
TEMP = os.environ.get("TEMP", "0.6")
SOCK = os.environ.get("VUI_SOCK", "/tmp/vui.sock")
MODEL = f"{HERE}/vui_nano_full.bin"
CACHE = f"{HERE}/prompt_{VOICE}_official.bin"
NQ = int(os.environ.get("VUI_NQ", "10"))
ONNX = f"{HERE}/codec_q{NQ}.onnx"
SR, DOWN = 24000, 1920
# tuned for fastest smooth start (warm daemon): ~250ms TTFA, 0 underruns on long utterances.
# chunk32 keeps lookback overhead low -> throughput comfortably >1x -> tiny prebuffer is safe.
CTX = int(os.environ.get("VUI_CTX", "4"))
FIRST = int(os.environ.get("VUI_FIRST", "4"))  # 4-frame first chunk -> fast first audio
CHUNK = int(
    os.environ.get("VUI_CHUNK", "32")
)  # large steady chunk -> low overhead -> smooth
PREBUFFER = float(os.environ.get("VUI_PREBUFFER", "0.35"))
FADE = int(0.05 * SR)

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])


def decode(fr):
    return sess.run(None, {"codes": np.array(fr, dtype=np.int64).T[None]})[0][0, 0]


buf = np.zeros(0, dtype=np.float32)
lock = threading.Lock()
started = False
pre = int(PREBUFFER * SR)
utt_done = False
underruns = 0


def cb(outdata, n, t, status):
    global buf, started, underruns
    with lock:
        if not started:
            if len(buf) >= pre or utt_done:
                started = True
            else:
                outdata[:] = 0
                return
        m = min(n, len(buf))
        outdata[:m, 0] = buf[:m]
        outdata[m:, 0] = 0
        buf = buf[m:]
        if m < n and not utt_done:
            underruns += 1  # buffer starved mid-stream = audible gap


stream = sd.OutputStream(
    samplerate=SR, channels=1, dtype="float32", callback=cb, latency="low"
)
stream.start()

# warm C backbone server (loads model+cache once)
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
        "--chunk-frames",
        str(CHUNK),
        "--first-chunk",
        str(FIRST),
        "--server",
    ],
    cwd=HERE,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    env={**os.environ, "OMP_NUM_THREADS": "4"},
    text=True,
    bufsize=1,
)


def wait_ready():
    while True:
        l = proc.stdout.readline()
        if not l:
            return False
        if l.strip() == "READY":
            return True


def speak(text):
    global buf, started, utt_done, underruns
    text = simple_clean(text)
    with lock:
        buf = np.zeros(0, dtype=np.float32)
    started = False
    utt_done = False
    underruns = 0
    frames, played, t0, first, allaudio = [], 0, None, None, []

    def flush(final=False):
        nonlocal played, first
        global buf
        if played >= len(frames):
            return
        a, b = played, len(frames)
        cs = min(CTX, a)
        new = decode(frames[a - cs : b])[cs * DOWN :]
        if final:
            k = min(FADE, len(new))
            if k:
                new[-k:] = new[-k:] * np.linspace(1.0, 0.0, k, dtype=np.float32)
        if first is None:
            first = time.perf_counter() - t0
        with lock:
            buf = np.concatenate([buf, new])
        allaudio.append(new)
        played = b

    proc.stdin.write(text + "\n")
    proc.stdin.flush()
    while True:
        l = proc.stdout.readline()
        if not l:
            break
        l = l.strip()
        if l == "END":
            break
        if l == "READY":
            continue  # trailing READY from the previous utterance
        if l == "CHUNK":
            flush()
            proc.stdin.write("\n")
            proc.stdin.flush()
            continue
        if l:
            if t0 is None:
                t0 = time.perf_counter()
            frames.append([int(x) for x in l.split()][:NQ])
    flush(final=True)
    utt_done = True
    if allaudio:
        import soundfile as _sf

        _sf.write("/tmp/say.wav", np.concatenate(allaudio), SR)
    compute = (time.perf_counter() - t0) if t0 else 0.0
    audio_s = played * DOWN / SR
    # wait for playback to drain before returning (so back-to-back calls don't overlap)
    while True:
        with lock:
            if started and len(buf) == 0:
                break
        time.sleep(0.03)
    rtf = audio_s / compute if compute else 0.0
    return (
        f"{text!r} | {audio_s:.1f}s in {compute:.1f}s => {rtf:.2f}x rt, TTFA {(first or 0)*1000:.0f}ms, "
        f"underruns {underruns} (Q{NQ} pre{PREBUFFER} first{FIRST} chunk{CHUNK})"
    )


wait_ready()  # block until model+cache loaded (first READY)
if os.path.exists(SOCK):
    os.remove(SOCK)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(SOCK)
srv.listen(8)
print(f"vui daemon ready on {SOCK} (voice={VOICE})", flush=True)
try:
    while True:
        conn, _ = srv.accept()
        text = conn.recv(65536).decode().strip()
        if text == "__QUIT__":
            conn.close()
            break
        try:
            msg = speak(text)
        except Exception as e:
            msg = f"error: {e}"
        try:
            conn.sendall(msg.encode())
        except Exception:
            pass
        conn.close()
finally:
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
    except Exception:
        pass
    os.path.exists(SOCK) and os.remove(SOCK)
    stream.stop()
    stream.close()
