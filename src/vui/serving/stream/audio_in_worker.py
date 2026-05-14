"""Audio input worker process: streaming codec encoder (GPU).

Receives raw 24kHz audio chunks, feeds to StreamingCodecEncoder,
sends back codec codes when available.
"""

import time
import traceback
from multiprocessing import Queue

import torch

from vui.qwen_codec import QwenCodecEncoder, StreamingCodecEncoder


def audio_in_process(cmd_queue: Queue, result_queue: Queue, n_quantizers: int = 16):
    """Main audio input worker loop."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("[AudioIn] Loading codec encoder...")
    codec_enc = QwenCodecEncoder.from_pretrained().cuda().half().eval()

    stream_enc = StreamingCodecEncoder(
        codec_enc, n_quantizers=n_quantizers, max_secs=10, min_chunk_frames=12
    )

    print("[AudioIn] Setting up CUDA graph...")
    stream_enc.setup_graph()

    print("[AudioIn] Ready!")
    result_queue.put({"type": "ready"})

    recording = False

    while True:
        try:
            msg = cmd_queue.get()
        except Exception:
            break

        cmd = msg.get("cmd")

        if cmd == "shutdown":
            break

        elif cmd == "start":
            recording = True
            stream_enc.reset()

        elif cmd == "feed":
            if not recording:
                continue
            try:
                audio = msg["audio"]  # numpy float32 array, 24kHz
                audio_t = torch.from_numpy(audio).float()
                codes = stream_enc.feed(audio_t)
                if codes is not None:
                    result_queue.put({"type": "codes", "codes": codes.cpu()})
            except Exception:
                traceback.print_exc()

        elif cmd == "stop":
            recording = False
            try:
                t0 = time.perf_counter()
                stream_enc.flush()
                torch.cuda.synchronize()
                total_frames = stream_enc._emitted
                codes = None
                if total_frames > 0:
                    if stream_enc._graph is not None:
                        stream_enc._graph.replay()
                        codes = (
                            stream_enc._codes_out[0, :, :total_frames].T.long().cpu()
                        )
                    else:
                        c = stream_enc.encoder(stream_enc._audio_buf)
                        codes = c[0, :, :total_frames].T.long().cpu()
                t_flush = (time.perf_counter() - t0) * 1000
                print(f"[AudioIn] Flush: {t_flush:.1f}ms, {total_frames} frames")

                if codes is not None:
                    result_queue.put({"type": "codes_final", "codes": codes})
                else:
                    result_queue.put({"type": "codes_final", "codes": None})
            except Exception:
                traceback.print_exc()
                result_queue.put({"type": "codes_final", "codes": None})

        elif cmd == "encode_full":
            # One-shot encode for prompt upload
            try:
                audio = msg["audio"]  # numpy float32 array, 24kHz
                audio_t = (
                    torch.from_numpy(audio).float().cuda().half().reshape(1, 1, -1)
                )
                t0 = time.perf_counter()
                codes = codec_enc.encode(audio_t)
                codes = codes[0].T.long().cpu()  # (1, Q, T) -> (T, Q)
                t_enc = (time.perf_counter() - t0) * 1000
                print(f"[AudioIn] Full encode: {t_enc:.1f}ms, {codes.shape[0]} frames")
                result_queue.put({"type": "encoded", "codes": codes})
            except Exception:
                traceback.print_exc()
                result_queue.put({"type": "encoded", "codes": None})

    print("[AudioIn] Shutting down")
