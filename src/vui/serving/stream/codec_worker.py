"""Codec decoder worker process for Apple Silicon using MPS.

Runs separately from TTS model to avoid GPU contention.
Handles KV-cached streaming decode with own Metal context.
"""

from __future__ import annotations

import time
import traceback

import numpy as np
import torch
from torch.multiprocessing import Queue


class CodecWorker:
    def __init__(self):
        import mlx.core as mx

        print("[CODEC] Loading MLX codec decoder...")
        from vui.mlx.tts.codec import load_codec_decoder_mlx

        self.codec = load_codec_decoder_mlx()
        self._mx = mx
        self._warmup()

    def _warmup(self):
        """Warm up MLX codec."""
        mx = self._mx
        dummy = mx.zeros((1, 16, 1), dtype=mx.int32)
        audio = self.codec.forward(dummy)
        mx.eval(audio)
        self.codec.reset_state()
        print("[CODEC] Warmed up")

    def prefill_context(self, codes_pt: torch.Tensor):
        """Prefill codec streaming context with prompt codes.

        codes_pt: (B, Q, T) on CPU torch tensor.
        """
        mx = self._mx
        t0 = time.perf_counter()
        codes_mx = mx.array(codes_pt.numpy().astype(np.int32))
        self.codec.reset_state()
        self.codec.prefill(codes_mx)
        mx.eval(self.codec.parameters())
        t_prefill = (time.perf_counter() - t0) * 1000
        print(f"[CODEC] Prefilled: {codes_pt.shape[2]} frames in {t_prefill:.1f}ms")

    def decode_frame(
        self, codes_pt: torch.Tensor, vocoder_ctx: int = 10
    ) -> torch.Tensor:
        """Decode single frame using streaming context.

        codes_pt: (1, Q, 1) on CPU torch tensor.
        Returns: (S,) audio on CPU torch tensor.
        """
        mx = self._mx
        codes_mx = mx.array(codes_pt.numpy().astype(np.int32))
        audio = self.codec.decode_frame(codes_mx)
        mx.eval(audio)
        return torch.from_numpy(np.array(audio.flatten())).float()

    def reset(self):
        """Reset codec state."""
        self.codec.reset_state()


def codec_process(cmd_queue: Queue, audio_queue: Queue):
    """Codec worker process entry point."""
    try:
        worker = CodecWorker()
        audio_queue.put({"type": "ready"})
    except Exception as e:
        traceback.print_exc()
        audio_queue.put({"type": "error", "msg": f"Failed to initialize: {str(e)}"})
        return

    try:
        while True:
            try:
                msg = cmd_queue.get()
            except Exception:
                break

            cmd = msg.get("cmd")

            if cmd == "shutdown":
                break

            elif cmd == "reset":
                worker.reset()
                audio_queue.put({"type": "codec_reset"})

            elif cmd == "prefill":
                try:
                    codes = msg["codes"]
                    if isinstance(codes, np.ndarray):
                        codes = torch.from_numpy(codes).long()
                    elif not isinstance(codes, torch.Tensor):
                        raise ValueError(
                            f"Expected Tensor or ndarray, got {type(codes)}"
                        )
                    else:
                        codes = codes.long()

                    if codes.dim() == 2:
                        codes = codes.unsqueeze(0)
                    worker.prefill_context(codes)
                    audio_queue.put({"type": "codec_prefilled"})
                except Exception as e:
                    print(f"[CODEC] prefill error: {e}")
                    traceback.print_exc()
                    audio_queue.put({"type": "error", "msg": str(e)})

            elif cmd == "decode_frame":
                try:
                    codes = msg["codes"]
                    vocoder_ctx = msg.get("vocoder_ctx", 10)

                    if isinstance(codes, np.ndarray):
                        codes = torch.from_numpy(codes).long()
                    elif not isinstance(codes, torch.Tensor):
                        raise ValueError(
                            f"Expected Tensor or ndarray, got {type(codes)}"
                        )
                    else:
                        codes = codes.long()

                    if codes.dim() == 2:
                        codes = codes.unsqueeze(0)
                    if codes.shape[0] != 1:
                        codes = codes.unsqueeze(0)

                    audio = worker.decode_frame(codes, vocoder_ctx=vocoder_ctx)
                    audio_queue.put({"type": "audio_frame", "audio": audio})
                except Exception as e:
                    print(f"[CODEC] decode_frame error: {e}")
                    traceback.print_exc()
                    audio_queue.put({"type": "error", "msg": str(e)})

            elif cmd == "decode_cached_frame":
                try:
                    codes = msg["codes"]
                    vocoder_ctx = msg.get("vocoder_ctx", 10)

                    # Handle both numpy arrays and torch tensors
                    if isinstance(codes, np.ndarray):
                        codes = torch.from_numpy(codes).long()
                    elif not isinstance(codes, torch.Tensor):
                        raise ValueError(
                            f"Expected Tensor or ndarray, got {type(codes)}"
                        )
                    else:
                        codes = codes.long()

                    if codes.dim() == 2:
                        codes = codes.unsqueeze(0)
                    if codes.shape[0] != 1:
                        codes = codes.unsqueeze(0)

                    audio = worker.decode_frame(codes, vocoder_ctx=vocoder_ctx)
                    audio_queue.put({"type": "audio_frame", "audio": audio})
                except Exception as e:
                    print(f"[CODEC] decode_cached_frame error: {e}")
                    traceback.print_exc()
                    audio_queue.put({"type": "error", "msg": str(e)})
    finally:
        print("[CODEC] Shutting down")
