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

from vui.qwen_codec import QwenCodecDecoder


class CodecWorker:
    def __init__(self):
        print("[CODEC] Loading QwenCodecDecoder on MPS...")
        self.codec_dec = QwenCodecDecoder.from_pretrained().to("mps").float().eval()
        self._quant_buf = None
        self._transformer_out = None
        self._warmup()

    def _warmup(self):
        """Warm up codec MPS shaders."""
        dummy_codes = torch.zeros(1, 16, 1, dtype=torch.long, device="mps")
        self.codec_dec.pre_transformer.reset_cache()
        with torch.inference_mode():
            self.codec_dec.decode_cached_frame(dummy_codes, vocoder_ctx=1)
        self.codec_dec.pre_transformer.reset_cache()
        self._quant_buf = None
        self._transformer_out = None
        print("[CODEC] Warmed up")

    def prefill_context(self, codes_pt: torch.Tensor):
        """Prefill codec KV cache with context codes.

        codes_pt: (B, Q, T) on CPU, will be moved to MPS.
        """
        t0 = time.perf_counter()
        codes_pt = codes_pt.to("mps")

        with torch.inference_mode():
            # decode_cached_prefill does all the work: quantizer decode, pre_conv, transformer
            self.codec_dec.decode_cached_prefill(codes_pt)

        t_prefill = (time.perf_counter() - t0) * 1000
        print(f"[CODEC] Prefilled: {codes_pt.shape[2]} frames in {t_prefill:.1f}ms")

    def decode_frame(
        self, codes_pt: torch.Tensor, vocoder_ctx: int = 10
    ) -> torch.Tensor:
        """Decode single frame using KV-cached path.

        codes_pt: (1, Q, 1) on CPU, will be moved to MPS.
        Returns: (1, 1, T*1920) audio on CPU.
        """
        codes_pt = codes_pt.to("mps")

        with torch.inference_mode(), torch.autocast("mps", enabled=False):
            audio = self.codec_dec.decode_cached_frame(
                codes_pt, vocoder_ctx=vocoder_ctx
            )

        return audio[0, 0].detach().float().cpu()

    def reset(self):
        """Reset codec state."""
        self.codec_dec.pre_transformer.reset_cache()
        self._quant_buf = None
        self._transformer_out = None


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
