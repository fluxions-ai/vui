"""IPC message types for multi-process streaming server.

All messages are plain dicts (must be picklable for multiprocessing.Queue).
"""

# -- main -> tts --
# {"cmd": "reset"}
# {"cmd": "set_cond", "wps": 1.0}
# {"cmd": "prefill_text", "text": "hello", "is_first": True}
# {"cmd": "prefill_audio", "codes": Tensor}  (T, Q) on CPU
# {"cmd": "generate", "chunk_text": str, "sc": bool, "is_first": bool,
#     "temperature": 0.8, "top_k": 300, "rep_penalty": 1.4, "rep_window": 24, "max_duration": 120}
# {"cmd": "cancel"}
# {"cmd": "warmup"}
# {"cmd": "save_kv", "name": str}
# {"cmd": "load_kv", "name": str, "path": str}
# {"cmd": "allocate_and_prefill", "codes": Tensor, "text": str|None, "settings": dict}
# {"cmd": "shutdown"}

# -- tts -> main --
# {"type": "audio", "data": Tensor, "T": int, "secs": float}  24kHz float32 on CPU
# {"type": "timing", **timing_dict}
# {"type": "chunk_done", "chunk_idx": int, "text": str, "secs": float, "frames": int, "gen_time": float, "ttfb": float}
# {"type": "done", "T": int, "total_secs": float}
# {"type": "cancelled", "T": int}
# {"type": "ready"}
# {"type": "kv_saved", "name": str, "ok": bool}
# {"type": "kv_loaded", "name": str, "ok": bool, "T": int, "text": str}
# {"type": "prompt_loaded", "T": int, "text": str}
# {"type": "error", "msg": str}

# -- main -> asr --
# {"cmd": "start"}
# {"cmd": "feed", "audio": ndarray}  16kHz float32
# {"cmd": "stop"}
# {"cmd": "shutdown"}

# -- asr -> main --
# {"type": "partial", "text": str}
# {"type": "final", "text": str}
# {"type": "ready"}

# -- main -> audio_in --
# {"cmd": "start"}
# {"cmd": "feed", "audio": ndarray}  24kHz int16
# {"cmd": "stop"}
# {"cmd": "shutdown"}

# -- audio_in -> main --
# {"type": "codes", "codes": Tensor}  (T, Q) on CPU
# {"type": "ready"}
