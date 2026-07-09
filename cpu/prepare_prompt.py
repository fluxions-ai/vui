"""Prepare a voice cloning prompt for C inference.

Takes a WAV file, transcribes it with faster-whisper, encodes the speaker
embedding + prompt text + audio codes into a KV cache file for vui_tts.

Usage:
    .venv/bin/python scripts/prepare_prompt.py ~/good_prompt.wav prompt_cache.bin
    ./vui_tts vui_full.bin --kv-cache prompt_cache.bin --text "Hello world." -o out.wav
"""

import struct
import sys
import wave

import numpy as np
import torch


def load_wav_24k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16 if sw == 2 else np.float32)
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch)[:, 0]
    audio = pcm.astype(np.float32) / (32768.0 if sw == 2 else 1.0)
    if sr != 24000:
        n_out = int(len(audio) * 24000 / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio


def transcribe(path: str) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel("tiny", compute_type="int8", device="cpu")
    segments, _ = model.transcribe(path, language="en")
    return " ".join(s.text.strip() for s in segments)


def prepare_prompt(
    audio_path: str,
    output_path: str,
    checkpoint: str = "checkpoints/0jiksor5_0100000.pt",
):
    print(f"Transcribing {audio_path}...")
    prompt_text = transcribe(audio_path)
    print(f"Transcript: '{prompt_text}'")

    print("Loading model...")
    from vui.model import Vui

    model = Vui.from_pretrained(checkpoint).float().eval()
    cfg = model.config.model

    print("Loading codec encoder...")
    from vui.qwen_codec import QwenCodecEncoder

    encoder = QwenCodecEncoder.from_pretrained().float().eval()

    print("Loading speaker encoder...")
    from vui.qwen_spk_enc import QwenSpeakerEncoder

    spk_enc = QwenSpeakerEncoder.from_pretrained().float().eval()

    audio_24k = load_wav_24k(audio_path)
    print(f"Audio: {len(audio_24k) / 24000:.1f}s")

    # Encode audio to codes
    audio_tensor = torch.from_numpy(audio_24k).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        codes = encoder.encode(audio_tensor)  # (1, n_q, T)
    prompt_codes = codes[0].T  # (T, n_q)
    print(f"Encoded: {prompt_codes.shape[0]} frames, {prompt_codes.shape[1]} codebooks")

    # Allocate static KV cache and prefill using PyTorch directly
    n_layers = cfg.n_layers
    n_kv_heads = cfg.n_kv_heads or cfg.n_heads
    head_dim = cfg.d_model // cfg.n_heads
    max_seq = model.decoder.max_seqlen
    Q = cfg.n_quantizers
    codebook_size = cfg.codebook_size

    model.decoder.allocate_inference_cache(1, "cpu", torch.float32)
    pos = 0

    def prefill_step(emb, n_tokens):
        nonlocal pos
        input_pos = torch.arange(pos, pos + n_tokens)
        with torch.inference_mode():
            model.decoder.forward(emb, input_pos)
        pos += n_tokens

    # 1. Speaker embedding token
    print("Prefilling speaker embedding...")
    with torch.inference_mode():
        spk_emb = spk_enc.embed(torch.from_numpy(audio_24k).float(), sr=24000)
        spk_token = model.spk_proj(spk_emb.unsqueeze(0)).unsqueeze(0)  # (1, 1, d_model)
    prefill_step(spk_token, 1)

    # 2. Prompt text + [SC] (no cond_bias)
    tok = model.text_tokenizer
    text_ids = tok.encode(prompt_text)
    sc_id = tok.special_to_id["[SC]"]
    text_ids = torch.cat([text_ids, torch.tensor([sc_id])])
    text_emb = model.token_emb(text_ids).unsqueeze(0)  # (1, T, d_model)
    print(f"Prefilling prompt text ({len(text_ids)} tokens)...")
    prefill_step(text_emb, len(text_ids))

    # 3. Prompt audio codes (no cond_bias)
    print(f"Prefilling audio ({prompt_codes.shape[0]} frames)...")
    for t in range(prompt_codes.shape[0]):
        frame_codes = prompt_codes[t, :Q]  # (Q,) - audio_emb adds offsets internally
        audio_emb = model.audio_emb(frame_codes.unsqueeze(0)).unsqueeze(
            0
        )  # (1, 1, d_model)
        prefill_step(audio_emb, 1)

    print(f"KV cache position: {pos}")

    # Extract KV cache and convert to C format
    # Static KV cache shape: (B, n_kv_heads, max_seqlen, head_dim)
    # C format: key_cache[layers, max_seq, kv_dim] where kv_dim = n_kv_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    key_cache = np.zeros((n_layers, max_seq, kv_dim), dtype=np.float32)
    value_cache = np.zeros((n_layers, max_seq, kv_dim), dtype=np.float32)

    for l in range(n_layers):
        kv = model.decoder.blocks[l].attn.kv_cache
        # kv.k_cache: (1, n_kv_heads, max_seqlen, head_dim)
        k = (
            kv.k_cache[0, :, :pos, :].detach().float().numpy()
        )  # (n_kv_heads, pos, head_dim)
        v = kv.v_cache[0, :, :pos, :].detach().float().numpy()
        # -> (pos, n_kv_heads * head_dim)
        key_cache[l, :pos] = k.transpose(1, 0, 2).reshape(pos, kv_dim)
        value_cache[l, :pos] = v.transpose(1, 0, 2).reshape(pos, kv_dim)

    with open(output_path, "wb") as f:
        f.write(struct.pack("i", pos))
        f.write(key_cache.tobytes())
        f.write(value_cache.tobytes())

    import os

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved {output_path} ({size_mb:.0f}MB, pos={pos})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <prompt.wav> <output_cache.bin> [checkpoint]")
        sys.exit(1)
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/0jiksor5_0100000.pt"
    prepare_prompt(sys.argv[1], sys.argv[2], ckpt)
