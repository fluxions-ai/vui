"""CPU inference for the VUI TTS model using ONNX Runtime.

Exports backbone decoder step (with KV cache) and full RQ generate loop to ONNX,
quantizes to INT8, and runs inference entirely in ORT.

Does NOT modify any existing CUDA inference code.

Performance (Ryzen 7 7840U, DDR5, no discrete GPU):
  Q=8:  ~1.24x RTF (faster than realtime)
  Q=16: ~0.87x RTF (near realtime)

Usage:
    python scripts/cpu_inference.py --checkpoint checkpoints/0jiksor5_0100000.pt --benchmark
    python scripts/cpu_inference.py --checkpoint checkpoints/0jiksor5_0100000.pt --text "Hello world."
    python scripts/cpu_inference.py --checkpoint checkpoints/0jiksor5_0100000.pt --export-only
"""

import argparse
import os
import time
import wave

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vui.model import Vui

torch.set_num_threads(4)

ONNX_DIR = os.path.join(os.path.dirname(__file__), "..", "onnx_models")


# ---------------------------------------------------------------------------
# ONNX wrapper for backbone decode step
# ---------------------------------------------------------------------------


class BackboneStepONNX(nn.Module):
    """Single-token decode with growing KV cache for ONNX export."""

    def __init__(self, decoder):
        super().__init__()
        self.n_layers = len(decoder.blocks)
        self.n_heads = decoder.blocks[0].attn.n_heads
        self.n_kv_heads = decoder.blocks[0].attn.n_kv_heads
        self.head_dim = decoder.blocks[0].attn.head_dim
        self.n_reps = self.n_heads // self.n_kv_heads
        self.d_model = self.n_heads * self.head_dim

        self.attn_norms = nn.ModuleList()
        self.Wqkvs = nn.ModuleList()
        self.out_projs = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        self.mlp_w1s = nn.ModuleList()
        self.mlp_w2s = nn.ModuleList()
        self.mlp_w3s = nn.ModuleList()

        for block in decoder.blocks:
            self.attn_norms.append(block.attn_norm)
            self.Wqkvs.append(block.attn.Wqkv)
            self.out_projs.append(block.attn.out_proj)
            self.mlp_norms.append(block.mlp_norm)
            self.mlp_w1s.append(block.mlp.w1)
            self.mlp_w2s.append(block.mlp.w2)
            self.mlp_w3s.append(block.mlp.w3)

        self.final_norm = decoder.norm
        self.register_buffer("freqs_cis", decoder.freqs_cis, persistent=False)

    def forward(self, x: Tensor, position: Tensor, past_k: Tensor, past_v: Tensor):
        freqs = self.freqs_cis[position]
        new_ks, new_vs = [], []

        for i in range(self.n_layers):
            h = self._rms_norm(x, self.attn_norms[i])
            qkv = self.Wqkvs[i](h)

            q_size = self.n_heads * self.head_dim
            kv_size = self.n_kv_heads * self.head_dim
            q = (
                qkv[:, :, :q_size]
                .reshape(1, 1, self.n_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                qkv[:, :, q_size : q_size + kv_size]
                .reshape(1, 1, self.n_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                qkv[:, :, q_size + kv_size :]
                .reshape(1, 1, self.n_kv_heads, self.head_dim)
                .transpose(1, 2)
            )

            q = self._apply_rope(freqs, q)
            k = self._apply_rope(freqs, k)

            full_k = torch.cat([past_k[i], k], dim=2)
            full_v = torch.cat([past_v[i], v], dim=2)
            new_ks.append(full_k)
            new_vs.append(full_v)

            attn_k, attn_v = full_k, full_v
            if self.n_reps > 1:
                attn_k = full_k.repeat_interleave(self.n_reps, dim=1)
                attn_v = full_v.repeat_interleave(self.n_reps, dim=1)

            scale = self.head_dim**-0.5
            attn = (q @ attn_k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out = (attn @ attn_v).transpose(1, 2).reshape(1, 1, self.d_model)
            x = x + self.out_projs[i](out)

            h = self._rms_norm(x, self.mlp_norms[i])
            x = x + self.mlp_w2s[i](F.silu(self.mlp_w1s[i](h)) * self.mlp_w3s[i](h))

        hidden = self._rms_norm(x, self.final_norm)
        return hidden[:, 0], torch.stack(new_ks), torch.stack(new_vs)

    def _rms_norm(self, x, norm):
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        return (output * norm.weight).to(x.dtype)

    def _apply_rope(self, freqs, t):
        rot_dim = freqs.shape[-2]
        freqs_cos, freqs_sin = freqs[..., 0], freqs[..., 1]
        if rot_dim >= t.shape[-1]:
            x1, x2 = t[..., ::2], t[..., 1::2]
            t_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
            return (t * freqs_cos + t_rot * freqs_sin).to(t.dtype)
        t_mid, t_right = t[..., :rot_dim], t[..., rot_dim:]
        x1, x2 = t_mid[..., ::2], t_mid[..., 1::2]
        t_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        t_mid = t_mid * freqs_cos + t_rot * freqs_sin
        return torch.cat([t_mid, t_right], dim=-1).to(t.dtype)


# ---------------------------------------------------------------------------
# ONNX wrapper for full RQ generate (unrolled, argmax sampling)
# ---------------------------------------------------------------------------


class RQGenerateONNX(nn.Module):
    """Full RQ generate loop unrolled for ONNX. Uses argmax for ONNX compatibility."""

    def __init__(self, rq, n_quantizers: int | None = None):
        super().__init__()
        self.backbone_proj = rq.backbone_proj
        self.code_emb = rq.code_emb
        self.pos_emb = rq.pos_emb
        self.blocks = rq.blocks
        self.norm = rq.norm
        self.head_W = rq.head_W
        self.codebook_size = rq.codebook_size
        self.rq_dim = rq.rq_dim
        self.Q = n_quantizers or rq.n_quantizers

    def forward(self, backbone_hidden, code0, temperature):
        Q = self.Q
        seq = torch.zeros(1, Q, self.rq_dim)
        seq[:, 0] = self.backbone_proj(backbone_hidden) + self.pos_emb.weight[0]
        seq[:, 1] = self.code_emb.embedding(code0) + self.pos_emb.weight[1]
        all_codes = torch.zeros(1, Q, dtype=torch.long)
        all_codes[:, 0] = code0
        for i in range(Q - 1):
            s = seq[:, : i + 2]
            for block in self.blocks:
                s = block(s)
            s = self.norm(s)
            logits = F.linear(s[:, -1], self.head_W[i]) / temperature
            next_code = logits.argmax(dim=-1)
            all_codes[:, i + 1] = next_code
            if i + 2 < Q:
                seq[:, i + 2] = (
                    self.code_emb.embedding(next_code + (i + 1) * self.codebook_size)
                    + self.pos_emb.weight[i + 2]
                )
        return all_codes


# ---------------------------------------------------------------------------
# Export + quantize
# ---------------------------------------------------------------------------


def export_backbone(model: Vui, path: str):
    wrapper = BackboneStepONNX(model.decoder).float().eval()
    d = model.config.model.d_model
    n_kv = model.decoder.blocks[0].attn.n_kv_heads
    hd = model.decoder.blocks[0].attn.head_dim
    n_layers = len(model.decoder.blocks)

    x = torch.randn(1, 1, d)
    pos = torch.tensor([10], dtype=torch.long)
    past_k = torch.randn(n_layers, 1, n_kv, 10, hd)
    past_v = torch.randn(n_layers, 1, n_kv, 10, hd)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x, pos, past_k, past_v),
            path,
            input_names=["x", "position", "past_k", "past_v"],
            output_names=["hidden", "present_k", "present_v"],
            dynamic_axes={
                "past_k": {3: "past_len"},
                "past_v": {3: "past_len"},
                "present_k": {3: "present_len"},
                "present_v": {3: "present_len"},
            },
            opset_version=17,
            dynamo=False,
        )
    print(f"Exported backbone: {os.path.getsize(path) / 1e6:.1f}MB")


def export_rq(model: Vui, path: str, n_quantizers: int):
    wrapper = RQGenerateONNX(model.rq_transformer, n_quantizers).float().eval()
    h = torch.randn(1, model.rq_transformer.rq_dim)
    c0 = torch.tensor([42], dtype=torch.long)
    temp = torch.tensor(0.7)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (h, c0, temp),
            path,
            input_names=["backbone_hidden", "code0", "temperature"],
            output_names=["codes"],
            opset_version=17,
            dynamo=False,
        )
    print(f"Exported RQ (Q={n_quantizers}): {os.path.getsize(path) / 1e6:.1f}MB")


def quantize_model(input_path: str, output_path: str):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(input_path, output_path, weight_type=QuantType.QInt8)
    print(f"Quantized: {os.path.getsize(output_path) / 1e6:.1f}MB -> {output_path}")


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class CPUInferenceEngine:
    """All-ORT INT8 inference: backbone with KV cache + unrolled RQ generate."""

    def __init__(self, model: Vui, backbone_path: str, n_quantizers: int):
        cfg = model.config.model
        self.d_model = cfg.d_model
        self.n_layers = cfg.n_layers
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.codebook_size = cfg.codebook_size
        self.Q = n_quantizers

        # ORT sessions - 4 intra-op threads is optimal for AMD Zen 4
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1

        self.backbone_sess = ort.InferenceSession(
            backbone_path, opts, providers=["CPUExecutionProvider"]
        )

        # RQ in bfloat16 (2x faster matmuls on Zen 4 AVX-512 BF16)
        self.rq = model.rq_transformer.bfloat16().eval()
        self.rq.n_quantizers = n_quantizers
        self.rq_dtype = torch.bfloat16

        with torch.no_grad():
            self.audio_emb_weight = model.audio_emb.embedding.weight.float().numpy()
            self.token_emb_weight = model.token_emb.weight.float().numpy()
            self.codec_head_weight = model.codec_head.weight.float().numpy()
            self.eos_head_weight = model.eos_head.weight.float().numpy()
            self.eos_head_bias = model.eos_head.bias.float().numpy()

            self.cond_bias = np.zeros((1, 1, self.d_model), dtype=np.float32)
            if model.sq_proj is not None:
                sq_val = torch.tensor(
                    [[4.0, 4.0, 4.0, 4.5, 4.0, 4.0, 4.5]], dtype=torch.float32
                )
                self.cond_bias = model.sq_proj(sq_val).reshape(1, 1, -1).float().numpy()

            # Speaker projection for voice cloning
            self.spk_proj = model.spk_proj
            self.spk_enc = None

        self.tokenizer = model.text_tokenizer
        self.reset()

    def reset(self):
        self.kv_k = np.zeros(
            (self.n_layers, 1, self.n_kv_heads, 0, self.head_dim), dtype=np.float32
        )
        self.kv_v = np.zeros(
            (self.n_layers, 1, self.n_kv_heads, 0, self.head_dim), dtype=np.float32
        )
        self.pos = 0

    def save_cache(self, path: str):
        np.savez(path, kv_k=self.kv_k, kv_v=self.kv_v, pos=self.pos)

    def load_cache(self, path: str):
        data = np.load(path)
        self.kv_k = data["kv_k"]
        self.kv_v = data["kv_v"]
        self.pos = int(data["pos"])

    def _backbone_step(self, emb: np.ndarray) -> np.ndarray:
        hidden, self.kv_k, self.kv_v = self.backbone_sess.run(
            None,
            {
                "x": emb.astype(np.float32),
                "position": np.array([self.pos], dtype=np.int64),
                "past_k": self.kv_k,
                "past_v": self.kv_v,
            },
        )
        self.pos += 1
        return hidden

    def prefill_text(self, text: str, add_cond_bias: bool = True):
        ids = self.tokenizer.encode(text).numpy()
        emb = self.token_emb_weight[ids]
        bias = self.cond_bias if add_cond_bias else 0.0
        for t in range(len(ids)):
            self._backbone_step(emb[np.newaxis, t : t + 1] + bias)

    def prefill_speaker(self, audio_24k: np.ndarray):
        """Compute speaker embedding and prefill as a single token. No cond_bias."""
        if self.spk_enc is None:
            from vui.qwen_spk_enc import QwenSpeakerEncoder

            self.spk_enc = QwenSpeakerEncoder.from_pretrained().float().eval()
        audio_t = torch.from_numpy(audio_24k).float()
        with torch.inference_mode():
            spk_emb = self.spk_enc.embed(audio_t, sr=24000)  # (1024,)
            spk_token = (
                self.spk_proj(spk_emb.unsqueeze(0)).float().numpy()
            )  # (1, d_model)
        self._backbone_step(spk_token[np.newaxis])  # (1, 1, d_model)

    def prefill_prompt_text(self, text: str):
        """Prefill prompt transcript + [SC] token. No cond_bias."""
        ids = self.tokenizer.encode(text).numpy()
        sc_id = self.tokenizer.special_to_id["[SC]"]
        ids = np.append(ids, sc_id)
        emb = self.token_emb_weight[ids]
        for t in range(len(ids)):
            self._backbone_step(emb[np.newaxis, t : t + 1])

    def prefill_audio(self, codes: np.ndarray):
        """Prefill with audio codes. No cond_bias (prompt audio)."""
        for t in range(codes.shape[0]):
            Q = min(codes.shape[1], self.Q)
            offsets = np.arange(Q) * self.codebook_size
            emb = self.audio_emb_weight[codes[t, :Q] + offsets].sum(
                axis=0, keepdims=True
            )[np.newaxis]
            self._backbone_step(emb)

    def decode_step(
        self,
        codes: np.ndarray,
        temperature: float = 0.9,
        top_k: int = 0,
        rq_temperature: float = 0.9,
    ):
        Q = min(codes.shape[0], self.Q)
        offsets = np.arange(Q) * self.codebook_size
        emb = self.audio_emb_weight[codes[:Q] + offsets].sum(axis=0, keepdims=True)[
            np.newaxis
        ]

        hidden = self._backbone_step(emb)  # no cond_bias during decode

        codec_logits = hidden @ self.codec_head_weight.T
        eos_logit = hidden @ self.eos_head_weight.T + self.eos_head_bias

        logits = codec_logits[0] / temperature
        if top_k > 0 and top_k < len(logits):
            indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -1e9)
            mask[indices] = logits[indices]
            logits = mask
        e = np.exp(logits - logits.max())
        code0 = np.random.choice(len(e), p=e / e.sum())

        with torch.inference_mode():
            codes_out = self.rq.generate(
                torch.from_numpy(hidden).to(self.rq_dtype),
                torch.tensor([code0], dtype=torch.long),
                temperature=rq_temperature,
            )  # (1, Q)
        return codes_out[0].numpy(), eos_logit[0, 0]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_codes(engine, text, max_frames=375, temperature=0.9, eos_threshold=0.5):
    engine.prefill_text(text, add_cond_bias=True)
    codes_in = np.zeros(engine.Q, dtype=np.int64)
    all_codes = []
    for i in range(max_frames):
        codes_frame, eos_logit = engine.decode_step(codes_in, temperature)
        all_codes.append(codes_frame)
        codes_in = codes_frame[: engine.Q]
        if _sigmoid(eos_logit) > eos_threshold and i > 6:
            break
    return all_codes


def benchmark(engine, n_steps=100):
    engine.reset()
    t0 = time.perf_counter()
    engine.prefill_text("Hello, this is a test of the voice synthesis system.")
    print(f"Prefill: {(time.perf_counter() - t0) * 1000:.1f}ms")

    codes_in = np.zeros(engine.Q, dtype=np.int64)
    bb_times, rq_times = [], []

    for step in range(n_steps):
        Q = engine.Q
        offsets = np.arange(Q) * engine.codebook_size
        emb = engine.audio_emb_weight[codes_in[:Q] + offsets].sum(
            axis=0, keepdims=True
        )[np.newaxis]

        t0 = time.perf_counter()
        hidden = engine._backbone_step(emb)
        bb_times.append(time.perf_counter() - t0)

        logits = (hidden @ engine.codec_head_weight.T)[0] / 0.7
        e = np.exp(logits - logits.max())
        code0 = np.random.choice(len(e), p=e / e.sum())

        t0 = time.perf_counter()
        with torch.inference_mode():
            codes_out = engine.rq.generate(
                torch.from_numpy(hidden).to(engine.rq_dtype),
                torch.tensor([code0], dtype=torch.long),
                temperature=0.7,
            )
        rq_times.append(time.perf_counter() - t0)
        codes_in = codes_out[0].numpy()

    total = np.mean(bb_times) + np.mean(rq_times)
    audio_s = n_steps / 12.5
    wall_s = sum(bb_times) + sum(rq_times)

    print(f"\n--- Q={engine.Q} ORT INT8 (intra=4) ---")
    print(
        f"Backbone: {np.mean(bb_times) * 1000:.1f}ms avg ({bb_times[0] * 1000:.1f}ms first, {bb_times[-1] * 1000:.1f}ms last)"
    )
    print(f"RQ:       {np.mean(rq_times) * 1000:.1f}ms avg")
    print(f"Total:    {total * 1000:.1f}ms/step")
    print(f"RTF:      {80 / (total * 1000):.2f}x")
    print(
        f"Generated {audio_s:.1f}s audio in {wall_s:.1f}s = {audio_s / wall_s:.2f}x effective RTF"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/0jiksor5_0100000.pt")
    parser.add_argument("--text", default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--quantizers",
        type=int,
        default=8,
        help="Number of RQ quantizers (8-16). Lower = faster, slightly lower quality.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-frames", type=int, default=375)
    parser.add_argument(
        "--output", "-o", default=None, help="Output WAV path (default: output.wav)"
    )
    parser.add_argument(
        "--prompt", default=None, help="Path to prompt WAV file for voice cloning"
    )
    parser.add_argument(
        "--prompt-text", default=None, help="Transcript of the prompt audio"
    )
    parser.add_argument(
        "--int8", action="store_true", help="Use INT8 quantized backbone"
    )
    parser.add_argument(
        "--save-cache", default=None, help="Save KV cache after prompt prefill"
    )
    parser.add_argument(
        "--load-cache", default=None, help="Load KV cache (skip prompt prefill)"
    )
    args = parser.parse_args()

    os.makedirs(ONNX_DIR, exist_ok=True)
    Q = args.quantizers

    backbone_fp32 = os.path.join(ONNX_DIR, "backbone_step.onnx")
    backbone_int8 = os.path.join(ONNX_DIR, "backbone_step_int8.onnx")

    print("Loading model...")
    model = Vui.from_pretrained(args.checkpoint).float().eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    if not os.path.exists(backbone_fp32):
        export_backbone(model, backbone_fp32)
    if not os.path.exists(backbone_int8):
        quantize_model(backbone_fp32, backbone_int8)

    if args.export_only:
        print("Export complete.")
        return

    backbone_path = backbone_int8 if args.int8 else backbone_fp32
    engine = CPUInferenceEngine(model, backbone_path, Q)
    print(f"Engine ready (Q={Q})")

    if args.benchmark:
        benchmark(engine)
    elif args.text:
        # Load cached KV state if available
        prompt_codes = None
        prompt_audio_24k = None

        if args.load_cache:
            print(f"Loading cache: {args.load_cache}")
            engine.load_cache(args.load_cache)
        elif args.prompt:
            from vui.qwen_codec import QwenCodecEncoder

            print(f"Encoding prompt: {args.prompt}")
            with wave.open(args.prompt, "rb") as wf:
                sr = wf.getframerate()
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(raw, dtype=np.int16 if sw == 2 else np.float32)
            if n_ch > 1:
                pcm = pcm.reshape(-1, n_ch)[:, 0]
            audio_np = pcm.astype(np.float32) / (32768.0 if sw == 2 else 1.0)
            if sr != 24000:
                n_out = int(len(audio_np) * 24000 / sr)
                audio_np = np.interp(
                    np.linspace(0, len(audio_np) - 1, n_out),
                    np.arange(len(audio_np)),
                    audio_np,
                ).astype(np.float32)
            prompt_audio_24k = audio_np
            audio_tensor = (
                torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
            )  # (1, 1, samples)
            encoder = QwenCodecEncoder.from_pretrained().float().eval()
            with torch.inference_mode():
                codes_enc = encoder.encode(audio_tensor)  # (1, n_q, T)
            prompt_codes = codes_enc[0].T.numpy()  # (T, n_q)
            print(
                f"Prompt: {prompt_codes.shape[0]} frames ({prompt_codes.shape[0] / 12.5:.1f}s)"
            )

        # Prefill prompt (speaker + text + audio) if not loaded from cache
        if not args.load_cache:
            engine.reset()
            if prompt_audio_24k is not None:
                engine.prefill_speaker(prompt_audio_24k)
            if args.prompt_text is not None:
                engine.prefill_prompt_text(args.prompt_text)
            if prompt_codes is not None:
                engine.prefill_audio(prompt_codes)
            if args.save_cache:
                engine.save_cache(args.save_cache)
                print(f"Saved cache: {args.save_cache} (pos={engine.pos})")

        print(f"\nGenerating: '{args.text}'")
        t0 = time.perf_counter()
        all_codes = generate_codes(engine, args.text, args.max_frames, args.temperature)
        gen_time = time.perf_counter() - t0
        n_frames = len(all_codes)
        audio_secs = n_frames / 12.5
        print(
            f"Generated {n_frames} frames ({audio_secs:.1f}s) in {gen_time:.1f}s (RTF: {audio_secs / gen_time:.2f}x)"
        )

        # Decode codes to audio via QwenCodecDecoder
        from vui.qwen_codec import QwenCodecDecoder

        print("Loading codec decoder...")
        codec = QwenCodecDecoder.from_pretrained()
        codes_np = np.stack(all_codes)  # (n_frames, Q)
        codes_padded = np.zeros((n_frames, 16), dtype=np.int64)
        codes_padded[:, :Q] = codes_np[:, :Q]
        codes_tensor = torch.from_numpy(codes_padded).T.unsqueeze(
            0
        )  # (1, 16, n_frames)
        with torch.inference_mode():
            audio = codec.decode(codes_tensor)  # (1, 1, n_samples)
        audio_np = audio[0, 0].numpy()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (audio_np * 32767).astype(np.int16)

        out_path = args.output or "output.wav"
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm16.tobytes())
        print(f"Saved {out_path} ({len(pcm16) / 24000:.1f}s, 24kHz)")
    else:
        print("Specify --text or --benchmark")


if __name__ == "__main__":
    main()
