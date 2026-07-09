"""Export VUI model + codec decoder + tokenizer to single binary for C inference.

Binary format (vui2):
  Header (256 bytes) - same as vui1 plus codec/tokenizer offsets
  Backbone + RQ weights (same as vui1)
  Codec decoder weights
  Tokenizer vocab + merges

Build C: gcc -O3 -march=native -ffast-math -fopenmp -o vui_tts csrc/vui_tts.c -lm
Usage: .venv/bin/python scripts/export_full.py checkpoints/0jiksor5_0100000.pt vui_full.bin
"""

import functools
import json
import os
import struct
import sys

import numpy as np
import torch


def serialize_fp32(f, tensor):
    d = tensor.detach().cpu().float().contiguous().view(-1).numpy()
    f.write(d.tobytes())


@functools.lru_cache()
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def export(checkpoint_path: str, output_path: str):
    from vui.model import Vui
    from vui.qwen_codec import QwenCodecDecoder
    from vui.tokenizer import TokenizerConfig, VuiTokenizer

    print("Loading TTS model...")
    m = Vui.from_pretrained(checkpoint_path).float().eval()
    print("Loading codec decoder...")
    codec = QwenCodecDecoder.from_pretrained().float().eval()
    print("Loading tokenizer...")
    tokenizer = VuiTokenizer(
        TokenizerConfig(base_tokenizer="HuggingFaceTB/SmolLM2-135M")
    )

    d = m.decoder
    rq = m.rq_transformer
    b0 = d.blocks[0]

    bb_dim = b0.attn.dim
    bb_heads = b0.attn.n_heads
    bb_kv_heads = b0.attn.n_kv_heads
    bb_hidden = b0.mlp.w1.weight.shape[0]
    bb_layers = len(d.blocks)
    bb_max_seq = d.max_seqlen

    rq_dim = rq.rq_dim
    rb0 = rq.blocks[0]
    rq_heads = rb0.n_heads
    rq_hidden = rb0.w1.weight.shape[0]
    rq_layers = len(rq.blocks)
    rq_n_q = rq.n_quantizers
    rq_cs = rq.codebook_size

    vocab_size = m.token_emb.weight.shape[0]
    audio_emb_size = m.audio_emb.embedding.weight.shape[0]
    rope_theta = m.config.model.rope_theta

    with torch.no_grad():
        # Use the official global cond_bias shipped in the release prompt files
        # (identical across voices). Falls back to SQ_P90 if unavailable.
        cond_src = os.environ.get("VUI_COND_BIAS", "_hf/prompts/maeve.safetensors")
        if os.path.exists(cond_src):
            from safetensors.torch import load_file

            cond_bias = load_file(cond_src)["cond_bias"].float().reshape(-1)
            print(f"cond_bias: loaded from {cond_src} (norm {cond_bias.norm():.3f})")
        else:
            cond_bias = m.sq_proj(
                torch.tensor([[3.58, 3.95, 3.90, 4.25, 3.75, 4.03]])
            ).reshape(-1)
            print(f"cond_bias: computed from SQ_P90 (norm {cond_bias.norm():.3f})")
        eos_bias = m.eos_head.bias.item()

    sc_id = tokenizer.special_to_id["[SC]"]

    print(
        f"Backbone: d={bb_dim} h={bb_hidden} L={bb_layers} heads={bb_heads} kv={bb_kv_heads}"
    )
    print(
        f"RQ: d={rq_dim} h={rq_hidden} L={rq_layers} heads={rq_heads} Q={rq_n_q} CS={rq_cs}"
    )
    print(f"Vocab={vocab_size} SC_id={sc_id}")

    with open(output_path, "wb") as f:
        # ==================== HEADER (256 bytes) ====================
        header = struct.pack(
            "Ii14i2f2i",
            0x76756932,  # magic "vui2"
            2,  # version
            bb_dim,
            bb_hidden,
            bb_layers,
            bb_heads,
            bb_kv_heads,
            bb_max_seq,
            rq_dim,
            rq_hidden,
            rq_layers,
            rq_heads,
            rq_n_q,
            rq_cs,
            vocab_size,
            audio_emb_size,
            rope_theta,
            eos_bias,
            sc_id,
            0,  # sc_token_id, reserved
        )
        f.write(header)
        f.write(b"\x00" * (256 - len(header)))

        # ==================== BACKBONE WEIGHTS ====================
        print("Writing backbone weights...")
        for l in range(bb_layers):
            bl = d.blocks[l]
            serialize_fp32(f, bl.attn_norm.weight)
            serialize_fp32(f, bl.attn.Wqkv.weight)
            serialize_fp32(f, bl.attn.out_proj.weight)
            serialize_fp32(f, bl.mlp_norm.weight)
            serialize_fp32(f, bl.mlp.w1.weight)
            serialize_fp32(f, bl.mlp.w2.weight)
            serialize_fp32(f, bl.mlp.w3.weight)
        serialize_fp32(f, d.norm.weight)
        serialize_fp32(f, d.freqs_cis[:bb_max_seq])

        # Heads
        serialize_fp32(f, m.codec_head.weight)
        serialize_fp32(f, m.eos_head.weight)

        # Embeddings
        serialize_fp32(f, m.token_emb.weight)
        serialize_fp32(f, m.audio_emb.embedding.weight)
        serialize_fp32(f, cond_bias)

        # ==================== RQ WEIGHTS ====================
        print("Writing RQ weights...")
        for l in range(rq_layers):
            bl = rq.blocks[l]
            serialize_fp32(f, bl.attn_norm.weight)
            serialize_fp32(f, bl.Wqkv.weight)
            serialize_fp32(f, bl.out_proj.weight)
            serialize_fp32(f, bl.mlp_norm.weight)
            serialize_fp32(f, bl.w1.weight)
            serialize_fp32(f, bl.w2.weight)
            serialize_fp32(f, bl.w3.weight)
        serialize_fp32(f, rq.norm.weight)
        serialize_fp32(f, rq.code_emb.embedding.weight)
        serialize_fp32(f, rq.pos_emb.weight)
        serialize_fp32(f, rq.head_W)

        # ==================== CODEC DECODER WEIGHTS ====================
        print("Writing codec decoder weights...")
        codec_start = f.tell()

        # Quantizer codebooks
        sem_cb = codec.quantizer.semantic.codebooks[0].embed  # [2048, 256]
        serialize_fp32(f, sem_cb)
        serialize_fp32(f, codec.quantizer.semantic.output_proj.weight)  # [512, 256, 1]

        for i in range(15):
            serialize_fp32(f, codec.quantizer.acoustic.codebooks[i].embed)
        serialize_fp32(f, codec.quantizer.acoustic.output_proj.weight)  # [512, 256, 1]

        # Pre-conv: CausalConv1d(512, 1024, k=3)
        serialize_fp32(f, codec.pre_conv.conv.weight)  # [1024, 512, 3]
        serialize_fp32(f, codec.pre_conv.conv.bias)  # [1024]

        # Pre-transformer
        pt = codec.pre_transformer
        serialize_fp32(f, pt.input_proj.weight)  # [512, 1024]
        serialize_fp32(f, pt.input_proj.bias)  # [512]

        for l in range(8):
            bl = pt.layers[l]
            serialize_fp32(f, bl.input_layernorm.weight)  # [512]
            serialize_fp32(f, bl.q_proj.weight)  # [1024, 512]
            serialize_fp32(f, bl.k_proj.weight)  # [1024, 512]
            serialize_fp32(f, bl.v_proj.weight)  # [1024, 512]
            serialize_fp32(f, bl.o_proj.weight)  # [512, 1024]
            serialize_fp32(f, bl.attn_scale)  # [512]
            serialize_fp32(f, bl.post_attention_layernorm.weight)  # [512]
            serialize_fp32(f, bl.gate_proj.weight)  # [1024, 512]
            serialize_fp32(f, bl.up_proj.weight)  # [1024, 512]
            serialize_fp32(f, bl.down_proj.weight)  # [512, 1024]
            serialize_fp32(f, bl.mlp_scale)  # [512]

        serialize_fp32(f, pt.norm.weight)  # [512]
        serialize_fp32(f, pt.output_proj.weight)  # [1024, 512]
        serialize_fp32(f, pt.output_proj.bias)  # [1024]

        # Pre-compute codec RoPE (non-interleaved, head_dim=64, theta=10000)
        codec_max_seq = 1024
        head_dim = 64
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(codec_max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_seq, head_dim/2]
        # Store as [max_seq, head_dim/2, 2] (cos, sin) to match backbone format
        codec_rope = torch.stack(
            [freqs.cos(), freqs.sin()], dim=-1
        )  # [max_seq, hd/2, 2]
        serialize_fp32(f, codec_rope)

        # Upsample: 2 stages, each TransConv1d + ConvNeXtBlock
        for i in range(2):
            stage = codec.upsample[i]
            tconv = stage[0]  # CausalTransConv1d
            cnext = stage[1]  # ConvNeXtBlock
            serialize_fp32(f, tconv.conv.weight)  # [1024, 1024, 2]
            serialize_fp32(f, tconv.conv.bias)  # [1024]
            serialize_fp32(f, cnext.dwconv.conv.weight)  # [1024, 1, 7]
            serialize_fp32(f, cnext.dwconv.conv.bias)  # [1024]
            serialize_fp32(f, cnext.norm.weight)  # [1024]
            serialize_fp32(f, cnext.norm.bias)  # [1024]
            serialize_fp32(f, cnext.pwconv1.weight)  # [4096, 1024]
            serialize_fp32(f, cnext.pwconv1.bias)  # [4096]
            serialize_fp32(f, cnext.pwconv2.weight)  # [1024, 4096]
            serialize_fp32(f, cnext.pwconv2.bias)  # [1024]
            serialize_fp32(f, cnext.gamma)  # [1024]

        # Waveform decoder: initial conv + 4 blocks + final
        dec = codec.decoder
        # dec[0]: initial CausalConv1d(1024, 1536, k=7)
        serialize_fp32(f, dec[0].conv.weight)  # [1536, 1024, 7]
        serialize_fp32(f, dec[0].conv.bias)  # [1536]

        # dec[1]-dec[4]: DecoderBlocks
        block_configs = [
            (1536, 768, 8),  # block 1
            (768, 384, 5),  # block 2
            (384, 192, 4),  # block 3
            (192, 96, 3),  # block 4
        ]
        for bi, (in_dim, out_dim, stride) in enumerate(block_configs):
            blk = dec[bi + 1].block
            # blk[0]: SnakeBeta(in_dim)
            serialize_fp32(f, blk[0].alpha)
            serialize_fp32(f, blk[0].beta)
            # blk[1]: CausalTransConv1d(in_dim, out_dim, k=stride*2, s=stride)
            serialize_fp32(f, blk[1].conv.weight)  # [in_dim, out_dim, stride*2]
            serialize_fp32(f, blk[1].conv.bias)  # [out_dim]
            # blk[2]-blk[4]: DecoderResUnit(out_dim, dilation=1,3,9)
            for ri in range(2, 5):
                ru = blk[ri]
                serialize_fp32(f, ru.act1.alpha)
                serialize_fp32(f, ru.act1.beta)
                serialize_fp32(f, ru.conv1.conv.weight)  # [out_dim, out_dim, 7]
                serialize_fp32(f, ru.conv1.conv.bias)
                serialize_fp32(f, ru.act2.alpha)
                serialize_fp32(f, ru.act2.beta)
                serialize_fp32(f, ru.conv2.conv.weight)  # [out_dim, out_dim, 1]
                serialize_fp32(f, ru.conv2.conv.bias)

        # dec[5]: SnakeBeta(96)
        serialize_fp32(f, dec[5].alpha)
        serialize_fp32(f, dec[5].beta)
        # dec[6]: CausalConv1d(96, 1, k=7)
        serialize_fp32(f, dec[6].conv.weight)  # [1, 96, 7]
        serialize_fp32(f, dec[6].conv.bias)  # [1]

        codec_bytes = f.tell() - codec_start
        print(f"Codec: {codec_bytes / 1e6:.1f}MB")

        # ==================== TOKENIZER ====================
        print("Writing tokenizer...")
        tok_start = f.tell()

        # Get HF tokenizer internals
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        tj = json.loads(hf_tok.backend_tokenizer.to_str())
        hf_vocab = tj["model"]["vocab"]  # str -> id
        merges = tj["model"]["merges"]  # list of [str1, str2]
        base_vocab_size = len(hf_vocab)  # 49152

        # Build byte-to-unicode and reverse
        b2u = bytes_to_unicode()
        u2b = {v: k for k, v in b2u.items()}

        # Convert vocab strings to byte strings
        id_to_bytes = {}
        for ustr, tid in hf_vocab.items():
            blist = []
            for ch in ustr:
                if ch in u2b:
                    blist.append(u2b[ch])
                else:
                    blist.extend(ch.encode("utf-8"))
            id_to_bytes[tid] = bytes(blist)

        # Compute merge scores (higher = merge earlier = higher priority)
        scores = np.zeros(base_vocab_size, dtype=np.float32)
        n_merges = len(merges)
        for i, merge_pair in enumerate(merges):
            merged_str = merge_pair[0] + merge_pair[1]
            if merged_str in hf_vocab:
                scores[hf_vocab[merged_str]] = float(n_merges - i)

        # Write: n_vocab, max_token_len
        max_token_len = max(len(b) for b in id_to_bytes.values())
        f.write(struct.pack("ii", base_vocab_size, max_token_len))

        # For each token: score(f32), len(i32), bytes
        for tid in range(base_vocab_size):
            b = id_to_bytes.get(tid, b"")
            f.write(struct.pack("fi", scores[tid], len(b)))
            f.write(b)

        # Write byte token offset and special token info
        byte_offset = tokenizer.byte_offset  # 49152
        special_offset = tokenizer.special_offset  # 49408
        n_specials = len(tokenizer.special_to_id)

        f.write(struct.pack("iii", byte_offset, special_offset, n_specials))
        for tok_name, tok_id in sorted(
            tokenizer.special_to_id.items(), key=lambda x: x[1]
        ):
            name_bytes = tok_name.encode("utf-8")
            f.write(struct.pack("ii", tok_id, len(name_bytes)))
            f.write(name_bytes)

        tok_bytes = f.tell() - tok_start
        print(f"Tokenizer: {tok_bytes / 1e6:.1f}MB")

    total = os.path.getsize(output_path)
    print(f"\nWrote {output_path} ({total / 1e6:.1f}MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt> <output.bin>")
        sys.exit(1)
    export(sys.argv[1], sys.argv[2])
