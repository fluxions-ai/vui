"""Equivalence tests for the pure-PyTorch flash-attn fallback.

Runs on CPU — no flash_attn, no GPU needed. Where flash_attn *is* importable
(Linux x86_64 + CUDA) `test_matches_flash_attn` also checks the fallback
against the real kernel.

    pytest tests/test_flash_compat.py
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vui.flash_compat import HAS_FLASH_ATTN, _sdpa_attn_with_kvcache

torch.manual_seed(0)

B, S, H, HKV, D = 3, 16, 8, 2, 32


def _cache(dtype=torch.float32, batch=B):
    return (
        torch.randn(batch, S, HKV, D, dtype=dtype),
        torch.randn(batch, S, HKV, D, dtype=dtype),
    )


def _reference(q, keys, vals, klen, causal, window_left=-1):
    """Dense reference: explicit softmax over the filled prefix only."""
    Bq, Tq = q.shape[0], q.shape[1]
    out = torch.zeros_like(q)
    n_reps = q.shape[2] // keys.shape[2]
    for b in range(Bq):
        n = int(klen[b])
        for t in range(Tq):
            j_max = n - Tq + t if causal else n - 1
            lo = 0 if window_left < 0 else max(0, (n - Tq + t) - window_left)
            idx = list(range(lo, j_max + 1))
            if not idx:
                continue
            k = keys[b, idx].repeat_interleave(n_reps, dim=1)  # (n_sel, H, D)
            v = vals[b, idx].repeat_interleave(n_reps, dim=1)
            scores = torch.einsum("hd,nhd->hn", q[b, t], k) / math.sqrt(D)
            out[b, t] = torch.einsum("hn,nhd->hd", scores.softmax(-1), v)
    return out


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("Tq", [1, 4])
def test_matches_dense_reference(causal: bool, Tq: int):
    """Cache append + masking + GQA == dense attention over the filled prefix."""
    k_cache, v_cache = _cache()
    q = torch.randn(B, Tq, H, D)
    k, v = torch.randn(B, Tq, HKV, D), torch.randn(B, Tq, HKV, D)
    seqlens = torch.tensor([0, 5, 9], dtype=torch.int32)

    expected_k = k_cache.clone()
    expected_v = v_cache.clone()
    for b in range(B):
        expected_k[b, seqlens[b] : seqlens[b] + Tq] = k[b]
        expected_v[b, seqlens[b] : seqlens[b] + Tq] = v[b]

    out = _sdpa_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, cache_seqlens=seqlens, causal=causal
    )

    # K/V were appended in place at cache_seqlens.
    torch.testing.assert_close(k_cache, expected_k)
    torch.testing.assert_close(v_cache, expected_v)

    ref = _reference(q, expected_k, expected_v, seqlens + Tq, causal)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_sliding_window():
    k_cache, v_cache = _cache()
    q = torch.randn(B, 1, H, D)
    k, v = torch.randn(B, 1, HKV, D), torch.randn(B, 1, HKV, D)
    seqlens = torch.tensor([2, 7, 11], dtype=torch.int32)
    w = 3

    out = _sdpa_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=seqlens,
        causal=True,
        window_size=(w, 0),
    )
    ref = _reference(q, k_cache, v_cache, seqlens + 1, causal=True, window_left=w)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_cache_batch_idx():
    """Query rows read/write arbitrary cache slots, and only those slots."""
    n_slots = 5
    k_cache, v_cache = _cache(batch=n_slots)
    before_k, before_v = k_cache.clone(), v_cache.clone()
    slots = torch.tensor([4, 0, 2], dtype=torch.int32)
    seqlens_all = torch.tensor([3, 0, 6, 0, 1], dtype=torch.int32)
    q = torch.randn(B, 1, H, D)
    k, v = torch.randn(B, 1, HKV, D), torch.randn(B, 1, HKV, D)

    out = _sdpa_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=seqlens_all[slots.long()],
        cache_batch_idx=slots,
        causal=True,
    )

    # Untouched slots (1, 3) are byte-identical.
    for s in (1, 3):
        torch.testing.assert_close(k_cache[s], before_k[s])
        torch.testing.assert_close(v_cache[s], before_v[s])

    ref = _reference(
        q,
        k_cache[slots.long()],
        v_cache[slots.long()],
        seqlens_all[slots.long()] + 1,
        causal=True,
    )
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_incremental_decode_matches_prefill():
    """Token-by-token decode == one prefill over the same sequence."""
    T = 7
    q = torch.randn(1, T, H, D)
    k, v = torch.randn(1, T, HKV, D), torch.randn(1, T, HKV, D)

    kc, vc = torch.zeros(1, S, HKV, D), torch.zeros(1, S, HKV, D)
    prefill = _sdpa_attn_with_kvcache(
        q, kc, vc, k=k, v=v, cache_seqlens=torch.zeros(1, dtype=torch.int32), causal=True
    )

    kc2, vc2 = torch.zeros(1, S, HKV, D), torch.zeros(1, S, HKV, D)
    seqlens = torch.zeros(1, dtype=torch.int32)
    steps = []
    for t in range(T):
        steps.append(
            _sdpa_attn_with_kvcache(
                q[:, t : t + 1],
                kc2,
                vc2,
                k=k[:, t : t + 1],
                v=v[:, t : t + 1],
                cache_seqlens=seqlens,
                causal=True,
            )
        )
        seqlens += 1

    torch.testing.assert_close(torch.cat(steps, dim=1), prefill, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(kc2, kc)


def test_empty_cache_returns_zeros():
    """No keys at all (nothing appended, seqlen 0) must not produce NaNs."""
    k_cache, v_cache = _cache(batch=1)
    q = torch.randn(1, 1, H, D)
    out = _sdpa_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=torch.zeros(1, dtype=torch.int32), causal=True
    )
    assert torch.equal(out, torch.zeros_like(out))


def test_decoder_forward_flash_matches_forward(monkeypatch):
    """Decoder.forward_flash (fallback kernel) == the SDPA training path."""
    import vui.flash_compat as fc
    from vui.model import Decoder

    # Runs on CPU, so force the fallback even where flash_attn is installed.
    monkeypatch.setattr(fc, "_impl", fc._sdpa_attn_with_kvcache)

    torch.manual_seed(1)
    dec = Decoder(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        bias=False,
        dropout=0.0,
        max_seqlen=32,
    ).eval()

    T = 6
    x = torch.randn(1, T, 64)
    input_pos = torch.arange(T)

    with torch.inference_mode():
        ref = dec(x.clone(), input_pos)

        dec.allocate_flash_kv_cache(1, device=x.device, dtype=torch.float32)
        flash = dec.forward_flash(x.clone(), input_pos)

        # ... and again one token at a time, reusing the KV cache.
        dec.allocate_flash_kv_cache(1, device=x.device, dtype=torch.float32)
        steps = [
            dec.forward_flash(x[:, t : t + 1].clone(), input_pos[t : t + 1])
            for t in range(T)
        ]

    torch.testing.assert_close(flash, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.cat(steps, dim=1), ref, atol=1e-5, rtol=1e-5)


def test_unsupported_kwarg_is_loud():
    k_cache, v_cache = _cache(batch=1)
    with pytest.raises(NotImplementedError, match="softcap"):
        _sdpa_attn_with_kvcache(
            torch.randn(1, 1, H, D), k_cache, v_cache, causal=True, softcap=0.5
        )


def test_falls_back_when_gpu_has_no_kernels(monkeypatch):
    """A launch failure on an unsupported arch switches to SDPA, once."""
    import vui.flash_compat as fc

    calls = []

    def fake_flash(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("no kernel image is available for execution on the device")

    monkeypatch.setattr(fc, "_impl", fake_flash)

    k_cache, v_cache = _cache(batch=1)
    q = torch.randn(1, 1, H, D)
    k, v = torch.randn(1, 1, HKV, D), torch.randn(1, 1, HKV, D)
    args = dict(k=k, v=v, cache_seqlens=torch.zeros(1, dtype=torch.int32), causal=True)

    out = fc.flash_attn_with_kvcache(q, k_cache, v_cache, **args)
    assert out.shape == (1, 1, H, D) and torch.isfinite(out).all()
    assert fc._impl is fc._sdpa_attn_with_kvcache

    fc.flash_attn_with_kvcache(q, k_cache, v_cache, **args)
    assert len(calls) == 1  # never retried the broken kernel


def test_unrelated_runtime_error_propagates(monkeypatch):
    import vui.flash_compat as fc

    def fake_flash(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(fc, "_impl", fake_flash)
    with pytest.raises(RuntimeError, match="out of memory"):
        fc.flash_attn_with_kvcache(torch.randn(1, 1, H, D), *_cache(batch=1))
    assert fc._impl is fake_flash


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_capture(monkeypatch):
    """The engine replays decode as a CUDA graph — the fallback must capture.

    Mirrors `Engine._backbone_decode_body` / `_backbone_decode_body_b`: warm,
    capture, replay, and check the replayed steps against eager execution.
    """
    import vui.flash_compat as fc
    from vui.model import Decoder

    monkeypatch.setattr(fc, "_impl", fc._sdpa_attn_with_kvcache)  # force fallback

    torch.manual_seed(2)
    dev, dt, d, Bs, T = "cuda", torch.bfloat16, 64, 2, 5
    dec = Decoder(
        n_layers=2,
        d_model=d,
        n_heads=4,
        n_kv_heads=2,
        bias=False,
        dropout=0.0,
        max_seqlen=32,
    ).to(dev, dt).eval()
    steps_in = [torch.randn(Bs, 1, d, device=dev, dtype=dt) for _ in range(T)]
    seq_lens = None

    def decode_body(x, slot_idx=None):
        pos = seq_lens[slot_idx.long()] if slot_idx is not None else seq_lens[:Bs]
        for i, (block, kv) in enumerate(zip(dec.blocks, dec.flash_kv_caches)):
            x = block.forward_flash(
                x,
                kv,
                dec._get_freqs(pos, i),
                per_sample_freqs=True,
                cache_batch_idx=slot_idx,
            )
        return dec.norm(x)[:, 0]

    for slot_idx in (None, torch.tensor([1, 0], device=dev, dtype=torch.int32)):
        # Eager reference.
        dec.allocate_flash_kv_cache(Bs, device=dev, dtype=dt, max_seqlen=32)
        seq_lens = dec.flash_kv_caches[0].seq_lens
        with torch.inference_mode():
            eager = []
            for x in steps_in:
                eager.append(decode_body(x, slot_idx).clone())
                seq_lens += 1

        # Warm, capture, replay.
        dec.allocate_flash_kv_cache(Bs, device=dev, dtype=dt, max_seqlen=32)
        seq_lens = dec.flash_kv_caches[0].seq_lens
        static_in = torch.zeros(Bs, 1, d, device=dev, dtype=dt)
        static_out = torch.zeros(Bs, d, device=dev, dtype=dt)

        def body():
            static_out.copy_(decode_body(static_in, slot_idx))
            seq_lens[:Bs] += 1

        with torch.inference_mode():
            for _ in range(3):
                body()
                seq_lens.zero_()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            body()
        seq_lens.zero_()

        replayed = []
        for t, x in enumerate(steps_in):
            static_in.copy_(x)
            seq_lens.fill_(t)
            graph.replay()
            replayed.append(static_out.clone())

        for t, (got, want) in enumerate(zip(replayed, eager)):
            torch.testing.assert_close(got, want, msg=f"step {t}, slots={slot_idx}")


@pytest.mark.skipif(
    not (HAS_FLASH_ATTN and torch.cuda.is_available()),
    reason="flash_attn / CUDA not available on this platform",
)
def test_matches_flash_attn():
    from flash_attn import flash_attn_with_kvcache

    dev, dt = "cuda", torch.bfloat16
    for Tq, causal, window in ((1, True, (-1, -1)), (4, True, (-1, -1)), (1, True, (3, 0))):
        q = torch.randn(B, Tq, H, D, device=dev, dtype=dt)
        k = torch.randn(B, Tq, HKV, D, device=dev, dtype=dt)
        v = torch.randn(B, Tq, HKV, D, device=dev, dtype=dt)
        seqlens = torch.tensor([0, 5, 9], dtype=torch.int32, device=dev)
        kc = torch.randn(B, S, HKV, D, device=dev, dtype=dt)
        vc = torch.randn(B, S, HKV, D, device=dev, dtype=dt)
        kc2, vc2 = kc.clone(), vc.clone()

        ref = flash_attn_with_kvcache(
            q, kc, vc, k=k, v=v, cache_seqlens=seqlens.clone(),
            causal=causal, window_size=window,
        )
        got = _sdpa_attn_with_kvcache(
            q, kc2, vc2, k=k, v=v, cache_seqlens=seqlens.clone(),
            causal=causal, window_size=window,
        )
        torch.testing.assert_close(kc2, kc)
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
