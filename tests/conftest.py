# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Shared helpers: synthetic calls, the dense scan's scores, and the exact canonical top-k of a score table."""

import pytest
import torch

from flashtopk.kernels import launch

TOPK = 2048


def gpu_arch():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()[0]


requires_gpu = pytest.mark.skipif(gpu_arch() != 9, reason="needs an SM90 GPU")
DTYPES = [pytest.param(torch.float8_e4m3fn, id="fp8"), pytest.param(torch.bfloat16, id="bf16")]


def make_call(rows, seq_len_kv, dtype, seed=0, windows="causal", spread=1.0):
    """A synthetic call: Gaussian q / kv, weights of both signs, kv scales for FP8. windows: "causal" (the rows
    are the last of the keys), "random" (any [ks, ke), some empty), or "short" (shorter than TOPK)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, 32, 128, generator=g, device="cuda") * spread
    kv = torch.randn(seq_len_kv, 128, generator=g, device="cuda")
    weights = torch.randn(rows, 32, generator=g, device="cuda") * 0.1
    if windows == "causal":
        ke = torch.arange(seq_len_kv - rows + 1, seq_len_kv + 1, device="cuda").clamp(min=1).int()
        ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    elif windows == "random":
        a = torch.randint(0, seq_len_kv, (rows,), generator=g, device="cuda")
        b = torch.randint(0, seq_len_kv + 1, (rows,), generator=g, device="cuda")
        ks, ke = torch.minimum(a, b).int(), torch.maximum(a, b).int()
        ke[::7] = ks[::7]  # some empty windows
    else:
        ks = torch.randint(0, seq_len_kv - TOPK, (rows,), generator=g, device="cuda").int()
        ke = ks + torch.randint(0, TOPK, (rows,), generator=g, device="cuda").int()
    if dtype == torch.bfloat16:
        return q.bfloat16(), kv.bfloat16(), None, weights, ks, ke
    scales = torch.rand(seq_len_kv, generator=g, device="cuda") * 0.02 + 0.01
    return q.to(torch.float8_e4m3fn), kv.to(torch.float8_e4m3fn), scales, weights, ks, ke


def dense_scores(q, kv, scales, weights, ks, ke, rows_at_once=512):
    """Every in-window score by the dense scan (the selection's own bits), -inf elsewhere: [rows, seq_len_kv]."""
    n, L = q.shape[0], kv.shape[0]
    out = torch.full((n, L), float("-inf"), device=q.device)
    scratch = torch.empty(n, dtype=torch.int32, device=q.device)
    for a in range(0, n, rows_at_once):
        b = min(n, a + rows_at_once)
        with launch.sm90_q_scope(q[a:b]):
            launch.scan_dense(q[a:b], kv, scales, weights[a:b], ks[a:b], ke[a:b], out[a:b], scratch[a:b])
    cols = torch.arange(L, device=q.device)
    inside = (cols[None] >= ks[:, None].long()) & (cols[None] < ke[:, None].long())
    return out.masked_fill_(~inside, float("-inf"))


def canonical_topk(dense, topk=TOPK):
    """The exact top-k of each row of a score table in canonical order (score descending, then index ascending;
    -0.0 ties +0.0), padded with (-inf, -1): (scores, indices int32)."""
    n, L = dense.shape
    k = min(topk, L)
    cols = torch.arange(L, device=dense.device)
    pivot = torch.topk(dense, k, dim=1).values[:, -1:]
    greater = dense > pivot
    equal = (dense == pivot) & torch.isfinite(pivot)
    need = k - greater.sum(1, keepdim=True)
    chosen = greater | (equal & (torch.cumsum(equal.int(), 1) <= need))
    key = torch.where(chosen, (L - cols)[None].int(), torch.zeros((), dtype=torch.int32, device=dense.device))
    top = torch.topk(key, k, dim=1).values
    idx = torch.where(top > 0, L - top, torch.full_like(top, -1))
    sc = torch.where(idx >= 0, dense.gather(1, idx.clamp(min=0).long()), torch.full_like(top, 0).float() - float("inf"))
    order = torch.sort(sc, dim=1, descending=True, stable=True).indices
    scores = torch.full((n, topk), float("-inf"), device=dense.device)
    indices = torch.full((n, topk), -1, dtype=torch.int32, device=dense.device)
    scores[:, :k] = sc.gather(1, order)
    indices[:, :k] = idx.gather(1, order).int()
    return scores, indices


def assert_same(got, ref):
    gs, gi = got
    rs, ri = ref
    bad = (~((gi == ri) & (gs.view(torch.int32) == rs.view(torch.int32))).all(1)).nonzero().flatten()
    assert bad.numel() == 0, f"{bad.numel()} rows differ, first {bad[:8].tolist()}"
