# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Layer 4, properties of the output that hold whatever the scores: canonical order, distinct in-window indices,
min(topk, window) of them, (-inf, -1) padding, the same bits on every run, and rows independent of their batch."""

import pytest
import torch
from conftest import DTYPES, TOPK, make_call, requires_gpu

import flashtopk

pytestmark = requires_gpu


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("windows", ["causal", "random", "short"])
def test_structure_of_rows(dtype, windows):
    q, kv, s, w, ks, ke = make_call(257, 11000, dtype, seed=31, windows=windows)
    scores, indices = flashtopk.Selector(TOPK, dtype=dtype, return_scores=True).run(q, kv, w, ks, ke, s)
    valid = indices >= 0
    n_valid = valid.sum(1)
    window = (ke - ks).clamp(min=0)
    assert torch.equal(n_valid, window.clamp(max=TOPK).long())
    # valid entries first, then (-inf, -1)
    assert bool((valid.int().diff(dim=1) <= 0).all())
    assert bool(torch.isneginf(scores[~valid]).all())
    # in window and distinct
    assert bool(((indices >= ks[:, None]) & (indices < ke[:, None]) | ~valid).all())
    srt = torch.where(valid, indices, torch.arange(TOPK, device="cuda").neg() - 2).sort(1).values
    assert bool((srt.diff(dim=1) != 0).all())
    # canonical order: scores non-increasing, equal scores by increasing index
    a, b = scores[:, :-1], scores[:, 1:]
    both = valid[:, :-1] & valid[:, 1:]
    assert bool(((a > b) | ((a == b) & (indices[:, :-1] < indices[:, 1:])) | ~both).all())


@pytest.mark.parametrize("dtype", DTYPES)
def test_deterministic(dtype):
    q, kv, s, w, ks, ke = make_call(400, 30000, dtype, seed=32)
    sel = flashtopk.Selector(TOPK, dtype=dtype, return_scores=True)
    first = [t.clone() for t in sel.run(q, kv, w, ks, ke, s)]
    for _ in range(3):
        again = sel.run(q, kv, w, ks, ke, s)
        assert torch.equal(first[1], again[1])
        assert torch.equal(first[0].view(torch.int32), again[0].view(torch.int32))


@pytest.mark.parametrize("dtype", DTYPES)
def test_rows_independent(dtype):
    """A row's selection does not depend on the other rows of its call."""
    q, kv, s, w, ks, ke = make_call(600, 25000, dtype, seed=33, windows="random")
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning="off", return_scores=True)
    whole = [t.clone() for t in sel.run(q, kv, w, ks, ke, s)]
    part = torch.arange(5, 600, 3, device="cuda")
    sub = sel.run(q[part], kv, w[part], ks[part], ke[part], s)
    assert torch.equal(sub[1], whole[1][part])
    assert torch.equal(sub[0].view(torch.int32), whole[0][part].view(torch.int32))


def test_windows_from_cu_seqlens():
    cu_q = torch.tensor([0, 3, 3, 8], device="cuda")
    cu_k = torch.tensor([0, 10, 12, 20], device="cuda")
    ks, ke = flashtopk.windows_from_cu_seqlens(cu_q, cu_k)
    want_ks, want_ke = [], []
    for i in range(3):
        lq, lk = int(cu_q[i + 1] - cu_q[i]), int(cu_k[i + 1] - cu_k[i])
        for t in range(lq):
            want_ks.append(int(cu_k[i]))
            want_ke.append(int(cu_k[i]) + lk - lq + t + 1)
    assert ks.tolist() == want_ks and ke.tolist() == want_ke
    assert ks.dtype == ke.dtype == torch.int32
