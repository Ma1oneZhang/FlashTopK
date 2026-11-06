# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Layer 1, self-consistent exactness: every row is the exact canonical top-k of the dense scan's scores (the same
arithmetic, so equality is bit for bit), on every path: the scan, pruning, adaptive pruning, and both recoveries."""

import pytest
import torch
from conftest import DTYPES, TOPK, assert_same, canonical_topk, dense_scores, make_call, requires_gpu

import flashtopk
from flashtopk import topk as topk_mod
from flashtopk.kernels import launch

pytestmark = requires_gpu

SHAPES = [
    pytest.param(1, 5000, "causal", id="1x5k"),
    pytest.param(7, 3000, "causal", id="7x3k-short"),
    pytest.param(129, 20000, "random", id="129x20k-random"),
    pytest.param(300, 9000, "short", id="300x9k-windows<topk"),
    pytest.param(515, 140000, "causal", id="515x140k"),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,seq_len_kv,windows", SHAPES)
@pytest.mark.parametrize("pruning", ["off", "always", "adaptive"])
def test_exact(dtype, rows, seq_len_kv, windows, pruning):
    q, kv, s, w, ks, ke = make_call(rows, seq_len_kv, dtype, seed=rows, windows=windows)
    got = flashtopk.Selector(TOPK, dtype=dtype, pruning=pruning, return_scores=True).run(q, kv, w, ks, ke, s)
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_len_kv", [20000, 70000])
def test_per_row_recovery(dtype, seq_len_kv, monkeypatch):
    """Nine rows' gates closed after calibration come up short; per-row recovery (sliced past 32K keys) fixes them."""
    q, kv, s, w, ks, ke = make_call(515, seq_len_kv, dtype, seed=3)
    rows = torch.arange(514, 514 - 9 * 37, -37, device="cuda")
    seed_prep = launch.seed_prep

    def closed(sample, ks_, ke_, lo, hi, topk, slack, k_min, origin, inv_delta, th_bucket, cand_cnt):
        seed_prep(sample, ks_, ke_, lo, hi, topk, slack, k_min, origin, inv_delta, th_bucket, cand_cnt)
        th_bucket[rows] = 0
        inv_delta[rows] = 255.0 / 1e-6

    monkeypatch.setattr(launch, "seed_prep", closed)
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning="off", return_scores=True)
    got = sel.run(q, kv, w, ks, ke, s)
    assert sel.last_info.bad_rows == 9
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_whole_call_recovery(dtype, monkeypatch):
    """A candidate buffer too small for most rows: whole-call recovery rescores every row."""
    q, kv, s, w, ks, ke = make_call(515, 20000, dtype, seed=4)
    monkeypatch.setattr(topk_mod, "candidate_cap", lambda topk, num_q, seq_len_kv: 2100)
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning="off", return_scores=True)
    got = sel.run(q, kv, w, ks, ke, s)
    assert sel.last_info.bad_rows > 64
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))


@pytest.mark.parametrize("topk", [1, 64, 777, 4096])
def test_other_topk(topk):
    q, kv, s, w, ks, ke = make_call(130, 12000, torch.float8_e4m3fn, seed=5, windows="random")
    sel = flashtopk.Selector(topk, dtype=torch.float8_e4m3fn, pruning="off", return_scores=True)
    got = sel.run(q, kv, w, ks, ke, s)
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke), topk))


def test_ties():
    """Identical keys tie exactly: equal scores go to the smallest indices, in index order."""
    q, kv, s, w, ks, ke = make_call(64, 9000, torch.bfloat16, seed=6, windows="causal")
    kv[1::3] = kv[0]  # a third of the keys share one score per row
    got = flashtopk.Selector(TOPK, dtype=torch.bfloat16, pruning="off", return_scores=True).run(q, kv, w, ks, ke, s)
    assert_same(got, canonical_topk(dense_scores(q, kv, s, w, ks, ke)))
