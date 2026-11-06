# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Layer 2, against FP64: every returned score is within a rounding bound of the exact score, and no key whose exact
score clears the row's cutoff by more than twice that bound is missed."""

import pytest
import torch
from conftest import DTYPES, TOPK, make_call, requires_gpu

import flashtopk

pytestmark = requires_gpu

# FP32 accumulation over the 128-term dot products and the 32-head sum, with room for tensor-core accumulators
# that keep fewer than 24 bits between steps: 2^-20 per term of absolute mass.
REL = 192 * 2.0**-20


def exact_scores(q, kv, scales, weights):
    """FP64 scores and their absolute mass s_t sum_h |w_h| sum_d |q_hd k_td|: [rows, L] each."""
    qd, kd = q.double(), kv.double()
    dots = torch.einsum("rhd,td->rht", qd, kd)
    mass = torch.einsum("rhd,td->rht", qd.abs(), kd.abs())
    w = weights.double()[:, :, None]
    s = 1.0 if scales is None else scales.double()[None, :]
    return s * (w * dots.clamp(min=0)).sum(1), s * (w.abs() * mass).sum(1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("pruning", ["off", "always"])
def test_fp64_bound(dtype, pruning):
    rows, L = 48, 6000
    q, kv, s, w, ks, ke = make_call(rows, L, dtype, seed=11, windows="causal")
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning=pruning, return_scores=True)
    scores, indices = sel.run(q, kv, w, ks, ke, s)
    exact, mass = exact_scores(q, kv, s, w)
    cols = torch.arange(L, device="cuda")
    inside = (cols[None] >= ks[:, None].long()) & (cols[None] < ke[:, None].long())
    exact = exact.masked_fill(~inside, float("-inf"))
    valid = indices >= 0
    at = indices.clamp(min=0).long()
    err = (scores.double() - exact.gather(1, at)).abs()
    bound = REL * mass.gather(1, at) + 1e-300
    assert bool(((err <= bound) | ~valid).all()), f"max error / bound {float((err / bound)[valid].max()):.3g}"

    # No clear miss: a key outside the selection does not beat the lowest selected key by more than both bounds.
    slack = 2 * REL * mass.max(1, keepdim=True).values
    lowest = exact.gather(1, at).masked_fill(~valid, float("inf")).min(1, keepdim=True).values
    chosen = torch.zeros(rows, L + 1, dtype=torch.bool, device="cuda")
    chosen.scatter_(1, torch.where(valid, indices.long(), L), True)
    missed = inside & ~chosen[:, :L] & (exact > lowest + slack)
    assert int(missed.sum()) == 0
    full = valid.sum(1) == (ke - ks).clamp(min=0).clamp(max=TOPK)
    assert bool(full.all())
