# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Layer 3, against an independent implementation: torch.topk over dense logits from DeepGEMM (FP8, when installed)
or from FP32 torch (BF16). A row is wrong if its index set differs from the reference's in a key whose reference
score is below the row's k-th by more than a relative 1e-5 (a tie under the other kernel's rounding is not wrong)."""

import pytest
import torch
from conftest import TOPK, make_call, requires_gpu

import flashtopk

pytestmark = requires_gpu
REL_TIE = 1e-5


def wrong_rows(indices, logits, kth):
    """Rows (every window at least TOPK keys) whose set differs from torch.topk's beyond reference-score ties."""
    sel = indices.long()
    assert bool((sel >= 0).all())
    picked = logits.gather(1, sel).min(1).values
    ref = torch.topk(logits, TOPK, dim=1).indices.sort(1).values
    differs = (sel.sort(1).values != ref).any(1)
    near = picked >= kth - REL_TIE * kth.abs()
    return int((differs & ~near).sum())


def torch_logits(q, kv, weights, ks, ke):
    dots = torch.einsum("rhd,td->rht", q.float(), kv.float())
    logits = (weights[:, :, None] * dots.clamp(min=0)).sum(1)
    cols = torch.arange(kv.shape[0], device=q.device)
    inside = (cols[None] >= ks[:, None].long()) & (cols[None] < ke[:, None].long())
    return logits.masked_fill(~inside, float("-inf"))


def deep_gemm_logits(q, kv, scales, weights, ks, ke):
    deep_gemm = pytest.importorskip("deep_gemm")
    if hasattr(deep_gemm, "fp8_mqa_logits"):
        logits = deep_gemm.fp8_mqa_logits(q, (kv, scales), weights, ks, ke, clean_logits=True)
    else:
        logits = deep_gemm.fp8_fp4_mqa_logits((q, None), (kv, scales), weights, ks, ke, clean_logits=True)
    return logits[:, : kv.shape[0]].float()


@pytest.mark.parametrize("pruning", ["off", "adaptive"])
def test_bf16_against_torch(pruning):
    q, kv, s, w, ks, ke = make_call(64, 8192, torch.bfloat16, seed=21, windows="causal")
    indices = flashtopk.Selector(TOPK, dtype=torch.bfloat16, pruning=pruning).run(q, kv, w, ks, ke, s)
    logits = torch_logits(q, kv, w, ks, ke)
    kth = torch.topk(logits, TOPK, dim=1).values[:, -1]
    assert wrong_rows(indices, logits, kth) == 0


@pytest.mark.parametrize("pruning", ["off", "adaptive"])
def test_fp8_against_deep_gemm(pruning):
    q, kv, s, w, ks, ke = make_call(64, 8192, torch.float8_e4m3fn, seed=22, windows="causal")
    indices = flashtopk.Selector(TOPK, dtype=torch.float8_e4m3fn, pruning=pruning).run(q, kv, w, ks, ke, s)
    logits = deep_gemm_logits(q, kv, s, w, ks, ke)
    kth = torch.topk(logits, TOPK, dim=1).values[:, -1]
    assert wrong_rows(indices, logits, kth) == 0
