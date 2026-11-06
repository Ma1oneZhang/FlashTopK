# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Layer 5, how a call runs: never synchronizes, keeps its memory, reports its health, replays from a CUDA graph,
works as a custom op under torch.compile, and rejects operands it cannot take."""

import pytest
import torch
from conftest import DTYPES, TOPK, assert_same, canonical_topk, dense_scores, make_call, requires_gpu

import flashtopk
from flashtopk import topk as topk_mod
from flashtopk.kernels import launch

pytestmark = requires_gpu


@pytest.mark.parametrize("pruning", ["off", "always", "adaptive"])
def test_no_synchronization(pruning):
    q, kv, s, w, ks, ke = make_call(300, 140000, torch.float8_e4m3fn, seed=41)
    sel = flashtopk.Selector(TOPK, dtype=torch.float8_e4m3fn, pruning=pruning).plan(300, 140000, warmup=False)
    sel.run(q, kv, w, ks, ke, s)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        sel.run(q, kv, w, ks, ke, s)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_memory_steady():
    q, kv, s, w, ks, ke = make_call(300, 50000, torch.bfloat16, seed=42)
    sel = flashtopk.Selector(TOPK, dtype=torch.bfloat16)
    out = torch.empty(300, TOPK, dtype=torch.int32, device="cuda")
    sel.run(q, kv, w, ks, ke, s, out=out)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(3):
        sel.run(q, kv, w, ks, ke, s, out=out)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


class Graphed:
    """One selection captured in a CUDA graph: static operands, outputs written in place; load() copies a call of
    the same shape into the operands before replay()."""

    def __init__(self, sel, call):
        self.sel = sel
        self.static = [None if t is None else t.clone() for t in call]
        rows = call[0].shape[0]
        indices = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
        self.out = (torch.empty(rows, TOPK, device="cuda"), indices) if sel.return_scores else indices
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._run()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._run()

    def _run(self):
        q, kv, s, w, ks, ke = self.static
        self.sel.run(q, kv, w, ks, ke, s, out=self.out)

    def replay(self, call):
        for dst, src in zip(self.static, call):
            if dst is not None:
                dst.copy_(src)
        self.graph.replay()
        torch.cuda.synchronize()
        return self.out


def reference(call):
    q, kv, s, w, ks, ke = call
    return canonical_topk(dense_scores(q, kv, s, w, ks, ke))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("pruning", ["off", "always", "adaptive"])
def test_cuda_graph(dtype, pruning):
    """Captured once, replayed on calls of the same shape with other operands and windows: every replay exact."""
    calls = [make_call(515, 140000, dtype, seed=61 + i, windows=windows)
             for i, windows in enumerate(["causal", "causal", "random"])]
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning=pruning, return_scores=True).plan(515, 140000)
    graphed = Graphed(sel, calls[0])
    for call in calls:
        assert_same(graphed.replay(call), reference(call))


@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_graph_adaptive_choice(dtype):
    """The adaptive choice is the replayed call's own: a call whose keys the bound mostly rules out prunes, a plain
    one scans, from one graph. The tight call keeps 5% of its keys at full size and shrinks the rest a hundredfold,
    with positive head weights, so the top-k sit among the full-size keys and the bound rules out the others."""
    plain = make_call(515, 140000, dtype, seed=71)
    q, kv, s, w, ks, ke = make_call(515, 140000, dtype, seed=72)
    kv = kv.clone()
    small = torch.rand(kv.shape[0], device="cuda") > 0.05
    kv[small] = (kv[small].float() * 0.01).to(dtype)
    tight = (q, kv, s, w.abs(), ks, ke)
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning="adaptive", return_scores=True).plan(515, 140000)
    graphed = Graphed(sel, plain)
    chosen = []
    for call in (plain, tight, plain):
        assert_same(graphed.replay(call), reference(call))
        chosen.append(sel.last_info.pruned)
    assert chosen == [False, True, False]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["per_row", "whole_call"])
def test_cuda_graph_recovery(dtype, kind, monkeypatch):
    """Recovery replays too: rows whose gates are closed after calibration (per row), or a candidate buffer too
    small for most rows (whole call), on calls other than the captured one."""
    rows = torch.arange(514, 514 - 9 * 37, -37, device="cuda")
    if kind == "per_row":
        seed_prep = launch.seed_prep

        def closed(sample, ks_, ke_, lo, hi, topk, slack, k_min, origin, inv_delta, th_bucket, cand_cnt):
            seed_prep(sample, ks_, ke_, lo, hi, topk, slack, k_min, origin, inv_delta, th_bucket, cand_cnt)
            th_bucket.index_fill_(0, rows, 0)  # index_fill_, not th_bucket[rows] = 0: no host copy under capture
            inv_delta.index_fill_(0, rows, 255.0 / 1e-6)

        monkeypatch.setattr(launch, "seed_prep", closed)
    else:
        monkeypatch.setattr(topk_mod, "candidate_cap", lambda topk, num_q, seq_len_kv: 2100)
    calls = [make_call(515, 20000, dtype, seed=81 + i) for i in range(3)]
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning="off", return_scores=True).plan(515, 20000)
    graphed = Graphed(sel, calls[0])
    for call in calls[1:]:
        assert_same(graphed.replay(call), reference(call))
        bad = sel.last_info.bad_rows
        assert bad == 9 if kind == "per_row" else bad > 64


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("pruning", ["off", "always", "adaptive"])
def test_chunked_same_bits(dtype, pruning):
    """A call run in slices of chunk_size rows selects the same bits as one pass, whether the slices divide the rows
    or leave a short last one (100 rows, which takes the small-call candidate buffer), and reports the slices'
    health together."""
    call = make_call(2100, 140000, dtype, seed=91)
    q, kv, s, w, ks, ke = call
    one = flashtopk.Selector(TOPK, dtype=dtype, pruning=pruning, chunk_size=4096, return_scores=True)
    whole = one.run(q, kv, w, ks, ke, s)
    torch.cuda.synchronize()
    health = one.last_info.health.clone()
    for chunk_size in (700, 1000):
        sel = flashtopk.Selector(TOPK, dtype=dtype, pruning=pruning, chunk_size=chunk_size, return_scores=True)
        got = sel.run(q, kv, w, ks, ke, s)
        torch.cuda.synchronize()
        assert sel.max_rows == chunk_size
        assert torch.equal(got[1], whole[1]) and torch.equal(got[0].view(torch.int32), whole[0].view(torch.int32))
        assert sel.last_info.bad_rows == int(health[7]) and sel.last_info.pruned == one.last_info.pruned
    via_op = torch.ops.flashtopk.select(q, kv, w, ks, ke, s, TOPK, pruning, 1000, True)
    assert torch.equal(via_op[1], whole[1]) and torch.equal(via_op[0].view(torch.int32), whole[0].view(torch.int32))


@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_graph_chunked(dtype):
    """A call in slices (200, 200 and 115 rows) captures and replays like one pass."""
    calls = [make_call(515, 140000, dtype, seed=95 + i) for i in range(3)]
    sel = flashtopk.Selector(TOPK, dtype=dtype, chunk_size=200, return_scores=True).plan(515, 140000)
    graphed = Graphed(sel, calls[0])
    for call in calls:
        assert_same(graphed.replay(call), reference(call))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("pruning", ["off", "always", "adaptive"])
def test_indices_only(dtype, pruning):
    """By default a call returns the indices alone, the ones it returns with the scores, in one pass or in slices;
    the custom op returns empty scores beside them."""
    q, kv, s, w, ks, ke = make_call(2100, 140000, dtype, seed=93)
    _, whole = flashtopk.Selector(TOPK, dtype=dtype, pruning=pruning, return_scores=True).run(q, kv, w, ks, ke, s)
    for chunk_size in (4096, 1000):
        sel = flashtopk.Selector(TOPK, dtype=dtype, pruning=pruning, chunk_size=chunk_size)
        assert torch.equal(sel.run(q, kv, w, ks, ke, s), whole)
    assert torch.equal(flashtopk.select(q, kv, w, ks, ke, s, pruning=pruning), whole)
    scores, indices = torch.ops.flashtopk.select(q, kv, w, ks, ke, s, TOPK, pruning)
    assert scores.shape == (2100, 0) and torch.equal(indices, whole)


@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_graph_indices_only(dtype):
    """Indices alone, in slices, capture and replay too."""
    calls = [make_call(515, 140000, dtype, seed=97 + i) for i in range(2)]
    sel = flashtopk.Selector(TOPK, dtype=dtype, chunk_size=200).plan(515, 140000)
    graphed = Graphed(sel, calls[0])
    for call in calls:
        assert torch.equal(graphed.replay(call), reference(call)[1])


def test_info():
    q, kv, s, w, ks, ke = make_call(200, 9000, torch.float8_e4m3fn, seed=43)
    sel = flashtopk.Selector(TOPK, dtype=torch.float8_e4m3fn, pruning="always")
    sel.run(q, kv, w, ks, ke, s)
    info = sel.last_info
    assert info.bad_rows == 0 and info.pruned is True
    assert info.health.dtype == torch.int64 and info.health.numel() == 8


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("return_scores", [False, True])
def test_custom_op_under_compile(dtype, return_scores):
    q, kv, s, w, ks, ke = make_call(100, 9000, dtype, seed=44)

    def f(q, kv, w, ks, ke, s):
        scores, indices = torch.ops.flashtopk.select(q, kv, w, ks, ke, s, TOPK, return_scores=return_scores)
        return scores * 1.0, indices

    eager = f(q, kv, w, ks, ke, s)
    compiled = torch.compile(f, fullgraph=True)(q, kv, w, ks, ke, s)
    assert torch.equal(eager[1], compiled[1])
    assert torch.equal(eager[0].view(torch.int32), compiled[0].view(torch.int32))


def test_rejects():
    q, kv, s, w, ks, ke = make_call(16, 4000, torch.float8_e4m3fn, seed=45)
    sel = flashtopk.Selector(TOPK, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        sel.run(q, kv, w, ks, ke, None)  # FP8 without scales
    with pytest.raises(ValueError):
        sel.run(q[:, :16], kv, w, ks, ke, s)  # 16 heads
    with pytest.raises(ValueError):
        sel.run(q, kv, w[:, :16], ks, ke, s)
    with pytest.raises(ValueError):
        flashtopk.Selector(TOPK, num_heads=64)
    with pytest.raises(ValueError):
        flashtopk.Selector(TOPK, pruning="sometimes")
    assert flashtopk.supported(q, kv) and not flashtopk.supported(q[:, :, :64])


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("rows", [5, 300, 2100, 20000])
def test_group_estimate(dtype, rows):
    """The device's group estimate is the torch reference's (plan.estimate_shares, pooled over the groups), its masks
    pack each estimate row's in-window samples, and the flag is the estimate against the threshold."""
    from flashtopk import plan as planner

    q, kv, s, w, ks, ke = make_call(rows, 30000, dtype, seed=46, windows="random")
    sel = flashtopk.Selector(TOPK, dtype=dtype, pruning="off")
    sel.run(q, kv, w, ks, ke, s)
    n = q.shape[0]
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    estimate = torch.zeros(1, dtype=torch.float32, device="cuda")
    scan_ks, prune_ks = torch.empty_like(ks), torch.empty_like(ks)
    plan = planner.adaptive_estimate(q, w, kv, s, ks, ke, sel._origin[:n], sel._inv_delta[:n], sel._th_bucket[:n],
                                     flag, scan_ks, prune_ks, estimate)
    union, row = planner.estimate_shares(plan.est)
    assert abs(float(estimate[0]) - float(union)) <= 1e-6
    assert 0.0 <= float(row) <= 1.0 and 0.0 <= float(union) <= 1.0
    threshold = planner.adaptive_threshold(q.device.index, s is None, kv.shape[0])
    assert bool(flag[0]) == (float(estimate[0]) <= threshold)
    stride = planner.estimate_stride(n)
    rows_est = [(e // 8) * 8 * stride + e % 8 for e in range(plan.est_rows)]
    bits = (plan.est[:, :8].unsqueeze(-1) >> torch.arange(32, device="cuda", dtype=torch.int32)) & 1
    for e, r in enumerate(rows_est):
        want = torch.zeros(256, dtype=torch.int32, device="cuda")
        if r < n:
            want = ((plan.cols >= ks[r]) & (plan.cols < ke[r])).int()
        assert torch.equal(bits[e].reshape(-1), want), f"estimate row {e} (row {r})"
    pruned = bool(flag[0])
    assert torch.equal(scan_ks, ke if pruned else ks) and torch.equal(prune_ks, ks if pruned else ke)


def test_workspace_bytes():
    sel = flashtopk.Selector(TOPK, dtype=torch.float8_e4m3fn).plan(4096, 131072, warmup=False)
    assert flashtopk.workspace_bytes(4096, 131072) >= sel._arena.numel() * 4
    sel = flashtopk.Selector(TOPK, dtype=torch.float8_e4m3fn, chunk_size=1000).plan(4096, 131072, warmup=False)
    assert sel.max_rows == 1000
    assert flashtopk.workspace_bytes(4096, 131072, chunk_size=1000) >= sel._arena.numel() * 4
    assert flashtopk.workspace_bytes(4096, 131072, chunk_size=1000) == flashtopk.workspace_bytes(1000, 131072)
    sel = flashtopk.Selector(TOPK, dtype=torch.float8_e4m3fn).plan(4096, 131072, warmup=False)
    without = flashtopk.workspace_bytes(4096, 131072)
    assert without == flashtopk.workspace_bytes(4096, 131072, return_scores=True) + 4096 * TOPK * 4
    assert without >= (sel._arena.numel() + sel._scores.numel()) * 4
