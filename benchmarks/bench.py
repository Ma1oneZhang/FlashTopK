#!/usr/bin/env python3
# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Time flashtopk on indexer calls and record a digest of each selection.

usage: bench.py --out FILE.jsonl [--pruning adaptive] [--chunk-size ROWS] [--scores] [--warmup 3] [--iters 10]
                [--shard I/N] [--cuda-graph] JOB ...

A JOB is one of
  synthetic:DTYPE:KEYS:ROWS[:SEED]  a generated call, e.g. synthetic:fp8:128K:4096: Gaussian q and kv, signed head
                                    weights, causal windows (the ROWS query rows are the last of the KEYS keys);
                                    random operands, on which adaptive pruning mostly scans
  OPS.pt                            a saved call: FP8 operands (q_fp8 [rows, 32, 128], kv_fp8 [nk, 128], kv_scales
                                    [nk], w_eff [rows, 32], starts, ends) or BF16 ones (q, kv, weights, starts, ends)
  @LIST                             jobs from a file, one a line: a job, or a path and a row count (the call's last
                                    rows, as a prefill chunk)
A call is timed with CUDA events around one selection (median of --iters after --warmup), as the GPU sees it back to
back; --chunk-size sets the selector's chunk_size (default flashtopk's DEFAULT_CHUNK_SIZE), and --scores returns the
scores with the indices (return_scores=True). With --cuda-graph the selection is also captured in a CUDA graph and
its replays timed the same way (graph_median_ms), and graph_same records whether a replay selected what the direct
call did.

The digest is SHA-256 of the selection sorted by index (the indices, then with --scores the scores' bits): two runs
agree on it exactly when they selected the same keys (with the same scores), whatever order they returned them in.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flashtopk  # noqa: E402

TOPK = 2048
SLICE = 65536  # rows generated or hashed at once


def load(path, rows=None):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "q_fp8" in d:
        ops = dict(q=d["q_fp8"], kv=d["kv_fp8"], kv_scales=d["kv_scales"].reshape(-1).float(),
                   weights=d["w_eff"].float())
    else:
        ops = dict(q=d["q"], kv=d["kv"], kv_scales=None, weights=d["weights"].float())
    ops.update(ks=d["starts"].int(), ke=d["ends"].int())
    if rows is not None:
        ops.update({k: ops[k][-rows:] for k in ("q", "weights", "ks", "ke")})
    return {k: (None if v is None else v.contiguous().cuda()) for k, v in ops.items()}


def count(text):
    scale = {"K": 1 << 10, "M": 1 << 20}.get(text[-1:].upper(), 1)
    return int(text[:-1] if scale > 1 else text) * scale


def synthetic(spec):
    _, dtype, keys, rows, *seed = spec.split(":")
    nk, rows = count(keys), count(rows)
    if dtype not in ("fp8", "bf16"):
        raise ValueError(f"{spec}: DTYPE is fp8 or bf16")
    elem = torch.float8_e4m3fn if dtype == "fp8" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(int(seed[0]) if seed else 0)
    q = torch.empty(rows, 32, 128, dtype=elem, device="cuda")
    for a in range(0, rows, SLICE):
        q[a:a + SLICE] = torch.randn(min(SLICE, rows - a), 32, 128, generator=g, device="cuda")
    kv = torch.randn(nk, 128, generator=g, device="cuda")
    weights = torch.randn(rows, 32, generator=g, device="cuda") * 0.1
    ke = torch.arange(nk - rows + 1, nk + 1, device="cuda").clamp(min=1).int()
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    if dtype == "bf16":
        return dict(q=q, kv=kv.bfloat16(), kv_scales=None, weights=weights, ks=ks, ke=ke)
    scales = torch.rand(nk, generator=g, device="cuda") * 0.02 + 0.01
    return dict(q=q, kv=kv.to(torch.float8_e4m3fn), kv_scales=scales, weights=weights, ks=ks, ke=ke)


def digest(scores, indices):
    h = hashlib.sha256()
    for a in range(0, indices.shape[0], SLICE):
        h.update(indices[a:a + SLICE].sort(dim=-1)[0].contiguous().cpu().numpy().tobytes())
    for a in range(0, indices.shape[0] if scores is not None else 0, SLICE):
        order = indices[a:a + SLICE].sort(dim=-1)[1]
        h.update(scores[a:a + SLICE].gather(1, order).contiguous().view(torch.int32).cpu().numpy().tobytes())
    return h.hexdigest()


def cuda_times(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def graph_times(sel, call, out, warmup, iters):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        sel.run(*call, out=out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sel.run(*call, out=out)
    return cuda_times(graph.replay, warmup, iters)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pruning", default="adaptive", choices=("off", "always", "adaptive"))
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--scores", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("ops", nargs="+")
    args = parser.parse_args()
    jobs = []
    for p in args.ops:
        lines = [line.split() for line in open(p[1:]) if line.strip()] if p.startswith("@") else [[p]]
        jobs += [(f[0], int(f[1]) if len(f) > 1 else None) for f in lines]
    shard, nshards = (int(x) for x in args.shard.split("/"))
    jobs = sorted(set(jobs), key=lambda j: (j[0], j[1] or 0))[shard::nshards]
    chunk_size = args.chunk_size or flashtopk.DEFAULT_CHUNK_SIZE
    return_scores = args.scores
    done = set()
    if os.path.exists(args.out):
        done = {(r["ops"], r["rows"], r.get("chunk_size"), r.get("scores", True))
                for r in map(json.loads, open(args.out))}
    env = dict(device=torch.cuda.get_device_name(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
               torch=torch.__version__, flashtopk=flashtopk.__version__)
    selectors = {}
    with open(args.out, "a") as out:
        for path, chunk in jobs:
            ops = synthetic(path) if path.startswith("synthetic:") else load(path, chunk)
            rows, nk = ops["q"].shape[0], ops["kv"].shape[0]
            if (path, rows, chunk_size, return_scores) in done:
                continue
            record = dict(ops=path, rows=rows, nk=nk, dtype="bf16" if ops["kv_scales"] is None else "fp8",
                          pruning=args.pruning, chunk_size=chunk_size, scores=return_scores, **env)
            try:
                started = time.time()
                key = ops["q"].dtype
                if key not in selectors:
                    selectors[key] = flashtopk.Selector(TOPK, dtype=key, pruning=args.pruning, chunk_size=chunk_size,
                                                        return_scores=return_scores)
                sel = selectors[key].plan(rows, nk, warmup=False)
                call = (ops["q"], ops["kv"], ops["weights"], ops["ks"], ops["ke"], ops["kv_scales"])
                times = cuda_times(lambda: sel.run(*call), args.warmup, args.iters)
                record.update(median_ms=statistics.median(times), min_ms=min(times),
                              times_ms=[round(t, 4) for t in times])
                scores, indices = sel.run(*call) if return_scores else (None, sel.run(*call))
                torch.cuda.synchronize()
                info = sel.last_info
                health = info.health.tolist()
                record.update(digest=digest(scores, indices), pruned=info.pruned, overflow_rows=health[0],
                              underfill_rows=health[1], bad_rows=health[7])
                if args.cuda_graph:
                    graphed = (torch.empty_like(scores), torch.empty_like(indices)) if return_scores else \
                        torch.empty_like(indices)
                    times = graph_times(sel, call, graphed, args.warmup, args.iters)
                    torch.cuda.synchronize()
                    same = digest(*graphed) if return_scores else digest(None, graphed)
                    record.update(graph_median_ms=statistics.median(times), graph_min_ms=min(times),
                                  graph_same=same == record["digest"])
                record["wall_s"] = round(time.time() - started, 2)
            except Exception as exc:  # recorded per call so one failure does not stop the sweep
                record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(json.dumps({k: v for k, v in record.items() if k != "times_ms"}), flush=True)
            del ops
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
