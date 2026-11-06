#!/usr/bin/env python3
# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Time one method on whole-prefill calls and check its rows against the exact top-k of the dense scores.

usage: bench.py --method M --out FILE.jsonl CALL.pt [CALL.pt ...] [--warmup 1] [--iters 3]

A CALL.pt is one causal prefill from get_inputs.py (row r of q sees keys [0, r + 1)): FP8 (q_fp8, kv_fp8,
kv_scales, w_eff, starts, ends) or BF16 (q, kv, weights, starts, ends). M is a method of methods.py. All methods
run on NVIDIA Hopper (SM90).

Timing: CUDA events around one whole prefill, the median of --iters after --warmup. Exactness, on the rows a check
covers (the last 4096 and 16 blocks of 256 spread over the rest, each against its own causal reference): a row is
exact if its set is the reference's, boundary if they differ only in keys within 1e-5 relative of the k-th score,
and wrong otherwise. The first 2047 rows see fewer than TOPK keys and must select all of them. A JSON line per call;
a rerun skips completed calls and retries failed calls. Exits nonzero if any call fails.
"""
import argparse
import json
import os
import statistics
import time

import torch

import methods
from methods import TOPK, causal_rows, dense_logits, is_bf16, nk_of, q_of
from records import completed, latest

REL_TIE = 1e-5


def load_operands(path):
    """The operand dict of a call file; BF16 weights rounded to BF16 values, as cuDNN takes them (every method
    gets the same)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d.pop("meta", {})
    if "q_fp8" not in d:
        w = d["weights"].float().bfloat16()
        d = dict(q_bf16=d["q"], kv_bf16=d["kv"], w=w.float(), w_bf16=w, starts=d["starts"], ends=d["ends"])
    return {k: v.contiguous().cuda() for k, v in d.items()}, meta


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


def sample_slices(rows, tail=4096, spread=16, block=256):
    """The rows the check covers: the last `tail`, and `spread` blocks of `block` rows spread over the rest from TOPK
    on (the earlier rows see fewer than TOPK keys)."""
    head = max(TOPK, rows - tail)
    out = [slice(head, rows)]
    for i in range(spread if head > TOPK else 0):
        a = TOPK + i * (head - TOPK) // spread
        out.append(slice(a, min(a + block, head)))
    return out


def compare(selected, nk, op, row_block=4096):
    """Exact / boundary / wrong of selected rows against the exact top-k of op's dense scores, a block at a time."""
    rows = selected.shape[0]
    tally = dict(rows=rows, exact=0, boundary=0, wrong=0)
    for r in range(0, rows, row_block):
        s = slice(r, min(rows, r + row_block))
        block = dense_logits(op, s)
        values, ref = torch.topk(block, TOPK, dim=1)
        kth = values[:, -1]
        sel = selected[s].long()
        invalid = ((sel < 0) | (sel >= nk)).any(1)
        a = sel.sort(1).values
        duplicate = (a[:, 1:] == a[:, :-1]).any(1)
        differs = (a != ref.sort(1).values).any(1)
        picked = block.gather(1, sel.clamp(0, nk - 1)).min(1).values
        near = picked >= kth - REL_TIE * kth.abs()
        bad = invalid | duplicate
        tally["exact"] += int((~differs & ~bad).sum())
        tally["boundary"] += int((differs & near & ~bad).sum())
        tally["wrong"] += int((bad | (differs & ~near)).sum())
        del block, values, ref
    return tally


def check_short(selected):
    """The rows whose window [0, r + 1) holds fewer than TOPK keys select every key of it: (rows, wrong). Entries
    outside the window count as padding."""
    n = min(TOPK - 1, selected.shape[0])
    sel = selected[:n].long()
    r = torch.arange(n, device=sel.device)[:, None]
    col = torch.arange(sel.shape[1], device=sel.device)[None]
    inside = torch.where((sel >= 0) & (sel <= r), sel, torch.full_like(sel, 1 << 40)).sort(1).values
    return n, int((~((inside == col) | (col > r)).all(1)).sum())


def check(op, selected):
    totals = dict(checked_rows=0, exact=0, boundary=0, wrong=0)
    for s in sample_slices(q_of(op).shape[0]):
        t = compare(selected[s], s.stop, causal_rows(op, s))
        totals["checked_rows"] += t.pop("rows")
        for k, v in t.items():
            totals[k] += v
    totals["short_rows"], totals["short_wrong"] = check_short(selected)
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=sorted(methods.METHODS), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("calls", nargs="+")
    a = p.parse_args()
    if a.warmup < 0 or a.iters < 1:
        p.error("--warmup must be nonnegative and --iters must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        p.error("the reproduction benchmarks require an SM90 GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    method = methods.METHODS[a.method](a)
    env = dict(device=torch.cuda.get_device_name(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
               torch=torch.__version__)
    import cutlass
    import flashtopk

    env.update(cutlass=cutlass.__version__, flashtopk=flashtopk.__version__)
    done = set()
    if os.path.exists(a.out):
        done = {path for path, r in latest(a.out).items() if completed(r) and r.get("method") == a.method}
    failed = False
    with open(a.out, "a") as out:
        for path in sorted(a.calls):
            if path in done:
                continue
            record = dict(ops=path, method=a.method, **env)
            op = selected = None
            try:
                op, meta = load_operands(path)
                record.update(nk=nk_of(op), rows=int(q_of(op).shape[0]), dtype="bf16" if is_bf16(op) else "fp8",
                              source=meta.get("source", path), full_prefill=True)
                started = time.time()
                if hasattr(method, "prepare"):
                    method.prepare(op)
                times = cuda_times(lambda: method(op), a.warmup, a.iters)
                record.update(median_ms=statistics.median(times), min_ms=min(times),
                              times_ms=[round(t, 4) for t in times])
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                selected = method(op).clone()
                torch.cuda.synchronize()
                record["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                if hasattr(method, "digest"):
                    record["digest"] = method.digest(selected)
                record.update(check(op, selected))
                record["stats"] = method.stats(op)
                record["wall_s"] = round(time.time() - started, 2)
            except Exception as exc:  # recorded per call, so one failure does not stop the sweep
                record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            failed |= not completed(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(json.dumps({k: v for k, v in record.items() if k != "times_ms"}), flush=True)
            del op, selected
            torch.cuda.empty_cache()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
