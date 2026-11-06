#!/usr/bin/env python3
# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Tables and figures of run.sh's records, optionally compared with a saved run.

usage: report.py [--results results] [--reference SAVED_RESULTS] [--tolerance 0.10]

For every input set, GPU, operand type and length: each method's speedup over dense scoring + FlashInfer top-k, the
geometric mean over the calls of that length the method finished; FlashTopK's time against dense's on the trained
indexer; each method's exactness on the checked rows. With --reference, a point of FlashTopK's speedup passes when it
is within tolerance of the saved run's for the same GPU, operand type, set and length. Figures, drawn as the repository's
(figures.py): results/performance_<set>_<gpu>.png.
"""
import argparse
import collections
import glob
import math
import os
import re
import statistics

from records import completed, latest

SETS = [("glm", "Trained indexer"), ("init", "Untrained indexer"), ("random", "Random indexer weights and keys")]
LENGTHS = ["128k", "192k", "256k", "512k", "768k", "1m"]
METHODS = [("flashtopk", "FlashTopK"), ("hpc_topk", "dense + HPC-Ops"), ("deepselect", "dense + DeepSelect")]


def load(root, failures=None):
    """(card, dtype, set) -> (length, call) -> method -> record, over the timed records under root."""
    data = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in glob.glob(os.path.join(root, "*_*", "*", "*.jsonl")):
        cd, subset = path.split(os.sep)[-3:-1]
        card, dtype = cd.rsplit("_", 1)
        for r in latest(path).values():
            if not completed(r):
                if failures is not None:
                    failures.append((path, r))
                continue
            m = re.search(r"(?:glm51|init|rand)_(\w+?)/(call\d+)\.pt$", r.get("ops", ""))
            if m:
                data[(card, dtype, subset)][m.groups()][r["method"]] = r
    return data


def geo(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def speedups(cells):
    """length -> method -> speedup over dense, over the calls of that length the method finished."""
    out = collections.defaultdict(dict)
    for length in LENGTHS:
        calls = [c for (L, _), c in cells.items() if L == length and "dense" in c]
        for m, _ in METHODS:
            v = geo([c["dense"]["median_ms"] / c[m]["median_ms"] for c in calls if m in c])
            if v:
                out[length][m] = v
    return out


def fmt_ms(ms):
    return f"{ms:.1f} ms" if ms < 100 else f"{ms:.0f} ms" if ms < 10000 else f"{ms / 1000:.1f} s"


def tables(data):
    for (card, dtype, subset), cells in sorted(data.items()):
        sp = speedups(cells)
        lengths = [L for L in LENGTHS if L in sp]
        print(f"\n### {card.upper()} {dtype.upper()}, {dict(SETS)[subset].lower()}: speedup over dense + FlashInfer\n")
        print("| method | " + " | ".join(L.upper() for L in lengths) + " |")
        print("|---|" + "---:|" * len(lengths))
        for m, label in METHODS:
            if any(m in sp[L] for L in lengths):
                print(f"| {label} | " + " | ".join(f"{sp[L][m]:.2f}x" if m in sp[L] else "—" for L in lengths) + " |")
        if subset == "glm":
            row = []
            for L in lengths:
                cs = [c for (l, _), c in cells.items() if l == L and "flashtopk" in c and "dense" in c]
                row.append(f"{fmt_ms(statistics.median(c['flashtopk']['median_ms'] for c in cs))} / "
                           f"{fmt_ms(statistics.median(c['dense']['median_ms'] for c in cs))}" if cs else "—")
            print("| FlashTopK / dense time | " + " | ".join(row) + " |")
    tally = collections.defaultdict(collections.Counter)
    for cells in data.values():
        for c in cells.values():
            for m, r in c.items():
                for k in ("checked_rows", "wrong", "boundary", "short_rows", "short_wrong"):
                    tally[m][k] += r.get(k, 0)
    print("\n### Exactness on the checked rows\n\n| method | checked | wrong | within 1e-5 of the k-th | short windows wrong |")
    print("|---|---:|---:|---:|---:|")
    for m, t in sorted(tally.items()):
        n = max(1, t["checked_rows"])
        print(f"| {m} | {t['checked_rows']} | {100 * t['wrong'] / n:.2f}% | {100 * t['boundary'] / n:.3f}% | "
              f"{t['short_wrong']} of {t['short_rows']} |")


def against(data, reference, tolerance):
    rows, failed = [], []
    for key, cells in sorted(data.items()):
        if key not in reference:
            continue
        ours, ref = speedups(cells), speedups(reference[key])
        for L in LENGTHS:
            if "flashtopk" in ours.get(L, {}) and "flashtopk" in ref.get(L, {}):
                ratio = ours[L]["flashtopk"] / ref[L]["flashtopk"]
                rows.append((key, L, ours[L]["flashtopk"], ref[L]["flashtopk"], ratio))
                if abs(ratio - 1) > tolerance:
                    failed.append(rows[-1])
    print(f"\n### FlashTopK's speedup against the reference (tolerance {tolerance:.0%})\n")
    if not rows:
        print("No point in common with the supplied reference: no matching GPU or completed comparison.")
    for (card, dtype, subset), L, o, r, ratio in rows:
        print(f"{'FAIL' if abs(ratio - 1) > tolerance else 'ok  '} {card} {dtype} {subset:6s} {L:4s} "
              f"{o:.2f}x (reference {r:.2f}x, {ratio:.3f})")
    if rows:
        print(f"\n{len(rows) - len(failed)} of {len(rows)} points within {tolerance:.0%}: "
              f"{'PASS' if not failed else 'FAIL'}")
    # With the reference's inputs (the random set; the published model sets) FlashTopK selects the same bits.
    same = total = 0
    for key, cells in data.items():
        for call, c in cells.items():
            ref = reference.get(key, {}).get(call, {}).get("flashtopk", {})
            if "digest" in c.get("flashtopk", {}) and "digest" in ref:
                total += 1
                same += c["flashtopk"]["digest"] == ref["digest"]
    if total:
        print(f"FlashTopK selected the reference's bits on {same} of {total} calls with a reference digest")
        if same != total:
            print("FAIL: FlashTopK selected different bits from the reference")
    return (not failed and same == total) if rows or total else None


def figures(data, out_dir):
    """The repository's figures (figures.py) of these records, one per input set and GPU."""
    import figures as f

    for card in sorted({card for card, _, _ in data}):
        for subset in f.SETS:
            path = os.path.join(out_dir, f"performance_{f.SETS[subset][0]}_{card}.png")
            if f.figure(data, card, subset, path):
                print(path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(here, "results"))
    p.add_argument("--reference", help="compare against records from a saved run")
    p.add_argument("--tolerance", type=float, default=0.10)
    a = p.parse_args()
    if not math.isfinite(a.tolerance) or a.tolerance < 0:
        p.error("--tolerance must be finite and nonnegative")
    failures = []
    data = load(a.results, failures)
    for path, record in failures:
        reason = record.get("error", "incomplete timing or failed correctness checks")
        print(f"FAIL {path}: {record['ops']}: {reason}")
    if not data:
        raise SystemExit(f"no records under {a.results}: run run.sh first")
    tables(data)
    passed = against(data, load(a.reference), a.tolerance) if a.reference else None
    figures(data, a.results)
    return int(bool(failures) or passed is False)


if __name__ == "__main__":
    raise SystemExit(main())
