#!/usr/bin/env python3
# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""The performance figures of FlashTopK's README and performance report, from run.sh's records: per
input set and GPU, each method's speedup over dense scoring + FlashInfer top-k on one full prefill, as grouped bars (one
group per length, one bar per method), FP8 above BF16. A bar is the geometric mean over the calls of that length that
the method finished; a method with no bar does not run on that GPU or operand type, or failed at that length. The look
follows the charts of Tencent Hy's Hy4 article: FlashTopK in Hy4's blue, the others in muted colors, no axis lines but a
dark baseline under the bars, faint grid lines, bars with slightly rounded tops, a dark navy bold title.

usage: figures.py [--records results] [--out results] [--cards h800,h20] [--sets glm,init,random] [--anonymous]
  -> OUT/performance_{trained,untrained,random}_{card}.png. With --anonymous and one card the figures name no GPU,
     OUT/performance_{trained,untrained,random}.png.
Run `figures.py --out ../assets` to redraw the repository's charts with your generated results.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

import report  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10})

CARDS = {"h800": "H800", "h20": "H20"}
DTYPES = [("fp8", "FP8"), ("bf16", "BF16")]
SETS = {"glm": ("trained", "Trained indexer", "GLM-5.1, layers 24 and 48"),
        "init": ("untrained", "Untrained indexer", "GLM-5.1 at Megatron-LM initialization, layers 24 and 48"),
        "random": ("random", "Random indexer weights and keys", "Gaussian q and kv")}
# key, label, color: FlashTopK in Hy4's blue; the other methods in muted colors.
METHODS = [("flashtopk", "FlashTopK", "#2A77FF"), ("hpc_topk", "dense + HPC-Ops top-k", "#8DBFB4"),
           ("deepselect", "dense + DeepSelect", "#DDC79E")]
DENSE = "dense + FlashInfer top-k"
INK, MUTED, GRID, BASE, DENSE_C = "#1B2540", "#7A8190", "#EDEFF2", "#1B2540", "#9AA0AA"
WIDTH, PANEL_H, HEADER, GAP, FOOTER, LEFT, RIGHT = 9.0, 1.45, 1.82, 0.46, 0.78, 0.62, 0.15
GROUP = 0.84
NCOLS = 4  # legend columns, read row by row


def bar(ax, x, h, w, color, aspect):
    """A bar with slightly rounded top corners, circular on the page: aspect is y data units per x data unit there."""
    rx = w * 0.2
    ry = min(rx * aspect, h)
    x0, x1 = x - w / 2, x + w / 2
    verts = [(x0, 0), (x0, h - ry), (x0, h), (x0 + rx, h), (x1 - rx, h), (x1, h), (x1, h - ry), (x1, 0), (x0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor=color, linewidth=0, zorder=3))


def figure(records, card, subset, out, anonymous=False):
    """One input set on one GPU, FP8 above BF16; False if the records hold neither."""
    data = {t: report.speedups(records.get((card, t, subset), {})) for t, _ in DTYPES}
    lengths = [(k, k.upper()) for k in report.LENGTHS if any(k in d for d in data.values())]
    if not lengths:
        return False
    drawn = {m for d in data.values() for row in d.values() for m in row}
    methods = [(m, label, color) for m, label, color in METHODS if m in drawn]
    bar_w = GROUP / len(methods)
    top = max(v for d in data.values() for row in d.values() for v in row.values())
    ymax = (int(top * 2) + 2) / 2

    height = HEADER + 2 * PANEL_H + GAP + FOOTER
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=LEFT / WIDTH, right=1 - RIGHT / WIDTH, top=1 - HEADER / height, bottom=FOOTER / height,
                        hspace=GAP / PANEL_H)
    aspect = (ymax / PANEL_H) / (len(lengths) / (WIDTH - LEFT - RIGHT))
    for ax, (dtype, dname) in zip(axes, DTYPES):
        d = data[dtype]
        ax.axhline(1, color=DENSE_C, lw=1.1, ls=(0, (4, 3)), zorder=2)
        for x, (k, _) in enumerate(lengths):
            row = d.get(k, {})
            x0 = x - len(methods) * bar_w / 2
            for j, (m, _, color) in enumerate(methods):
                if m not in row:
                    continue
                bx = x0 + (j + 0.5) * bar_w
                bar(ax, bx, row[m], bar_w * 0.86, color, aspect)
                if m == METHODS[0][0]:
                    ax.annotate(f"{row[m]:.2f}", (bx, row[m]), textcoords="offset points", xytext=(0, 3),
                                ha="center", fontsize=8, fontweight="bold", color=color, zorder=7)
        ax.set_xlim(-0.5, len(lengths) - 0.5)
        ax.set_ylim(0, ymax)
        ax.set_xticks(range(len(lengths)), [name for _, name in lengths])
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if v == 0 else f"{v:g}\u00d7"))
        ax.grid(axis="y", color=GRID, lw=0.9)
        ax.set_axisbelow(True)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.spines["bottom"].set_linewidth(1.3)
        ax.spines["bottom"].set_zorder(5)
        ax.tick_params(length=0, pad=5, labelsize=8.5, colors=MUTED)
        ax.tick_params(axis="x", labelsize=9.5, colors=INK, pad=6)
        ax.set_title(dname, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=6)

    handles = [Patch(facecolor=c, edgecolor=c, linewidth=0, label=label) for _, label, c in methods]
    handles.append(Line2D([], [], color=DENSE_C, lw=1.1, ls=(0, (4, 3)), label=f"{DENSE} (1\u00d7)"))
    rows = -(-len(handles) // NCOLS)
    handles = [handles[r * NCOLS + c] for c in range(NCOLS) for r in range(rows) if r * NCOLS + c < len(handles)]
    left = LEFT / WIDTH
    _, title, subtitle = SETS[subset]
    if not anonymous:
        title = f"{title} on {CARDS.get(card, card.upper())}"
    lead = "Speedup of one full prefill over dense scoring + FlashInfer top-k (higher is better), topk = 2048."
    fig.text(left, 1 - 0.32 / height, title, fontsize=18, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(left, 1 - 0.56 / height, f"{lead}\n{subtitle}.", fontsize=9.5, color=MUTED, ha="left", va="top",
             linespacing=1.45)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left - 0.006, 1 - 1.0 / height), ncols=NCOLS,
               frameon=False, fontsize=9, handlelength=1.3, handleheight=0.9, columnspacing=1.6, labelcolor=INK)
    notes = ("Each bar is the geometric mean over its calls.\nA method without a bar does not run on that operand "
             "type or failed at that length.")
    fig.text(left, 0.08 / height, notes, fontsize=8, color=MUTED, ha="left", va="bottom", linespacing=1.5)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", default=os.path.join(HERE, "results"))
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--cards", default=",".join(CARDS))
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--anonymous", action="store_true", help="name no GPU (one card)")
    a = p.parse_args()
    cards, subsets = a.cards.split(","), a.sets.split(",")
    if a.anonymous and len(cards) != 1:
        raise SystemExit("--anonymous draws one card")
    records = report.load(a.records)
    os.makedirs(a.out, exist_ok=True)
    for card in cards:
        for subset in subsets:
            name = f"performance_{SETS[subset][0]}" + ("" if a.anonymous else f"_{card}") + ".png"
            if figure(records, card, subset, os.path.join(a.out, name), a.anonymous):
                print(os.path.join(a.out, name))


if __name__ == "__main__":
    main()
