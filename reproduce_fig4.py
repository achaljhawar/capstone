#!/usr/bin/env python3
"""Figure 4: performance deviation of MAI and GRD against the scaling parameter h, for Cases I, II and III.

Reads the rows the C++ (or simulate.py) wrote for each case and plots them in the paper's form. The paper's
claim is that MAI's deviation is under 3% for all h >= 30; that line is drawn for reference.

Usage (from python/, after build_cases.sh and the per-case gen.py runs):
    ~/patrol-venv/bin/python reproduce_fig4.py
    ~/patrol-venv/bin/python reproduce_fig4.py --root .. --out benchmarks/fig4.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

CASES = {1: ("Case I (I=6)", 300), 2: ("Case II (I=10)", 395), 3: ("Case III (I=14)", 51)}
# Validated categorical slots: one hue per case, solid = MAI, dashed = GRD.
COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb"


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 10 and f[0].strip():
            rows.append([float(v) for v in f[:10]])
    if not rows:
        return None
    a = np.array(sorted(rows))
    return {"h": a[:, 0], "greedy_dev": a[:, 7], "mai_dev": a[:, 8], "lb": a[:, 9][0]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="..", help="directory holding test33_case<N>/")
    ap.add_argument("--out", default="benchmarks/fig4.png")
    args = ap.parse_args()
    root = Path(args.root)

    data = {}
    for case in CASES:
        d = load(root / f"test33_case{case}" / "long-term-performance-cost.out")
        if d:
            data[case] = d
    if not data:
        print("no per-case results found; run build_cases.sh, gen.py and the h sweep first", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=10, length=3, color=GRID)

    for case, d in sorted(data.items()):
        label, _ = CASES[case]
        c = COLORS[case]
        ax.plot(d["h"], d["mai_dev"], color=c, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"MAI, {label}", zorder=3)
        ax.plot(d["h"], d["greedy_dev"], color=c, linewidth=2, linestyle="--", marker="s", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, alpha=0.75, label=f"GRD, {label}", zorder=2)

    ax.axhline(0.03, color=INK3, linewidth=1, linestyle=":", zorder=1)
    ax.annotate("paper's 3% claim for h >= 30", (ax.get_xlim()[1], 0.03), xytext=(-6, 5),
                textcoords="offset points", ha="right", fontsize=9, color=INK3)

    all_h = sorted({int(v) for d in data.values() for v in d["h"]})
    ax.xaxis.set_major_locator(FixedLocator(all_h))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_xlabel("scaling parameter h", color=INK3, fontsize=10)
    ax.set_ylabel("deviation from the lower bound", color=INK3, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_title("Performance deviation of MAI and GRD against h", color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, ncol=2, labelcolor=INK2)
    bounds = ", ".join(f"{CASES[c][0].split(' ')[1]} {d['lb']:.2f}" for c, d in sorted(data.items()))
    fig.text(0.01, 0.005, f"1,000 runs per policy per point; lower bounds: {bounds}", color=INK3, fontsize=8.5)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)

    print(f"\n{'case':>5} {'h':>4} {'GRD':>9} {'MAI':>9}   paper: MAI < 3% for h >= 30")
    for case, d in sorted(data.items()):
        for k, h in enumerate(d["h"]):
            flag = "" if h < 30 else ("  PASS" if d["mai_dev"][k] < 0.03 else "  FAIL")
            print(f"{case:>5} {int(h):>4} {d['greedy_dev'][k]*100:>8.2f}% {d['mai_dev'][k]*100:>8.2f}%{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
