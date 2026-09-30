#!/usr/bin/env python3
"""Figure 3: cumulative distribution of the performance deviation over random settings.

Reads the CSV written by reproduce_fig3.py and draws one panel per case.  Within a panel the five
curves are the five horizons h = 1, 5, 10, 20, 40; h is an ordered magnitude, so it is encoded as a
single-hue sequential ramp running light (h=1) to dark (h=40), stepped on contrast against the chart
surface (2.6:1 to 9.5:1) and monotone in lightness.  The case keeps its categorical hue from
reproduce_fig4.py, so a case is the same colour in both figures.

The paper's claim is marked on Case I: at h = 40, MAI's deviation is under 3% in over 75% of settings.

Usage:
    ~/patrol-venv/bin/python plot_fig3.py --csv benchmarks/fig3.csv
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

H_VALUES = (1, 5, 10, 20, 40)
CASE_LABEL = {1: "Case I (I=6)", 2: "Case II (I=10)", 3: "Case III (I=14)"}
# Same categorical slots as reproduce_fig4.py: one hue per case.
COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb"
CONTRAST_STEPS = (2.6, 3.6, 5.0, 7.0, 9.5)   # targets for h = 1, 5, 10, 20, 40


def _rgb(h: str) -> tuple[float, float, float]:
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _lum(rgb) -> float:
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ramp(base: str) -> list[tuple[float, float, float]]:
    """Five steps of `base`, light to dark, each hitting a target contrast against SURFACE.

    Contrast falls as lightness rises, so bisect on lightness with hue and saturation held.
    """
    ls = _lum(_rgb(SURFACE))

    def contrast(rgb):
        return (ls + 0.05) / (_lum(rgb) + 0.05)

    hh, _, ss = colorsys.rgb_to_hls(*_rgb(base))
    out = []
    for target in CONTRAST_STEPS:
        lo, hi = 0.02, 0.98
        for _ in range(60):
            mid = (lo + hi) / 2
            if contrast(colorsys.hls_to_rgb(hh, mid, ss)) > target:
                lo = mid
            else:
                hi = mid
        out.append(colorsys.hls_to_rgb(hh, (lo + hi) / 2, ss))
    return out


def load(path: Path) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """(case, h) -> sorted deviation arrays for both policies."""
    acc: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: {"mai": [], "grd": []})
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (int(r["case"]), int(r["h"]))
            acc[key]["mai"].append(float(r["mai_dev"]))
            acc[key]["grd"].append(float(r["greedy_dev"]))
    return {k: {p: np.sort(np.array(v)) for p, v in d.items()} for k, d in acc.items()}


def cdf(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF as a step: P(deviation <= x)."""
    return xs, np.arange(1, xs.size + 1) / xs.size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="benchmarks/fig3.csv")
    ap.add_argument("--out", default="benchmarks/fig3.png")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"{path} not found; run reproduce_fig3.py first", file=sys.stderr)
        return 1
    data = load(path)
    if not data:
        print(f"{path} has no rows", file=sys.stderr)
        return 1

    cases = sorted({c for c, _ in data})
    n = max(d["mai"].size for d in data.values())

    # a single panel still needs room for the title and footer, so floor the width
    width = max(4.6 * len(cases), 8.0)
    fig, axes = plt.subplots(1, len(cases), figsize=(width, 5.2), dpi=150,
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for col, case in enumerate(cases):
        ax = axes[0][col]
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(True, axis="both", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=10, length=3, color=GRID)

        shades = ramp(COLORS[case])
        for k, h in enumerate(H_VALUES):
            d = data.get((case, h))
            if d is None:
                continue
            x, y = cdf(d["mai"])
            ax.step(x, y, where="post", color=shades[k], linewidth=2,
                    label=f"MAI, h = {h}" if col == 0 else None, zorder=3 + k)

        # GRD at the largest h, as a neutral dashed reference: a different mark type, not a new hue.
        d_last = data.get((case, H_VALUES[-1]))
        if d_last is not None:
            x, y = cdf(d_last["grd"])
            ax.step(x, y, where="post", color=INK3, linewidth=1.6, linestyle="--",
                    label=f"GRD, h = {H_VALUES[-1]}" if col == 0 else None, zorder=2)

        # Six series, so identity rests on the legend: the MAI curves run too close together near
        # the origin for a direct label to land without covering its neighbours.

        ax.axvline(0.03, color=INK3, linewidth=1, linestyle=":", zorder=1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.set_xlabel("deviation from the lower bound", color=INK3, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        ax.set_title(CASE_LABEL[case], color=INK, fontsize=11, loc="left", pad=8)

        # the paper's claim, annotated where it is made
        if case == 1 and (1, 40) in data:
            frac = float((data[(1, 40)]["mai"] < 0.03).mean())
            ax.axhline(0.75, color=INK3, linewidth=1, linestyle=":", zorder=1)
            # right edge, just above the 75% line: clear of every curve at that height
            ax.annotate(f"paper: >75% under 3% at h = 40\nmeasured {frac*100:.0f}%",
                        (ax.get_xlim()[1], 0.75), xytext=(-6, 7), textcoords="offset points",
                        ha="right", fontsize=9, color=INK2)

    axes[0][0].set_ylabel("fraction of settings", color=INK3, fontsize=10)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.suptitle("Cumulative distribution of MAI's performance deviation",
                 color=INK, fontsize=12.5, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.935, "over randomly generated settings, by horizon h",
             color=INK2, fontsize=10)
    fig.text(0.012, 0.012, f"{n} settings per case; 1,000 simulation runs per policy per setting; "
                           f"dotted lines mark 3% deviation and the 75% level",
             color=INK3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.92))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)

    # the table view the contrast WARN on the categorical green obliges
    print(f"\n{'case':>5} {'h':>4} {'n':>6} {'MAI median':>11} {'MAI p75':>9} {'MAI <3%':>9} {'GRD median':>11}")
    for case in cases:
        for h in H_VALUES:
            d = data.get((case, h))
            if d is None:
                continue
            mai, grd = d["mai"], d["grd"]
            print(f"{case:>5} {h:>4} {mai.size:>6} {np.median(mai)*100:>10.2f}% "
                  f"{np.percentile(mai, 75)*100:>8.2f}% {float((mai < 0.03).mean())*100:>8.1f}% "
                  f"{np.median(grd)*100:>10.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
