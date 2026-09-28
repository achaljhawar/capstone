#!/usr/bin/env python3
"""Draw the charts in benchmarks.md from benchmarks/results.tsv (light and dark SVG of each).

Usage: python3 benchmarks/plot.py      (needs matplotlib)
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=["#3987e5", "#d95926", "#199e70"]),
}
LINE_PT = 1.5          # 2 px
MARKER_PT = 6          # 8 px across
RING_PT = 1.5          # 2 px surface ring around markers


def load() -> list[dict]:
    with open(HERE / "results.tsv") as f:
        rows = [r for r in csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t")]
    for r in rows:
        for k, v in r.items():
            if k != "value_function":
                r[k] = float(v)
    return rows


def style(ax, th, title: str, subtitle: str) -> None:
    fig = ax.figure
    fig.patch.set_facecolor(th["surface"])
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["axis"])
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], labelsize=9, length=0, pad=6)
    ax.grid(True, axis="y", color=th["grid"], linewidth=0.75, linestyle="-")
    ax.set_axisbelow(True)
    fig.text(0.012, 0.965, title, color=th["ink"], fontsize=12.5, fontweight="semibold", va="top")
    fig.text(0.012, 0.905, subtitle, color=th["ink2"], fontsize=9.5, va="top")


def line(ax, th, x, y, color, label):
    ax.plot(x, y, color=color, linewidth=LINE_PT, solid_capstyle="round", solid_joinstyle="round",
            marker="o", markersize=MARKER_PT, markerfacecolor=color, markeredgecolor=th["surface"],
            markeredgewidth=RING_PT, label=label, zorder=3)


def end_label(ax, th, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(10, dy), textcoords="offset points", va="center", fontsize=9.5,
                color=th["ink"], annotation_clip=False)


def legend(ax, th):
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=3, frameon=False, fontsize=9.5,
                    handlelength=1.6, columnspacing=1.6, labelcolor=th["ink2"])
    for h in leg.legend_handles:
        h.set_markeredgecolor(th["surface"])


def gap_chart(rows, mode: str) -> None:
    th = THEMES[mode]
    lp = [r for r in rows if r["value_function"] == "lp"]
    dp = [r for r in rows if r["value_function"] == "dp"]
    x = [r["scaler"] for r in lp]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.subplots_adjust(left=0.075, right=0.80, top=0.77, bottom=0.12)
    style(ax, th, "Gap to the lower bound as the system grows",
          "(average cost − lower bound) / lower bound; shaded = 95% confidence interval; 1,000 runs per point")
    series = [("Greedy", lp, "greedy", th["series"][0]),
              ("MAI, --value-function lp", lp, "mai", th["series"][1]),
              ("MAI, --value-function dp", dp, "mai", th["series"][2])]
    for name, rs, key, color in series:
        lb = rs[0]["lower_bound"]
        y = [100 * r[f"{key}_dev"] for r in rs]
        ci = [100 * r[f"{key}_ci"] / lb for r in rs]
        ax.fill_between(x, [a - b for a, b in zip(y, ci)], [a + b for a, b in zip(y, ci)], color=color, alpha=0.12,
                        linewidth=0, zorder=2)
        line(ax, th, x, y, color, name)
    end_label(ax, th, x[-1], 100 * lp[-1]["greedy_dev"], f"Greedy  {100 * lp[-1]['greedy_dev']:.1f}%")
    end_label(ax, th, x[-1], 100 * lp[-1]["mai_dev"], f"MAI (lp)  {100 * lp[-1]['mai_dev']:.1f}%", dy=3)
    end_label(ax, th, x[-1], 100 * dp[-1]["mai_dev"], f"MAI (dp)  {100 * dp[-1]['mai_dev']:.1f}%", dy=-3)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{int(v)}" for v in x])
    ax.set_xlim(0.85, 118)
    ax.set_ylim(0, 16)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_yticklabels(["0%", "4%", "8%", "12%", "16%"])
    ax.set_xlabel("scaler (sub-areas per area; log scale)", color=th["ink2"], fontsize=9.5, labelpad=6)
    ax.annotate("0% = the LP lower bound, 3407.97", (1, 0), xytext=(2, 5), textcoords="offset points",
                fontsize=8.5, color=th["muted"])
    legend(ax, th)
    fig.savefig(HERE / f"gap-{mode}.svg", facecolor=th["surface"])
    plt.close(fig)


def runtime_chart(rows, mode: str) -> None:
    th = THEMES[mode]
    lp = [r for r in rows if r["value_function"] == "lp"]
    x = [r["scaler"] for r in lp]
    py, cpp = [r["python_seconds"] for r in lp], [r["cpp_seconds"] for r in lp]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.subplots_adjust(left=0.085, right=0.84, top=0.77, bottom=0.12)
    style(ax, th, "Simulation time: Python port vs the C++",
          "1,000 runs of each policy (Greedy + MAI); wall-clock seconds, Apple M4; both axes log scale")
    line(ax, th, x, py, th["series"][0], "Python port")
    line(ax, th, x, cpp, th["series"][1], "C++ (GCC 15, -O2)")
    end_label(ax, th, x[-1], py[-1], f"Python  {py[-1]:.0f} s")
    end_label(ax, th, x[-1], cpp[-1], f"C++  {cpp[-1]:.0f} s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{int(v)}" for v in x])
    ax.set_xlim(0.85, 118)
    ax.yaxis.set_minor_locator(NullLocator())
    ticks = [0.1, 1, 10, 100, 1000]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels(["0.1 s", "1 s", "10 s", "100 s", "1,000 s"])
    ax.set_ylim(0.1, 1000)
    ax.set_xlabel("scaler (sub-areas per area; log scale)", color=th["ink2"], fontsize=9.5, labelpad=6)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=2, frameon=False, fontsize=9.5,
                    handlelength=1.6, columnspacing=1.6, labelcolor=th["ink2"])
    for h in leg.legend_handles:
        h.set_markeredgecolor(th["surface"])
    fig.savefig(HERE / f"runtime-{mode}.svg", facecolor=th["surface"])
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "svg.fonttype": "path", "svg.hashsalt": "benchmarks", "axes.unicode_minus": True})
    rows = load()
    for mode in THEMES:
        gap_chart(rows, mode)
        runtime_chart(rows, mode)
    print(f"wrote {HERE}/gap-{{light,dark}}.svg and runtime-{{light,dark}}.svg")


if __name__ == "__main__":
    main()
