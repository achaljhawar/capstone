#!/usr/bin/env python3
"""Theorem 3: does MAI's deviation decay exponentially in the scaling parameter h?

The paper (p. 24) states that "the performance deviation between phi and an optimal solution of
MAB-ML diminishes exponentially in the scaling parameter h", and presents this as stronger than the
O(sqrt(N)) bound of the prior work it cites.  It is a numerical claim and nothing in the project
tested it, so this does.

What can actually be measured is deviation from the LP *lower bound*, not from the optimum:

    measured(h) = (cost(h) - bound) / bound
                = (cost(h) - optimum)/bound  +  (optimum - bound)/bound
                   \\___ what Theorem 3 bounds ___/   \\___ the relaxation's own gap ___/

The second term is not something we can observe at scale, so every model fitted here carries a free
floor C to absorb it.  That is the honest reading, and it is why the floor is reported: a model is
only interesting if it fits *and* its floor is plausible.  tests/check_lower_bound.py showed the
relaxation is very tight on instances small enough to solve exactly (0.00-0.05% slack), which is the
evidence that the floor is small -- but that was at h = 1 on three areas, so it is suggestive, not
proof at these scales.

Two candidates are compared:

    exponential   dev(h) = A exp(-k h) + C      <- what the paper claims
    power law     dev(h) = A h^(-p)   + C       <- the weaker, sqrt-like regime

They are compared on RMSE and on AIC (which penalises nothing here, both having three parameters, so
it reduces to the likelihood -- but it is reported so the comparison is explicit).  Points are
weighted by the inverse square of each point's confidence interval, because the interval shrinks
steeply with h and an unweighted fit would let the noisy small-h points dominate.

Usage:
    ~/patrol-venv/bin/python analyse_theorem3.py --dir ~/patrol/theorem3
    ~/patrol-venv/bin/python analyse_theorem3.py --dir benchmarks/theorem3 --plot benchmarks/theorem3.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

CASE_LABEL = {1: "Case I (I=6)", 2: "Case II (I=10)", 3: "Case III (I=14)"}
COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb"


def exponential(h, A, k, C):
    return A * np.exp(-k * h) + C


def power_law(h, A, p, C):
    return A * h ** (-p) + C


MODELS = (("exponential", exponential, "rate k"), ("power law", power_law, "exponent p"))


def load(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """case -> arrays, from the 10-column rows simulate.py writes."""
    out = {}
    for c in (1, 2, 3):
        f = path / f"case{c}_sweep.out"
        if not f.exists():
            continue
        rows = []
        for line in f.read_text().splitlines():
            cols = line.rstrip("\t").split("\t")
            if len(cols) >= 10 and cols[0].strip():
                rows.append([float(x) for x in cols[:10]])
        if not rows:
            continue
        a = np.array(sorted(rows))
        h, mai, mai_ci, dev, bound = a[:, 0], a[:, 4], a[:, 5], a[:, 8], a[:, 9]
        out[c] = dict(h=h, mai=mai, ci=mai_ci, dev=dev, bound=bound,
                      dev_ci=mai_ci / bound)          # the interval, carried onto the deviation scale
    return out


def fit(h, dev, dev_ci, f, p0):
    """Weighted least squares; sigma is each point's CI on the deviation scale."""
    sigma = np.where(dev_ci > 0, dev_ci, np.nanmedian(dev_ci))
    popt, _ = curve_fit(f, h, dev, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=400000)
    resid = (dev - f(h, *popt)) / sigma
    chi2 = float((resid ** 2).sum())
    rmse = float(np.sqrt(((dev - f(h, *popt)) ** 2).mean()))
    n = len(h)
    aic = n * np.log(max(chi2 / n, 1e-300)) + 2 * len(popt)
    r2 = 1 - float(((dev - f(h, *popt)) ** 2).sum() / ((dev - dev.mean()) ** 2).sum())
    return popt, dict(chi2=chi2, rmse=rmse, aic=aic, r2=r2, dof=max(n - len(popt), 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="benchmarks/theorem3", help="directory holding case<N>_sweep.out")
    ap.add_argument("--plot", default="benchmarks/theorem3.png")
    args = ap.parse_args()

    data = load(Path(args.dir).expanduser())
    if not data:
        print(f"no case<N>_sweep.out found in {args.dir}", file=sys.stderr)
        return 1

    verdicts = {}
    for c, d in sorted(data.items()):
        print(f"\n=== {CASE_LABEL[c]} — {len(d['h'])} points, h = {int(d['h'][0])}..{int(d['h'][-1])} ===")
        print(f"{'h':>5} {'MAI cost':>11} {'deviation':>11} {'+- CI':>10} {'CI/signal':>10}")
        for k in range(len(d["h"])):
            sig = d["dev"][k]
            print(f"{int(d['h'][k]):>5} {d['mai'][k]:>11.3f} {sig * 100:>10.4f}% "
                  f"{d['dev_ci'][k] * 100:>9.4f}% {d['dev_ci'][k] / sig if sig else float('nan'):>10.3f}")

        print(f"\n{'model':>12} {'shape':>12} {'floor C':>11} {'RMSE':>10} {'chi2/dof':>9} {'AIC':>9} {'R2':>8}")
        res = {}
        for name, f, _ in MODELS:
            p0 = (d["dev"][0], 0.15, d["dev"][-1]) if name == "exponential" else (d["dev"][0], 1.0, d["dev"][-1])
            try:
                popt, st = fit(d["h"], d["dev"], d["dev_ci"], f, p0)
            except Exception as e:                       # noqa: BLE001
                print(f"{name:>12}   fit failed: {e}")
                continue
            res[name] = (popt, st)
            print(f"{name:>12} {popt[1]:>12.4f} {popt[2]:>11.6f} {st['rmse']:>10.2e} "
                  f"{st['chi2'] / st['dof']:>9.2f} {st['aic']:>9.2f} {st['r2']:>8.4f}")

        if len(res) == 2:
            d_aic = res["exponential"][1]["aic"] - res["power law"][1]["aic"]
            better = "exponential" if d_aic < 0 else "power law"
            # a gap under ~2 AIC units is not a distinction worth drawing
            verdict = better if abs(d_aic) >= 2 else "inconclusive"
            verdicts[c] = (verdict, d_aic, res)
            print(f"\n  AIC(exponential) - AIC(power law) = {d_aic:+.2f}"
                  f"   ->  {'favours ' + better if verdict != 'inconclusive' else 'inconclusive (gap < 2)'}")

    if verdicts:
        print("\n" + "=" * 72)
        print("Theorem 3 claims exponential decay. Across the cases:")
        for c, (v, d_aic, _) in sorted(verdicts.items()):
            print(f"  {CASE_LABEL[c]:<18} {v:<14} (dAIC {d_aic:+.2f})")
        print("\nDeviation here is measured against the LP lower bound, so each fit carries a floor")
        print("absorbing the relaxation's own gap; a power-law win is evidence about what is visible")
        print("at h <= 100, not a refutation of an asymptotic theorem.")

    if args.plot and data:
        try:
            make_plot(data, verdicts, Path(args.plot))
        except Exception as e:                           # noqa: BLE001
            print(f"\n(plot skipped: {e})")
    return 0


def make_plot(data, verdicts, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(data), figsize=(max(4.6 * len(data), 8.0), 5.2), dpi=150,
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for col, (c, d) in enumerate(sorted(data.items())):
        ax = axes[0][col]
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=10, length=3, color=GRID)

        ax.errorbar(d["h"], d["dev"], yerr=d["dev_ci"], fmt="o", color=COLORS[c], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, elinewidth=1.2,
                    ecolor=INK3, capsize=2, label="measured", zorder=3)
        grid = np.linspace(d["h"].min(), d["h"].max(), 400)
        if c in verdicts:
            _, _, res = verdicts[c]
            for (name, f, _), style in zip(MODELS, ("-", "--")):
                if name in res:
                    ax.plot(grid, f(grid, *res[name][0]), style, color=INK2, linewidth=1.6,
                            label=name, zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel("scaling parameter h", color=INK3, fontsize=10)
        ax.set_title(CASE_LABEL[c], color=INK, fontsize=11, loc="left", pad=8)
        if col == 0:
            ax.set_ylabel("MAI deviation from the lower bound (log scale)", color=INK3, fontsize=10)
            ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Does MAI's deviation decay exponentially in h?", color=INK, fontsize=12.5,
                 x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.012, "Log y-axis; error bars are 95% CIs. Both models carry a floor absorbing the "
                           "relaxation's own gap, so both flatten — what separates them is the shape of "
                           "the approach, not the plateau.", color=INK3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
