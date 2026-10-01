#!/usr/bin/env python3
"""Theorem 3: what the paper claims about convergence in h, and what the data shows.

READ THIS BEFORE INTERPRETING THE OUTPUT. An earlier version of this script got the premise
wrong in two ways, both corrected here after checking the paper directly.

**What Gamma^{h,*} is.** Page 22, immediately after equation (35):

    define Gamma^{h,*} := min_{phi in Phi-hat_h} Gamma^{h,phi} as the minimum of the RELAXED
    MAB-ML problem described in (32) and (31)

So Gamma^{h,*} is the relaxed optimum -- the LP's optimum, which is exactly the lower bound these
runs report -- and NOT the true optimum of the original problem. (The paper then notes "Gamma^{h,*}
is a lower bound to the minimum of MAB-ML".) The quantity in Theorem 3 is therefore the gap we can
actually measure, provided our LP is that relaxation. There is no unobservable optimum-to-bound gap
to absorb, and so no floor term is needed on the theorem's own terms.

**What Theorem 3 actually says.** Page 23, verbatim:

    Theorem 3 (Exponential Convergence): If the MAB-ML process is strictly indexable and a policy
    phi in Phi_h satisfies (29), (30) and (38), then, for any delta > 0, there exist H > 0 and
    s > 0 such that, for all h > H,  Gamma^{h,phi} - Gamma^{h,*} <= e^{-sh} + delta.   (40)

Note the quantifier order: delta is chosen first, then H and s may depend on it. Any non-negative
sequence converging to zero satisfies this -- given delta, pick H beyond which the gap is below
delta, and then any s > 0 works because e^{-sh} > 0. h^{-1/2} satisfies it. So inequality (40) is
equivalent to "the gap converges to zero" and does NOT pin down an exponential rate, even though the
theorem's title and the surrounding prose say "exponential". A curve fit therefore cannot contradict
(40); the gap between the prose and the formal statement is a reading of the paper, not a
measurement.

**What this script can honestly do**, then, is a restricted empirical comparison: over the range of h
actually reachable, does the measured gap look more like exponential decay or like a power law? That
is worth knowing -- it speaks to the paper's prose claim and to the O(sqrt(N)) comparison it draws --
but it is not a test of Theorem 3.

Method: weighted least squares on

    exponential   gap(h) = A exp(-k h) + C
    power law     gap(h) = A h^(-p)   + C

with A, C >= 0 and rate > 0. Because Gamma^{h,*} is the relaxed optimum, the gap should tend to 0, so
C = 0 is the model the theory implies; C >= 0 free is reported as a sensitivity. Errors are the
Monte-Carlo standard errors, obtained from the reported 95% half-widths by dividing by the Student-t
quantile -- a half-width is not one sigma, and treating it as one understates chi-square by t^2
(about 3.85x at 1,000 runs). With equal parameter counts, AIC differences reduce to chi-square
differences, so that is what is reported, alongside an absolute goodness-of-fit p-value: a model can
win the comparison and still be a bad description of the data.

**Tail windows are part of the standard output, not an afterthought.** A convergence claim is about
large h, so the comparison is repeated with the smallest scales dropped. On the h <= 50 data the
full-range preference for the power law came almost entirely from h = 1 and vanished for h >= 10.

Caveat this script cannot address: `simulate.py` passes the same seed at each h, and `sim.py` builds
run seeds without h, so errors across h may be correlated. The weighting here assumes independence.
Matched run-level costs would be needed to estimate the covariance.

Usage:
    ~/patrol-venv/bin/python analyse_theorem3.py --dir benchmarks/theorem3 --runs 10000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist
from scipy.stats import t as student_t

CASE_LABEL = {1: "Case I (I=6)", 2: "Case II (I=10)", 3: "Case III (I=14)"}
COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb"
TAIL_CUTOFFS = (1, 2, 5, 10)          # predefined, not chosen after seeing the result


def exponential(h, A, k, C):
    return A * np.exp(-k * h) + C


def power_law(h, A, p, C):
    return A * h ** (-p) + C


MODELS = (("exponential", exponential), ("power law", power_law))


def load(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """case -> arrays, from the rows simulate.py writes (col 0 = h, 4 = MAI cost, 5 = MAI CI, 9 = bound)."""
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
        out[c] = dict(h=a[:, 0], mai=a[:, 4], ci=a[:, 5], gap=a[:, 8], bound=a[:, 9])
    return out


def standard_errors(ci: np.ndarray, bound: np.ndarray, runs: int) -> np.ndarray:
    """Reported 95% half-width -> standard error on the gap scale.

    The gap is (mean - bound)/bound with bound deterministic, so it is an affine transform of the
    mean and SE(gap) = SE(mean)/bound exactly -- no ratio bias. The half-width is t * SE, hence the
    division by the Student-t quantile.

    (The C++ this port reproduces computes SE as sqrt(v/n) with v the divisor-n variance, where the
    sample SE would be sqrt(v/(n-1)); that makes these about 0.05% small at n = 1000. It is the
    authors' convention, reproduced bit-for-bit on purpose, and immaterial at this size.)
    """
    return ci / (student_t.ppf(0.975, runs - 1) * bound)


def fit(h, gap, sigma, f, free_floor: bool):
    """Weighted least squares with A, C >= 0 and rate > 0. Returns (params, chi2, dof)."""
    hi_C = np.inf if free_floor else 1e-12
    p0 = [max(gap[0] - gap[-1], 1e-6), 0.3 if f is exponential else 0.7,
          max(gap[-1], 1e-9) if free_floor else 0.0]
    p0[2] = min(p0[2], hi_C)
    popt, _ = curve_fit(f, h, gap, p0=p0, sigma=sigma, absolute_sigma=True,
                        bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, hi_C]), maxfev=400000)
    chi2 = float((((gap - f(h, *popt)) / sigma) ** 2).sum())
    k_free = 3 if free_floor else 2
    return popt, chi2, max(len(h) - k_free, 1)


def compare(h, gap, sigma, free_floor: bool):
    out = {}
    for name, f in MODELS:
        try:
            popt, chi2, dof = fit(h, gap, sigma, f, free_floor)
            out[name] = dict(p=popt, chi2=chi2, dof=dof,
                             gof=float(chi2_dist.sf(chi2, dof)))
        except Exception as e:                                   # noqa: BLE001
            out[name] = dict(error=str(e))
    return out


def print_comparison(res: dict, label: str) -> float | None:
    """Prints one model comparison; returns dchi2 = chi2(exp) - chi2(power), or None."""
    print(f"\n  {label}")
    print(f"    {'model':>12} {'shape':>9} {'floor C':>11} {'chi2':>9} {'dof':>4} {'gof p':>9}")
    for name in ("exponential", "power law"):
        d = res.get(name, {})
        if "error" in d:
            print(f"    {name:>12}   fit failed: {d['error']}")
            return None
        A, rate, C = d["p"]
        print(f"    {name:>12} {rate:>9.4f} {C:>11.6f} {d['chi2']:>9.2f} {d['dof']:>4d} {d['gof']:>9.3g}")
    dchi2 = res["exponential"]["chi2"] - res["power law"]["chi2"]
    better = "power law" if dchi2 > 0 else "exponential"
    verdict = better if abs(dchi2) >= 2 else "neither (|dAIC| < 2)"
    print(f"    dAIC = dchi2 (exp - power) = {dchi2:+.2f}  ->  {verdict}")
    return dchi2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="benchmarks/theorem3", help="directory holding case<N>_sweep.out")
    ap.add_argument("--runs", type=int, default=1000, help="Monte-Carlo runs behind each point")
    ap.add_argument("--plot", default="benchmarks/theorem3.png", help='empty string to skip')
    args = ap.parse_args()

    data = load(Path(args.dir).expanduser())
    if not data:
        print(f"no case<N>_sweep.out found in {args.dir}", file=sys.stderr)
        return 1

    print("Theorem 3 states: for any delta > 0 there exist H, s > 0 with gap(h) <= e^{-sh} + delta")
    print("for all h > H. Any sequence tending to zero satisfies that, so the comparison below is an")
    print("empirical description over the h actually measured -- NOT a test of the theorem.")
    print(f"Standard errors from the reported 95% half-widths, {args.runs} runs per point.\n")

    fits = {}
    for c, d in sorted(data.items()):
        sigma = standard_errors(d["ci"], d["bound"], args.runs)
        print(f"=== {CASE_LABEL[c]} — {len(d['h'])} points, h = {int(d['h'][0])}..{int(d['h'][-1])} ===")
        print(f"{'h':>5} {'MAI cost':>11} {'gap':>10} {'SE':>10} {'SE/gap':>8}")
        for k in range(len(d["h"])):
            g = d["gap"][k]
            print(f"{int(d['h'][k]):>5} {d['mai'][k]:>11.3f} {g*100:>9.4f}% "
                  f"{sigma[k]*100:>9.4f}% {sigma[k]/g if g else float('nan'):>8.3f}")

        res0 = compare(d["h"], d["gap"], sigma, free_floor=False)
        resC = compare(d["h"], d["gap"], sigma, free_floor=True)
        print_comparison(res0, "C = 0 (the theory's asymptote: the gap tends to zero)")
        print_comparison(resC, "C >= 0 free (sensitivity)")
        fits[c] = resC

        # tail windows: a convergence claim is about large h
        print(f"\n    tail windows, C >= 0 free:")
        print(f"      {'h >=':>5} {'n':>3} {'dchi2 (exp - power)':>21}  favours")
        for cut in TAIL_CUTOFFS:
            m = d["h"] >= cut
            if m.sum() < 5:
                continue
            r = compare(d["h"][m], d["gap"][m], sigma[m], free_floor=True)
            if "error" in r.get("exponential", {}) or "error" in r.get("power law", {}):
                continue
            dd = r["exponential"]["chi2"] - r["power law"]["chi2"]
            fav = "power law" if dd >= 2 else ("exponential" if dd <= -2 else "neither")
            print(f"      {cut:>5} {int(m.sum()):>3} {dd:>21.2f}  {fav}")
        print()

    print("=" * 78)
    print("Reading: a model winning the comparison is not the same as fitting the data -- check the")
    print("goodness-of-fit p. And if the preference disappears once the smallest h are dropped, it")
    print("is a small-scale effect, not evidence about convergence.")

    if args.plot and data:
        try:
            make_plot(data, fits, Path(args.plot), args.runs)
        except Exception as e:                                   # noqa: BLE001
            print(f"\n(plot skipped: {e})")
    return 0


def make_plot(data, fits, out: Path, runs: int) -> None:
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

        sigma = standard_errors(d["ci"], d["bound"], runs)
        ax.errorbar(d["h"], d["gap"], yerr=sigma, fmt="o", color=COLORS[c], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.0, elinewidth=1.2,
                    ecolor=INK3, capsize=2, label="measured (±1 SE)", zorder=3)
        grid = np.linspace(d["h"].min(), d["h"].max(), 400)
        for (name, f), style in zip(MODELS, ("-", "--")):
            r = fits.get(c, {}).get(name)
            if r and "p" in r:
                ax.plot(grid, f(grid, *r["p"]), style, color=INK2, linewidth=1.6, label=name, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("scaling parameter h", color=INK3, fontsize=10)
        ax.set_title(CASE_LABEL[c], color=INK, fontsize=11, loc="left", pad=8)
        if col == 0:
            ax.set_ylabel("gap from the relaxed optimum", color=INK3, fontsize=10)
            ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    hs = sorted({int(v) for d in data.values() for v in d["h"]})
    fig.suptitle("Measured gap against h, with both candidate decays", color=INK, fontsize=12.5,
                 x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.012, f"Log-log axes, so a power law is a straight line and an exponential bends "
                           f"down; h = {hs[0]}–{hs[-1]}, {runs:,} runs per point. Theorem 3 does not "
                           f"pin a rate — see the module docstring.", color=INK3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
