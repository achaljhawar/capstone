#!/usr/bin/env python3
"""Does any case flatten to a nonzero floor?  That, and only that, bears on Theorem 3.

Inequality (40) is equivalent to "the gap converges to zero": if the gap tended to c > 0,
taking delta = c/2 would require gap <= e^{-sh} + c/2 for all large h, and e^{-sh} -> 0.
So the rate the theorem's title advertises is not testable, but a persistent positive
asymptote would contradict (40) outright.  That asymptote is what this script measures,
and it is the complement to analyse_theorem3.py, which compares decay shapes.

Per case and per tail window it reports the fitted floor C with a profile-likelihood 95%
interval (dchi2 = 3.841), so "includes zero" and "excludes zero" become statable.  It then
then reports what the gap actually did between the two largest h, which assumes no curve at all,
alongside what the full-range C = 0 power law predicted.  That second comparison is NOT model-free
and is not a calibrated test: the predicted drop comes from a fit that may itself be rejected, and
the discrepancy ignores uncertainty in the fitted exponent and any dependence across h.

Usage:
    ~/patrol-venv/bin/python check_theorem3_floor.py --dir benchmarks/theorem3_h100
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t as student_t, chi2

TAIL_CUTOFFS = (1, 10, 20, 28)


def load(path: str, runs: int):
    """h, MAI gap and its Monte-Carlo SE from one case<N>_sweep.out.

    Columns, as simulate.py writes them: 0=h, 1=GRD cost, 2=GRD ci, 3=GRD aux, 4=MAI cost,
    5=MAI ci, 6=MAI aux, 7=GRD gap, 8=MAI gap, 9=lower bound, 10=slackness.  The ci columns
    are 95% half-widths in cost units, so they become a ratio to the bound and then an SE by
    dividing out the Student-t quantile -- a half-width is not one sigma.
    """
    rows = np.loadtxt(path)
    tcrit = student_t.ppf(0.975, runs - 1)
    h = rows[:, 0]
    gap = rows[:, 8]
    se = (rows[:, 5] / rows[:, 9]) / tcrit
    return h, gap, se


def fit(h, gap, se, floor_free):
    if floor_free:
        f = lambda x, a, p, c: a * x ** (-p) + c
        p0, lo, hi = [gap[0], 0.7, 0.0], [0, 1e-6, 0], [np.inf, 50, np.inf]
    else:
        f = lambda x, a, p: a * x ** (-p)
        p0, lo, hi = [gap[0], 0.7], [0, 1e-6], [np.inf, 50]
    popt, _ = curve_fit(f, h, gap, p0=p0, sigma=se, absolute_sigma=True,
                        bounds=(lo, hi), maxfev=200000)
    r = (gap - f(h, *popt)) / se
    return popt, float(r @ r)


def profile_floor(h, gap, se, chi2_best):
    """95% interval for C by scanning it and refitting the rest."""
    out = []
    for c in np.linspace(0.0, 0.03, 601):
        f = lambda x, a, p: a * x ** (-p) + c
        try:
            popt, _ = curve_fit(f, h, gap, p0=[gap[0], 0.7], sigma=se,
                                absolute_sigma=True, bounds=([0, 1e-6], [np.inf, 50]),
                                maxfev=200000)
        except Exception:
            continue
        r = (gap - f(h, *popt)) / se
        if float(r @ r) - chi2_best <= 3.841:
            out.append(c)
    return (min(out), max(out)) if out else (float("nan"), float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="benchmarks/theorem3_h100",
                    help="directory holding case<N>_sweep.out")
    ap.add_argument("--runs", type=int, default=10000,
                    help="Monte-Carlo runs behind each point")
    args = ap.parse_args()

    for case in (1, 2, 3):
        h, gap, se = load("%s/case%d_sweep.out" % (args.dir, case), args.runs)
        print("=" * 78)
        print("Case %d" % case)
        for cut in TAIL_CUTOFFS:
            m = h >= cut
            if m.sum() < 4:
                continue
            (a, p, c), x2 = fit(h[m], gap[m], se[m], True)
            lo, hi = profile_floor(h[m], gap[m], se[m], x2)
            dof = m.sum() - 3
            print("  h>=%-4d n=%-3d floor C = %.4f%%  95%% [%.4f%%, %.4f%%]   exponent %.3f  "
                  "chi2 %.1f/%d  p %.3g"
                  % (cut, m.sum(), c * 100, lo * 100, hi * 100, p, x2, dof,
                     chi2.sf(x2, dof) if dof > 0 else float("nan")))
        # model-free: the two largest h
        g1, g2, s1, s2 = gap[-2], gap[-1], se[-2], se[-1]
        (a0, p0), _ = fit(h, gap, se, False)
        pred = g1 * (h[-1] / h[-2]) ** (-p0)
        d = g1 - g2
        sd = np.hypot(s1, s2)
        print("  h %d -> %d: gap %.4f%% -> %.4f%%, moved %.4f%% +- %.4f%% (%.1f sigma)"
              % (h[-2], h[-1], g1 * 100, g2 * 100, d * 100, sd * 100, d / sd))
        # The discrepancy is observed-minus-predicted at the final h.  Dividing the PREDICTED
        # drop by the SE instead -- which this line did until an adversarial review caught it --
        # silently omits the observed drop and understated Case II at 5.8 sigma instead of 6.7.
        print("            a C=0 power law (exponent %.3f) predicts %.4f%%, a drop of %.4f%%;"
              % (p0, pred * 100, (g1 - pred) * 100))
        print("            observed minus predicted at h=%d: %+.4f%% (%.1f sigma, uncalibrated)"
              % (h[-1], (g2 - pred) * 100, (g2 - pred) / sd))


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
