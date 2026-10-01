#!/usr/bin/env python3
"""Are the Theorem 3 convergence results properties of the method, or of one random setting?

benchmarks.md sections 8.3 and 8.4 fit a decay shape and an asymptote to three cases, each of
which is a single random setting: Case I is seed 300 on the six-area graph, Case II seed 395 on
the ten-area graph.  This script re-measures the tail on fresh seeds for the same graph sizes, so
those fits can be read against the spread they sit in.

For each (graph size, seed) it prints the MAI gap at h = 20, 55, 100 and the change from 55 to
100 in sigma.  An instance still converging keeps falling; one at a floor does not.  What the
run showed: Case II's floor is seed 395's alone -- every fresh ten-area seed keeps falling -- while
two of three fresh six-area seeds flatten and one rises, so a non-falling tail is common and
Case I's clean power law is the atypical draw.

Usage:
    ~/patrol-venv/bin/python check_theorem3_seeds.py --dir benchmarks/theorem3_seeds
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.stats import t as student_t

# The sweep points these are compared against, from benchmarks/theorem3_h100 at 10,000 runs:
# graph size -> (label in benchmarks.md, sweep seed, gaps at h = 20, 55, 100 in per cent).
REFERENCE = {6: ("Case I", 300, (1.0125, 0.5042, 0.3428)),
             10: ("Case II", 395, (1.2825, 1.0177, 0.9888))}
HS = (20, 55, 100)


def rows(path: str, runs: int):
    """h, MAI gap in per cent, and its Monte-Carlo SE, sorted by h.

    Columns as simulate.py writes them: 0=h, 5=MAI ci (a 95% half-width in cost units),
    8=MAI gap, 9=lower bound.  A half-width is not one sigma, hence the Student-t divisor.
    """
    r = np.atleast_2d(np.loadtxt(path))
    tcrit = student_t.ppf(0.975, runs - 1)
    h = r[:, 0]
    gap = r[:, 8] * 100
    se = (r[:, 5] / r[:, 9]) / tcrit * 100
    o = np.argsort(h)
    return h[o], gap[o], se[o]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="benchmarks/theorem3_seeds",
                    help="directory holding a<areas>_s<seed>.out")
    ap.add_argument("--runs", type=int, default=1000,
                    help="Monte-Carlo runs behind each fresh-seed point")
    args = ap.parse_args()

    for areas in sorted(REFERENCE):
        label, ref_seed, ref = REFERENCE[areas]
        print("=" * 76)
        print("I = %d   (%s in benchmarks.md, seed %d, 10,000 runs)" % (areas, label, ref_seed))
        print("  %-10s %9s %9s %9s   %s" % ("seed", "h=20", "h=55", "h=100", "55 -> 100"))
        print("  %-10s %8.3f%% %8.3f%% %8.3f%%   %+.3f pts"
              % ("%d*" % ref_seed, ref[0], ref[1], ref[2], ref[2] - ref[1]))
        for path in sorted(glob.glob("%s/a%d_s*.out" % (args.dir, areas))):
            seed = os.path.basename(path).split("_s")[1].split(".")[0]
            h, gap, se = rows(path, args.runs)
            d = {int(k): (v, s) for k, v, s in zip(h, gap, se)}
            if not set(HS) <= set(d):
                print("  %-10s incomplete: has h = %s" % (seed, sorted(d)))
                continue
            g20, g55, g100 = (d[k][0] for k in HS)
            sd = float(np.hypot(d[55][1], d[100][1]))
            print("  %-10s %8.3f%% %8.3f%% %8.3f%%   %+.3f pts  (%+.1f sigma)"
                  % (seed, g20, g55, g100, g100 - g55, (g100 - g55) / sd))
        print("  * the sweep point, at 10,000 runs; the rest are 1,000 runs, so SE ~3x larger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
