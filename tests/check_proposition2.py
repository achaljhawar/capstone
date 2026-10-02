#!/usr/bin/env python3
"""Proposition 2: does a mixture of threshold policies actually attain the relaxed optimum?

Proposition 2 (p. 13, equation (23)) asserts that there exist policies satisfying (21)-(22) and a
probability vector over them such that, for every (i, j, t), the expected number of agents moving
into (i, j) equals the expected number available. Equivalently: the complementary-slackness residual
`sum_{t,i,j} gradient(t,i,j) * mu(t,i,j)` is zero at the LP optimum.

This is the one claim outside Section VI that the shared package already instruments -- the last
column of `long-term-performance-cost.out` is that residual -- and the authors' `Readme.pdf` says it
should be zero. It is not: `main.cpp` reports values in the hundreds.

That is a defect in the diagnostic, not in the proposition. The LP being solved is the dual, so at a
vertex the Bellman inequalities are tight for several actions at once and the true primal optimum
randomizes over them. `main.cpp` instead rebuilds a single deterministic policy by taking the argmin
of the Bellman recursion, which breaks those ties arbitrarily. Complementary slackness holds between
the LP's own primal and dual, not between the dual and an arbitrarily tie-broken policy, so a large
residual is exactly what you would expect. (DECISIONS.md part 2 records how this was found, and
`main_fixed.cpp` routes around it on the C++ side by recovering the occupation measures from CPLEX.)

`patrol.gradients.gradients_randomized` does the same thing in the port, using the randomized policy
the LP solve already produces. This script checks both, so the contrast is visible:

  * under the deterministic argmin policy, the residual is large -- reproducing main.cpp;
  * under the LP's randomized policy, it should vanish.

The stronger check is on the individual gradients rather than their weighted sum: a sum near zero
could be cancellation between large terms, whereas every gradient at machine epsilon means each
constraint in (23) is satisfied on its own.

Scope: the randomized policy comes from the LP's row duals, and LP duality makes these flow residuals
zero for any dual-feasible optimal pair. A pass therefore confirms the solver's output and the forward
propagation, a necessary condition for Proposition 2. It does not check the threshold structure of
(21)-(22), and a feasible but suboptimal policy can also have zero residuals (benchmarks.md section 9).

Usage:
    ~/patrol-venv/bin/python tests/check_proposition2.py
    ~/patrol-venv/bin/python tests/check_proposition2.py --case all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patrol.generate import generate_instance
from patrol.gradients import (gradients, gradients_randomized, slackness,
                              slackness_randomized)
from patrol.lp import solve_relaxed

INSTANCES = {
    "seed395": dict(label="seed 395 (I=10, the package's own instance)", areas=10,
                    graph="graph/adjacent_matrix_10.in", seed=395),
    "1": dict(label="Case I (I=6)", areas=6, graph="graph/adjacent_matrix_6.in", seed=300),
    "2": dict(label="Case II (I=10)", areas=10, graph="graph/adjacent_matrix_10.in", seed=395),
    "3": dict(label="Case III (I=14)", areas=14, graph="graph/adjacent_matrix_14.in", seed=51),
}

# every gradient must be this small for (23) to hold constraint by constraint
TOL = 1e-10


def check(key: str) -> bool:
    cfg = INSTANCES[key]
    inst = generate_instance("process/test2_alpha.in", "process/test2_beta.in",
                             cfg["graph"], seed=cfg["seed"], area_num=cfg["areas"])
    sol = solve_relaxed(inst)

    gd = gradients(sol)
    gr = gradients_randomized(sol)
    sd, sr = slackness(sol, gd), slackness_randomized(sol, gr)

    print(f"\n=== {cfg['label']} ===")
    print(f"  {'policy used for the gradient':<36} {'residual':>14} {'max |gradient|':>16}")
    print(f"  {'deterministic argmin (main.cpp)':<36} {sd:>14.6g} {np.abs(gd).max():>16.4g}")
    print(f"  {'LP randomized policy (Prop. 2)':<36} {sr:>14.6g} {np.abs(gr).max():>16.4g}")

    ok = float(np.abs(gr).max()) < TOL
    print(f"  -> every gradient under the LP policy below {TOL:g}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        worst = np.unravel_index(int(np.argmax(np.abs(gr))), gr.shape)
        print(f"     worst at (t, i, j) = {worst}: {gr[worst]:.6g}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="seed395",
                    help="seed395, 1, 2, 3 or 'all'")
    args = ap.parse_args()

    keys = ["seed395", "1", "2", "3"] if args.case == "all" else [args.case]
    if any(k not in INSTANCES for k in keys):
        print(f"unknown case; pick from {list(INSTANCES)} or 'all'", file=sys.stderr)
        return 2

    results = {k: check(k) for k in keys}
    print()
    if all(results.values()):
        print("Every instance checked has zero flow residuals, constraint by constraint, under the LP's")
        print("own randomized policy: the solver returned a dual-feasible optimal pair and the forward")
        print("propagation agrees with it. LP duality guarantees this for any optimal solution, so it is a")
        print("necessary condition for Proposition 2, not a test of its threshold structure. The large")
        print("residual the shared code reports comes from its deterministic tie-break.")
        return 0
    print("At least one instance left a residual above tolerance under the LP's own policy, which")
    print("would be a real discrepancy rather than a diagnostic artefact. Investigate before trusting")
    print("the lower bound on that instance.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
