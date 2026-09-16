#!/usr/bin/env python3
"""Solve the relaxed LP, then Monte-Carlo the Greedy and MAI policies.

Usage: python3 python/simulate.py <scaler> [<scaler> ...] [--iters 1000] [--out test33/long-term-performance-cost.py.out]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patrol.instance import load_from_cpp_outputs
from patrol.lp import solve_relaxed
from patrol.sim import ITER_MAX, monte_carlo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scalers", type=int, nargs="+")
    ap.add_argument("--iters", type=int, default=ITER_MAX)
    ap.add_argument("--seed", type=int, default=395)
    ap.add_argument("--log", default="output.txt")
    ap.add_argument("--init-probs", default="test33/store/initProbs-scaler1-seed395.out")
    ap.add_argument("--graph", default="graph/adjacent_matrix_10.in")
    ap.add_argument("--out", default="test33/long-term-performance-cost.py.out")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    inst = load_from_cpp_outputs(args.log, args.init_probs, args.graph)
    t0 = time.time()
    sol = solve_relaxed(inst)
    print(f"lower bound = {sol.lower_bound:.6f}   [LP {time.time() - t0:.1f} s]", file=sys.stderr)

    for scaler in args.scalers:
        t0 = time.time()
        row = monte_carlo(inst, sol, scaler, seed=args.seed, iter_max=args.iters, progress=True)
        line = row.as_line()
        print(f"scaler={scaler}: greedy {row.greedy_avg:.2f} +-{row.greedy_ci:.2f}  MAI {row.mai_avg:.2f} +-{row.mai_ci:.2f}"
              f"  dev {row.greedy_dev:.4f} / {row.mai_dev:.4f}   [{time.time() - t0:.1f} s]", file=sys.stderr)
        print(line)
        if not args.no_write:
            with open(args.out, "a") as f:
                f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
