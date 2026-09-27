#!/usr/bin/env python3
"""Solve the relaxed LP (or load a store file), then Monte-Carlo the Greedy and MAI policies, as main.cpp does.

Usage: python3 simulate.py <scaler> [<scaler> ...] [--iters 1000] [--value-function lp|dp]
                           [--store test23/store/store_valueFuncs-seed395.out] [--trajectory test33/store]
                           [--out test33/long-term-performance-cost.py.out]

Each row is the one main.cpp appends to long-term-performance-cost.out, including the slackness column.
With --store, V and mu come from that file exactly as main.cpp imports them (so a CPLEX-written file reproduces
the C++ numbers), and --value-function lp means "use the file's V".
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patrol.cli import ROW_BOUND_HELP, VALUE_FUNCTION_HELP, add_instance_args, instance_from_args
from patrol.gradients import slackness
from patrol.lp import VALUE_FUNCTIONS, RelaxedSolution, solve_relaxed
from patrol.sim import ITER_MAX, monte_carlo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scalers", type=int, nargs="+")
    add_instance_args(ap)
    ap.add_argument("--iters", type=int, default=ITER_MAX)
    ap.add_argument("--value-function", choices=VALUE_FUNCTIONS, default="lp", help=VALUE_FUNCTION_HELP)
    ap.add_argument("--no-row-lower-bound", action="store_true", help=ROW_BOUND_HELP)
    ap.add_argument("--store", help="use this store_valueFuncs file instead of solving the LP")
    ap.add_argument("--trajectory", metavar="DIR",
                    help="write the per-step trajectory files (simulation()'s outFlag output) into DIR")
    ap.add_argument("--out", default="test33/long-term-performance-cost.py.out")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    inst = instance_from_args(args)
    t0 = time.time()
    if args.store:
        sol = RelaxedSolution.from_store(inst, args.store)
        source = args.store
    else:
        sol = solve_relaxed(inst, ranged_rows=not args.no_row_lower_bound)
        source = "LP"
    lb = sol.lower_bound_cxx(args.value_function)
    slack = slackness(sol)
    print(f"lower bound = {lb:.6f}, slackness = {slack:.6g}   [{source}, V = {args.value_function}, "
          f"{time.time() - t0:.1f} s]", file=sys.stderr)

    for scaler in args.scalers:
        t0 = time.time()
        row = monte_carlo(inst, sol, scaler, seed=args.seed, iter_max=args.iters, progress=True,
                          value_function=args.value_function, slackness=slack, trajectory_dir=args.trajectory)
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
