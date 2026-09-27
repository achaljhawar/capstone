#!/usr/bin/env python3
"""Generate the instance, solve the relaxed LP and write the store file main.cpp imports.

Usage: python3 gen.py [--seed 395] [--value-function lp|dp] [--out-dir test23/store] [--policy] [-v]

By default the store file holds the LP vertex's V, as the C++/CPLEX path writes it. Its off-support entries
depend on the solver, so HiGHS and CPLEX files can give different MAI results; --value-function dp writes the
Bellman V for the optimal mu instead.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patrol.cli import ROW_BOUND_HELP, VALUE_FUNCTION_HELP, add_instance_args, instance_from_args
from patrol.generate import knowledge_set_lines
from patrol.lp import VALUE_FUNCTIONS, solve_relaxed
from patrol.storefile import write_policy, write_value_funcs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_instance_args(ap)
    ap.add_argument("--value-function", choices=VALUE_FUNCTIONS, default="lp", help=VALUE_FUNCTION_HELP)
    ap.add_argument("--no-row-lower-bound", action="store_true", help=ROW_BOUND_HELP)
    ap.add_argument("--out-dir", default="test23/store", help="main.cpp reads test23/store/store_valueFuncs-seed<seed>.out")
    ap.add_argument("--policy", action="store_true",
                    help="also write store_lpPolicy (the LP's randomized policy; a Python-only extra)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    inst = instance_from_args(args)
    if args.verbose:
        print("\n".join(knowledge_set_lines(inst)))
    print(f"instance: {inst.area_num} areas x {inst.type_num} agent types, T={inst.maxtime}, varNum={inst.var_num()}")

    t0 = time.time()
    sol = solve_relaxed(inst, verbose=args.verbose, ranged_rows=not args.no_row_lower_bound)
    for a in sol.agents:
        print(f"agent type {a.agent_type}: {a.n_rows} rows x {a.n_cols} cols, nnz={a.nnz}  ->  "
              f"objective = {a.objective:.6f}   (randomized states={a.randomized_states}, unreached={a.unreached_states})")
    print(f"lower bound = {sol.lower_bound:.6f}   [{time.time() - t0:.1f} s]")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vf = out / f"store_valueFuncs-seed{args.seed}.out"
    write_value_funcs(sol, vf, args.value_function)
    print(f"wrote {vf}  (V = {args.value_function})")
    if args.policy:
        pol = out / f"store_lpPolicy-seed{args.seed}.out"
        write_policy(sol, pol)
        print(f"wrote {pol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
