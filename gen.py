#!/usr/bin/env python3
"""Solve the relaxed LP and write the store files.

Usage: python3 python/gen.py [--log output.txt] [--seed 395] [--out-dir test23/store]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patrol.instance import load_from_cpp_outputs
from patrol.lp import solve_relaxed
from patrol.storefile import write_policy, write_value_funcs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="output.txt", help="C++ run log containing the knowledgeSets lines")
    ap.add_argument("--init-probs", default="test33/store/initProbs-scaler1-seed395.out")
    ap.add_argument("--graph", default="graph/adjacent_matrix_10.in")
    ap.add_argument("--seed", type=int, default=395)
    ap.add_argument("--out-dir", default="test23/store")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    inst = load_from_cpp_outputs(args.log, args.init_probs, args.graph)
    print(f"instance: {inst.area_num} areas x {inst.type_num} agent types, T={inst.maxtime}, varNum={inst.var_num()}")

    t0 = time.time()
    sol = solve_relaxed(inst, verbose=args.verbose)
    for a in sol.agents:
        print(f"agent type {a.agent_type}: {a.n_rows} rows x {a.n_cols} cols, nnz={a.nnz}  ->  "
              f"objective = {a.objective:.6f}   (randomized states={a.randomized_states}, unreached={a.unreached_states})")
    print(f"lower bound = {sol.lower_bound:.6f}   [{time.time() - t0:.1f} s]")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vf, pol = out / f"store_valueFuncs-seed{args.seed}.out", out / f"store_lpPolicy-seed{args.seed}.out"
    write_value_funcs(sol, vf)
    write_policy(sol, pol)
    print(f"wrote {vf}\nwrote {pol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
