#!/usr/bin/env python3
"""Validate the Python LP against the C++/CPLEX results.

Usage: python3 validate_lp.py [--cpp-store test23/store_cplex/store_valueFuncs-seed395.out] [instance options]

--cpp-store is a store file written by the C++/CPLEX run (keep it apart from test23/store, which gen.py
overwrites); check [3] is skipped without it. Check [4] is a Python-only sanity check of the LP's randomized
policy, not main.cpp's gradients (those are in patrol.gradients and tests/test_against_cpp.py).

The reference objectives in check [2], and the default --cpp-log and --cpp-store, belong to the default
instance (seed 395, 10 areas). For any other instance check [2] is skipped, and checks [1] and [3] run only
when --cpp-log / --cpp-store are given for that instance's own C++ run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patrol.cli import add_instance_args, instance_from_args
from patrol.generate import generate_instance, knowledge_set_lines
from patrol.lp import solve_relaxed
from patrol.storefile import read_value_funcs

CPP_OBJECTIVES = {0: 371.653, 1: 3036.31}   # "Solution value =" from the C++ CPLEX runs
CPP_LOWER_BOUND = 3407.97                    # "lower_bound =" in output.txt
DEFAULT_CPP_LOG = "output.txt"
DEFAULT_CPP_STORE = "test23/store_cplex/store_valueFuncs-seed395.out"


def gradients_under_policy(inst, sol, j):
    """E[present at i] - E[moves into i] under the LP's randomized policy (zero at an LP optimum)."""
    T, N = inst.maxtime, inst.area_num
    dist = [inst.init_probs[i][j].copy() for i in range(N)]
    g = np.zeros((T, N))
    for t in range(T):
        pol = sol.agents[j].policy[t]
        for i in range(N):
            nb = inst.neighbourhood[i]
            present = sum(dist[i][s] for s in range(inst.state_num(i, j)) if s % 2 == 1)
            moves_in = 0.0
            for i2 in nb:
                nb2 = inst.neighbourhood[i2]
                a_to_i = [a for a, n in enumerate(nb2) if n == i]
                for s2 in range(inst.state_num(i2, j)):
                    moves_in += dist[i2][s2] * sum(pol[i2][s2, a] for a in a_to_i)
            g[t, i] = present - moves_in
        if t < T - 1:
            new = []
            for i in range(N):
                ap, nb = inst.processes[i][j], inst.neighbourhood[i]
                nxt = np.zeros(ap.state_num)
                for s in range(ap.state_num):
                    if dist[i][s] == 0:
                        continue
                    p_move = pol[i][s, :len(nb)].sum()
                    p_stay = pol[i][s, len(nb)]
                    for move_in, w in ((1, p_move), (0, p_stay)):
                        if w > 0:
                            for tr in ap.next_states(s, move_in):
                                nxt[tr.s_next] += w * dist[i][s] * tr.prob
                new.append(nxt)
            dist = new
    return g


def check_log(cpp_log, inst) -> bool:
    cpp_lines = [line for line in Path(cpp_log).read_text().splitlines() if line.startswith("knowledgeSets")]
    ours = knowledge_set_lines(inst)
    bad = [k for k, (a, b) in enumerate(zip(ours, cpp_lines)) if a != b] + ([-1] if len(ours) != len(cpp_lines) else [])
    print(f"[1] knowledge sets / C parameters match the C++ log for {len(ours) - len(bad)}/{len(cpp_lines)} (i,j) pairs"
          + (f"   MISMATCH at lines {bad}" if bad else ""))
    return not bad


def check_objectives(sol) -> bool:
    ok = True
    for a in sol.agents:
        ref = CPP_OBJECTIVES[a.agent_type]
        diff = abs(a.objective - ref)
        print(f"[2] agent type {a.agent_type}: objective {a.objective:.6f}  vs C++ {ref}  (|diff| = {diff:.2e})"
              + ("" if diff < 5e-3 else "   MISMATCH"))
        ok &= diff < 5e-3
    diff = abs(sol.lower_bound - CPP_LOWER_BOUND)
    print(f"[2] lower bound {sol.lower_bound:.6f}  vs C++ {CPP_LOWER_BOUND}  (|diff| = {diff:.2e})"
          + ("" if diff < 5e-3 else "   MISMATCH"))
    return ok and diff < 5e-3


def check_store(cpp_store, inst, sol) -> bool:
    ok = True
    Vc, muc = read_value_funcs(cpp_store, inst)
    dmu = max(abs(sol.multiplier(t, i, j) - muc[t, i, j])
              for t in range(inst.maxtime) for i in range(inst.area_num) for j in range(inst.type_num))
    print(f"[3] max|mu_py - mu_cpp| = {dmu:.3e}" + ("" if dmu < 1e-6 else "   MISMATCH"))
    ok &= dmu < 1e-6
    for j in range(inst.type_num):
        n_bound = sum(int((np.abs(Vc[t][i][j]) > 9e4).sum()) for t in range(inst.maxtime) for i in range(inst.area_num))
        n_all = inst.maxtime * sum(inst.state_num(i, j) for i in range(inst.area_num))
        print(f"[3] agent type {j}: CPLEX file has {n_bound}/{n_all} V entries at the +-1e5 bound "
              "(off the LP's support, so solver-specific)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_instance_args(ap)
    ap.add_argument("--cpp-log", default=DEFAULT_CPP_LOG, help="the C++ run's stdout, for check [1]")
    ap.add_argument("--cpp-store", default=DEFAULT_CPP_STORE)
    args = ap.parse_args()

    inst = instance_from_args(args)
    is_default = inst.fingerprint() == generate_instance().fingerprint()
    if not is_default:
        print("note: not the default seed-395 instance, so the hard-coded C++/CPLEX references do not apply")
    ok = True

    # 1. knowledge sets and C parameters
    if not is_default and args.cpp_log == DEFAULT_CPP_LOG:
        print(f"[1] skipped: {DEFAULT_CPP_LOG} is the default instance's log; pass --cpp-log for this instance")
    else:
        ok &= check_log(args.cpp_log, inst)

    # 2. objective
    sol = solve_relaxed(inst)
    if is_default:
        ok &= check_objectives(sol)
    else:
        print("[2] skipped: the reference objectives are the default instance's")

    # 3. mu vs the CPLEX file
    if not is_default and args.cpp_store == DEFAULT_CPP_STORE:
        print(f"[3] skipped: {DEFAULT_CPP_STORE} is the default instance's; pass --cpp-store for this instance")
    elif Path(args.cpp_store).exists():
        ok &= check_store(args.cpp_store, inst, sol)
    else:
        print(f"[3] skipped: {args.cpp_store} not found")

    # 4. gradients under the LP policy
    for j in range(inst.type_num):
        g = gradients_under_policy(inst, sol, j)
        slack = float((g * sol.agents[j].mu).sum())
        print(f"[4] agent type {j}: max|gradient| = {np.abs(g).max():.2e},  slackness contribution = {slack:.2e}"
              + ("" if np.abs(g).max() < 1e-6 else "   NOT ZERO"))
        ok &= np.abs(g).max() < 1e-6

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
