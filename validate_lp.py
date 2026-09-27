#!/usr/bin/env python3
"""Validate the Python LP against the C++/CPLEX results.

Usage: python3 validate_lp.py [--cpp-store test23/store_cplex/store_valueFuncs-seed395.out] [instance options]

--cpp-store is a store file written by the C++/CPLEX run (keep it apart from test23/store, which gen.py
overwrites); check [3] is skipped without it. Check [4] is a Python-only sanity check of the LP's randomized
policy, not main.cpp's gradients (those are in patrol.gradients and tests/test_against_cpp.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patrol.cli import add_instance_args, instance_from_args
from patrol.generate import knowledge_set_lines
from patrol.lp import solve_relaxed, dp_value_functions
from patrol.storefile import read_value_funcs

CPP_OBJECTIVES = {0: 371.653, 1: 3036.31}   # "Solution value =" from the original CPLEX run
CPP_LOWER_BOUND = 3407.97                    # "lower_bound =" in output.txt


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_instance_args(ap)
    ap.add_argument("--cpp-log", default="output.txt", help="the C++ run's stdout, for check [1]")
    ap.add_argument("--cpp-store", default="test23/store_cplex/store_valueFuncs-seed395.out")
    args = ap.parse_args()

    inst = instance_from_args(args)
    ok = True

    # 1. knowledge sets and C parameters
    cpp_lines = [line for line in Path(args.cpp_log).read_text().splitlines() if line.startswith("knowledgeSets")]
    ours = knowledge_set_lines(inst)
    bad = [k for k, (a, b) in enumerate(zip(ours, cpp_lines)) if a != b] + ([-1] if len(ours) != len(cpp_lines) else [])
    print(f"[1] knowledge sets / C parameters match the C++ log for {len(ours) - len(bad)}/{len(cpp_lines)} (i,j) pairs"
          + (f"   MISMATCH at lines {bad}" if bad else ""))
    ok &= not bad

    # 2. objective
    sol = solve_relaxed(inst)
    for a in sol.agents:
        ref = CPP_OBJECTIVES[a.agent_type]
        diff = abs(a.objective - ref)
        print(f"[2] agent type {a.agent_type}: objective {a.objective:.6f}  vs C++ {ref}  (|diff| = {diff:.2e})"
              + ("" if diff < 5e-3 else "   MISMATCH"))
        ok &= diff < 5e-3
    diff = abs(sol.lower_bound - CPP_LOWER_BOUND)
    print(f"[2] lower bound {sol.lower_bound:.6f}  vs C++ {CPP_LOWER_BOUND}  (|diff| = {diff:.2e})"
          + ("" if diff < 5e-3 else "   MISMATCH"))
    ok &= diff < 5e-3

    # 3. mu and V vs the CPLEX file
    if Path(args.cpp_store).exists():
        Vc, muc = read_value_funcs(args.cpp_store, inst)
        dmu = max(abs(sol.multiplier(t, i, j) - muc[t, i, j])
                  for t in range(inst.maxtime) for i in range(inst.area_num) for j in range(inst.type_num))
        print(f"[3] max|mu_py - mu_cpp| = {dmu:.3e}" + ("" if dmu < 1e-6 else "   MISMATCH"))
        ok &= dmu < 1e-6
        for j in range(inst.type_num):
            a = sol.agents[j]
            Vdp_cpp = dp_value_functions(inst, j, muc[:, :, j])
            dV = max(np.abs(a.V[t] - Vdp_cpp[t]).max() for t in range(inst.maxtime))
            n_bound = sum(int((np.abs(Vc[t][i][j]) > 9e4).sum()) for t in range(inst.maxtime) for i in range(inst.area_num))
            n_all = inst.maxtime * sum(inst.state_num(i, j) for i in range(inst.area_num))
            print(f"[3] agent type {j}: max|V_py - DP(mu_cpp)| = {dV:.3e}" + ("" if dV < 1e-6 else "   MISMATCH")
                  + f"   (CPLEX file: {n_bound}/{n_all} raw V entries at the +-1e5 bound -- undefined off-support)")
            ok &= dV < 1e-6
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
