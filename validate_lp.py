#!/usr/bin/env python3
"""Validate the Python LP against the C++/CPLEX results."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patrol.instance import load_from_cpp_outputs
from patrol.lp import solve_relaxed, dp_value_functions
from patrol.storefile import read_value_funcs

CPP_OBJECTIVES = {0: 371.653, 1: 3036.31}   # from gen_log.txt "Solution value ="
CPP_LOWER_BOUND = 3407.97


def gradients_under_policy(inst, sol, j):
    """updateGradients_agent_randomized: E[present at i] - E[moves into i] under the LP policy."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="output.txt")
    ap.add_argument("--init-probs", default="test33/store/initProbs-scaler1-seed395.out")
    ap.add_argument("--graph", default="graph/adjacent_matrix_10.in")
    ap.add_argument("--cpp-store", default="test23/store/store_valueFuncs-seed395.out")
    args = ap.parse_args()

    inst = load_from_cpp_outputs(args.log, args.init_probs, args.graph)
    ok = True

    # 1. knowledge sets
    cpp_sets = {}
    for line in Path(args.log).read_text().splitlines():
        m = re.match(r"knowledgeSets\[(\d+)\]\[(\d+)\].*size=\d+(.*)$", line)
        if m:
            cpp_sets[(int(m.group(1)), int(m.group(2)))] = m.group(3).strip()
    bad = [(i, j) for (i, j), txt in cpp_sets.items() if inst.processes[i][j].knowledge_str().strip() != txt]
    print(f"[1] knowledge sets match C++ for {len(cpp_sets) - len(bad)}/{len(cpp_sets)} (i,j) pairs"
          + (f"   MISMATCH at {bad}" if bad else ""))
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
