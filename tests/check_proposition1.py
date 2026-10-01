#!/usr/bin/env python3
"""Proposition 1: at fixed multipliers, the Lagrangian decomposes over areas.

Why this script exists, and why the obvious version of it was worthless.  DECISIONS.md 9.4 once
recorded that Proposition 1 "is implicitly exercised every time the per-type LPs' objectives sum to
the reported lower bound".  It is not: `RelaxedSolution.lower_bound` (`patrol/lp.py:207`) is *defined*
as `sum(a.objective for a in self.agents)`, and a sum cannot disagree with its own definition.  The
replacement first proposed -- solve the per-(i, j) subproblems and compare their total against the
per-type LP objective -- was also wrong, as two independent reviews pointed out: the per-type LP
couples areas through mu (`mu_idx(t, nb[a])` in `build_and_solve`), so areas do not decompose at the
LP level at all.  The decomposition holds for the inner minimisation at **fixed mu**, which is the
quantifier the proposition actually carries.

So the check is: fix mu, compute the Lagrangian minimum **jointly** over the product state space of
all areas -- enumerating joint actions and propagating product transitions, assuming no factorisation
whatsoever -- and compare it against the sum of the per-area recursions that `dp_with_actions`
produces.  The identity under test is

    J(t, (s_1, ..., s_N))  ==  sum_i V_i(t, s_i)        for every t and every product state

and it is tested at several arbitrary mu, including mu that no LP would return, because "arbitrary
mu" is the content of the claim.  A joint value lower than the sum would mean the areas can be served
better together than apart, i.e. the relaxation's objective does not decompose and every per-area
number in this project is measuring the wrong thing.

What this does NOT establish, stated because the sibling script `check_lower_bound.py` overclaimed the
same thing: the joint DP reads `AreaProcess.next_states` and `ap.cost` from `patrol/model.py`, which
is also where `dp_with_actions` reads them.  This is independent of the *recursion*, not of the
*model*.  An error in the dynamics or the cost rate sits in both and cancels.

Usage:
    ~/patrol-venv/bin/python tests/check_proposition1.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patrol.instance import Instance
from patrol.lp import dp_with_actions
from patrol.model import AreaProcess

# Two areas by default: the product state space is the cost here, and two areas already couple
# through mu in both directions, which is what the decomposition has to survive.
ADJACENCY_2 = np.array([[0, 1],
                        [1, 0]], dtype=int)
ADJACENCY_3 = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=int)
# (alpha0, beta0, D1a, D2a, Db), asymmetric so a bug cannot hide behind symmetry
AREA_PARAMS = [(10, 40, 3, 7, 2),
               (25, 25, 4, 6, 3),
               (40, 10, 5, 8, 1)]


def build_instance(horizon: int, n_areas: int) -> Instance:
    procs = [[AreaProcess(*AREA_PARAMS[i])] for i in range(n_areas)]
    init = []
    for i in range(n_areas):
        p = np.zeros(procs[i][0].state_num)
        p[AreaProcess.flat(0, 1 if i == 0 else 0)] = 1.0
        init.append([p])
    adj = ADJACENCY_2 if n_areas == 2 else ADJACENCY_3
    return Instance(area_num=n_areas, type_num=1, adjacency=adj,
                    processes=procs, init_probs=init, maxtime=horizon)


def offsets(inst: Instance, j: int) -> tuple[list[int], list[int]]:
    """Per-area state counts and the offsets dp_with_actions packs V with."""
    sizes = [inst.processes[i][j].state_num for i in range(inst.area_num)]
    offs, acc = [], 0
    for s in sizes:
        offs.append(acc)
        acc += s
    return sizes, offs


def joint_lagrangian(inst: Instance, j: int, mu: np.ndarray) -> dict:
    """The Lagrangian minimum over the FULL product state space, assuming no decomposition.

    An area's own options are the ones the relaxed problem gives it: stay unvisited, or be visited by
    an agent arriving from a neighbour n, which credits -mu[t][n].  The mu[t][i] * indicator term is
    charged on the current state either way.  Actions are chosen jointly here -- the point is that
    nothing in the construction assumes the choice for one area is independent of another's.
    """
    T, N = inst.maxtime, inst.area_num
    procs = [inst.processes[i][j] for i in range(N)]
    sizes, _ = offsets(inst, j)
    # per-area (immediate cost, successors) for each (state, action)
    per_area: list[dict] = []
    for i in range(N):
        ap, nb = procs[i], inst.neighbourhood[i]
        tbl = {}
        for s in range(sizes[i]):
            ind = s % 2
            cost = ap.cost[s // 2]
            opts = []
            # action index len(nb) is "no move", matching dp_with_actions
            for a in range(len(nb) + 1):
                if a == len(nb):
                    imm = cost + mu[:, i] * ind
                    trs = tuple((tr.s_next, tr.prob) for tr in ap.next_states(s, 0))
                else:
                    imm = cost + mu[:, i] * ind - mu[:, nb[a]]
                    trs = tuple((tr.s_next, tr.prob) for tr in ap.next_states(s, 1))
                opts.append((imm, trs))
            tbl[s] = opts
        per_area.append(tbl)

    states = list(itertools.product(*[range(n) for n in sizes]))
    action_sets = [range(len(inst.neighbourhood[i]) + 1) for i in range(N)]
    joint_actions = list(itertools.product(*action_sets))

    J = [{} for _ in range(T)]
    for t in range(T - 1, -1, -1):
        nxt = None if t == T - 1 else J[t + 1]
        for sv in states:
            best = None
            for av in joint_actions:
                total = 0.0
                succ = [None] * N
                for i in range(N):
                    imm, trs = per_area[i][sv[i]][av[i]]
                    total += imm[t]
                    succ[i] = trs
                if nxt is not None:
                    for combo in itertools.product(*succ):
                        p = 1.0
                        nxt_state = []
                        for s_next, prob in combo:
                            p *= prob
                            nxt_state.append(s_next)
                        if p:
                            total += p * nxt[tuple(nxt_state)]
                if best is None or total < best:
                    best = total
            J[t][sv] = best
    return {"J": J, "states": states, "sizes": sizes}


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--horizon", type=int, default=3)
    # 3 areas is accepted but impractical: 102**3 product states x 36 joint actions does not
    # finish in minutes, and 2 areas already credits -mu from a neighbour in both directions,
    # which is the coupling the decomposition has to survive.
    ap_.add_argument("--areas", type=int, default=2, choices=(2, 3))
    ap_.add_argument("--tol", type=float, default=1e-9)
    args = ap_.parse_args()

    inst = build_instance(args.horizon, args.areas)
    T, N = inst.maxtime, inst.area_num
    sizes, offs = offsets(inst, 0)
    print(f"  {N} areas, T = {T}, per-area states {sizes}, "
          f"product states {int(np.prod(sizes))}")

    rng = np.random.default_rng(20261017)
    mus = {
        "all zero": np.zeros((T, N)),
        "constant 5": np.full((T, N), 5.0),
        "ramp in t": np.outer(np.arange(T, dtype=float), np.ones(N)) * 3.0,
        "asymmetric": np.array([[2.0 * i + 1.5 * k for i in range(N)] for k in range(T)]),
        "random": rng.uniform(0, 20, size=(T, N)),
        "negative": -rng.uniform(0, 10, size=(T, N)),
    }

    worst = 0.0
    ok = True
    for name, mu in mus.items():
        V, _ = dp_with_actions(inst, 0, mu)
        res = joint_lagrangian(inst, 0, mu)
        diff = 0.0
        for t in range(T):
            for sv in res["states"]:
                summed = sum(V[t][offs[i] + sv[i]] for i in range(N))
                diff = max(diff, abs(res["J"][t][sv] - summed))
        worst = max(worst, diff)
        good = diff <= args.tol
        ok &= good
        # the value at t = 0 under the instance's own initial distribution, for scale
        v0 = sum(float(inst.init_probs[i][0] @ V[0][offs[i]:offs[i] + sizes[i]]) for i in range(N))
        print(f"  mu = {name:<12} joint == sum of per-area: {'PASS' if good else 'FAIL'}"
              f"   max |diff| {diff:.3e}   (value at t=0 {v0:.6f})")

    print(f"\n  worst discrepancy over {len(mus)} multiplier settings, "
          f"all t and all product states: {worst:.3e}")
    if not ok:
        print("  A FAIL means the Lagrangian does not decompose over areas at fixed mu, so the")
        print("  per-area recursion every result here relies on is solving a different problem.")
        return 1
    print("  Proposition 1's decomposition holds at fixed mu, to machine precision, including at")
    print("  multipliers no LP would return. Independent of the recursion, not of the model:")
    print("  both sides read next_states and cost from patrol/model.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
