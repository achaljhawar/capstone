#!/usr/bin/env python3
"""Is the LP lower bound actually a lower bound?

Everything else in this project checks the port against the C++, which shares the LP formulation
with it.  A modelling error in the relaxation itself -- a sign, a missing constraint, a mis-stated
objective -- would sit in both and show up as nothing.  The check that catches it is to solve a
problem small enough to solve *exactly*, by backward induction over the full joint state space with
the agent-movement constraint enforced, and confirm

    LP bound  <=  exact optimum  <=  cost of any policy

The middle quantity is computed here from the dynamics alone, without the LP, so it is independent
of the thing it is testing.

The instance is deliberately tiny: 3 areas, 1 agent type, T = 4, h = 1, one or two agents.  The
knowledge reachable from a fixed (alpha0, beta0) in four steps is small, so the joint space is a few
thousand states and the recursion is exact -- no sampling, no truncation.

Usage:
    ~/patrol-venv/bin/python tests/check_lower_bound.py
    ~/patrol-venv/bin/python tests/check_lower_bound.py --agents 2 --horizon 5
"""
from __future__ import annotations

import argparse
import itertools
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patrol.instance import Instance
from patrol.lp import solve_relaxed
from patrol.model import AreaProcess
from patrol.sim import DP_INDEX, GREEDY, Simulator

# three areas in a path 0 - 1 - 2, so movement is constrained and the middle area is special
ADJACENCY = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]], dtype=int)
# (alpha0, beta0) differ per area to break symmetry; increments inside the paper's ranges
# (D1a in {2..6}, D2a in {5..9}, Db in {1..5})
AREA_PARAMS = [(10, 40, 3, 7, 2),
               (25, 25, 4, 6, 3),
               (40, 10, 5, 8, 1)]


def build_instance(horizon: int, agent_areas: tuple[int, ...]) -> Instance:
    """A point-mass instance: area i starts in knowledge 0, occupied iff i is in agent_areas."""
    n = len(AREA_PARAMS)
    procs = [[AreaProcess(a0, b0, ca1, ca2, cb)] for (a0, b0, ca1, ca2, cb) in AREA_PARAMS]
    init = []
    for i in range(n):
        p = np.zeros(procs[i][0].state_num)
        p[AreaProcess.flat(0, 1 if i in agent_areas else 0)] = 1.0
        init.append([p])
    return Instance(area_num=n, type_num=1, adjacency=ADJACENCY,
                    processes=procs, init_probs=init, maxtime=horizon)


def knowledge_transitions(ap: AreaProcess) -> dict:
    """(k, move_in) -> ((k', prob), ...), read off the area process's own transition table.

    next_states carries the indicator too, but the knowledge part depends only on k and move_in.
    """
    out = {}
    for k in range(ap.K):
        for move_in in (0, 1):
            trs = ap.next_states(AreaProcess.flat(k, 0), move_in)
            out[(k, move_in)] = tuple((tr.s_next // 2, tr.prob) for tr in trs)
    return out


def successors(occupied: frozenset, neighbourhood: list[list[int]]) -> list[frozenset]:
    """Every occupancy the agents can produce in one step.

    Each agent moves to a neighbour of its area or stays; h = 1 caps one agent per area, so the
    targets of a move must be distinct.  This is exactly Constraint (4), enforced rather than relaxed.
    """
    agents = sorted(occupied)
    seen = set()
    for targets in itertools.product(*(neighbourhood[i] for i in agents)):
        if len(set(targets)) == len(targets):
            seen.add(frozenset(targets))
    return sorted(seen, key=sorted)


def exact_optimum(inst: Instance, agent_areas: tuple[int, ...]) -> float:
    """Minimum expected total cost, by backward induction over (knowledge vector, occupancy).

    Stage cost is the crime rate of the current knowledge, which the step's movement cannot change;
    the movement chosen at t determines the transition into t+1.  That is the simulator's accounting
    (decide -> charge current_cost -> transition), so the two are comparable.
    """
    n, T = inst.area_num, inst.maxtime
    procs = [inst.processes[i][0] for i in range(n)]
    trans = [knowledge_transitions(ap) for ap in procs]
    cost = [ap.cost for ap in procs]
    nb = inst.neighbourhood

    @lru_cache(maxsize=None)
    def value(t: int, ks: tuple, occupied: frozenset) -> float:
        if t == T:
            return 0.0
        here = sum(cost[i][ks[i]] for i in range(n))
        best = None
        for nxt in successors(occupied, nb):
            # knowledge evolves independently per area given who is where
            per_area = [trans[i][(ks[i], 1 if i in nxt else 0)] for i in range(n)]
            expected = 0.0
            for combo in itertools.product(*per_area):
                p = 1.0
                for _, q in combo:
                    p *= q
                if p == 0.0:
                    continue
                expected += p * value(t + 1, tuple(c[0] for c in combo), nxt)
            if best is None or expected < best:
                best = expected
        return here + (best or 0.0)

    return value(0, tuple(0 for _ in range(n)), frozenset(agent_areas))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--agents", type=int, default=1, choices=(1, 2))
    ap.add_argument("--runs", type=int, default=2000, help="simulation runs per policy")
    args = ap.parse_args()

    agent_areas = (0,) if args.agents == 1 else (0, 2)
    inst = build_instance(args.horizon, agent_areas)

    print(f"3 areas in a path, 1 agent type, T = {args.horizon}, h = 1, "
          f"{args.agents} agent(s) starting at {agent_areas}\n")

    sol = solve_relaxed(inst)
    bound = sol.lower_bound                    # the LP optimum itself
    bound_cxx = sol.lower_bound_cxx("lp")      # getLowerBound's p0 . V_0, the number the runs report

    exact = exact_optimum(inst, agent_areas)

    sim = Simulator(inst, sol, 1, value_function="dp")
    costs = {}
    for policy, name in ((GREEDY, "Greedy"), (DP_INDEX, "MAI")):
        each = []
        for it in range(args.runs):
            sim.init_agents(it + 1)
            each.append(sim.run(policy, it + 1).total_cost)
        arr = np.asarray(each, dtype=float)
        # SE of the mean; with one run there is no spread to estimate, so it is unknown.
        se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
        costs[name] = (float(arr.mean()), se)

    print(f"  LP lower bound (LP optimum)    {bound:>12.6f}")
    print(f"  LP lower bound (p0 . V_0)      {bound_cxx:>12.6f}")
    print(f"  exact optimum (joint DP)       {exact:>12.6f}")
    for name, (c, se) in costs.items():
        tail = f"+- {se:.6f} SE" if se == se else "SE unknown, one run"
        print(f"  {name + ', simulated':<30} {c:>12.6f}   ({args.runs} runs, {tail})")

    tol = 1e-9
    ok_bound = bound <= exact + tol
    print(f"\n  LP bound <= exact optimum      {'PASS' if ok_bound else 'FAIL'}"
          f"   (slack {exact - bound:+.6f}, {(exact - bound) / exact * 100:+.2f}%)")

    # `exact <= expected cost of any policy` is a statement about the EXPECTATION.  A sample mean
    # can sit below the optimum by chance -- at --runs 1 it routinely does -- so this half is a
    # comparison against sampling error, not an assertion.  Treating it as one produced a false
    # diagnosis ("the relaxation or the accounting is wrong") on a one-run invocation.
    ok_policies = True
    for name, (c, se) in costs.items():
        short = exact - c                      # positive means the sample beat the optimum
        if short <= tol:
            verdict, note = "PASS", f"gap {c - exact:+.6f}, {(c - exact) / exact * 100:+.2f}%"
        elif se != se:
            verdict, note = "n/a ", f"sample {short:.6f} below optimum; one run, no SE to judge it"
        elif short <= 3 * se:
            verdict, note = "PASS", f"sample {short:.6f} below optimum, within {short / se:.1f} SE"
        else:
            verdict, note = "FAIL", f"sample {short:.6f} below optimum, {short / se:.1f} SE -- too far"
            ok_policies = False
        print(f"  exact optimum <= {name:<13} {verdict}   ({note})")

    if not ok_bound:
        print("\n  LP bound > exact optimum is deterministic and cannot be sampling noise: the")
        print("  relaxation or the simulator's accounting is wrong.")
        return 1
    if not ok_policies:
        print("\n  A sample mean below the exact optimum by more than three standard errors is not")
        print("  luck. Suspect the cost accounting or the movement constraint, not the policy.")
        return 1
    print("\n  The ordering holds, so the LP relaxation and the simulator agree with a model solved")
    print("  exactly and independently of both. Note the second comparison is statistical: it")
    print("  checks the sample mean is not implausibly below the optimum, which needs enough runs")
    print("  to mean anything -- use --runs 1 for timing, not for verification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
