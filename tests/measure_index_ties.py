#!/usr/bin/env python3
"""How much of the MAI ranking is decided by the tie-break the shared code leaves undocumented?

Section V of the paper attaches a small eps_{i,j}(s,a,t) > 0 to the cost rate c_{i,j}(s,a,t) to
obtain strict indexability, and calls those values "crucial to determining the ranking of all the
movements".  The shared code has no eps: `costRateAAPair` returns p*100 regardless of the action, so
c(s,1,t) == c(s,0,t), the immediate term of vartheta vanishes identically, and whatever ties remain
fall through to the order `std::make_heap` happens to produce.  This script measures how many ties
that rule is actually resolving.

Two kinds of tie have to be told apart:

  * A movement's MAI index is vartheta[j][t][to_area][state] - mu[j][t][from_area].  It does not
    depend on which sub-area k the agent occupies, so movements sharing a (from_area, to_area,
    state) triple always tie.  Those ties are harmless -- the movements are interchangeable by
    symmetry -- and eps would not break them either, being indexed by (i,j,s,a,t) and not by k.
  * A tie between *distinct* triples is the one that matters: there eps would impose a strict order
    that the shared code decides by heap position instead.

Only the second kind is reported as "harmful".

Usage:
    ~/patrol-venv/bin/python tests/measure_index_ties.py --case 1
    ~/patrol-venv/bin/python tests/measure_index_ties.py --case all --runs 20
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patrol import generate
from patrol.lp import solve_relaxed
from patrol.sim import BASE_INIT_AGENTS, BASE_SIMULATION_SEED, DP_INDEX, Simulator

CASES = {
    1: dict(areas=6, graph="graph/adjacent_matrix_6.in", seed=300),
    2: dict(areas=10, graph="graph/adjacent_matrix_10.in", seed=395),
    3: dict(areas=14, graph="graph/adjacent_matrix_14.in", seed=51),
}


def measure(case: int, scaler: int, runs: int, value_function: str) -> dict:
    cfg = CASES[case]
    inst = generate.generate_instance("process/test2_alpha.in", "process/test2_beta.in",
                                      cfg["graph"], seed=cfg["seed"], area_num=cfg["areas"])
    sim = Simulator(inst, solve_relaxed(inst), scaler, value_function=value_function)

    stats = collections.Counter()
    groups = collections.Counter()
    orig = sim._rank_movements

    def counting_rank(t, j, policy):
        out = orig(t, j, policy)
        if policy == DP_INDEX:
            stats["rankings"] += 1
            stats["movements"] += len(out)
            by_value = collections.defaultdict(set)
            for m in out:
                by_value[m.index].add((m.from_area, m.to_area, m.state))
            stats["distinct_values"] += len(by_value)
            stats["distinct_triples"] += len({t3 for s in by_value.values() for t3 in s})
            harmful = {v: len(s) for v, s in by_value.items() if len(s) > 1}
            stats["harmful_values"] += len(harmful)
            stats["triples_tied"] += sum(harmful.values())
            if harmful:
                stats["rankings_with_harmful_tie"] += 1
                groups[max(harmful.values())] += 1
        return out

    sim._rank_movements = counting_rank
    for it in range(runs):
        sim.init_agents(it + cfg["seed"] + BASE_INIT_AGENTS)
        sim.run(DP_INDEX, it + cfg["seed"] + BASE_SIMULATION_SEED)

    r = stats["rankings"]
    return dict(case=case, rankings=r,
                movements_per_ranking=stats["movements"] / r,
                triples_per_ranking=stats["distinct_triples"] / r,
                values_per_ranking=stats["distinct_values"] / r,
                frac_rankings_tied=stats["rankings_with_harmful_tie"] / r,
                harmful_values_per_ranking=stats["harmful_values"] / r,
                triples_tied_per_ranking=stats["triples_tied"] / r,
                largest_group=max(groups) if groups else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="1", help="1, 2, 3 or 'all'")
    ap.add_argument("--scaler", type=int, default=40, help="h")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--value-function", choices=("lp", "dp"), default="dp")
    args = ap.parse_args()

    cases = [1, 2, 3] if args.case == "all" else [int(args.case)]
    rows = [measure(c, args.scaler, args.runs, args.value_function) for c in cases]

    print(f"MAI index = vartheta[j][t][to_area][state] - mu[j][t][from_area]")
    print(f"h = {args.scaler}, {args.runs} runs per case, V = {args.value_function}\n")
    hdr = (f"{'case':>5} {'movements':>10} {'triples':>8} {'distinct':>9} {'tied':>7} "
           f"{'tie vals':>9} {'triples':>8} {'largest':>8}")
    sub = (f"{'':>5} {'/ranking':>10} {'/ranking':>8} {'values':>9} {'rankings':>7} "
           f"{'/ranking':>9} {'tied':>8} {'group':>8}")
    print(hdr); print(sub); print("-" * len(hdr))
    for d in rows:
        print(f"{d['case']:>5} {d['movements_per_ranking']:>10.1f} {d['triples_per_ranking']:>8.1f} "
              f"{d['values_per_ranking']:>9.1f} {d['frac_rankings_tied']*100:>6.1f}% "
              f"{d['harmful_values_per_ranking']:>9.2f} {d['triples_tied_per_ranking']:>8.1f} "
              f"{d['largest_group']:>8}")
    print("\n'triples tied' counts distinct (from, to, state) triples sharing an index value with")
    print("another triple -- the ranking decisions eps would make and the heap makes instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
