#!/usr/bin/env python3
"""Figure 3: cumulative distribution of the performance deviation over randomly generated settings.

The paper (Section VI) draws 1,000 settings per case and evaluates each at h = 1, 5, 10, 20, 40 with 1,000
simulation runs per policy. A setting is one draw of the increment parameters and the agent placement, i.e. one
seed; the initial risk (Table I) and the topology are fixed per case.

Claim to check: for Case I at h = 40, MAI's deviation is under 3% in over 75% of settings.

Full scale is 1,000 settings x 3 cases = 3,000 LP solves and 30 million simulation runs, on the order of 100
hours single-threaded. Use --settings to run a smaller sample first; the CDF converges long before 1,000.

Usage:
    ~/patrol-venv/bin/python reproduce_fig3.py --case 1 --settings 50
    ~/patrol-venv/bin/python reproduce_fig3.py --case all --settings 1000 --jobs 8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patrol import generate
from patrol.lp import solve_relaxed
from patrol.sim import monte_carlo

H_VALUES = (1, 5, 10, 20, 40)

# Case -> (areas, graph, seed used for the Figure 4 instance). Figure 3 varies the seed instead.
CASES = {
    1: dict(areas=6, graph="graph/adjacent_matrix_6.in", fig4_seed=300),
    2: dict(areas=10, graph="graph/adjacent_matrix_10.in", fig4_seed=395),
    3: dict(areas=14, graph="graph/adjacent_matrix_14.in", fig4_seed=51),
}


def one_setting(args: tuple) -> list[dict]:
    """Solve the LP for one seed and evaluate both policies at every h."""
    case, seed, iters, value_function, alpha, beta = args
    cfg = CASES[case]
    inst = generate.generate_instance(alpha, beta, cfg["graph"], seed=seed, area_num=cfg["areas"])
    sol = solve_relaxed(inst)
    rows = []
    for h in H_VALUES:
        row = monte_carlo(inst, sol, h, seed=seed, iter_max=iters, value_function=value_function)
        rows.append(dict(case=case, seed=seed, h=h, lower_bound=row.lower_bound,
                         greedy=row.greedy_avg, mai=row.mai_avg,
                         greedy_dev=row.greedy_dev, mai_dev=row.mai_dev))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="2", help="1, 2, 3 or 'all'")
    ap.add_argument("--settings", type=int, default=50, help="number of random settings (paper uses 1000)")
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1000, help="simulation runs per policy per setting")
    ap.add_argument("--value-function", choices=("lp", "dp"), default="dp")
    ap.add_argument("--alpha", default="process/test2_alpha.in")
    ap.add_argument("--beta", default="process/test2_beta.in")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="benchmarks/fig3.csv")
    args = ap.parse_args()

    cases = [1, 2, 3] if args.case == "all" else [int(args.case)]
    work = [(c, s, args.iters, args.value_function, args.alpha, args.beta)
            for c in cases
            for s in range(args.first_seed, args.first_seed + args.settings)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    with out.open("w", newline="") as f:
        writer = None
        runner = ProcessPoolExecutor(args.jobs) if args.jobs > 1 else None
        results = runner.map(one_setting, work) if runner else map(one_setting, work)
        for rows in results:
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            f.flush()
            done += 1
            print(f"  {done}/{len(work)} settings   [{time.time() - t0:.0f} s]", file=sys.stderr, flush=True)
        if runner:
            runner.shutdown()

    print(f"\nwrote {out}  ({done} settings, {done * len(H_VALUES)} rows, {time.time() - t0:.0f} s)")
    summarise(out, cases)
    return 0


def summarise(path: Path, cases: list[int]) -> None:
    """The paper's claim: Case I at h=40 is under 3% in over 75% of settings."""
    import collections
    buckets = collections.defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            buckets[(int(r["case"]), int(r["h"]))].append((float(r["mai_dev"]), float(r["greedy_dev"])))
    print(f"\n{'case':>5} {'h':>4} {'n':>5} {'MAI median':>11} {'MAI <3%':>9} {'GRD median':>11}")
    for case in cases:
        for h in H_VALUES:
            vals = buckets.get((case, h))
            if not vals:
                continue
            mai = sorted(v[0] for v in vals)
            grd = sorted(v[1] for v in vals)
            frac = sum(1 for v in mai if v < 0.03) / len(mai)
            print(f"{case:>5} {h:>4} {len(mai):>5} {mai[len(mai)//2]*100:>10.2f}% {frac*100:>8.1f}% "
                  f"{grd[len(grd)//2]*100:>10.2f}%")


if __name__ == "__main__":
    raise SystemExit(main())
