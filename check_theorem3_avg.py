#!/usr/bin/env python3
"""Does MAI's gap keep decreasing in h once the instance is averaged out?

Sections 8.3-8.6 of benchmarks.md measure one instance per case, and 8.6 shows why that cannot
answer the question: at h = 100 the gap spans 0.006%-2.641% across instances, several times any
instance's own variation over 20 <= h <= 100, so a per-instance tail is read against the wrong
yardstick.  This script reads the averaged-settings sweep instead -- reproduce_fig3.py over many
random settings at each h, which is Figure 3's design applied to the convergence question.

The statistic that matters is **paired**.  Each setting is evaluated at every h with one LP solve and
one seed, so the per-setting change from h to h' has none of the across-instance spread in it, and
its standard error is far below that of the difference of two ensemble means.  That pairing is the
whole reason 200 settings is enough.

Reported per h-pair: the mean paired change with its standard error and a paired t-test, the fraction
of settings that improved, and the ensemble local log-log slope.  A decreasing gap shows up as a
negative mean change with the fraction above a half; convergence stalling shows up as a mean change
indistinguishable from zero with the fraction at a half.

Neither outcome tests Theorem 3, which section 8.1 establishes has no finite-sample content.  What
this bounds is the local decay rate at reachable h, on an ensemble rather than one draw.

Usage:
    ~/patrol-venv/bin/python check_theorem3_avg.py --csv benchmarks/avg_sweep/case1_avg.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def load(path: Path) -> tuple[list[int], dict[int, np.ndarray], list[int]]:
    """h values, seed -> gap per h, and the seeds present at every h."""
    by_seed: dict[int, dict[int, float]] = defaultdict(dict)
    hs: set[int] = set()
    with path.open() as f:
        for r in csv.DictReader(f):
            h = int(r["h"])
            hs.add(h)
            by_seed[int(r["seed"])][h] = float(r["mai_dev"])
    order = sorted(hs)
    complete = sorted(s for s, d in by_seed.items() if len(d) == len(order))
    table = {s: np.array([by_seed[s][h] for h in order]) for s in complete}
    return order, table, complete


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="benchmarks/avg_sweep/case1_avg.csv")
    args = ap.parse_args()

    path = Path(args.csv)
    hs, table, seeds = load(path)
    n = len(seeds)
    if n < 2:
        print("need at least two complete settings")
        return 1
    G = np.vstack([table[s] for s in seeds]) * 100.0      # settings x h, per cent
    print(f"  {path.name}: {n} complete settings, h = {hs}")

    print(f"\n  {'h':>5} {'mean gap':>10} {'SE':>8} {'median':>9} {'p10':>8} {'p90':>8} {'<3%':>7}")
    for k, h in enumerate(hs):
        col = G[:, k]
        print(f"  {h:>5} {col.mean():>9.4f}% {col.std(ddof=1) / math.sqrt(n):>7.4f}% "
              f"{np.median(col):>8.4f}% {np.percentile(col, 10):>7.4f}% "
              f"{np.percentile(col, 90):>7.4f}% {(col < 3).mean() * 100:>6.1f}%")

    print(f"\n  paired changes, {n} settings each (negative = the gap fell)")
    print(f"  {'from':>5} {'to':>5} {'mean change':>12} {'SE':>8} {'t':>8} {'p':>10} "
          f"{'improved':>9} {'local slope':>12}")
    tcrit = student_t.ppf(0.975, n - 1)
    pairs = [(k, k + 1) for k in range(len(hs) - 1)]
    if len(hs) > 2:
        pairs.append((0, len(hs) - 1))
    for a, b in pairs:
        d = G[:, b] - G[:, a]
        se = d.std(ddof=1) / math.sqrt(n)
        tstat = d.mean() / se if se else float("nan")
        pval = 2 * student_t.sf(abs(tstat), n - 1)
        # per-setting local log-log slope, over settings where both gaps are positive
        ok = (G[:, a] > 0) & (G[:, b] > 0)
        slope = -np.log(G[ok, b] / G[ok, a]) / math.log(hs[b] / hs[a])
        s_se = slope.std(ddof=1) / math.sqrt(ok.sum())
        print(f"  {hs[a]:>5} {hs[b]:>5} {d.mean():>11.4f}% {se:>7.4f}% {tstat:>8.2f} "
              f"{pval:>10.2e} {(d < 0).mean() * 100:>8.1f}% "
              f"{slope.mean():>7.3f} +-{s_se:.3f}")

    a, b = 0, len(hs) - 1
    d = G[:, b] - G[:, a]
    se = d.std(ddof=1) / math.sqrt(n)
    lo, hi = d.mean() - tcrit * se, d.mean() + tcrit * se
    print(f"\n  over h = {hs[a]} -> {hs[b]}: mean change {d.mean():+.4f}% "
          f"95% CI [{lo:+.4f}%, {hi:+.4f}%], {(d < 0).mean() * 100:.1f}% of settings improved")
    tail = G[:, -1] - G[:, -2]
    tse = tail.std(ddof=1) / math.sqrt(n)
    print(f"  over the last octave {hs[-2]} -> {hs[-1]}: mean change {tail.mean():+.4f}% "
          f"95% CI [{tail.mean() - tcrit * tse:+.4f}%, {tail.mean() + tcrit * tse:+.4f}%]")
    print("\n  Reading: a mean change whose CI excludes zero is a decrease measurable on the")
    print("  ensemble; one that straddles zero means the gap has stopped falling detectably over")
    print("  that range. Neither bears on Theorem 3, which has no finite-sample content (8.1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
