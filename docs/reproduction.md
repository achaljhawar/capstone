# Reproducing the paper's numerical results

This page covers Section VI of the paper: what it specifies, what this repository has rerun, how to rerun each
experiment, and what cannot be reproduced from the authors' package. Every result here is a result about the
shared code's model, which differs from the paper's text in the ways listed in [departures.md](departures.md).

## What Section VI specifies

- Two agent types (J = 2) and horizon T = 10.
- Three patrol regions with I = 6, 10 and 14 areas (Cases I, II and III), with topologies in Appendix IX.
- Crime risk as a Beta belief with α + β = 50, initialised from Table I, cost 100 α / (α + β).
- Increments Δ¹α ∈ {2, …, 6}, Δ²α ∈ {5, …, 9} and Δβ ∈ {1, …, 5}, drawn uniformly.
- For each type, the number of agents has mean M⁰ⱼ h with M⁰ⱼ uniform on {1, …, I/2}; Table II gives the initial
  placement used for Figure 4.
- Figure 3: for h = 1, 5, 10, 20 and 40, the distribution of the deviation (Γ^{h,ϕ} − Γ*) / Γ* over 1,000 random
  settings, each evaluated with 1,000 simulation runs. Claim: for Case I at h = 40, MAI is under 3% in more than
  75% of settings.
- Figure 4: the deviation against h for one fixed setting per case. Claim: MAI is under 3% for all h ≥ 30, and
  Greedy does not improve with h.
- All 95% confidence intervals within ±3% of the mean.
- g++ 11.4.0 and CPLEX 22.1.1.

The authors' package contains the C++ source, the graph and risk input files, a readme and a compiled binary. It
does not contain an LP solution, a run log, the random settings behind Figure 3, or the data behind either figure.

## The three cases

| Case | I | Graph | Seed | Lower bound |
|---|---:|---|---:|---:|
| I | 6 | `graph/adjacent_matrix_6.in` | 300 | 2613.70 |
| II | 10 | `graph/adjacent_matrix_10.in` | 395 | 3407.97 |
| III | 14 | `graph/adjacent_matrix_14.in` | 51 | 5333.08 |

All three cases read `process/test2_alpha.in` and `process/test2_beta.in` and use their first I entries. The
package's `test1_*.in` files look like Case I data, but they saturate the C++'s initial-risk scaling and give a
degenerate α₀ = 50, β₀ = 0, so they are not used. `adjacent_matrix_6.in` is the package's
`adjacent_matrix_test1.in` under a clearer name. The three graphs match the paper's Figure 5.

The seeds come from `tests/find_paper_seeds.py`, which replays the random stream and looks for a seed whose agent
placement matches Table II. The placement loop starts after 3·I·J draws for the increments, so its position
depends on I and each case needs its own seed. Seed 395 is the one hard-coded in `main.cpp`.

With these seeds, the generated placements match Table II for all three cases, and the generated initial risk
matches Table I in every entry but one: for Case III, area 13, type 2, the paper prints (22, 46), which does not
sum to 50, and the input file gives (4, 46). See [departures.md](departures.md#table-i-case-iii).

To run a case, pass its settings to any entry point:

```bash
python3 simulate.py 1 5 10 --areas 6  --graph graph/adjacent_matrix_6.in  --seed 300
python3 simulate.py 1 5 10 --areas 10 --graph graph/adjacent_matrix_10.in --seed 395   # the default
python3 simulate.py 1 5 10 --areas 14 --graph graph/adjacent_matrix_14.in --seed 51
```

## Figure 4

| h | Case I GRD / MAI | Case II GRD / MAI | Case III GRD / MAI |
|---:|---:|---:|---:|
| 1 | 12.05% / 7.58% | 14.73% / 8.42% | 15.54% / 7.63% |
| 10 | 10.16% / 1.97% | 10.45% / 3.05% | 12.46% / 1.90% |
| 30 | 9.76% / 1.42% | 10.19% / 1.83% | 12.24% / 0.98% |
| 50 | 9.88% / 1.28% | 10.09% / 2.29% | 12.27% / 1.15% |

The plot is `benchmarks/fig4.png`, on h = 1, 5, 10, …, 50 with 1,000 runs per point. MAI is under 3% at every
plotted h from 30 to 50 in all three cases, Greedy stays between about 10% and 12.5% from h = 10 on, and the
largest relative confidence half-width is 0.91%. That supports both of the paper's Figure 4 claims on this grid.
It does not establish the claim for every integer h ≥ 30.

How it was produced: one C++ binary per case was built from `main_fixed.cpp`/`stdafx_fixed.cpp` (modified
versions of the shared source) against CPLEX, `gen.py --policy` wrote each case's HiGHS solution, and the binaries
ran the h grid; `reproduce_fig4.py` then reads `../test33_case<N>/long-term-performance-cost.out` and draws the
plot. MAI used the LP solution's V (`lp`). The modified sources, their build script and the per-case rows are not
in this repository, so the figure cannot be rebuilt from a clone as it stands. With `dp` V, the same grid gives
lower MAI gaps at every h ≥ 30, by 0.4 to 1.3 points.

To get comparable numbers with the Python alone:

```bash
python3 simulate.py 1 5 10 15 20 25 30 35 40 45 50 --areas 6 --graph graph/adjacent_matrix_6.in --seed 300 \
    --no-write
```

These will differ somewhat from the plotted values, which came from the `*_fixed` build. For Case II at h = 10,
the plot has an MAI gap of 3.05%, and the unmodified C++ and this code give 2.86% on the same HiGHS solution
([benchmarks.md](../benchmarks.md) §1).

## Figure 3

```bash
python3 reproduce_fig3.py --case 1 --settings 1000 --jobs 8 --out benchmarks/fig3_case1_full.csv
python3 plot_fig3.py --csv benchmarks/fig3_case1_full.csv --out benchmarks/fig3.png
```

`reproduce_fig3.py` takes settings with seeds 1 to N. A seed changes the increments and the agent placement;
Table I stays fixed. Each setting gets one LP solve, reused across h, and 1,000 simulation runs per policy at each
h. It defaults to `--value-function dp` and to the LP without the C++'s row lower bound. Output is one CSV row per
(case, seed, h) with the bound, both policies' mean costs and both deviations.

Case I has been run at the paper's full 1,000 settings:

| h | MAI median | MAI 75th percentile | MAI under 3% | Greedy median |
|---:|---:|---:|---:|---:|
| 1 | 4.35% | 5.74% | 21.0% | 7.81% |
| 5 | 1.49% | 2.25% | 87.0% | 6.85% |
| 10 | 0.93% | 1.55% | 91.5% | 6.64% |
| 20 | 0.60% | 1.22% | 92.7% | 6.53% |
| 40 | 0.41% | 1.03% | 92.8% | 6.45% |

The paper's claim, more than 75% of settings under 3% at h = 40, holds with 92.8%. A 95% Wilson interval on that
proportion is about 91.0% to 94.2%. That interval covers the variation across settings only; it does not include
the Monte Carlo error in classifying each setting.

What is missing:

- Cases II and III have not been run at 1,000 settings. The closest data are the 200-setting ensembles in
  `benchmarks/avg_sweep/`, on a different h grid (10, 20, 40, 55, 100).
- `plot_fig3.py` draws Greedy at h = 40 only. The paper draws all five.
- The CSV does not store confidence intervals, so the paper's ±3% condition cannot be checked per setting from it.

## Convergence in h (Theorem 3)

Theorem 3 says that for any δ > 0 there exist H and s such that the gap is at most e^{−sh} + δ for all h > H.
Because δ is chosen first, any non-negative gap that tends to zero satisfies the inequality, so no finite set of
measurements can confirm or refute an exponential rate from it. The theorem also assumes strict indexability,
which the shared code does not implement (no ε). The experiments below are therefore measurements of how this
code's gap behaves at reachable h, not tests of the theorem.

| Experiment | Data | Command |
|---|---|---|
| One instance per case, 13 values of h up to 100, 10,000 runs | `benchmarks/theorem3_h100/` | `python3 analyse_theorem3.py --dir benchmarks/theorem3_h100 --runs 10000 --plot benchmarks/theorem3_h100.png` |
| Fitted floor with a profile interval | same | `python3 check_theorem3_floor.py --dir benchmarks/theorem3_h100 --runs 10000` |
| Six more seeds at h = 20, 55, 100, at 1,000 and 10,000 runs | `benchmarks/theorem3_seeds/`, `benchmarks/theorem3_seeds10k/` | `python3 check_theorem3_seeds.py --dir benchmarks/theorem3_seeds10k --runs 10000` |
| 200 random settings per case, paired changes in h | `benchmarks/avg_sweep/` | `python3 check_theorem3_avg.py --csv benchmarks/avg_sweep/case1_avg.csv` |

The sweep files are rows written by `simulate.py` with `--value-function dp`; the ensembles come from
`reproduce_fig3.py --settings 200 --h 10,20,40,55,100`.

Every instance and both ensembles show the large fall from h = 1 to h ≈ 20 that Figure 4 shows. Past
that, behaviour depends on the instance. On the 200-setting ensembles, Case I's mean gap stops falling measurably
after h ≈ 40 (mean paired change from h = 55 to 100 of −0.013 points, 95% interval −0.042 to +0.017), while
Case II pauses between 40 and 55 and then falls again (−0.116 points from 55 to 100). Single-instance fits, such
as Case I's power law h^−0.70, describe that one seed and do not generalise. [benchmarks.md](../benchmarks.md) §8
has the full analysis, including the statistical caveats (correlated errors across h, power, and fits within a
chosen model family) and the corrections made in review.

## What cannot be reproduced from the package

- The paper's exact curves. There is no figure data to compare against, and no LP solution, so the V the
  authors' MAI used is unknown. Under `lp` V, MAI's results depend on the solver's vertex by several points
  ([value-function.md](value-function.md)).
- The 1,000 random settings behind Figure 3. Seeds 1 to 1,000 are this repository's choice.
- The ε perturbation of Section V, which the paper specifies only as "sufficiently small".
- The paper's CPLEX 22.1.1 environment. The CPLEX runs here used 22.2.
