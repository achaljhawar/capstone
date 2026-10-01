# Benchmarks

Results of the Python port on the paper's synthetic instance (10 areas, 2 agent types,
10 time steps, seed 395), with a check against the original C++ at every point.

## Setup

| | |
|---|---|
| Machine | Apple M4 (10 cores), 16 GB RAM, macOS 27.0.1 |
| Python | 3.14.5, NumPy 2.4.6, SciPy 1.17.1 (HiGHS) |
| C++ | the shared `main.cpp`/`stdafx.cpp`, built by `tests/cpp_harness/build.py` with Homebrew GCC 15.2 (`-O2`) |
| Runs | 1,000 Monte-Carlo runs per policy per scaler |
| LP solution | `tests/data/store_lp-seed395.out.gz` (identical to what `gen.py` produces on this setup) |

Sections 1–3 were measured on the machine above. Sections 4–8 came later, on the paper's three cases
rather than the single seed-395 instance, and on two other machines:

| | |
|---|---|
| x86-64 | WSL Ubuntu 22.04, Python 3.10, NumPy 2.2.6, SciPy 1.15.3 |
| aarch64 | NVIDIA DGX Spark (GB10, 20× Cortex-X925, 121 GB), Ubuntu 24.04, Python 3.12, same library versions |

Section 7 is the check that this does not matter: with `--value-function dp` the two architectures
give bit-identical policy costs. Note that the GPU is unused throughout — the LP is 3% of a run's
time and the other 97% is a bit-exact reimplementation of glibc `rand()` consumed strictly in
sequence, which is not GPU-shaped work.

## 1. Results: Greedy vs MAI as the system grows

"Gap" is (average cost − lower bound) / lower bound, with lower bound 3407.97. The paper's claim is that MAI's gap
shrinks towards 0 as the scaler grows.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/gap-dark.svg">
  <img src="benchmarks/gap-light.svg" alt="Line chart of the gap to the lower bound against the scaler, 1 to 100 on a log scale. Greedy falls from 14.7% to 9.9%; MAI falls from 8.4% to 2.2%." width="760">
</picture>

### How to read it

Each line is one policy. Lower is better: 0% would mean the policy does as well as the LP lower bound, and no
policy can go below 0 on average.

The x-axis is the scaler, the number of sub-areas (and agents) each area is split into. It's on a log scale so
that 1, 2 and 5 get as much room as 50 and 100.

The shaded band around each line is the 95% confidence interval from the 1,000 runs. It narrows as the scaler
grows because each run averages over more sub-areas. Where two bands don't overlap, the difference isn't noise.

### What it shows

MAI beats Greedy at every scaler, by 6 to 8 percentage points of gap. Greedy levels off at about a 10% gap.
MAI's gap shrinks from 8.4% at scaler 1 to 2.2% at scaler 50 and stays there at scaler 100. These numbers use
the value function of one particular LP solution (HiGHS), and the entries the LP leaves arbitrary affect MAI
(see the README); section 3 shows how much MAI's results depend on the solver.

Movement adaptions (agents re-routed to keep the moves feasible) average 0.08 per run for Greedy and 0.21 for
MAI at scaler 1, and are 0 for both policies by scaler 20. Slackness (column 11) is −58.15.

### Exact numbers

Costs are the average total cost ± the 95% confidence half-width.

| scaler | Greedy | MAI | Greedy gap | MAI gap |
|---:|---:|---:|---:|---:|
| 1 | 3910.04 ± 35.41 | 3694.55 ± 29.37 | 14.73% | 8.41% |
| 2 | 3852.71 ± 25.96 | 3599.99 ± 19.70 | 13.05% | 5.63% |
| 5 | 3806.05 ± 17.52 | 3537.33 ± 12.11 | 11.68% | 3.80% |
| 10 | 3764.07 ± 13.37 | 3505.49 ± 8.60 | 10.45% | 2.86% |
| 20 | 3764.24 ± 9.90 | 3498.76 ± 6.44 | 10.45% | 2.66% |
| 50 | 3751.84 ± 6.27 | 3483.76 ± 3.86 | 10.09% | 2.22% |
| 100 | 3746.69 ± 4.28 | 3481.93 ± 2.67 | 9.94% | 2.17% |

## 2. Python matches the C++ at every point

At all 7 scalers above, the C++ program, run on the same store file, gives the same row, bit for bit, in
every column. That covers both averages, both confidence intervals,
both adaption counts, both gaps, the lower bound and the slackness.

## 3. With a real CPLEX solution

The sections above use LP solutions from HiGHS. To check the port against the authors' own setup, IBM CPLEX 22.2
(academic edition) was installed on two Linux machines and the authors' unmodified `main.cpp`/`stdafx.cpp` were
built on each with real CPLEX:

- ARM64: an NVIDIA DGX Spark, Ubuntu 24.04, GCC 13.3 (`-O2 -ffp-contract=off`), Boost 1.83;
- x86-64: a Windows laptop under WSL, Ubuntu 22.04, GCC 11.4 (`-O2`).

On each, a short driver called their `cplexEquivalentProblem` with CPLEX's default settings to solve the LP and
write the store file, and their simulator ran on it. The two machines returned different optimal solutions.
`output.txt` in this repository is the x86 run's log at scaler 1.

### The port against the authors' build

On both CPLEX solutions the Python port and the authors' build agree bit for bit on eight of the ten columns:
both averages, both adaption counts, both gaps, the lower bound and the slackness. The two confidence intervals
differ in the last bit (about 2e-16 relative) because the C++ takes the t-quantile from Boost and the port from
SciPy. On the x86 solution the port also prints every `Gradients for t=...` line and the slackness in
`output.txt` exactly. `tests/test_against_cpp.py` checks all of this for scalers 1, 2 and 5.

### CPLEX against HiGHS

Both CPLEX runs give the objective values 371.653 and 3036.31, and where the LP has a unique answer CPLEX agrees
with HiGHS: the multipliers μ match to 2e-11. The value function is another matter. On the ARM solution, 8,039 and 7,674 of the 10,200 entries per agent type differ from HiGHS, and
CPLEX leaves 23 and 3,243 entries at the ±100,000 bound where HiGHS leaves 70 and 115. On the x86 solution, 7,664
and 7,885 differ.

That shows up directly in MAI. Greedy is the same in every case, since it doesn't use the value function.

| scaler | MAI gap, CPLEX ARM | MAI gap, CPLEX x86 | MAI gap, HiGHS |
|---:|---:|---:|---:|
| 1 | 11.20% | 10.93% | 8.41% |
| 2 | 9.45% | 9.15% | 5.63% |
| 5 | 8.04% | 7.70% | 3.80% |
| 10 | 7.42% | 7.19% | 2.86% |
| 20 | 7.36% | 7.05% | 2.66% |
| 50 | 6.92% | 5.73% | 2.22% |
| 100 | 6.86% | 5.47% | 2.17% |

MAI's gap at scaler 100 ranges from 2.2% to 6.9% depending on which optimal LP solution the solver happened to
return.

### Which optimal solution comes back

The gradients and slackness depend on the exact optimal solution, and CPLEX's settings and machine change which
one it returns. On ARM, nine settings gave seven different slackness values; the x86 default run gave another:

| CPLEX 22.2 run | slackness |
|---|---:|
| x86, default (the run in `output.txt`) | 142.982 |
| ARM, default (concurrent, 20 threads; barrier wins) | 109.672 |
| ARM, 8 threads | 128.300 |
| ARM, 4 threads | 138.285 |
| ARM, 2 threads | −63.638 |
| ARM, 1 thread | −92.542 |
| ARM, dual simplex | −92.542 |
| ARM, primal simplex | 191.118 |
| ARM, barrier | 126.087 |
| ARM, barrier, 1 thread | −63.638 |

The authors' package has no run log and no LP solution file, and their `Readme.pdf` gives no reference numbers
(it only says the slackness should be zero). Their own LP solution, and with it their exact MAI numbers, can't be
recovered from what they shared.

## 4. Figure 3: the paper's 75% claim

Section VI claims that for Case I at h = 40, MAI's deviation is under 3% in **over 75%** of randomly
generated settings. Each setting is one seed — its own LP solve plus 1,000 simulation runs per policy
at each of five horizons.

50 settings (seeds 1–50), Case I, `--value-function dp`, on the DGX Spark at 16-way parallelism;
544 s wall.

| h | MAI median | MAI p75 | **MAI under 3%** | GRD median |
|---:|---:|---:|---:|---:|
| 1 | 5.11% | 6.09% | 22.0% | 7.89% |
| 5 | 1.69% | 2.38% | 88.0% | 7.40% |
| 10 | 1.04% | 1.51% | 90.0% | 7.29% |
| 20 | 0.59% | 1.08% | 90.0% | 7.12% |
| 40 | **0.46%** | 0.92% | **90.0%** | 7.25% |

**The claim holds, at 90% against the stated 75%**, and with margin — the 75th percentile of the
deviation is 0.92%, well inside 3%. The two qualitative features of the figure also reproduce: MAI's
distribution marches left and steepens as h grows (22% → 88% under 3% between h = 1 and h = 5, then
flat), while GRD sits near 7.1–7.9% at every h and converges to nothing.

Sample size was never the binding constraint here: at n = 50 the proportion is 90% ± 8.3% (95%),
clear of 75%. A 1,000-setting run at the paper's stated sample size is in progress on the same seeds.

CDF: `benchmarks/fig3_pilot.png`, drawn by `plot_fig3.py` from the CSV.

## 5. Is the lower bound a lower bound?

Every other check in this project compares the port against the C++, and the two share the LP
formulation — so an error in the relaxation itself would sit in both and show up as nothing.

`tests/check_lower_bound.py` tests it against something that is not the LP: a 3-area path graph, one
agent type, h = 1, solved **exactly** by backward induction over the full joint state space
(knowledge vector × occupancy), with the agent-movement constraint — the one the relaxation drops —
enforced rather than relaxed. With T small the reachable knowledge is small, so the recursion is
exact: no sampling, no truncation. It reads only the dynamics and never consults the LP.

| Instance | LP bound | Exact optimum | Slack | Greedy | MAI |
|---|---:|---:|---:|---:|---:|
| 1 agent, T = 4 | 589.2532 | 589.3433 | +0.02% | 599.800 | 589.428 |
| 1 agent, T = 5 | 731.1051 | 731.4471 | +0.05% | 753.128 | 732.036 |
| 2 agents, T = 4 | 547.2863 | 547.2863 | **0.00%** | 565.044 | 564.120 |

`LP bound ≤ exact optimum ≤ policy cost` holds in all three. The relaxation is very tight, and in the
two-agent case the bound lands *exactly* on the true optimum — which a mis-stated relaxation would be
unlikely to do. `sol.lower_bound` (the LP optimum) and `sol.lower_bound_cxx` (`getLowerBound`'s
p₀·V₀, the number the runs report) agree to every printed digit.

The policy gaps at h = 1 are the expected ones: both policies sit a few percent above the optimum in
the two-agent case, the far-from-asymptotic regime where MAI has no advantage yet — the same thing
§4 shows at h = 1, where only 22% of settings are under 3%.

## 6. The missing ε, and how little it moves the number

Section V attaches a small ε(i,j,s,a,t) > 0 to the cost rate for strict indexability, and calls those
values "crucial to determining the ranking of all the movements". The shared code has no ε. Note the
index set: ε depends on the **action**, and `costRateAAPair` returns `p*100` regardless of action, so
`c(s,1,t) − c(s,0,t)` is identically zero and the immediate term drops out of the MAI index entirely.

### How much of the ranking is ties

A movement's index is `vartheta[j][t][to_area][state] − mu[j][t][from_area]`, so it does not depend on
which sub-area the agent occupies. Movements sharing a (from, to, state) triple tie harmlessly — they
are interchangeable by symmetry, and ε would not separate them either. Ties between *distinct* triples
are the ones ε would resolve. `tests/measure_index_ties.py`, h = 40, 20 runs, V = dp:

| Case | distinct triples / ranking | distinct index values | rankings with a tie | triples tied | largest group |
|---|---:|---:|---:|---:|---:|
| I | 74.9 | 26.6 | 100% | 67.3 (90%) | 155 |
| II | 187.5 | 61.4 | 100% | 166.9 (89%) | 354 |
| III | 276.7 | 104.2 | 100% | 230.7 (83%) | 603 |

Those counts use exact float equality. The code's comparator is looser still — `compare_index_reverse`
calls two indices equal within `EPSILON_PRECISION` = 1e-10 and then falls back to `from_area` — and on
Case I the smallest gap between values counted as *distinct* is 8.9e-16, about 10⁵ times **inside** that
band. So the comparator treats even more as tied than the table shows. (That fallback is also not a
strict weak ordering, since near-equality is not transitive, which is why `make_heap` can return an
order that depends on insertion sequence rather than on any key.)

### Whether that changes the result

It does not follow from the counts, so it was measured. `tests/measure_tiebreak_sensitivity.py` leaves
the index values exactly as computed and permutes only the order in which movements enter the heap —
precisely what the shared code leaves undetermined — with runs paired on seeds, so any difference is
the tie-break and not Monte-Carlo noise. Case I, h = 40, 60 runs per rule, lower bound 2613.70:

| Tie-break rule | MAI cost | Deviation |
|---|---:|---:|
| heap order (as shipped) | 2616.52 | 0.11% |
| construction order reversed | 2616.79 | 0.12% |
| construction order shuffled #1 | 2616.05 | 0.09% |
| construction order shuffled #2 | 2616.23 | 0.10% |
| construction order shuffled #3 | 2616.55 | 0.11% |
| **spread** | **0.73** | **0.03 pts** |

The ranking is overwhelmingly ties, and resolving them differently moves MAI's deviation by 0.03
percentage points — against the whole percentage points that the choice of LP vertex moves it by (§3).
Movements that tie are tied because they carry equal marginal value, so exchanging them costs about
what the equality implies.

The paper also never gives ε: "appropriately attaching a sufficiently small" ε is an existence
argument, not a specification, so ε cannot be reproduced from the paper even in principle. It remains a
real departure from the published method — but it is worth 0.03 points, not percentage points, and the
missing LP solution file is where the irreducible uncertainty actually lives.

## 7. The same answer on x86-64 and aarch64

Only the LP vertex is architecture-sensitive, so `--value-function dp` — which depends on the
multipliers rather than on which optimal vertex came back — should make a run reproducible across
machines. That was an argument, so it was measured: seed 1 of Case I run on x86-64 (WSL Ubuntu 22.04,
Python 3.10) and aarch64 (DGX Spark, Ubuntu 24.04, Python 3.12), NumPy 2.2.6 / SciPy 1.15.3 pinned on
both, compared column by column over all five horizons.

| Column | Max abs difference | Max rel difference | Byte-identical |
|---|---:|---:|---|
| `greedy` | 0 | 0 | **yes** |
| `mai` | 0 | 0 | **yes** |
| `lower_bound` | 4.5e-13 | 1.7e-16 | no |
| `greedy_dev`, `mai_dev` | 1.9e-16 | 9.1e-15 | no |

Both policies' simulated costs are **bit-identical across the two architectures** at every h. The only
thing that moves is the LP objective's last bit — about one ulp — and the deviation columns inherit it.
MAI matching exactly means the value function, the index ranking and every movement decision agreed on
both machines.

Rows: `benchmarks/fig3_xcheck_local_s1.csv` (x86-64), `benchmarks/fig3_xcheck_spark_s1.csv` (aarch64).

One transfer caveat: copy result files with `scp` and check the md5, not with `ssh 'cat file' > local` —
the latter silently corrupted a row in a 250-row CSV here.

## 8. Theorem 3: is the convergence exponential?

Section V-D (p. 24) states that the deviation from an optimal solution "diminishes exponentially in the
scaling parameter h", presented as stronger than the O(√N) bound of the prior work it cites. It is a
numerical claim, and nothing in this project tested it.

What is measurable is deviation from the LP *lower bound*, not from the optimum:

```
measured(h) = (cost(h) - bound)/bound
            = (cost(h) - optimum)/bound  +  (optimum - bound)/bound
               \___ what Theorem 3 bounds ___/   \___ the relaxation's gap ___/
```

The second term is not observable at scale, so every model fitted carries a free floor `C` to absorb
it. `analyse_theorem3.py` fits an exponential and a power law, each with a floor, weighting points by
their confidence intervals (the interval shrinks steeply with h, so an unweighted fit lets the noisy
small-h points dominate) and comparing by AIC.

On the existing Figure 4 data — h ≤ 50, 1,000 runs, 11 points, V = dp:

| Case | exp. rate k | exp. R² | power exponent p | power R² | ΔAIC (exp − power) | Favours |
|---|---:|---:|---:|---:|---:|---|
| I | 0.2034 | 0.9744 | 0.6703 | 0.9976 | **+17.85** | power law |
| II | 0.2870 | 0.9871 | 0.8025 | 0.9957 | **+8.61** | power law |
| III | 0.2972 | 0.9837 | 0.8315 | 0.9933 | **+4.05** | power law |

The power law fits better in all three cases, with exponents clustering at 0.67–0.83 — the √N-ish
regime the paper says it improves on.

This is a signal, not a verdict, and three things limit it:

1. The fit is against the lower bound, so the floor absorbs an unknown and possibly h-dependent gap.
   (§5 shows the relaxation is very tight at small scale, which is why the floor is believed small —
   but that was h = 1 on three areas.) Case I's power-law floor comes out slightly **negative**
   (−0.000476), which is physically impossible for a bound gap and suggests mild over-fit.
2. The confidence intervals on this data are 10–26% of the decaying signal — too loose to call.
3. Theorem 3 is asymptotic; h ≤ 50 may be below where it bites.

A denser sweep is running to settle it: h out to 100 on 13 log-spaced points at 10,000 runs each,
which cuts the interval to roughly 6% of the signal. **The table above will be superseded by it.**

## Reproducing

```bash
# LP solve
python3 gen.py

# Python simulation on the stored LP solution
python3 simulate.py 20 --store tests/data/store_lp-seed395.out.gz --no-write
```

The numbers in this file are in `benchmarks/results.tsv` (full precision). To redraw the charts
from it (needs matplotlib):

```bash
python3 benchmarks/plot.py
```

Sections 4–8 are reproduced by their own scripts, all run from `python/`:

```bash
# section 4 -- Figure 3, Case I: the 75% claim (50 settings ~ 9 min on 16 cores)
python3 reproduce_fig3.py --case 1 --settings 50 --jobs 16 --out benchmarks/fig3_pilot_case1.csv
python3 plot_fig3.py --csv benchmarks/fig3_pilot_case1.csv --out benchmarks/fig3_pilot.png

# section 5 -- the lower bound against an exact joint DP (~1 min each; exits non-zero on failure)
python3 tests/check_lower_bound.py
python3 tests/check_lower_bound.py --horizon 5
python3 tests/check_lower_bound.py --agents 2

# section 6 -- how much of the ranking is ties, and whether the tie-break moves the number
python3 tests/measure_index_ties.py --case all
python3 tests/measure_tiebreak_sensitivity.py --case 1 --scaler 40 --runs 60

# section 8 -- the convergence-rate fits, over any directory of case<N>_sweep.out rows
python3 analyse_theorem3.py --dir benchmarks/theorem3 --plot benchmarks/theorem3.png
```

`check_lower_bound.py` is a pass/fail test. The other three are measurements: they print tables and
do not assert.

For section 7, run one setting on each machine with the library versions pinned and diff the rows:

```bash
python3 reproduce_fig3.py --case 1 --settings 1 --first-seed 1 --out benchmarks/fig3_xcheck.csv
```

The ARM CPLEX solution is in `tests/data/store_cplex22-seed395.out.gz`, and the authors' build's rows for it are
in `benchmarks/cplex_results.tsv` and `tests/data/cpp_reference_cplex.tsv`. The x86 solution is in
`tests/data/store_cplex22-x86-seed395.out.gz`, with its rows in `tests/data/cpp_reference_cplex_x86.tsv`. Both
row files cover every scaler from 1 to 100.

For the C++ side, build it with `tests/cpp_harness/build.py` (see the README), copy one of the
`tests/data/store_*.out.gz` files, uncompressed, to `<build dir>/test23/store/store_valueFuncs-seed395.out`, and
run `./edit <scaler>` from the build directory. The result row is appended to
`test33/long-term-performance-cost.out` with 17 significant digits.
