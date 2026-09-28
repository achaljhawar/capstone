# Benchmarks

Results and running times of the Python port on the paper's synthetic instance (10 areas, 2 agent types,
10 time steps, seed 395), with a check against the original C++ at every point.

## Setup

| | |
|---|---|
| Machine | Apple M4 (10 cores), 16 GB RAM, macOS 27.0.1 |
| Python | 3.14.5, NumPy 2.4.6, SciPy 1.17.1 (HiGHS) |
| C++ | the shared `main.cpp`/`stdafx.cpp`, built by `tests/cpp_harness/build.py` with Homebrew GCC 15.2 (`-O2`) |
| Runs | 1,000 Monte-Carlo runs per policy per scaler, one job at a time |
| LP solution | the store files in `tests/data/` (identical to what `gen.py` produces on this setup) |

## 1. Results: Greedy vs MAI as the system grows

"Gap" is (average cost − lower bound) / lower bound, with lower bound 3407.97. The paper's claim is that MAI's gap
shrinks towards 0 as the scaler grows.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/gap-dark.svg">
  <img src="benchmarks/gap-light.svg" alt="Line chart of the gap to the lower bound against the scaler, 1 to 100 on a log scale. Greedy falls from 14.7% to 9.9%; MAI with --value-function lp falls from 8.4% to 2.2%; MAI with --value-function dp falls from 7.4% to 1.0%." width="760">
</picture>

### How to read it

Each line is one policy. Lower is better: 0% would mean the policy does as well as the LP lower bound, and no
policy can go below 0 on average.

The x-axis is the scaler, the number of sub-areas (and agents) each area is split into. It's on a log scale so
that 1, 2 and 5 get as much room as 50 and 100.

The shaded band around each line is the 95% confidence interval from the 1,000 runs. It narrows as the scaler
grows because each run averages over more sub-areas. Where two bands don't overlap, the difference isn't noise.

The two MAI lines are the same policy fed two different value functions (see the README). Greedy doesn't use
the value function, so it has one line.

### What it shows

MAI beats Greedy at every scaler, by 6 to 9 percentage points of gap. Greedy levels off at about a 10% gap,
and it's the same in both tables because it doesn't use the value function.

With `dp`, MAI's gap keeps shrinking, from 7.4% at scaler 1 to 1.0% at scaler 100, in line with the paper's
claim. With `lp` it shrinks more slowly and stays around 2.2% between scaler 50 and 100. The difference comes
from the value-function entries the LP leaves arbitrary (see the README), so results should say which option
was used. Section 4 shows how much those entries depend on the solver.

Movement adaptions (agents re-routed to keep the moves feasible) average 0.08 per run for Greedy and 0.16 to
0.21 for MAI at scaler 1, and are 0 for both policies by scaler 20. Slackness (column 11) is −58.15 for both
store files.

### Exact numbers

Costs are the average total cost ± the 95% confidence half-width.

#### `--value-function lp` (default; what the C++ does)

| scaler | Greedy | MAI | Greedy gap | MAI gap |
|---:|---:|---:|---:|---:|
| 1 | 3910.04 ± 35.41 | 3694.55 ± 29.37 | 14.73% | 8.41% |
| 2 | 3852.71 ± 25.96 | 3599.99 ± 19.70 | 13.05% | 5.63% |
| 5 | 3806.05 ± 17.52 | 3537.33 ± 12.11 | 11.68% | 3.80% |
| 10 | 3764.07 ± 13.37 | 3505.49 ± 8.60 | 10.45% | 2.86% |
| 20 | 3764.24 ± 9.90 | 3498.76 ± 6.44 | 10.45% | 2.66% |
| 50 | 3751.84 ± 6.27 | 3483.76 ± 3.86 | 10.09% | 2.22% |
| 100 | 3746.69 ± 4.28 | 3481.93 ± 2.67 | 9.94% | 2.17% |

#### `--value-function dp`

| scaler | Greedy | MAI | Greedy gap | MAI gap |
|---:|---:|---:|---:|---:|
| 1 | 3910.04 ± 35.41 | 3658.71 ± 30.02 | 14.73% | 7.36% |
| 2 | 3852.71 ± 25.96 | 3569.18 ± 20.00 | 13.05% | 4.73% |
| 5 | 3806.05 ± 17.52 | 3501.08 ± 12.32 | 11.68% | 2.73% |
| 10 | 3764.07 ± 13.37 | 3461.67 ± 8.61 | 10.45% | 1.58% |
| 20 | 3764.24 ± 9.90 | 3455.61 ± 6.22 | 10.45% | 1.40% |
| 50 | 3751.84 ± 6.27 | 3444.06 ± 3.69 | 10.09% | 1.06% |
| 100 | 3746.69 ± 4.28 | 3440.92 ± 2.43 | 9.94% | 0.97% |

## 2. Python matches the C++ at every point

For all 14 configurations above (2 value functions × 7 scalers), the C++ program, run on the same store file,
gives the same row, bit for bit, in every column. That covers both averages, both confidence intervals,
both adaption counts, both gaps, the lower bound and the slackness.

## 3. Running time

### Simulation (1,000 runs of each policy)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/runtime-dark.svg">
  <img src="benchmarks/runtime-light.svg" alt="Line chart of simulation time against the scaler, both on log scales. Python rises from 4.8 s at scaler 1 to 392 s at scaler 100; the C++ rises from 0.23 s to 40 s." width="760">
</picture>

Both axes are log scales, so a straight line means time grows in proportion to the scaler, and the vertical
distance between the two lines is the speed ratio. The lines run parallel from scaler 5 on, where Python is
about 10× slower.

Python's line is flatter at the left. Each run has some per-step bookkeeping in Python that doesn't grow with the
scaler (roughly half of the 4.8 s at scaler 1), so the ratio is worst for small systems, 21× at scaler 1, and
settles once the work that grows with the scaler dominates.

The chart shows the `lp` runs. The `dp` runs take almost the same time (table below).

| scaler | Python `lp` | Python `dp` | C++ `lp` | C++ `dp` | Python / C++ |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.8 s | 4.8 s | 0.23 s | 0.22 s | 21× |
| 2 | 7.0 s | 6.9 s | 0.49 s | 0.49 s | 14× |
| 5 | 14.2 s | 14.0 s | 1.46 s | 1.44 s | 10× |
| 10 | 28.4 s | 28.1 s | 3.21 s | 3.19 s | 9× |
| 20 | 61.6 s | 61.0 s | 7.02 s | 6.94 s | 9× |
| 50 | 179.8 s | 177.9 s | 18.9 s | 18.7 s | 9× |
| 100 | 391.6 s | 386.7 s | 39.7 s | 39.0 s | 10× |

Python times cover only the simulation loop (`monte_carlo`). C++ times are for the whole program, which also
loads the store file and computes the gradients (about 0.1 s). Both grow roughly linearly with the scaler, and
the value-function option doesn't change the speed.

### LP solve (`gen.py`, both agent types)

| Variant | Time | Lower bound |
|---|---:|---:|
| Default: with the C++'s row lower bound | 85.7 s | 3407.967568 |
| `--no-row-lower-bound` | 22.1 s | 3407.967568 |

The C++ solves this LP with CPLEX, which isn't available here, so there's no C++ time to compare.

## 4. With a real CPLEX solution

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
with HiGHS: the multipliers μ match to 2e-11 and the `dp` value function to 4e-11. The raw LP value function is
another matter. On the ARM solution, 8,039 and 7,674 of the 10,200 entries per agent type differ from HiGHS, and
CPLEX leaves 23 and 3,243 entries at the ±100,000 bound where HiGHS leaves 70 and 115. On the x86 solution, 7,664
and 7,885 differ.

That shows up directly in MAI. Greedy is the same in every case, since it doesn't use the value function.

| scaler | MAI gap, CPLEX ARM raw V | MAI gap, CPLEX x86 raw V | MAI gap, HiGHS raw V | MAI gap, `dp` (any solver) |
|---:|---:|---:|---:|---:|
| 1 | 11.20% | 10.93% | 8.41% | 7.36% |
| 2 | 9.45% | 9.15% | 5.63% | 4.73% |
| 5 | 8.04% | 7.70% | 3.80% | 2.73% |
| 10 | 7.42% | 7.19% | 2.86% | 1.58% |
| 20 | 7.36% | 7.05% | 2.66% | 1.40% |
| 50 | 6.92% | 5.73% | 2.22% | 1.06% |
| 100 | 6.86% | 5.47% | 2.17% | 0.97% |

With the value function the C++ uses, MAI's gap at scaler 100 ranges from 2.2% to 6.9% depending on which
optimal LP solution the solver happened to return. With `dp` it is 1.0% for any of them.

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

## Reproducing

```bash
# LP solve
time python3 gen.py
time python3 gen.py --no-row-lower-bound

# Python simulation on the stored LP solution (one scaler at a time for clean timings)
time python3 simulate.py 20 --store tests/data/store_lp-seed395.out.gz --value-function lp --no-write
time python3 simulate.py 20 --store tests/data/store_dp-seed395.out.gz --value-function lp --no-write
```

The second simulation reads the `dp` store file with `--value-function lp`. That means "use the V in the file",
which gives the same results as the `dp` table.

The numbers in this file are in `benchmarks/results.tsv` (full precision, with timings). To redraw the charts
from it (needs matplotlib):

```bash
python3 benchmarks/plot.py
```

The ARM CPLEX solution is in `tests/data/store_cplex22-seed395.out.gz`, and the authors' build's rows for it are
in `benchmarks/cplex_results.tsv` and `tests/data/cpp_reference_cplex.tsv`. The x86 solution is in
`tests/data/store_cplex22-x86-seed395.out.gz`, with its rows in `tests/data/cpp_reference_cplex_x86.tsv`. Both
row files cover every scaler from 1 to 100.

For the C++ side, build it with `tests/cpp_harness/build.py` (see the README), copy one of the
`tests/data/store_*.out.gz` files, uncompressed, to `<build dir>/test23/store/store_valueFuncs-seed395.out`, and
run `./edit <scaler>` from the build directory. The result row is appended to
`test33/long-term-performance-cost.out` with 17 significant digits.
