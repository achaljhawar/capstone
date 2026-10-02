# Verification

There are two kinds of checks here. The first compares the Python with the C++ and establishes that the
reimplementation is faithful. The second tests the model and the relaxation against something other than the
C++, because a mistake shared by both programs would pass every comparison between them.

## Agreement with the C++

```bash
python3 tests/test_against_cpp.py      # or: python3 -m pytest tests/test_against_cpp.py
```

| Test | What it compares |
|---|---|
| `test_glibc_rand_matches_glibc` | `GlibcRand` against reference values from glibc for ten seeds, including 2³¹ − 1 to 2³² − 1 and −1, and positions 1,000 and 100,001 of one stream |
| `test_heap_sort_is_a_valid_sort` | The libstdc++ heap reimplementation sorts correctly under a valid comparator |
| `test_generated_instance_matches_cpp_log` | Every knowledge set, transition increment and initial placement against the C++ run's `output.txt` and `initProbs` file, as text |
| `test_lp_objective_and_multipliers` | HiGHS's optimum against CPLEX's printed objectives (to their printed precision) and μ against the stored solution (to 1e-8) |
| `test_store_file_is_checked_against_its_instance` | A solution file's sidecar matches its instance, loading it into a seed-396 instance fails, and a truncated file is refused |
| `test_slackness_matches_cpp` | The slackness diagnostic, exactly |
| `test_simulation_matches_cpp` | All ten numeric columns at scalers 1, 2 and 5 on the HiGHS solution |
| `test_simulation_matches_authors_build_on_cplex_solution` | The same, against the authors' build with real CPLEX on ARM64 |
| `test_simulation_matches_authors_build_on_x86_cplex_solution` | The same, against the authors' build with real CPLEX on x86-64 |
| `test_gradients_match_cpp_log` | Every `Gradients for t=...` line and the slackness line of `output.txt`, as text |
| `test_trajectory_files_match_cpp` | The four trajectory files, byte for byte, for two runs at scaler 2 |

In the simulation tests, eight columns must be equal as doubles. The two confidence half-widths may differ by at
most 1e-15 relative, because the C++ takes its t-quantile from Boost and this code from SciPy; the measured
difference is about 5e-16.

[benchmarks.md](../benchmarks.md) §2 extends the HiGHS comparison to scalers 10, 20, 50 and 100, and
`tests/data/cpp_reference_cplex*.tsv` holds the authors' build's rows at all seven scalers on both CPLEX
solutions.

### Reference data

| File | Produced by |
|---|---|
| `tests/data/store_lp-seed395.out.gz` | `gen.py` (HiGHS), seed 395 |
| `tests/data/cpp_reference.tsv`, `cpp_trajectory.txt` | The shared C++, built by `tests/cpp_harness/`, run on `store_lp-seed395.out.gz` |
| `tests/data/store_cplex22-seed395.out.gz`, `cpp_reference_cplex.tsv` | The authors' unmodified source built with CPLEX 22.2 on Linux ARM64; LP solved with default settings |
| `tests/data/store_cplex22-x86-seed395.out.gz`, `cpp_reference_cplex_x86.tsv` | The same on Linux x86-64 |
| `output.txt`, `test33/store/initProbs-scaler1-seed395.out` | The x86-64 CPLEX run's log at scaler 1 |

All of these were produced for this project. None of them came from the authors, whose package contains no
run log, LP solution or result table. The CPLEX version used here (22.2) is also not the one the paper reports
(22.1.1).

### Regenerating the references

`tests/cpp_harness/make_reference.py` builds the shared C++ without CPLEX or Boost and reruns it on the stored
solution:

```bash
CXX=g++-15 CC=gcc-15 python3 tests/cpp_harness/make_reference.py "path/to/code for sharing"
```

It needs GNU g++, GMP and MPFR. The changes the harness makes to the sources are listed in
[reimplementation.md](reimplementation.md#what-the-c-build-harness-changes). To run the C++ by hand, copy one of
the `tests/data/store_*.out.gz` files, uncompressed, to `<build dir>/test23/store/store_valueFuncs-seed395.out`
and run `./edit <scaler>` from the build directory.

### Scope

The comparison covers one instance (Case II, seed 395), three LP solutions and the scalers listed above. Equality
on those is strong evidence that the code paths are the same, but it does not exercise every graph, seed or
state. The paper's other two cases have not been compared against the C++.

## Checks of the model and the relaxation

These do not consult the C++. Each reads the transition and cost model from `patrol/model.py`, so an error in that
model would cancel on both sides of the comparison. They test the relaxation, the recursion and the
decomposition on top of that model.

### The lower bound is a lower bound

```bash
python3 tests/check_lower_bound.py
python3 tests/check_lower_bound.py --horizon 5
python3 tests/check_lower_bound.py --agents 2
```

On a three-area path graph with one agent type and h = 1, the script solves the original problem exactly by
backward induction over the joint state space, with the movement constraint enforced. It then checks
LP bound ≤ exact optimum ≤ simulated policy cost:

| Instance | LP bound | Exact optimum |
|---|---:|---:|
| 1 agent, T = 4 | 589.2532 | 589.3433 |
| 1 agent, T = 5 | 731.1051 | 731.4471 |
| 2 agents, T = 4 | 547.2863 | 547.2863 |

The script exits non-zero if the ordering fails. The policy costs are Monte Carlo means, so the second inequality
holds in expectation, not necessarily for every sample.

### Proposition 1: decomposition at fixed multipliers

```bash
python3 tests/check_proposition1.py
```

For two areas and T = 3, the script computes the Lagrangian minimum jointly over all 10,404 product states and
compares it with the sum of the per-area recursions from `dp_with_actions`, at six multiplier settings including
ones no LP would return. The largest difference is 2.3e-13. This tests the decomposition and the recursion for
the instances run; it is not a proof of the proposition.

### Proposition 2: flow residuals under the LP's policy

```bash
python3 tests/check_proposition2.py --case all
```

The script recomputes the gradients (expected agent presence minus expected agents moving in) under the
randomized policy given by the LP's row duals, and finds every one at about 1e-15 on all three cases. That is
what LP duality guarantees for any dual-feasible optimal pair, so the check confirms the solver's output and the
forward propagation. It does not test the threshold structure that Proposition 2 asserts, and a feasible but
suboptimal policy can also have zero residuals. The script's closing message says this;
[benchmarks.md](../benchmarks.md) §9 has the full discussion.

### Measurements

`tests/measure_index_ties.py` counts how much of the MAI ranking is ties, and
`tests/measure_tiebreak_sensitivity.py` reruns MAI with the tied movements entered in different orders. They print
tables and do not assert. Results and their uncertainty are in [benchmarks.md](../benchmarks.md) §6.

## Slackness

Column 11 of the result row is the C++'s "total Lagrange cost". The authors' readme says it should be zero.
`main.cpp` computes it by rebuilding one deterministic policy from the argmin of the Bellman recursion at the LP's
μ, propagating the initial distribution under it, and summing gradient × μ over every (t, i, j), where the
gradient is E[agent present] − E[agents moving in] (`patrol.gradients.slackness`).

At an LP vertex, several actions are usually tied for the minimum, and the relaxed optimum randomises over them.
A deterministic tie-broken policy does not satisfy complementary slackness, so the value is not zero in general,
and it changes with the tie-break: a change in the 12th decimal of μ can move it from about −58 to +146. The same
quantity under the LP's randomized policy (`patrol.gradients.gradients_randomized`) is zero to about 1e-13.

The reimplementation reports the C++'s value, because that is what the C++ prints. Treat it as a diagnostic of
the tie-break and not as a test of the LP solution.
