# Coordinated multi-agent patrolling in Python

A Python port of the C++ code for

> J. Fu, Z. Wang and J. Chen, "Coordinated Multiagent Patrolling With State-Dependent Cost Rates:
> Asymptotically Optimal Policies for Large-Scale Systems", *IEEE Transactions on Automatic Control*,
> vol. 70, no. 6, pp. 3800-3815, June 2025.

The original C++ (by Jing Fu, RMIT University) needs IBM CPLEX, Boost, GMP and MPFR. This port needs only
NumPy and SciPy and uses the free HiGHS solver that ships with SciPy. Given the same inputs, it produces the
same numbers as the C++, bit for bit (see [Verification](#verification)).

If you use this code, please cite the paper above.

## What it does

A city is split into areas. Each area has a crime risk that rises when no patrol agent is there and falls when
one is. There are two types of agents, and they can only move between neighbouring areas. The goal is to keep
the total cost (the sum of all areas' risk over 10 time steps) low.

The code builds the paper's synthetic test city (10 areas, 2 agent types, `patrol/generate.py`) and solves the
relaxed problem as a linear program (LP) in `patrol/lp.py`. The LP gives a lower bound, a cost no policy can beat
on average, and the values the MAI policy uses.

`patrol/sim.py` then simulates two policies 1,000 times each. Greedy sends agents to the riskiest neighbouring
areas first. MAI is the paper's index policy, built from the LP solution. The output is each policy's average
cost and its gap to the lower bound. The paper claims MAI's gap shrinks as the system is scaled up (the `scaler`
argument).

## Requirements

Python 3 with NumPy and SciPy (tested with Python 3.14, NumPy 2.4.6, SciPy 1.17.1).

## Quick start

```bash
# 1. Solve the LP and write the solution file (~90 s)
python3 gen.py
#    -> test23/store/store_valueFuncs-seed395.out

# 2. Simulate Greedy and MAI at scalers 1, 2 and 5 (1,000 runs each)
python3 simulate.py 1 2 5
#    -> appends one row per scaler to test33/long-term-performance-cost.py.out
```

`simulate.py` solves the LP itself, so step 1 is only needed if you want the solution file (for example, to feed
it to the C++ program).

## Output

Each row of `long-term-performance-cost*.out` is tab-separated, in the same format as the C++:

| # | Column |
|---|---|
| 1 | scaler |
| 2-4 | Greedy: average total cost, 95% confidence half-width, average number of adapted movements |
| 5-7 | MAI: the same three |
| 8 | Greedy's gap to the lower bound, (cost − bound) / bound |
| 9 | MAI's gap to the lower bound |
| 10 | Lower bound |
| 11 | "Slackness", a diagnostic only (see below) |

With the default settings you should get:

| scaler | Greedy | MAI | Lower bound |
|---|---|---|---|
| 1 | 3910.04 | 3694.55 | 3407.97 |
| 2 | 3852.71 | 3599.99 | 3407.97 |
| 5 | 3806.05 | 3537.33 | 3407.97 |

## Why MAI results depend on the LP solver

The MAI policy uses a value function V from the LP, exactly as the C++ does. The LP only pins V down for
situations it expects to happen. For the rest, the solver returns arbitrary numbers, often its ±100,000 variable
bound, and those entries depend on the solver. So HiGHS and CPLEX give different MAI results even though they
agree on the lower bound and the multipliers. At scaler 20, MAI's gap to the lower bound is 2.7% on a HiGHS
solution and 7.1% to 7.4% on two CPLEX 22.2 solutions (see [benchmarks.md](benchmarks.md)). Say which solver
produced the LP solution when you report results.

To reproduce the original C++/CPLEX numbers exactly, run on the solution file the C++ wrote. The one behind
`output.txt` is in `tests/data/`:

```bash
python3 simulate.py 1 2 5 --store tests/data/store_cplex22-x86-seed395.out.gz
```

## Command-line options

`gen.py` and `simulate.py` share these instance options:

| Option | Default | Meaning |
|---|---|---|
| `--seed` | 395 | Random seed for building the city and for the simulations |
| `--alpha`, `--beta` | `process/test2_alpha.in`, `process/test2_beta.in` | Initial risk data per area |
| `--graph` | `graph/adjacent_matrix_10.in` | Which areas are neighbours |
| `--areas` | 10 | Number of areas |
| `--no-row-lower-bound` | off | Skip one LP constraint the C++ includes: ~4x faster with the same lower bound, but slightly different V off the LP's support |

`simulate.py` also takes:

| Option | Meaning |
|---|---|
| `--iters N` | Runs per policy (default 1000) |
| `--store FILE` | Use an existing solution file instead of solving the LP |
| `--trajectory DIR` | Write step-by-step logs of every run (costs, agent positions, moves) |
| `--no-write` | Print rows only; don't append to the output file |

`gen.py --policy` also writes the LP's randomized policy (`store_lpPolicy-seed<seed>.out`). This is a Python-only
extra that the C++ never reads.

The simulator is pure Python: scaler 5 takes about 15 s, and large scalers take minutes.

## Verification

```bash
python3 tests/test_against_cpp.py      # or: python3 -m pytest tests/
```

The tests check that:

- the generated city matches the C++ run's own output (`output.txt`, `test33/store/initProbs-*`);
- the LP's optimum matches the CPLEX objective values;
- for scalers 1, 2 and 5, every output column matches the C++ program exactly, at full double precision;
- the step-by-step trajectory logs match the C++ program byte for byte;
- on two LP solutions written by real CPLEX 22.2 (one on ARM, one on x86-64), the authors' own Linux build and
  the port agree bit for bit, except the last bit of the two confidence intervals (the C++ takes its t-quantile
  from Boost, the port from SciPy);
- on the x86 solution, the port prints the gradient and slackness lines of `output.txt` exactly.

The C++ reference results are in `tests/data/`. To regenerate them, build the shared C++ without CPLEX or Boost
(this needs GNU g++, GMP and MPFR; on macOS, Homebrew's `g++-15`):

```bash
CXX=g++-15 CC=gcc-15 python3 tests/cpp_harness/make_reference.py "path/to/code for sharing"
```

`tests/cpp_harness/build.py` lists every change it makes to the C++ sources. None of them affect what the
simulation computes.

[benchmarks.md](benchmarks.md) has results for scalers 1 to 100 and the Python-vs-C++ check
at each point.

## Repository layout

| Path | Contents |
|---|---|
| `gen.py`, `simulate.py`, `validate_lp.py` | Command-line entry points |
| `patrol/model.py` | One area's risk process: states, costs, transitions |
| `patrol/generate.py` | Builds the test city the way `main.cpp` does |
| `patrol/instance.py` | The problem instance (can also be rebuilt from a C++ run log) |
| `patrol/lp.py` | The relaxed LP, the lower bound, and the C++'s dynamic program for given multipliers |
| `patrol/sim.py` | The Greedy and MAI simulations |
| `patrol/gradients.py` | The slackness diagnostic |
| `patrol/cxx_compat.py` | Exact copies of glibc's `rand()` and the C++ heap sort, so random draws and tie-breaks match |
| `patrol/storefile.py` | Reads and writes the C++ solution-file format |
| `process/`, `graph/` | Input data |
| `output.txt`, `test33/store/initProbs-*` | Output of a C++/CPLEX run (x86-64, CPLEX 22.2), used by the tests |
| `tests/` | Tests, C++ reference data, and the C++ build harness |

## Differences from the C++

The LP is solved with HiGHS instead of CPLEX. The lower bound and multipliers agree, but the arbitrary
off-support values differ (see above).

Slackness (column 11) is computed exactly as in the C++, but it depends on how exact ties are broken: a change in
the 12th decimal of the multipliers can swing it from about −58 to +146. Treat it as a diagnostic. The C++
documentation also describes it as a test value.

The C++ hard-codes 1,000 runs in its confidence intervals. The Python uses the actual `--iters` count.
