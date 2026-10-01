# How the C++ was reimplemented

The target was agreement with the shared C++ at full double precision, not approximate agreement. A Monte Carlo
simulation that draws one different random number, breaks one tie differently or adds floating-point terms in a
different order drifts away from the original within a few steps, and then the only possible comparison is
statistical. Exact agreement makes every disagreement a bug with a location.

This page lists what that required, then every place the Python deliberately differs from the C++.

## What exact agreement required

### The random-number stream

The C++ calls `srand(seed)` and `rand()` from glibc. `patrol.cxx_compat.GlibcRand` reimplements glibc's TYPE_3
additive feedback generator (degree 31, separation 3): the same seeding recurrence, the same 310 discarded
outputs, and `rand()` returning the sum shifted right by one. `uniform()` computes `rand() * 1.0 / RAND_MAX` in the
same double arithmetic as the C++.

The stream is consumed in exactly the C++'s order. Instance generation draws the transition increments area by
area and type by type, then the agent counts and placements. Every simulation run reseeds twice, once for the
initial occupancy (`run + seed + 100`) and once for the transitions (`run + seed + 200`).

macOS's libc `rand()` is a different generator, so the C++ build harness force-includes a glibc-compatible
`rand()` (`tests/cpp_harness/glibc_rand.c`) to make the C++ draw the same numbers on any platform.

The reimplementation does not handle seeds of 2³¹ or more correctly; see
[departures.md](departures.md#large-seeds).

### Movement ranking

The C++ ranks movements by building a `std::make_heap` and popping it with `std::pop_heap`. Its comparator,
`compare_index_reverse`, treats indices within 1e-10 as equal and then compares source areas. That relation is
not a strict weak ordering, so the order the heap produces depends on the algorithm's internal steps and on the
order the movements were inserted, not on any sort key. A call to `sorted()` gives a different order.

`patrol.cxx_compat` therefore reimplements libstdc++'s `__adjust_heap`, `__push_heap`, `make_heap` and `pop_heap`
step by step, and `heap_sort_cxx` runs the C++'s idiom (make the heap, then pop until empty). Movements are
generated in the C++'s loop order: target area, then source neighbour, then target sub-area.

### Floating-point order

Sums are accumulated in the C++'s order everywhere a result is compared:

- `dp_with_actions` adds the cost, the multiplier term and the expected next value in the same sequence as
  `dynamicProgrammingWithGivenMultipliers_agent`, and changes action only on a strictly smaller value;
- `lower_bound_cxx` sums p₀ · V₀ over areas, then types, then states, as `getLowerBound` does;
- `monte_carlo` computes the mean and the variance with the C++'s running updates
  (`m += (x − m) / n`), not with NumPy reductions.

The C++ harness builds with `-ffp-contract=off` so that an ARM64 compiler does not fuse multiply-adds that an
x86-64 build would leave separate.

### Input parsing

The C++ reads its input files with `getline` and `strtok`. Two quirks of that loop are reproduced: a last line
without a trailing newline is dropped, and missing adjacency entries count as 0. Initial (α, β) values go through
the C++'s scaling and capping before normalisation (`patrol.generate.initial_alpha`).

### The state layout

Knowledge index 0 is the initial (α₀, β₀) and the other 50 values follow in increasing α; flat states are
`2k + indicator`. The LP's columns, its rows (in (t, i, s, a) order) and the solution file all use this layout, so
a solution file written by either program can be read by the other.

### The LP

`build_and_solve` builds the same constraint matrix as `cplexEquivalentProblem_agent`, with the same ±1e5 bounds
on every variable. The C++ also bounds every row below by −1e5. That bound does not change the optimum or μ, but
it does change which optimal vertex the solver returns and therefore V off the LP's support. `gen.py` and
`simulate.py` impose it by default, as the C++ does; `--no-row-lower-bound` drops it. The library function
`solve_relaxed` defaults to no row bound, and `reproduce_fig3.py` uses that default.

The LP itself is solved by HiGHS instead of CPLEX. Where the LP has a unique answer the two agree. Where it does
not, which is most of V, they differ; [value-function.md](value-function.md) covers the consequences.

### The solution file

`patrol/storefile.py` reads and writes the C++'s `store_valueFuncs-seed<seed>.out` layout: a header
`N J T varNum`, one line of V for every (t, i, j), then one line of μ per t. Values are written with `repr`, so a
round trip is exact. Files ending in `.gz` are decompressed on read.

### Output

Result rows use the C++'s tab-separated layout and its default stream precision (`%g`, six significant digits).
With `--trajectory DIR`, the simulator writes the four per-step files the C++ writes when its `outFlag` is on
(`-cost`, `-graph`, `-agents` and `-actions`), in the same format.

## Where the Python deliberately differs

| Difference | Why | Effect on results |
|---|---|---|
| HiGHS instead of CPLEX | No commercial dependency | Same optimum and μ; different V off the support, and so different MAI results under `lp` V |
| t-quantile from SciPy instead of Boost | No Boost dependency | The two confidence half-widths differ in the last bit (about 2e-16 relative) |
| Confidence interval uses the actual run count | The C++ hard-codes 1,000 runs in its quantile and divisor | None at the default `--iters 1000` |
| `--value-function dp` | Gives MAI numbers that do not depend on the solver | Optional; the default `lp` behaves as the C++ |
| Randomized-policy diagnostics (`AgentLP.policy`, `gradients_randomized`, `gen.py --policy`) | The C++'s slackness diagnostic uses a tie-broken deterministic policy and cannot be zero in general | Additions only; the C++'s own slackness is still computed and reported |
| `--from-cpp-log` | Rebuild the exact instance a C++ run used from its stdout | Used by the tests |

Everything the C++ computes is still computed the C++'s way. The additions sit beside the faithful code paths
and do not replace them, which is what keeps the bit-exact tests meaningful.

## What the C++ build harness changes

`tests/cpp_harness/build.py` compiles the authors' unmodified `main.cpp` and `stdafx.cpp` with these changes:

- the CPLEX model-building functions are removed (`main.cpp` never calls them; it imports the LP solution from
  its store file);
- the single Boost t-quantile is computed once with SciPy and inlined;
- glibc's `rand()` is force-included, as above;
- the result row is written with 17 significant digits instead of 6, so a comparison is at full precision;
- `ITER_MAX` and the trajectory flag can be overridden.

None of these touch what the simulation computes. The harness needs GNU g++ (for libstdc++'s heap algorithms),
GMP and MPFR. The comparisons against CPLEX solutions in `tests/data/` used the authors' own unmodified build with
real CPLEX 22.2 on Linux.
