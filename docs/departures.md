# Departures from the paper, and known limitations

A reimplementation can follow the code or the paper. Where the two disagree, this one follows the code, so that
its output can be checked against the C++ number for number. The first part of this page lists the places where
the shared C++ and the paper's text disagree. In each of them the Python does what the C++ does. These are not
translation errors, and the results in this repository describe the shared code's model.

The second part lists limitations of this repository itself.

C++ line numbers refer to the authors' `code for sharing` package. Paper page and equation numbers refer to
arXiv:2309.13388v3.

## Where the shared code differs from the paper

### Transition direction

The paper (Section VI, p. 25) says that when a patrolled sub-area reports no crime, α becomes N(α + Δ¹α, β), with
Δ¹α a positive increment. The C++ uses N(max(0, α − C_a1), β) (`stdafx.cpp:337` and `:484`), and so does
`patrol/model.py`. For α = 10, β = 40 and an increment of 3, the paper gives a next α of 12 and the code gives 7.

In the code, a patrol that finds nothing lowers the estimated risk; in the paper's formula it raises it. Which
one the authors intended needs their confirmation.

### Zero risk

The paper's cost is 100 α / (α + β), which is 0 at α = 0. The C++ floors the probability at 0.01 when α = 0
(`MIN_CRIME_RATE`, `stdafx.h:67` and `:215`), so the cost there is 1 and a crime can still be reported. This
affects both the transitions and the minimum cost.

### Normalisation

The paper defines N(α, β) = ⌊50α / (α + β)⌋. The C++ computes `int(alpha * 1.0 / (alpha + beta) * 50)`
(`stdafx.h:220`), dividing first in floating point. The rounding can land just below an integer: `normalize(29, 21)`
returns (28, 22), where the exact floor is 29.

### No ε perturbation

Section V attaches a small ε(i, j, s, a, t) > 0 to each cost rate to make the process strictly indexable, and calls
these values "crucial to determining the ranking of all the movements". Strict indexability is a premise of
Theorem 3.

The C++'s `costRateAAPair` (`stdafx.cpp:202`) takes the action and the time as arguments and uses neither:

```cpp
double AreaProcess::costRateAAPair(AreaState currentState, int moveIn, int time)
{
    Knowledge knowledge_current = knowledgeSet.at(currentState.indexOfKnowledge);
    double p = knowledge_current.probability();

    return p*100;	//normalize to avoid precision issue
}
```

So c(s, 1, t) − c(s, 0, t) is zero, the immediate-cost term drops out of the MAI index, and many distinct movements
get the same index. At h = 40 with `dp` V, 83% to 90% of the distinct (source, target, state) triples in a ranking
are tied with another triple (`tests/measure_index_ties.py`). Resolving those ties in other orders moved Case I's
MAI gap by no more than about ±0.05 points at 95% confidence (`tests/measure_tiebreak_sensitivity.py`), but that
experiment changes only the tie order. A real ε would also shift index values. The paper gives no value for ε, so
it cannot be reproduced. See [benchmarks.md](../benchmarks.md) §6 and §11.

### Movement comparator

The ranking comparator (`compare_index_reverse`, `stdafx.h:445`) calls two indices equal when they are within 1e-10
and then orders by source area. Near-equality is not transitive, and the comparator returns true when an element
is compared with itself, so it is not a strict weak ordering and passing it to `std::make_heap` is undefined
behaviour. In practice libstdc++ returns an order that depends on the insertion sequence. The Python reproduces
libstdc++'s heap algorithm step by step to get the same order
([reimplementation.md](reimplementation.md#movement-ranking)). Agreement therefore depends on the heap
implementation; a different standard library could order the C++'s own ties differently.

### Confidence-interval divisor

`main.cpp:608` divides the variance by n and takes the Student t-quantile at n − 1 degrees of freedom:

```cpp
boost::math::students_t dist(ITER_MAX-1);
double T=quantile(complement(dist,0.025));
double cv1,cv2;
cv1 = sqrt(cv_average1/ITER_MAX) * T;
```

The usual estimator divides by n − 1. At n = 1,000 the half-width comes out about 0.05% smaller. `patrol/sim.py`
does the same as the C++. With very few runs the difference matters, and with one run the interval is undefined.

### Table I, Case III

For area 13, type 2, Table I (p. 41) prints (α⁰, β⁰) = (22, 46), which sums to 68 although every other entry sums
to 50. `process/test2_alpha.in` and `test2_beta.in` give (4, 46), and that is what the code uses. The printed 22 is
probably a typo, but the intended value needs the authors' confirmation. Every other Table I entry matches.

### What this means for the results

The bit-exact agreement shows that the shared package's outputs came from code with these properties. It does not
show that the paper's figures came from the same build, because the figure data is not in the package. The
results in this repository should be described as results for the shared code, with the differences above stated.

## Known limitations of this repository

### Large seeds

`GlibcRand.srand` masks the seed to 32 bits and then runs glibc's seeding recurrence on it as a non-negative Python
integer. glibc treats the initial word as a signed 32-bit integer and uses C's truncating division. Seeds below 2³¹
agree; seeds at or above 2³¹, or negative seeds, do not:

| Seed | glibc first output | `GlibcRand` first output |
|---:|---:|---:|
| 395 | 1171090355 | 1171090355 |
| 2147483648 | 1336741213 | 722473798 |
| 4294967295 | 254925627 | 1788141860 |

Every seed behind the results here is below 1,500, so no result is affected.

### Solution files are not tied to an instance

`RelaxedSolution.from_store` checks the header dimensions and the total number of values per line, but nothing
that identifies the instance. A solution file for seed 395 loads into a seed-396 instance of the same size without
error and reports a different bound. Keep solution files next to the settings that produced them.

### Figure 4 cannot be rebuilt from a clone

`reproduce_fig4.py` reads per-case rows from `../test33_case<N>/`, which are not in the repository. The rows were
written by C++ binaries built from `main_fixed.cpp`, `stdafx_fixed.cpp` and `stdafx_fixed.h`, and neither those
sources nor their build script are in the repository either. The saved `benchmarks/fig4.png` and the table in
[reproduction.md](reproduction.md#figure-4) are the only record.

### Other

- `validate_lp.py` accepts the instance options but compares against hard-coded seed-395 reference values, so it
  is meaningful only for the default instance.
- `tests/check_proposition2.py` prints that Proposition 2 holds, which is more than its residual check shows
  ([verification.md](verification.md#proposition-2-flow-residuals-under-the-lps-policy)).
- The Figure 3 CSVs do not record the value function, solver version, run count or confidence intervals; the
  scripts' defaults and [benchmarks.md](../benchmarks.md) are the record of those settings.
- `reproduce_fig3.py` calls h a "horizon" in places. The horizon is T = 10; h is the scaling parameter.
- The simulator is pure Python and single-threaded.
