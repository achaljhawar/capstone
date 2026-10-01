# The value function

MAI ranks movements with ϑ, which is built from the value function V at the next time step
([model.md](model.md#the-two-policies)). There are two ways to get V, and on this problem they give MAI results
that differ by several percentage points. This page explains why and when to use each.

## Where V comes from in the C++

The C++ solves the LP of eqs. (17)–(19) and writes the V block of the solution to its store file. The simulator
reads that V back and computes ϑ from it. This code does the same with `--value-function lp`, the default.

## Why the LP leaves most of V undetermined

The LP maximises Σ p₀ · V[0], where p₀ is the initial distribution. Its constraints say V[t][s] is at most the
cost of each action plus the expected next value. At the optimum, a constraint has to be tight only along states
the optimal policy actually reaches from p₀ with positive probability. At an unreached state, V can take any
value that keeps every constraint satisfied, and the objective does not change. The solver returns whichever
vertex its algorithm lands on, which is often the −1e5 or +1e5 variable bound.

Most of the state space is unreached: 51 knowledge levels × 2 indicators × 10 areas × 10 time steps, and from a
single initial knowledge state only a few knowledge levels are reachable in the early steps. On the seed-395
instance, the HiGHS and CPLEX solutions have the same optimum and multipliers that agree to 2e-11, but differ in
7,664 to 8,039 of the 10,200 V entries per agent type. CPLEX on ARM64 leaves 23 and 3,243 entries at the bound
where HiGHS leaves 70 and 115.

## Why that moves MAI

ϑ[t][i][s] compares E[V[t+1]] after moving in with E[V[t+1]] after not moving. When a sub-area is in a state
the LP never expected, the next states can be unreached ones as well, and ϑ then compares two arbitrary numbers.
The simulated system can reach such states because it keeps the movement constraints that the relaxation
drops. Wherever it does, the solver's arbitrary values decide MAI's ranking.

The effect on seed 395, with 1,000 runs per point:

| h | MAI gap, HiGHS | MAI gap, CPLEX x86-64 | MAI gap, CPLEX ARM64 |
|---:|---:|---:|---:|
| 1 | 8.41% | 10.93% | 11.20% |
| 10 | 2.86% | 7.19% | 7.42% |
| 100 | 2.17% | 5.47% | 6.86% |

Greedy does not read V and is identical in all three. Even the two CPLEX runs, same version and default
settings on two architectures, returned different vertices. On one machine, nine combinations of CPLEX thread
count and algorithm gave seven different values of the slackness diagnostic, so at least seven different
solutions ([benchmarks.md](../benchmarks.md) §3).

The authors' package contains no LP solution, so the V behind the paper's figures cannot be recovered.

## The alternative: V from the Bellman recursion

`--value-function dp` keeps the LP's multipliers μ and recomputes V by the backward recursion of eq. (15) at
those multipliers (`patrol.lp.dp_with_actions`). This is how the paper defines V in Section III-A. The
recursion:

- is defined at every state, reached or not;
- equals the LP's V at every state where the LP fixes it, so the lower bound is unchanged;
- is itself an optimal solution of the same LP;
- depends only on μ, and μ is what the solvers agree on, so the result does not depend on the solver or the
  machine.

The last point was measured: seed 1 of Case I run on x86-64 and on ARM64 gave bit-identical Greedy and MAI costs
at every h ([benchmarks.md](../benchmarks.md) §7). The recursion removes the dependence on an arbitrary vertex
for fixed μ. It does not prove that every solver returns the same μ on every instance; that held on the instances
checked here.

## Which to use

| Goal | Option |
|---|---|
| Reproduce a particular C++ run, or compare against the C++ | `lp` (default), with `--store` pointing at that run's solution file |
| Numbers that another person can reproduce with a different solver or machine | `dp` |
| The paper's definition of the index | `dp` |

Whichever you use, say so when you report MAI results, and for `lp` say which solver produced the solution and
whether the row lower bound was imposed. On Figure 4's grid, `dp` gave a lower MAI gap than `lp` at every
h ≥ 30, by 0.4 to 1.3 points.

`gen.py --value-function dp` writes the recursion's V into the store file, so the C++ can also be run with it.
