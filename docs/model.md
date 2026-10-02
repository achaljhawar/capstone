# The model and the method

This page describes what the code computes. Where the shared C++ and the paper disagree, it describes the C++,
because that is what this reimplementation follows; the disagreements are listed in
[departures.md](departures.md). Equation numbers refer to the arXiv v3 preprint (2309.13388v3).

## Map from the paper to the code

| Paper | C++ (`code for sharing`) | Python |
|---|---|---|
| Section II and Section VI: the per-sub-area process | `AreaProcess`, `Knowledge` in `stdafx.h`/`stdafx.cpp` | `patrol/model.py` |
| Section VI and Appendix X: the test instance | `main.cpp` lines 20 to 187, `constructKnowledgeSets` | `patrol/generate.py`, `patrol/instance.py` |
| Eqs. (17) to (19): the relaxed problem as an LP | `cplexEquivalentProblem_agent` (`stdafx.cpp:2191`) | `patrol.lp.build_and_solve` |
| Eq. (15): Bellman recursion at fixed multipliers | `dynamicProgrammingWithGivenMultipliers_agent` (`stdafx.cpp:596`) | `patrol.lp.dp_with_actions` |
| Eq. (22): ϑ, the value of moving an agent in | `getVartheta` (`stdafx.cpp:3707`) | `patrol.sim.Simulator._vartheta_table` |
| Eq. (24) and Algorithm 1: the index ranking | `updateMovementRankingLagrangianIndex`, `decisionMakingLagrangianIndex` | `Simulator._rank_movements`, `Simulator._decide` |
| Algorithm 2 and eq. (26): movement adaption (MAI) | `movementAdaption`, `updateDistances` | `Simulator._movement_adaption`, `Simulator._update_distances` |
| Section VI: the greedy baseline | `updateMovementRankingGreedy`, `decisionMakingGreedy` | same functions, `policy=GREEDY` |
| Section VI: deviation (Γ^{h,ϕ} − Γ*) / Γ* | `main.cpp` lines 533 to 640 | `patrol.sim.monte_carlo` |

## The patrol region

There are I areas. `graph/adjacent_matrix_<I>.in` gives the adjacency matrix, and an area counts as its own
neighbour. There are J = 2 agent types, one per crime type, and the horizon is T = 10 steps. The scaling
parameter h splits every area into h identical sub-areas, and the number of agents grows in proportion to h.
Areas are indexed from 0 in the code and from 1 in the paper.

The paper's three cases are I = 6, 10 and 14 (Cases I, II and III). The default instance is Case II with seed 395,
the seed hard-coded in `main.cpp`.

## One sub-area's process

Each (sub-area, agent type) pair is a small Markov chain. Its state has two parts:

- the knowledge (α, β), a Beta-distribution belief about the crime rate with α + β = 50 always, so there are 51
  knowledge states;
- an indicator of whether a type-j agent is in the sub-area.

That gives 102 states, stored flat as `s = 2k + indicator`, where k indexes the knowledge set. Knowledge index 0
is always the initial (α₀, β₀); the other 50 follow in increasing α. This ordering comes from
`constructKnowledgeSets` and fixes the layout of every array and file.

The crime probability is p = α / (α + β), except that p = 0.01 when α = 0. The cost per step is 100p
(`costRateAAPair`). It depends only on the knowledge; the action and the time do not enter it.

### Transitions

Write N(α, β) for the projection back onto α + β = 50, computed as `int(α / (α + β) * 50)` in double precision.
Each (area, type) pair has three integer increments drawn at instance generation: C_a1 ∈ {2, …, 6},
C_a2 ∈ {5, …, 9} and C_b ∈ {1, …, 5}.

| Agent moves in? | Crime reported (probability p) | No crime reported (probability 1 − p) |
|---|---|---|
| yes | (α, β + C_b), normalised | (max(0, α − C_a1), β), normalised |
| no | (α + C_a2, β), normalised | unchanged |

When both outcomes lead to the same knowledge state, the code stores a single transition with probability 1. The
indicator of the next state equals "agent moves in". The paper writes the patrolled, no-crime case as α + Δ¹α; see
[departures.md](departures.md#transition-direction).

## The instance

`patrol/generate.py` builds the instance with the same calls to glibc `rand()`, in the same order, as `main.cpp`:

1. For every (area, type), in area-major order, draw C_a1 = `rand() % 5 + 2`, C_a2 = `rand() % 5 + 5` and
   C_b = `rand() % 5 + 1`.
2. Read the initial risk from `process/test2_alpha.in` and `process/test2_beta.in` (paper Table I; every case uses
   the first I entries) and normalise it to α₀ + β₀ = 50.
3. For each type, draw the number of agents as `(rand() % max(1, I/2) + 1) × 20` and place them one at a time
   on random areas, in units of 1/20 of an area. This gives each area's initial share of occupied sub-areas
   (paper Table II).

The initial distribution of each (area, type) puts that share on state 1 (knowledge 0, agent present) and the rest
on state 0. In the simulation, each run samples the occupancy of every sub-area from this distribution.

Because the placement loop's position in the random stream depends on I, one seed cannot reproduce Table II for
all three cases. `tests/find_paper_seeds.py` searches for seeds that do: 300 for Case I, 395 for Case II and 51 for
Case III.

## The relaxed problem

Relaxing the constraint that agents only move between neighbours (and conserve their number) with multipliers μ
splits the problem by agent type and by area. For each agent type j, `build_and_solve` sets up the LP of
eqs. (17) to (19):

- variables V[t][i][s] for every time, area and state, and μ[t][i] for every time and area;
- maximise Σᵢ p₀ᵢ · V[0][i];
- subject to, for every t, i, s and every action a (move in from neighbour i′, or no move):

  V[t][i][s] ≤ c(s) + μ[t][i] · indicator(s) − μ[t][i′] · [a moves in] + Σ_{s′} P(s, a, s′) V[t+1][i][s′]

  with V[T] = 0;
- every variable in [−1e5, 1e5], and, in the C++ and in `gen.py`/`simulate.py`, every row bounded below by −1e5.

The rows are built in the C++'s order (t, i, s, a). The lower bound is the sum of the two types' optima.
`RelaxedSolution.lower_bound_cxx` recomputes it as `getLowerBound` does, Σ p₀ · V[0] in the C++'s loop order,
which is the number the result rows report.

The LP is solved with HiGHS through `scipy.optimize.linprog`. The row duals are the occupation measures of the
relaxed problem's optimal randomized policy; `build_and_solve` keeps them as `AgentLP.policy`. The C++ does not
compute this policy. It is used only by the extra diagnostics in `patrol/gradients.py` and by `gen.py --policy`.

For fixed μ, the inner problem is the Bellman recursion of eq. (15). `dp_with_actions` solves it backwards from
t = T − 1, starts each state from "no move", and switches action only on a strictly smaller value, in the same
floating-point order as the C++. Its V is the `dp` value function ([value-function.md](value-function.md)).

## The two policies

Both policies work the same way at each step t, for each agent type separately:

1. List every candidate movement (from area i′ to sub-area k of area i, where i′ is a neighbour of i and holds at
   least one agent).
2. Rank the movements by an index, using the C++'s heap sort and comparator.
3. Go down the ranking and accept a movement if its source area still has an unassigned agent and its target
   sub-area has no movement yet.
4. Repair the plan so every agent is accounted for (movement adaption, Algorithm 2). An agent left without an
   assignment is routed along a path of reassignments to the nearest vacant sub-area. Each repair adds one to the
   run's adaption count.

The only difference is the index:

- Greedy uses 1 − p of the target sub-area, so the riskiest sub-areas come first.
- MAI uses ϑ[t][i][s] − μ[t][i′], where
  ϑ[t][i][s] = c(s, 1, t) − c(s, 0, t) + E[V[t+1] | move in] − E[V[t+1] | no move] (eq. 22). Because the cost does
  not depend on the action, the first difference is always zero in this code, and at t = T − 1 the index is −μ.

The comparator treats two indices within 1e-10 of each other as equal and then orders by source area. Lower
index values come first.

## The simulation

Each run draws the initial occupancy of every sub-area, then for t = 0, …, T − 1: decide the movements, apply
movement adaption, add the current total cost divided by h, and sample every sub-area's transition. Dividing by h
keeps costs comparable across scalers.

For run r the occupancy seed is r + seed + 100 and the transition seed is r + seed + 200. Greedy and MAI use the
same two seeds, so their difference is not inflated by independent sampling noise.

After `--iters` runs (1,000 by default), `monte_carlo` reports each policy's mean cost, its 95% confidence
half-width (with the C++'s variance convention; see [departures.md](departures.md#confidence-interval-divisor)),
the mean adaption count, and the gap (mean − bound) / bound.
