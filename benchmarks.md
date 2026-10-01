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

Long runs go on the Spark, which is reachable two ways, and **both are worth knowing before one of
them fails**: `100.101.143.122` over Tailscale, and `192.168.0.40` on the LAN, which needs no overlay
when both machines are on `192.168.0.0/24`. A session was lost treating a one-directional Tailscale
failure as evidence about the machine while the LAN address worked on the first attempt; the Spark had
been up throughout, and had been reaching the host fine in the other direction. Jobs there go under
`setsid nohup` so they survive the launching session, which they have done through a host reboot.

Section 7 is the check that the architecture does not matter: with `--value-function dp` the two
architectures give bit-identical policy costs. Note that the GPU is unused throughout — the LP is 3% of a run's
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

**Read that table with section 9 in hand.** Those are the *deterministic* diagnostic's values — the one
`main.cpp` computes by taking an argmin of the Bellman recursion. They differ across configurations because the
tie-break differs, not because the relaxation is differently satisfied. Computed under each solution's own
randomized policy, the residual is zero (1e-14) for every vertex. What genuinely varies with the vertex, and what
moves MAI by percentage points, is `V` off the support — not the slackness.

## 4. Figure 3: the paper's 75% claim

Section VI claims that for Case I at h = 40, MAI's deviation is under 3% in **over 75%** of randomly
generated settings. Each setting is one seed — its own LP solve plus 1,000 simulation runs per policy
at each of five horizons.

**1,000 settings** (seeds 1–1000) — the paper's stated sample size — Case I, `--value-function dp`,
on the DGX Spark at 18-way parallelism; 9,323 s wall.

| h | MAI median | MAI p75 | **MAI under 3%** | GRD median |
|---:|---:|---:|---:|---:|
| 1 | 4.35% | 5.74% | 21.0% | 7.81% |
| 5 | 1.49% | 2.25% | 87.0% | 6.85% |
| 10 | 0.93% | 1.55% | 91.5% | 6.64% |
| 20 | 0.60% | 1.22% | 92.7% | 6.53% |
| 40 | **0.41%** | 1.03% | **92.8%** | 6.45% |

**The claim holds, at 92.8% against the stated 75%**, with the 75th percentile of the deviation at
1.03% — well inside 3%. The two qualitative features of the figure also reproduce: MAI's distribution
marches left and steepens as h grows (21% → 87% under 3% between h = 1 and h = 5, then flat), while
GRD sits near 6.5–7.8% at every h and converges to nothing.

A 50-setting pilot run first on the same seeds gave 90.0% at h = 40, against 92.8% here; sample size
was never the binding constraint, since even n = 50 puts the proportion at 90% ± 8.3% (95%), clear of
75%. The full run was worth doing to match the paper's stated protocol exactly, not to rescue the
conclusion.

CDF: `benchmarks/fig3.png`, drawn by `plot_fig3.py` from the CSV.

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

## 8. Theorem 3: what it claims, and what the data shows

Theorem 3 is titled "Exponential Convergence" and the surrounding prose says the deviation
"diminishes exponentially in the scaling parameter h". Nothing in this project had tested that, so
this section does — and the first two things it found were errors in the testing, not in the paper.

### 8.1 Two corrections to the premise

**Γ^{h,∗} is the relaxed optimum, not the true optimum.** Page 22, immediately after equation (35):

> define Γ^{h,∗} := min_{ϕ∈Φ̂h} Γ^{h,ϕ} as the minimum of the **relaxed** MAB-ML problem described in
> (32) and (31)

and the paper then notes "Γ^{h,∗} is a lower bound to the minimum of MAB-ML". So the quantity
Theorem 3 bounds is the gap against the *relaxation* — which is exactly the lower bound these runs
report. An earlier version of this section claimed the opposite: that only part of the measured
deviation was theorem-bounded and the rest was an unobservable optimum-to-bound gap needing a floor
term to absorb. That was wrong. On the theorem's own terms no floor is needed, and the gap we measure
is the gap the theorem talks about.

**The inequality does not pin down a rate.** Page 23, verbatim:

> Theorem 3 (Exponential Convergence): If the MAB-ML process is strictly indexable and a policy
> ϕ ∈ Φh satisfies (29), (30) and (38), then, for any δ > 0, there exist H > 0 and s > 0 such that,
> for all h > H, Γ^{h,ϕ} − Γ^{h,∗} ≤ e^{−sh} + δ.  (40)

Note the quantifier order: δ is chosen first, and H and s may depend on it. **Any** non-negative
sequence converging to zero satisfies this — given δ, choose H beyond which the gap is below δ, and
then any s > 0 works, because e^{−sh} > 0. A gap decaying like h^{−1/2} satisfies (40). So inequality
(40) is equivalent to "the gap converges to zero" and does not establish an exponential rate, despite
the theorem's title and the prose around it.

That is a reading of the paper rather than a measurement, and it is the more substantial observation
of the two. It does not make (40) unfalsifiable, though, and an earlier version of this section
overstated that. (40) is *equivalent* to "the gap converges to zero": if the gap tended to some
c > 0, choosing δ = c/2 would require gap ≤ e^{−sh} + c/2 for all h > H, and since e^{−sh} → 0 the
right-hand side eventually falls below c. So a fit can contradict (40) in exactly one way — by
establishing a **nonzero asymptote** — and cannot contradict it by finding the wrong rate. §8.4
measures that asymptote, which is where the one interesting result of this section turns out to be.

### 8.2 Method

`analyse_theorem3.py` fits, by weighted least squares with A, C ≥ 0 and rate > 0:

```
exponential   gap(h) = A exp(-k h) + C
power law     gap(h) = A h^(-p)   + C
```

Since Γ^{h,∗} is the relaxed optimum the gap should tend to zero, so **C = 0 is the model the theory
implies**; C ≥ 0 free is reported as a sensitivity. Errors are Monte-Carlo standard errors, obtained
from the reported 95% half-widths by dividing by the Student-t quantile — a half-width is not one
sigma, and an earlier version treated it as one, understating χ² by t² ≈ 3.85×. With equal parameter
counts AIC differences reduce to χ² differences, so that is what is reported, together with an
absolute goodness-of-fit p: **a model can win the comparison and still describe the data badly.**

`analyse_theorem3.py` compares the two decay *shapes*. The quantity that actually bears on (40) is
the asymptote, so `check_theorem3_floor.py` fits the floor on its own and reports it with a
profile-likelihood 95% interval (Δχ² = 3.841) per tail window, plus a check that assumes neither
curve: what the gap did between the two largest h, against what a C = 0 power law says it should
have done. The two scripts are deliberately separate — one answers "which shape", the other answers
"does it reach zero", and only the second is a claim about the paper.

### 8.3 Results, h ≤ 100, 10,000 runs per point

The numbers below come from a sweep of 39 points — 3 cases × h ∈ {1,2,3,5,7,10,14,20,28,40,55,75,100},
10,000 runs each, `--value-function dp`, seeds 300/395/51 — run 18-way parallel on the Spark,
06:02:27 → 07:48:14 on 2026-10-17, all 39 tasks exiting `rc=0`. It supersedes an earlier h ≤ 50,
1,000-run version of this section; where the conclusions changed, they changed because those error
bars were ~19% of the decaying signal and these are ~2–4%.

MAI's deviation from the relaxed optimum, in per cent (SE 0.012–0.14):

| h | Case I | Case II | Case III |
|---:|---:|---:|---:|
| 1 | 8.43 | 7.48 | 6.81 |
| 2 | 5.30 | 4.48 | 4.04 |
| 3 | 4.00 | 3.45 | 2.96 |
| 5 | 2.80 | 2.46 | 2.04 |
| 7 | 2.13 | 2.05 | 1.53 |
| 10 | 1.68 | 1.69 | 1.42 |
| 14 | 1.33 | 1.40 | 0.87 |
| 20 | 1.01 | 1.28 | 1.10 |
| 28 | 0.80 | 1.15 | 0.58 |
| 40 | 0.64 | 1.07 | 0.92 |
| 55 | 0.50 | 1.02 | 0.39 |
| 75 | 0.44 | 0.97 | 0.51 |
| 100 | 0.34 | **0.99** | 0.31 |

With C = 0, the asymptote the theory implies:

| Case | power-law χ²/dof | gof p | exponent | exponential χ²/dof | ΔAIC |
|---|---:|---:|---:|---:|---|
| I | 8.64/11 | 0.655 | 0.70 | 4189/11 | +4180 → power law |
| II | 1009/11 | 1.8e−209 | 0.40 | 3869/11 | +2859 → power law |
| III | 577/11 | 1.1e−116 | 0.63 | 3323/11 | +2746 → power law |

**Case I is now a clean positive result**: a pure power law h^−0.70 with no floor describes 13 points
over two decades of h at goodness-of-fit p = 0.655, while the exponential is excluded at χ² = 4189.
The margin is +4180, where the old data managed +22. The earlier finding that the preference
*reversed* at h ≥ 10 (ΔAIC −0.42) does not survive the smaller error bars: the tail windows now read
+979, +478, +118, **+28.5** at h ≥ 1, 2, 5, 10. That reversal was a small-sample artefact, and the
honest reading of the earlier section is that it could not tell — not that it had found the
exponential.

Cases II and III reject C = 0 outright, for two different and more interesting reasons.

### 8.4 One case does not converge to the bound

Fitting a floor and profiling it (95% interval from Δχ² = 3.841), per tail window:

| Case | window | floor C | 95% interval | exponent | χ²/dof | gof p |
|---|---|---:|---|---:|---:|---:|
| I | h ≥ 1 | 0.022% | [0, 0.060] | 0.71 | 7.6/10 | 0.668 |
| I | h ≥ 20 | 0.008% | [0, 0.185] | 0.67 | 2.1/3 | 0.545 |
| **II** | h ≥ 1 | **0.859%** | **[0.835, 0.880]** | 0.92 | 20.5/10 | 0.025 |
| **II** | h ≥ 20 | **0.953%** | **[0.880, 0.985]** | 1.60 | 3.4/3 | 0.336 |
| III | h ≥ 1 | 0.174% | [0.135, 0.210] | 0.74 | 524/10 | 3.0e−106 |
| III | h ≥ 20 | 0.000% | [0, 0.075] | 0.66 | 388/3 | 8.8e−84 |

**Case I's floor includes zero in every window. Case II's excludes it in every window**, and on the
tail the floor-plus-power-law fit is good (p = 0.336) with the floor at 0.95%.

A model-free version of the same thing, which does not depend on either curve. Between the two
largest h, 75 and 100:

| Case | what the gap did | what a C = 0 power law predicts |
|---|---|---|
| I | 0.436% → 0.343%, a fall of 0.094 ± 0.020 (**4.8σ**) | a fall of 0.080 — consistent |
| II | 0.973% → 0.989%, a change of **−0.016 ± 0.018 (−0.9σ)** | a fall of 0.105 — **5.8σ away** |
| III | 0.510% → 0.313%, a fall of 0.197 ± 0.020 (10.1σ) | a fall of 0.084 — falls faster |

Case I is still descending at h = 100. **Case II is flat**: a third more h buys a change
indistinguishable from zero, where decay to the bound would have required five sigma of improvement.

Since §8.1 establishes that (40) is equivalent to the gap converging to zero, a genuine nonzero
asymptote is the one thing a measurement here could hold against Theorem 3. Three reasons this is
not yet that — each a limit of this reproduction rather than a defence of the paper:

1. **The premise is not implemented.** (40) requires strict indexability, which §6 shows the shared
   code does not provide: the ε perturbation of §V is absent, so `c(s,1,t) − c(s,0,t)` is identically
   zero. A plateau in code that does not satisfy the hypothesis says nothing about the theorem. It
   says something about the shared code, and a floor is exactly where a missing ε would show up.
2. **One instance, one seed.** Each "case" here is a single random setting (Case II is seed 395),
   not an ensemble. §8.6 ran that check and it is decisive: all three fresh I = 10 seeds keep falling,
   so this floor is seed 395's and not the case's — while two of three fresh I = 6 seeds flatten and
   one rises, so floors are common and the clean Case I fit is the atypical draw. Read §8.4 as a
   statement about seed 395 only.
3. **Correlated errors across h.** `simulate.py` passes the same seed at each h and `sim.py` builds
   run seeds without h, so the points are not independent while the fit assumes they are. That
   inflates the apparent significance of a smooth trend; it does not explain a flat tail.

**Case III's residual structure is not noise.** Its gap is non-monotone at large h — 0.58% at 28,
0.92% at 40, 0.39% at 55 — and with SEs near 0.02% those swings are 10–25σ, so no curve of either
family can fit it, which is what χ² = 388 on 3 dof is reporting. The likely mechanism is already
measured elsewhere in this document: each h is a separate LP solve, and §3 shows MAI's gap moving
2.2%–6.9% on nothing but which vertex the solver returns. `dp` removes the solver's `V`, but the
multipliers μ still come from that vertex, so vertex-to-vertex variation across h survives, and at
I = 14 it evidently dominates. That makes Case III's fitted parameters uninterpretable, and is a
more useful finding than the fit it invalidates.

### 8.5 What can honestly be said

> Over h = 1–100 at 10,000 runs per point, Case I's deviation from the relaxed optimum is described
> by a pure power law h^−0.70 with no floor (gof p = 0.655), and an exponential is excluded by
> Δχ² ≈ 4180. Case II instead flattens to a floor of 0.95% [0.88, 0.99] and is statistically flat
> between h = 75 and h = 100, which over this range is inconsistent with convergence to the bound.
> Case III decreases but non-monotonically by 10–25σ, consistent with the LP-vertex sensitivity of
> §3, and is fit by neither family. None of this contradicts Theorem 3, whose premise of strict
> indexability the shared code does not implement. **All three are single instances, and §8.6 shows
> the spread across instances at h = 100 (0.006%–2.641%) dwarfs the h-dependence fitted within one:
> the exponent and the floor are properties of seeds 300 and 395, not of the method.** The claim that
> generalises is the one Figure 4 makes — a large fall from h = 1 to h ≈ 20 in every instance.

The exponents (0.63–0.92) still should not be read as a √N law: that comparison needs the
normalisations matched, and the simulator already divides cost by h.

### 8.6 The seed check, which reframes all of the above

Both graph sizes were re-run on three fresh seeds each at h = 20, 55, 100 (1,000 runs; 18 tasks, all
`rc=0`, 12:48 → 12:55). The question was whether Case II's floor belongs to the instance or to the
case size. The answer is neither of the two outcomes anticipated.

| I | seed | h = 20 | h = 55 | h = 100 | 55 → 100 |
|---:|---|---:|---:|---:|---|
| 6 | **300** (Case I) | 1.012% | 0.504% | 0.343% | −0.161 — falling |
| 6 | 301 | 2.378% | 2.494% | **2.641%** | **+0.148 (+3.7σ) — rising** |
| 6 | 302 | 0.159% | 0.110% | 0.168% | +0.058 (+1.2σ) — flat |
| 6 | 400 | 0.455% | 0.201% | 0.240% | +0.039 (+0.6σ) — flat |
| 10 | **395** (Case II) | 1.282% | 1.018% | 0.989% | −0.029 — flat |
| 10 | 396 | 0.523% | 0.202% | 0.140% | −0.062 (−1.0σ) — falling |
| 10 | 397 | 0.422% | 0.220% | 0.134% | −0.087 (−1.4σ) — falling |
| 10 | 500 | 0.248% | 0.141% | **0.006%** | −0.135 (−3.0σ) — falling |

Two findings, and the second is the one that matters.

**Case II's floor is seed 395's, not the I = 10 case's.** All three fresh I = 10 seeds keep falling
through h = 100 and reach 0.006%–0.140%, lower than any point in the sweep. So §8.4 cannot be read as
a property of the ten-area case.

**But a non-decreasing tail is common, and the clean case is the unusual one.** In the I = 6 family —
whose sweep seed produced the textbook h^−0.70 fit — two of three fresh seeds are flat from h = 55 to
100 and one *rises* by 3.7σ to 2.641%. Four of the six fresh instances have a tail that is not
falling. Seed 300's clean power law is one draw, not the typical behaviour, and §8.3's goodness-of-fit
p = 0.655 describes that draw rather than the model class.

**At h = 100 the gap ranges from 0.006% to 2.641% across eight instances — a factor of 400**, which is
far larger than any h-dependence measured within an instance. That is the result the sweep really
produced, and it has a mechanism already measured in §3: each h is a separate LP solve, and the vertex
the solver returns moves MAI's gap by 2.2%–6.9% on its own. `dp` removes the solver's `V`, but μ still
comes from the vertex, so every point in every one of these curves carries a per-h offset of the same
order as the gap being fitted. Seed 301's *rising* gap and Case III's 10–25σ non-monotonicity are the
same phenomenon seen twice.

**Consequence for §8.3 and §8.4.** Both are single-instance measurements, and the instance-to-instance
spread exceeds the effect they fit. Case I's exponent and Case II's floor should be read as properties
of seeds 300 and 395, not of the method. Settling convergence per se needs the design Figure 3 already
uses — averaging over many random settings at each h — rather than more h on one setting, and that is
the natural next experiment rather than a deeper sweep.

What survives unqualified: the gap falls substantially from h = 1 to h ≈ 20 in every instance measured
(8.43% → 1.01% for seed 300, 7.48% → 1.28% for seed 395), which is the paper's qualitative claim and
the regime Figure 4 plots. What does not survive is any single-instance statement about the asymptote.

### 8.7 Provenance

The two premise errors in 8.1, the half-width-as-sigma error, the AIC formula, and the h = 1
dependence were all found in an independent review by Deepak of the
first version of this section. Its corrected χ² figures and the tail-window numbers were reproduced
exactly by the rewritten script, which is the cross-check that they agree. The review also confirmed
that the gap-to-bound ratio transformation is unbiased, and that the negative floor the first version
worried about was only 0.36 SE below zero — a non-issue, where the real problems lay elsewhere.

The h ≤ 100 sweep then changed two of the conclusions that review had helped establish, which is the
point of running it: the h ≥ 10 reversals disappeared, having been small-sample noise, and the floor
term — introduced in the first version for the wrong reason, demoted to a sensitivity in the second —
turned out to be where the only theorem-relevant result lives. This is also the first version in
which the floor is reported with an interval rather than a point estimate, which is what makes
Case I's "includes zero" and Case II's "excludes zero" statable at all.

## 9. Proposition 2: the residual the package reports, and the one it should

Proposition 2 (p. 13, equation (23)) says a mixture of threshold policies attains the relaxed
optimum — equivalently, that the complementary-slackness residual is zero at the LP optimum. That
residual is **the last column of `long-term-performance-cost.out`**, and the authors' `Readme.pdf`
says it should be zero. The shipped code reports values in the hundreds.

The proposition is not at fault. The LP being solved is the dual, so at a vertex the Bellman
inequalities are tight for several actions at once and the true primal optimum randomises over them.
`main.cpp` instead rebuilds one deterministic policy from the argmin of the Bellman recursion, and
complementary slackness does not hold between the LP's dual and an arbitrarily tie-broken policy.

The port reproduced that faithfully and therefore inherited a diagnostic that cannot check anything.
`patrol.gradients.gradients_randomized` adds the missing path, computing the same gradients under the
randomized policy the LP solve already produces. `tests/check_proposition2.py` runs both, so the
contrast is visible rather than asserted:

| Instance | residual, argmin policy | residual, LP policy | max abs gradient, LP policy |
|---|---:|---:|---:|
| seed 395 (I=10) | 103.713 | −9.49e−14 | 5.55e−16 |
| Case I (I=6) | −3.093 | 2.67e−14 | 3.33e−16 |
| Case II (I=10) | 103.713 | −9.49e−14 | 5.55e−16 |
| Case III (I=14) | 267.481 | 6.64e−14 | 1.06e−15 |

The check is on the **individual** gradients, not their weighted sum. A sum near zero can be
cancellation between large terms; every gradient at machine epsilon means each constraint in (23) is
satisfied on its own, which is the actual content of the proposition.

**Proposition 2 holds on all three paper cases and on the package's own instance.** It is a claim of
the paper verified from the shared material alone, and it closes the port's last functional gap
against `main_fixed.cpp` — which had the equivalent fix on the C++ side via CPLEX's row duals.

Two things worth noting about how this was found. It came out of auditing the paper for checkable
claims rather than from the code: the residual had been computed, printed and discussed since the
first day of the project without anyone framing it as *a claim of the paper that could be tested*.
And the new functions sit beside the deterministic ones rather than replacing them — those are part
of what makes the port bit-exact with `main.cpp`, and the suite still passes 10/10 afterwards.

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
# section 4 -- Figure 3, Case I: the 75% claim (1,000 settings ~ 2.6 h on 18 cores)
python3 reproduce_fig3.py --case 1 --settings 1000 --jobs 18 --out benchmarks/fig3_case1_full.csv
python3 plot_fig3.py --csv benchmarks/fig3_case1_full.csv --out benchmarks/fig3.png

# section 5 -- the lower bound against an exact joint DP (~1 min each; exits non-zero on failure)
python3 tests/check_lower_bound.py
python3 tests/check_lower_bound.py --horizon 5
python3 tests/check_lower_bound.py --agents 2

# section 6 -- how much of the ranking is ties, and whether the tie-break moves the number
python3 tests/measure_index_ties.py --case all

# section 9 -- Proposition 2: residual under the LP's own policy (~4 min for all four)
python3 tests/check_proposition2.py --case all
python3 tests/measure_tiebreak_sensitivity.py --case 1 --scaler 40 --runs 60

# section 8 -- the convergence-rate fits, over any directory of case<N>_sweep.out rows.
# --runs must match the runs per point, since the standard errors are derived from the
# reported 95% half-widths via the Student-t quantile.
python3 analyse_theorem3.py --dir benchmarks/theorem3_h100 --runs 10000 \
    --plot benchmarks/theorem3_h100.png

# sections 8.4 and 8.5 -- the floor, with an interval, and the model-free tail check.
python3 check_theorem3_floor.py --dir benchmarks/theorem3_h100 --runs 10000

# section 8.6 -- the same tails on fresh seeds, which is what shows that 8.3 and 8.4 are
# single-instance results.  Reads benchmarks/theorem3_seeds/a<areas>_s<seed>.out.
python3 check_theorem3_seeds.py --dir benchmarks/theorem3_seeds --runs 1000
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
