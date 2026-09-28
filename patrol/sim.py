"""Monte-Carlo simulator for the Greedy and MAI patrolling policies (PatrollingProcess, stdafx.cpp)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from .cxx_compat import GlibcRand, heap_sort_cxx
from .instance import Instance
from .lp import RelaxedSolution
from .model import EPSILON_PRECISION

GREEDY, DP_INDEX = 1, 0            # stdafx.h
ITER_MAX = 1000                     # stdafx.h
BASE_INIT_AGENTS = 100              # stdafx.h
BASE_SIMULATION_SEED = 200          # stdafx.h
INT_MAX = 2**31 - 1
POLICY_NAMES = {DP_INDEX: "index", GREEDY: "greedy"}     # simulation(): names used in the trajectory files


@dataclass
class Movement:
    from_area: int
    to_area: int
    sub_area: int
    state: int
    index: float


def _compare_index_reverse(m1: Movement, m2: Movement) -> bool:
    """PatrollingProcess::Movement::compare_index_reverse (stdafx.h): the heap comparator."""
    d = m1.index - m2.index
    if d < -EPSILON_PRECISION:
        return False
    if -EPSILON_PRECISION <= d <= EPSILON_PRECISION:
        return not (m1.from_area - m2.from_area < 0)
    return True


@dataclass
class RunResult:
    total_cost: float
    movement_adaptions: int


class Simulator:
    """PatrollingProcess's simulation half."""

    def __init__(self, inst: Instance, sol: RelaxedSolution, scaler: int):
        self.inst, self.sol, self.scaler = inst, sol, scaler
        self.N, self.J, self.T = inst.area_num, inst.type_num, inst.maxtime
        self.nb = inst.neighbourhood
        self.procs = inst.processes
        # getVartheta (stdafx.cpp:3707), precomputed
        self.mu = [sol.agents[j].mu for j in range(self.J)]
        self.vartheta = [self._vartheta_table(j) for j in range(self.J)]
        self.state: list[list[np.ndarray]] = []
        self.move_from: list[list[np.ndarray]] = []
        self.total_movement_adaptions = 0
        # agents[j][a] = [area, sub-area] and link[i][j] = indices of the agents in area i, as the C++ keeps them
        self.agents: list[list[list[int]]] = []
        self.link: list[list[list[int]]] = []

    # ------------------------------------------------------------------ MAI index
    def _vartheta_table(self, j: int) -> list[list[list[float]]]:
        a = self.sol.agents[j]
        V = a.V
        table = []
        for t in range(self.T):
            row = []
            for i in range(self.N):
                ap = self.procs[i][j]
                vals = []
                for s in range(ap.state_num):
                    cost1 = cost0 = ap.cost[s // 2]
                    if t == self.T - 1:
                        vals.append(cost1 - cost0)
                        continue
                    Vn = V[t + 1]
                    off = a.w_i[i]
                    v = cost1 - cost0
                    tmp = 0.0
                    for tr in ap.next_states(s, 1):
                        tmp += tr.prob * Vn[off + tr.s_next]
                    v += tmp
                    tmp = 0.0
                    for tr in ap.next_states(s, 0):
                        tmp += tr.prob * Vn[off + tr.s_next]
                    v -= tmp
                    vals.append(v)
                row.append(vals)
            table.append(row)
        return table

    # ------------------------------------------------------------- initialisation
    def init_agents(self, seed: int) -> None:
        """initAgentsForSimulation (stdafx.cpp:3503)."""
        rng = GlibcRand(seed)
        self.state = [[np.zeros(self.scaler, dtype=int) for _ in range(self.N)] for _ in range(self.J)]
        self.move_from = [[np.full(self.scaler, -1, dtype=int) for _ in range(self.N)] for _ in range(self.J)]
        self.agents = [[] for _ in range(self.J)]
        self.link = [[[] for _ in range(self.J)] for _ in range(self.N)]
        for j in range(self.J):
            for i in range(self.N):
                probs = self.inst.init_probs[i][j]
                for k in range(self.scaler):
                    tmp = rng.uniform()
                    count = 0.0
                    for s in range(probs.size):
                        count += probs[s]
                        if count > tmp:
                            self.state[j][i][k] = s
                            if s % 2:
                                self.agents[j].append([i, k])
                                self.link[i][j].append(len(self.agents[j]) - 1)
                            break

    # ----------------------------------------------------------------- policies
    def _rank_movements(self, t: int, j: int, policy: int) -> list[Movement]:
        """updateMovementRanking{Greedy,LagrangianIndex} (stdafx.cpp:3406 / 3307)."""
        movements = []
        for i in range(self.N):
            for n in self.nb[i]:
                if int((self.state[j][n] % 2).sum()) == 0:
                    continue
                for k in range(self.scaler):
                    s = int(self.state[j][i][k])
                    if policy == GREEDY:
                        idx = 1 - self.procs[i][j].p[s // 2]
                    else:
                        idx = self.vartheta[j][t][i][s] - self.mu[j][t, n]
                    movements.append(Movement(n, i, k, s, idx))
        return heap_sort_cxx(movements, _compare_index_reverse)

    def _decide(self, t: int, policy: int) -> None:
        """decisionMaking{Greedy,LagrangianIndex} (stdafx.cpp:3986 / 3855)."""
        for j in range(self.J):
            ordered = self._rank_movements(t, j, policy)
            occupation = [int((self.state[j][i] % 2).sum()) for i in range(self.N)]
            for i in range(self.N):
                self.move_from[j][i][:] = -1
            for m in ordered:
                if occupation[m.from_area] > 0 and self.move_from[j][m.to_area][m.sub_area] < 0:
                    self.move_from[j][m.to_area][m.sub_area] = m.from_area
                    occupation[m.from_area] -= 1

    # ---------------------------------------------------------- movement adaption
    def _update_distances(self, j: int, eligible: list[int], is_eligible: list[bool]) -> list[int]:
        """updateDistances (stdafx.cpp:4260)."""
        dist = [INT_MAX // 10] * self.N
        mf = self.move_from[j]
        stack = []
        for i in eligible:
            if any((mf[n] < 0).any() for n in self.nb[i]):
                dist[i] = 0
                stack.append(i)
        offsprings: list[int] = []
        while stack:
            area = stack.pop()
            move_to = [ip for ip in self.nb[area] if (mf[ip] == area).any()]
            for ip in move_to:
                for n in self.nb[ip]:
                    if is_eligible[n] and n != area and dist[area] + 1 < dist[n]:
                        dist[n] = dist[area] + 1
                        offsprings.append(n)
            if not stack:
                stack.extend(offsprings)
                offsprings = []
        return dist

    def _movement_adaption(self) -> None:
        """movementAdaption (stdafx.cpp:4413)."""
        for j in range(self.J):
            mf = self.move_from[j]
            is_eligible = [bool((self.state[j][i] % 2).any()) for i in range(self.N)]
            eligible = [i for i in range(self.N) if is_eligible[i]]
            dist = self._update_distances(j, eligible, is_eligible)
            for i in eligible:
                adaption_num = int((self.state[j][i] % 2).sum())
                for n in self.nb[i]:
                    adaption_num -= int((mf[n] == i).sum())
                self.total_movement_adaptions += adaption_num
                for _ in range(adaption_num):
                    i_bar = i
                    while i_bar >= 0:
                        i_star = k_star = -1
                        min_d = INT_MAX
                        origin = -1
                        for n in self.nb[i_bar]:
                            for k in range(self.scaler):
                                origin = int(mf[n][k])
                                if origin < 0:
                                    continue
                                if min_d > dist[origin]:
                                    i_star, k_star, min_d = n, k, dist[origin]
                        if origin < 0:
                            raise RuntimeError("movementAdaption: origin < 0")
                        origin = int(mf[i_star][k_star])
                        mf[i_star][k_star] = i_bar
                        if min_d == 0:
                            placed = False
                            for n in self.nb[origin]:
                                for k in range(self.scaler):
                                    if mf[n][k] < 0:
                                        mf[n][k] = origin
                                        placed = True
                                        break
                                if placed:
                                    break
                            if not placed:
                                raise RuntimeError("movementAdaption: no vacant sub-area found")
                            i_bar = -1
                        else:
                            i_bar = origin
                    dist = self._update_distances(j, eligible, is_eligible)

    # ------------------------------------------------------------- dynamics/cost
    def _current_cost(self) -> float:
        """currentCostRate (stdafx.cpp:4688)."""
        cost = 0.0
        for i in range(self.N):
            for j in range(self.J):
                c = self.procs[i][j].cost
                for k in range(self.scaler):
                    cost += c[int(self.state[j][i][k]) // 2]
        return cost

    def _transition(self, rng: GlibcRand) -> None:
        """stateTransition (stdafx.cpp:4088)."""
        for i in range(self.N):
            for j in range(self.J):
                ap = self.procs[i][j]
                st, mf = self.state[j][i], self.move_from[j][i]
                for k in range(self.scaler):
                    move_in = 1 if mf[k] >= 0 else 0
                    trs = ap.next_states(int(st[k]), move_in)
                    tmp = rng.uniform()
                    tmp2 = 0.0
                    s = 0
                    while s < len(trs) and tmp2 < tmp:
                        tmp2 += trs[s].prob
                        s += 1
                    s -= 1
                    st[k] = trs[s].s_next
                    if move_in:
                        origin = self.link[int(mf[k])][j]
                        if not origin:
                            raise RuntimeError(f"stateTransition: no agent in area [i={int(mf[k])}][j={j}]")
                        a = origin.pop(0)
                        self.link[i][j].append(a)
                        self.agents[j][a] = [i, k]

    # -------------------------------------------------------------------- run
    def run(self, policy: int, seed: int, trajectory_dir: str | Path | None = None) -> RunResult:
        """simulation (stdafx.cpp:4715). trajectory_dir: write the per-step files the C++ writes with outFlag."""
        rng = GlibcRand(seed)
        self.total_movement_adaptions = 0
        cost = 0.0
        for t in range(self.T):
            self._decide(t, policy)
            self._movement_adaption()
            step = self._current_cost() / self.scaler
            cost += step
            if trajectory_dir is not None:
                self._write_trajectory(Path(trajectory_dir), policy, seed, t, step, cost)
            self._transition(rng)
        return RunResult(cost, self.total_movement_adaptions)

    def _write_trajectory(self, out_dir: Path, policy: int, seed: int, t: int, step: float, cost: float) -> None:
        """The outFlag block of simulation(): appends to trajectory-<policy>-seed<seed>-scaler<h>-*.out."""
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"trajectory-{POLICY_NAMES[policy]}-seed{seed}-scaler{self.scaler}"

        def append(suffix: str, text: str) -> None:
            with open(f"{stem}-{suffix}.out", "a") as f:
                f.write(text)

        append("cost", f"{t + 1}\t{_cxx(step)}\t{_cxx(cost / (t + 1))}\t{_cxx(cost)}\n")
        append("graph", f"For time = {t + 1}, the numbers of agents of different types in each area:\n"
               + "".join(f"Area <{i}>: " + "".join(f"{len(self.link[i][j])} " for j in range(self.J)) + "\n"
                         for i in range(self.N)) + "\n")
        append("agents", f"For time = {t + 1}, positions of the agents (i,k):\n"
               + "".join(f"Type-{j} agents: " + "".join(f"({a},{k}) " for a, k in self.agents[j]) + "\n"
                         for j in range(self.J)) + "\n")
        name = POLICY_NAMES[policy]
        lines = [f"For time = {t + 1}, taking movements under the {name} policy:\n"]
        for j in range(self.J):
            lines.append(f"Type-{j} agents: \n")
            for i in range(self.N):
                for k in range(self.scaler):
                    if self.move_from[j][i][k] >= 0:
                        lines.append(f"\t({int(self.move_from[j][i][k])})-->({i},{k})\n")
            lines.append("\n")
        append("actions", "".join(lines) + "\n")


def _cxx(x: float) -> str:
    """A double through `std::ostream <<` at the default precision (6 significant digits, %g)."""
    return f"{x:.6g}"


@dataclass
class SweepRow:
    scaler: int
    greedy_avg: float
    greedy_ci: float
    greedy_adaptions: float
    mai_avg: float
    mai_ci: float
    mai_adaptions: float
    greedy_dev: float
    mai_dev: float
    lower_bound: float
    slackness: float = 0.0

    def as_line(self, slackness: float | None = None) -> str:
        """Same tab-separated row main.cpp appends to long-term-performance-cost.out."""
        f = _cxx
        slackness = self.slackness if slackness is None else slackness
        return "\t".join([str(self.scaler), f(self.greedy_avg), f(self.greedy_ci), f(self.greedy_adaptions),
                          f(self.mai_avg), f(self.mai_ci), f(self.mai_adaptions),
                          f(self.greedy_dev), f(self.mai_dev), f(self.lower_bound), f(slackness)]) + "\t"


def monte_carlo(inst: Instance, sol: RelaxedSolution, scaler: int, seed: int = 395,
                iter_max: int = ITER_MAX, progress: bool = False, slackness: float = 0.0,
                trajectory_dir: str | Path | None = None) -> SweepRow:
    """Monte-Carlo loop and confidence intervals (main.cpp:533-640).

    The lower bound is getLowerBound's p0 . V_0 (equal to the LP optimum up to rounding); slackness is passed through to the row (see patrol.gradients).
    """
    sim = Simulator(inst, sol, scaler)
    costs1, costs2, adapt1, adapt2 = [], [], [], []
    for it in range(iter_max):
        init_seed, run_seed = it + seed + BASE_INIT_AGENTS, it + seed + BASE_SIMULATION_SEED
        sim.init_agents(init_seed)
        r1 = sim.run(GREEDY, run_seed, trajectory_dir)
        sim.init_agents(init_seed)
        r2 = sim.run(DP_INDEX, run_seed, trajectory_dir)
        costs1.append(r1.total_cost); costs2.append(r2.total_cost)
        adapt1.append(r1.movement_adaptions); adapt2.append(r2.movement_adaptions)
        if progress and (it + 1) % 50 == 0:
            print(f"  iter {it + 1}/{iter_max}", file=sys.stderr, flush=True)

    def running_mean(xs):
        m = 0.0
        for n, x in enumerate(xs, 1):
            m += (x - m) / n
        return m

    def ci(xs, m):
        v = 0.0
        for n, x in enumerate(xs, 1):
            v += ((x - m) * (x - m) - v) / n
        return float(np.sqrt(v / len(xs)) * student_t.ppf(0.975, len(xs) - 1))

    a1, a2 = running_mean(costs1), running_mean(costs2)
    lb = sol.lower_bound_cxx()
    return SweepRow(scaler, a1, ci(costs1, a1), running_mean(adapt1), a2, ci(costs2, a2), running_mean(adapt2),
                    (a1 - lb) / lb, (a2 - lb) / lb, lb, slackness)
