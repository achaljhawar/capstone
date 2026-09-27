"""The relaxed problem as a sparse LP (cplexEquivalentProblem_agent, stdafx.cpp:2191), solved with HiGHS.

Two value functions come out of a solve, and they differ off the LP's support:

* ``"lp"``: the V block of the LP vertex itself. This is what the C++ writes to store_valueFuncs and what
  main.cpp's MAI index (getVartheta) reads. For states the LP never reaches, V is not pinned down and sits at
  whatever the solver returned (often the +-1e5 variable bound), so it is solver-specific: HiGHS and CPLEX can
  give different values there, and the MAI simulation results move with them.
* ``"dp"``: the Bellman recursion for the optimal mu (dynamicProgrammingWithGivenMultipliers_agent). It equals
  the LP V on the support and is well defined everywhere else.

Library calls default to ``"dp"`` and to rows without the C++'s -1e5 lower bound (the previous behaviour); the
gen.py/simulate.py command lines default to ``"lp"`` with ranged rows, to match the C++.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .instance import Instance

VAR_BOUND = 1e5   # C++: minSols = -1e5, maxSols = 1e5 on every variable, and constraintsMin = -1e5 on every row
VALUE_FUNCTIONS = ("lp", "dp")


@dataclass
class AgentLP:
    """LP solution for one agent type."""
    agent_type: int
    objective: float                       # optimal value (== contribution to the lower bound)
    V: list[np.ndarray]                    # Bellman V for mu; V[t] concatenates the areas, area i at offset w_i[i]
    mu: np.ndarray                         # (T, N)
    policy: list[list[np.ndarray]] | None  # policy[t][i] shape (stateNum, |N(i)|+1), rows sum to 1
    w_i: np.ndarray                        # area offsets into V[t]
    randomized_states: int
    unreached_states: int
    status: str
    n_rows: int
    n_cols: int
    nnz: int
    V_lp: list[np.ndarray] | None = None   # V of the LP vertex (what the C++ stores); same layout as V

    def values(self, value_function: str = "dp") -> list[np.ndarray]:
        if value_function not in VALUE_FUNCTIONS:
            raise ValueError(f"value_function must be one of {VALUE_FUNCTIONS}, got {value_function!r}")
        return self.V_lp if value_function == "lp" else self.V


def _offsets(inst: Instance, j: int) -> tuple[list[int], np.ndarray, int]:
    S = [inst.state_num(i, j) for i in range(inst.area_num)]
    w_i = np.concatenate([[0], np.cumsum(S)[:-1]]).astype(int)      # C++ weight_i
    return S, w_i, int(sum(S))                                          # C++ weight_t


def _linprog(c, A, b, verbose):
    res = linprog(c, A_ub=A, b_ub=b, bounds=(-VAR_BOUND, VAR_BOUND), method="highs", options={"disp": verbose})
    return res, (-res.ineqlin.marginals if res.status == 0 else None)


def build_and_solve(inst: Instance, j: int, verbose: bool = False, ranged_rows: bool = False) -> AgentLP:
    """ranged_rows: also impose the C++'s row lower bound, -1e5 <= A x (constraintsMin). It binds for thousands of
    rows, so it changes which optimal vertex comes back (and so the "lp" V off the support), not the optimum.
    About 4x slower, since linprog needs it as a second copy of A."""
    T, N = inst.maxtime, inst.area_num
    procs = [inst.processes[i][j] for i in range(N)]
    S, w_i, W = _offsets(inst, j)
    n_V, n_mu = T * W, T * N
    n_cols = n_V + n_mu

    def v_idx(t, i, s):
        return t * W + w_i[i] + s

    def mu_idx(t, i):
        return n_V + t * N + i

    # --- objective: maximise sum p0 V_0  ->  minimise -p0 . V_0
    c = np.zeros(n_cols)
    for i in range(N):
        c[v_idx(0, i, 0):v_idx(0, i, 0) + S[i]] = -inst.init_probs[i][j]

    # --- rows, in the C++ order (t, i, s, a)
    rows, cols, vals, b = [], [], [], []
    r = 0
    for t in range(T):
        for i in range(N):
            ap, nb = procs[i], inst.neighbourhood[i]
            n_act = len(nb) + 1
            for s in range(S[i]):
                ind = s % 2
                for a in range(n_act):
                    move_in = 1 if a < len(nb) else 0
                    rows.append(r); cols.append(v_idx(t, i, s)); vals.append(1.0)
                    if t < T - 1:
                        for tr in ap.next_states(s, move_in):
                            rows.append(r); cols.append(v_idx(t + 1, i, tr.s_next)); vals.append(-tr.prob)
                    if ind:
                        rows.append(r); cols.append(mu_idx(t, i)); vals.append(-1.0)
                    if move_in:
                        rows.append(r); cols.append(mu_idx(t, nb[a])); vals.append(1.0)
                    b.append(ap.cost[s // 2])
                    r += 1
    n_rows = r
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
    A.sum_duplicates()
    b = np.asarray(b)

    if verbose:
        print(f"[lp] agent type {j}: {n_rows} rows x {n_cols} cols, nnz={A.nnz}")

    if ranged_rows:      # -A x <= 1e5 as extra rows; a row's dual is then (upper side) - (lower side)
        res, d = _linprog(c, sp.vstack([A, -A]).tocsr(), np.concatenate([b, np.full(n_rows, VAR_BOUND)]), verbose)
        duals = None if d is None else d[:n_rows] - d[n_rows:]
    else:
        res, duals = _linprog(c, A, b, verbose)
    if res.status != 0:
        raise RuntimeError(f"HiGHS failed for agent type {j}: {res.message}")

    x = res.x
    V_lp = [x[t * W:(t + 1) * W].copy() for t in range(T)]
    mu = x[n_V:].reshape(T, N).copy()

    # --- occupation measures from the row duals (Python-only extra: the C++ does not output a policy)
    duals[duals < 0] = 0.0
    policy, randomized, unreached = [], 0, 0
    r = 0
    for t in range(T):
        pol_t = []
        for i in range(N):
            n_act = len(inst.neighbourhood[i]) + 1
            block = duals[r:r + S[i] * n_act].reshape(S[i], n_act).copy()
            r += S[i] * n_act
            tot = block.sum(axis=1)
            reached = tot > 1e-12
            block[reached] /= tot[reached, None]
            block[~reached] = 0.0
            block[~reached, n_act - 1] = 1.0       # unreached: default to "no move"
            randomized += int(((block > 1e-9).sum(axis=1) > 1)[reached].sum())
            unreached += int((~reached).sum())
            pol_t.append(block)
        policy.append(pol_t)

    # --- V from the Bellman recursion for this mu
    V = dp_value_functions(inst, j, mu)
    if verbose:
        obj_dp = sum(float(inst.init_probs[i][j] @ V[0][w_i[i]:w_i[i] + S[i]]) for i in range(N))
        n_bound = sum(int((np.abs(V_lp[t]) > 0.9 * VAR_BOUND).sum()) for t in range(T))
        print(f"[lp] agent type {j}: p0.DP(mu) = {obj_dp:.6f} (LP {-res.fun:.6f}); raw LP V had {n_bound} entries at the bound")

    return AgentLP(j, float(-res.fun), V, mu, policy, w_i, randomized, unreached,
                   res.message, n_rows, n_cols, A.nnz, V_lp)


def dp_with_actions(inst: Instance, j: int, mu: np.ndarray) -> tuple[list[np.ndarray], list[list[list[int]]]]:
    """dynamicProgrammingWithGivenMultipliers_agent (stdafx.cpp:596): the Bellman recursion for fixed mu.

    Same floating-point operation order as the C++, and the same tie-breaking (start from "no move", switch only
    on a strictly smaller value), so V and the optimal actions are bit-identical to the C++ ones.
    Returns V[t] (layout as AgentLP.V) and opt_actions[t][i][s] (index into neighbourhood[i]; |N(i)| = no move).
    """
    T, N = inst.maxtime, inst.area_num
    procs = [inst.processes[i][j] for i in range(N)]
    S, w_i, W = _offsets(inst, j)
    V = [np.zeros(W) for _ in range(T)]
    actions: list[list[list[int]]] = [[[] for _ in range(N)] for _ in range(T)]
    for t in range(T - 1, -1, -1):
        Vt, last = V[t], t == T - 1
        Vn = None if last else V[t + 1]
        for i in range(N):
            ap, nb, off = procs[i], inst.neighbourhood[i], w_i[i]
            mu_t = mu[t]
            for s in range(S[i]):
                ind = s % 2
                cost = ap.cost[s // 2]
                v = cost
                v += mu_t[i] * ind
                if not last:
                    for tr in ap.next_states(s, 0):
                        v += tr.prob * Vn[off + tr.s_next]
                opt = len(nb)
                for a, n in enumerate(nb):
                    value = cost + mu_t[i] * ind
                    value -= mu_t[n]
                    if not last:
                        for tr in ap.next_states(s, 1):
                            value += tr.prob * Vn[off + tr.s_next]
                    if value < v:
                        v, opt = value, a
                Vt[off + s] = v
                actions[t][i].append(opt)
    return V, actions


def dp_value_functions(inst: Instance, j: int, mu: np.ndarray) -> list[np.ndarray]:
    return dp_with_actions(inst, j, mu)[0]


@dataclass
class RelaxedSolution:
    """Both agent types, in the layouts the C++ files use."""
    inst: Instance
    agents: list[AgentLP]

    @property
    def lower_bound(self) -> float:
        return sum(a.objective for a in self.agents)

    def lower_bound_cxx(self, value_function: str = "dp") -> float:
        """getLowerBound (stdafx.cpp:1385): sum_i sum_j p0 . V_0 in the C++ loop order, for the V the MAI index uses."""
        total = 0.0
        for i in range(self.inst.area_num):
            for j in range(self.inst.type_num):
                p0, v0 = self.inst.init_probs[i][j], self.value_function(0, i, j, value_function)
                for s in range(p0.size):
                    total += float(p0[s]) * float(v0[s])
        return total

    def value_function(self, t: int, i: int, j: int, value_function: str = "dp") -> np.ndarray:
        a = self.agents[j]
        return a.values(value_function)[t][a.w_i[i]:a.w_i[i] + self.inst.state_num(i, j)]

    def multiplier(self, t: int, i: int, j: int) -> float:
        return float(self.agents[j].mu[t, i])

    @classmethod
    def from_store(cls, inst: Instance, path: str | Path) -> "RelaxedSolution":
        """Load a store_valueFuncs file (e.g. one the C++/CPLEX run wrote), the way main.cpp imports it.

        The file's V becomes the "lp" value function; "dp" is recomputed from the file's mu.
        """
        from .storefile import read_value_funcs
        Vf, muf = read_value_funcs(path, inst)
        agents = []
        for j in range(inst.type_num):
            S, w_i, W = _offsets(inst, j)
            V_file = [np.concatenate([Vf[t][i][j] for i in range(inst.area_num)]) for t in range(inst.maxtime)]
            if any(v.size != W for v in V_file):
                raise ValueError(f"{path}: value-function lines do not match the instance's state counts")
            mu = np.ascontiguousarray(muf[:, :, j])
            obj = sum(float(inst.init_probs[i][j] @ V_file[0][w_i[i]:w_i[i] + S[i]]) for i in range(inst.area_num))
            agents.append(AgentLP(j, obj, dp_value_functions(inst, j, mu), mu, None, w_i, 0, 0,
                                  f"loaded from {path}", 0, 0, 0, V_file))
        return cls(inst, agents)


def solve_relaxed(inst: Instance, verbose: bool = False, ranged_rows: bool = False) -> RelaxedSolution:
    return RelaxedSolution(inst, [build_and_solve(inst, j, verbose, ranged_rows) for j in range(inst.type_num)])
