"""The relaxed problem as a sparse LP (cplexEquivalentProblem_agent, stdafx.cpp:2191), solved with HiGHS."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .instance import Instance

VAR_BOUND = 1e5   # C++: minSols = -1e5, maxSols = 1e5 on every variable


@dataclass
class AgentLP:
    """LP solution for one agent type."""
    agent_type: int
    objective: float                       # optimal value (== contribution to the lower bound)
    V: list[np.ndarray]                    # V[t] is a concatenation over areas, area i at offset w_i[i], length W
    mu: np.ndarray                         # (T, N)
    policy: list[list[np.ndarray]]         # policy[t][i] shape (stateNum, |N(i)|+1), rows sum to 1
    w_i: np.ndarray                        # area offsets into V[t]
    randomized_states: int
    unreached_states: int
    status: str
    n_rows: int
    n_cols: int
    nnz: int
    V_lp: list[np.ndarray] | None = None   # raw LP vertex's V


def build_and_solve(inst: Instance, j: int, verbose: bool = False) -> AgentLP:
    T, N = inst.maxtime, inst.area_num
    procs = [inst.processes[i][j] for i in range(N)]
    S = [ap.state_num for ap in procs]
    w_i = np.concatenate([[0], np.cumsum(S)[:-1]]).astype(int)      # C++ weight_i
    W = int(sum(S))                                                     # C++ weight_t
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

    res = linprog(c, A_ub=A, b_ub=b, bounds=(-VAR_BOUND, VAR_BOUND), method="highs",
                  options={"disp": verbose})
    if res.status != 0:
        raise RuntimeError(f"HiGHS failed for agent type {j}: {res.message}")

    x = res.x
    V_lp = [x[t * W:(t + 1) * W].copy() for t in range(T)]
    mu = x[n_V:].reshape(T, N).copy()

    # --- occupation measures from the row duals
    duals = -res.ineqlin.marginals
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
            block[~reached, n_act - 1] = 1.0       # unreached: default to "no move", as the C++ does
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


def dp_value_functions(inst: Instance, j: int, mu: np.ndarray) -> list[np.ndarray]:
    """dynamicProgrammingWithGivenMultipliers_agent (stdafx.cpp:596): the Bellman recursion for fixed mu."""
    T, N = inst.maxtime, inst.area_num
    procs = [inst.processes[i][j] for i in range(N)]
    S = [ap.state_num for ap in procs]
    w_i = np.concatenate([[0], np.cumsum(S)[:-1]]).astype(int)
    W = int(sum(S))
    V = [np.zeros(W) for _ in range(T)]
    for t in range(T - 1, -1, -1):
        for i in range(N):
            ap, nb = procs[i], inst.neighbourhood[i]
            for s in range(S[i]):
                base = ap.cost[s // 2] + (s % 2) * mu[t, i]
                best = None
                for a in range(len(nb) + 1):
                    move_in = 1 if a < len(nb) else 0
                    v = base - (mu[t, nb[a]] if move_in else 0.0)
                    if t < T - 1:
                        v += sum(tr.prob * V[t + 1][w_i[i] + tr.s_next] for tr in ap.next_states(s, move_in))
                    if best is None or v < best:
                        best = v
                V[t][w_i[i] + s] = best
    return V


@dataclass
class RelaxedSolution:
    """Both agent types, in the layouts the C++ files use."""
    inst: Instance
    agents: list[AgentLP]

    @property
    def lower_bound(self) -> float:
        return sum(a.objective for a in self.agents)

    def value_function(self, t: int, i: int, j: int) -> np.ndarray:
        a = self.agents[j]
        return a.V[t][a.w_i[i]:a.w_i[i] + self.inst.state_num(i, j)]

    def multiplier(self, t: int, i: int, j: int) -> float:
        return float(self.agents[j].mu[t, i])


def solve_relaxed(inst: Instance, verbose: bool = False) -> RelaxedSolution:
    return RelaxedSolution(inst, [build_and_solve(inst, j, verbose) for j in range(inst.type_num)])
