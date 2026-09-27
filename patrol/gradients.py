"""Gradients and slackness as main.cpp reports them (main.cpp:456-530, updateGradients_agent at stdafx.cpp:982).

The C++ takes the deterministic argmin actions of the Bellman recursion for the stored mu, propagates the
initial distribution under them, and reports gradient[t, i, j] = E[agent present] - E[agents moving in].
"slackness" (the last column of long-term-performance-cost.out) is sum gradient * mu. At an LP optimum many
actions tie exactly, so which one the argmin keeps depends on rounding; the value is a diagnostic, not a bound.
"""
from __future__ import annotations

import numpy as np

from .instance import Instance
from .lp import RelaxedSolution, dp_with_actions


def agent_gradients(inst: Instance, j: int, mu: np.ndarray) -> np.ndarray:
    """dynamicProgrammingWithGivenMultipliers_agent + updateGradients_agent for agent type j; returns (T, N)."""
    T, N = inst.maxtime, inst.area_num
    _, actions = dp_with_actions(inst, j, mu)
    nb = inst.neighbourhood
    g = np.zeros((T, N))
    prob_org = [[float(x) for x in inst.init_probs[i][j]] for i in range(N)]

    def gradient(t: int, i: int, probs: list[list[float]]) -> float:
        val = 0.0
        for s, p in enumerate(probs[i]):
            val += (s % 2) * p
        e_a = 0.0
        for i2 in nb[i]:
            e_a2 = 0.0
            for s2, p in enumerate(probs[i2]):
                a = actions[t][i2][s2]
                move_to = 1 if 0 <= a < len(nb[i2]) and nb[i2][a] == i else 0
                e_a2 += move_to * p
            e_a += e_a2
        return val - e_a

    for i in range(N):
        g[0, i] = gradient(0, i, prob_org)
    for t in range(1, T):
        probs = []
        for i in range(N):
            ap = inst.processes[i][j]
            nxt = [0.0] * ap.state_num
            for s in range(ap.state_num):
                move_in = 1 if actions[t - 1][i][s] < len(nb[i]) else 0
                for tr in ap.next_states(s, move_in):
                    nxt[tr.s_next] += prob_org[i][s] * tr.prob
            probs.append(nxt)
        for i in range(N):
            g[t, i] = gradient(t, i, probs)
        prob_org = probs
    return g


def gradients(sol: RelaxedSolution) -> np.ndarray:
    """All agent types, shape (T, N, J), as main.cpp's `gradients` array."""
    inst = sol.inst
    return np.stack([agent_gradients(inst, j, sol.agents[j].mu) for j in range(inst.type_num)], axis=2)


def slackness(sol: RelaxedSolution, grads: np.ndarray | None = None) -> float:
    """main.cpp's total_lagrange_cost: sum over (t, i, j), in that loop order, of gradient * mu."""
    inst = sol.inst
    grads = gradients(sol) if grads is None else grads
    total = 0.0
    for t in range(inst.maxtime):
        for i in range(inst.area_num):
            for j in range(inst.type_num):
                total += float(grads[t, i, j]) * sol.multiplier(t, i, j)
    return total
