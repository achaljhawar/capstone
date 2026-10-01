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


# ---------------------------------------------------------------------------------------------
# The same gradients under the LP's own randomized policy.
#
# Everything above reproduces main.cpp exactly: it takes the deterministic argmin of the Bellman
# recursion, which at an LP optimum breaks ties arbitrarily, so the residual it reports is not the
# complementary-slackness residual of the LP. main_fixed.cpp routes around that by recovering the
# primal occupation measures from the solver and computing the gradient under the resulting
# randomized policy; these functions are the Python side of the same idea, using AgentLP.policy.
#
# Their purpose is to check Proposition 2 (equation (23), p. 13), which asserts that a mixture of
# threshold policies attains the relaxed optimum -- i.e. that this residual is zero. They are new
# code, not a change to anything above, so the bit-exact reproduction of main.cpp is untouched.
# ---------------------------------------------------------------------------------------------


def agent_gradients_randomized(inst: Instance, j: int, policy) -> np.ndarray:
    """agent_gradients, but every action is taken with the LP's probability rather than argmin.

    policy[t][i] has shape (state_num, |N(i)| + 1) with rows summing to 1. Column a < |N(i)| is
    "an agent moves into i from N(i)[a]"; the last column is "no agent moves in".
    """
    T, N = inst.maxtime, inst.area_num
    nb = inst.neighbourhood
    g = np.zeros((T, N))
    prob_org = [[float(x) for x in inst.init_probs[i][j]] for i in range(N)]

    def gradient(t: int, i: int, probs: list[list[float]]) -> float:
        val = sum((s % 2) * p for s, p in enumerate(probs[i]))
        e_a = 0.0
        for i2 in nb[i]:
            if i not in nb[i2]:
                continue
            a = nb[i2].index(i)                       # the action "i2 pulls an agent from i"
            block = policy[t][i2]
            e_a += sum(p * float(block[s2][a]) for s2, p in enumerate(probs[i2]))
        return val - e_a

    for i in range(N):
        g[0, i] = gradient(0, i, prob_org)
    for t in range(1, T):
        probs = []
        for i in range(N):
            ap = inst.processes[i][j]
            block = policy[t - 1][i]
            nxt = [0.0] * ap.state_num
            for s in range(ap.state_num):
                p_in = float(block[s][:len(nb[i])].sum())      # any incoming action
                for move_in, weight in ((1, p_in), (0, 1.0 - p_in)):
                    if weight == 0.0:
                        continue
                    for tr in ap.next_states(s, move_in):
                        nxt[tr.s_next] += prob_org[i][s] * weight * tr.prob
            probs.append(nxt)
        for i in range(N):
            g[t, i] = gradient(t, i, probs)
        prob_org = probs
    return g


def gradients_randomized(sol: RelaxedSolution) -> np.ndarray:
    """All agent types, under the LP's randomized policy. Shape (T, N, J)."""
    inst = sol.inst
    if any(sol.agents[j].policy is None for j in range(inst.type_num)):
        raise ValueError("this solution carries no randomized policy; solve the LP rather than "
                         "loading a store file, which holds only V and mu")
    return np.stack([agent_gradients_randomized(inst, j, sol.agents[j].policy)
                     for j in range(inst.type_num)], axis=2)


def slackness_randomized(sol: RelaxedSolution, grads: np.ndarray | None = None) -> float:
    """sum gradient * mu under the LP's randomized policy: the residual Proposition 2 says is zero."""
    inst = sol.inst
    grads = gradients_randomized(sol) if grads is None else grads
    total = 0.0
    for t in range(inst.maxtime):
        for i in range(inst.area_num):
            for j in range(inst.type_num):
                total += float(grads[t, i, j]) * sol.multiplier(t, i, j)
    return total
