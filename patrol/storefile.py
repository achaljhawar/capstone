"""Read/write the store_valueFuncs / store_lpPolicy files in the layout main.cpp imports."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .instance import Instance
from .lp import RelaxedSolution


def write_value_funcs(sol: RelaxedSolution, path: str | Path) -> None:
    inst = sol.inst
    T, N, J = inst.maxtime, inst.area_num, inst.type_num
    lines = [f"{N} {J} {T} {inst.var_num()}"]
    for t in range(T):
        for i in range(N):
            for j in range(J):
                lines.append(" ".join(repr(float(v)) for v in sol.value_function(t, i, j)))
    for t in range(T):
        lines.append(" ".join(repr(sol.multiplier(t, i, j)) for i in range(N) for j in range(J)))
    Path(path).write_text("\n".join(lines) + "\n")


def write_policy(sol: RelaxedSolution, path: str | Path) -> None:
    inst = sol.inst
    T, N, J = inst.maxtime, inst.area_num, inst.type_num
    lines = [f"{N} {J} {T}"]
    for t in range(T):
        for i in range(N):
            for j in range(J):
                nb = len(inst.neighbourhood[i])
                block = sol.agents[j].policy[t][i]
                lines.append(f"{nb} {block.shape[0]} " + " ".join(repr(float(v)) for v in block.ravel()))
    Path(path).write_text("\n".join(lines) + "\n")


def read_value_funcs(path: str | Path, inst: Instance):
    """Parse a C++-format store file into (V[t][i][j] arrays, mu (T, N, J))."""
    lines = Path(path).read_text().splitlines()
    N, J, T, var_num = map(int, lines[0].split())
    if (N, J, T) != (inst.area_num, inst.type_num, inst.maxtime):
        raise ValueError(f"file header {(N, J, T)} != instance {(inst.area_num, inst.type_num, inst.maxtime)}")
    V = [[[None] * J for _ in range(N)] for _ in range(T)]
    k = 1
    for t in range(T):
        for i in range(N):
            for j in range(J):
                V[t][i][j] = np.array(list(map(float, lines[k].split())))
                k += 1
    mu = np.zeros((T, N, J))
    for t in range(T):
        vals = list(map(float, lines[k].split()))
        k += 1
        mu[t] = np.array(vals).reshape(N, J)
    return V, mu
