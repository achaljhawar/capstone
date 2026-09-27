"""Instance generation from main.cpp (lines 20-187) and constructKnowledgeSets (stdafx.cpp:3085).

Uses the same glibc rand() stream as the C++, so for the same inputs and seed it builds the same instance:
transition parameters C_a1, C_a2, C_b per (area, type), initial knowledge from the alpha/beta files, and the
initial agent placement, with initProbs as the exact doubles the C++ computes.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .cxx_compat import GlibcRand
from .instance import MAXTIME, Instance
from .model import EPSILON_PRECISION, MAX_SUM_ALPHA_BETA, NORMALIZE_CRIME_RATE, AreaProcess, normalize

# main.cpp / stdafx.h defaults
SEED = 395
AREA_NUM = 10
AGENT_TYPE_NUM = 2
SCALER_INIT = 20
ALPHA_PATH = "process/test2_alpha.in"
BETA_PATH = "process/test2_beta.in"
NEIGHBOURHOOD_PATH = "graph/adjacent_matrix_10.in"

_DELIMS = re.compile(r"[ ,;\t\n]+")      # the strtok delimiters the C++ readers use


def _cxx_lines(path: str | Path) -> list[str]:
    """`while (!eof) { getline; if (eof) break; ... }`: a last line without a trailing newline is dropped."""
    return Path(path).read_text().split("\n")[:-1]


def _tokens(line: str) -> list[str]:
    return [tok for tok in _DELIMS.split(line) if tok]


def read_rows(path: str | Path, n_rows: int) -> list[list[float]]:
    """constructKnowledgeSets' reader: one row of numbers per agent type, at most n_rows rows."""
    rows = [[float(tok) for tok in _tokens(line)] for line in _cxx_lines(path)[:n_rows]]
    return rows + [[] for _ in range(n_rows - len(rows))]


def read_adjacency_cxx(path: str | Path, area_num: int) -> np.ndarray:
    """initNeighbourhood(std::string*)'s reader: at most area_num rows and columns; missing entries count as 0."""
    adj = np.zeros((area_num, area_num), dtype=int)
    for r, line in enumerate(_cxx_lines(path)[:area_num]):
        for c, tok in enumerate(_tokens(line)[:area_num]):
            adj[r, c] = int(float(tok))
    return adj


def initial_alpha(alpha: list[list[float]], beta: list[list[float]], i: int, j: int) -> int:
    """The first knowledge state of (i, j), normalised to alpha + beta = 50."""
    if i >= len(alpha[j]) or i >= len(beta[j]):
        a = MAX_SUM_ALPHA_BETA // 2
    else:
        prob = alpha[j][i] / (alpha[j][i] + beta[j][i]) * NORMALIZE_CRIME_RATE
        if prob > 1:
            prob = 1
        a = int(prob * MAX_SUM_ALPHA_BETA)
    return normalize(a, MAX_SUM_ALPHA_BETA - a)[0]


def generate_instance(alpha_path: str | Path = ALPHA_PATH, beta_path: str | Path = BETA_PATH,
                      graph_path: str | Path = NEIGHBOURHOOD_PATH, seed: int = SEED, area_num: int = AREA_NUM,
                      type_num: int = AGENT_TYPE_NUM, scaler_init: int = SCALER_INIT,
                      maxtime: int = MAXTIME) -> Instance:
    rng = GlibcRand(seed)

    c = {}
    for i in range(area_num):
        for j in range(type_num):
            c_a1 = rng.rand() % 5 + 2        # _GENERATE_TRANSITION_C_ALPHA1
            c_a2 = rng.rand() % 5 + 5        # _GENERATE_TRANSITION_C_ALPHA2
            c_b = rng.rand() % 5 + 1         # _GENERATE_TRANSITION_C_BETA
            c[i, j] = (c_a1, c_a2, c_b)

    alpha, beta = read_rows(alpha_path, type_num), read_rows(beta_path, type_num)
    processes = []
    for i in range(area_num):
        row = []
        for j in range(type_num):
            a0 = initial_alpha(alpha, beta, i, j)
            row.append(AreaProcess(a0, MAX_SUM_ALPHA_BETA - a0, *c[i, j]))
        processes.append(row)

    # initial proportion of sub-areas holding an agent; state 0 = (knowledge 0, no agent), 1 = (knowledge 0, agent)
    p_agent = [[0.0] * type_num for _ in range(area_num)]
    p_empty = [[1.0] * type_num for _ in range(area_num)]
    for j in range(type_num):
        agent_num = (rng.rand() % int(max(1, area_num // 2)) + 1) * scaler_init     # _GENERATE_AGENT_NUMBER
        occupied = list(range(area_num))
        for _ in range(agent_num):
            k = rng.rand() % len(occupied)
            i = occupied[k]
            p_agent[i][j] += 1.0 / scaler_init
            p_empty[i][j] = 1 - p_agent[i][j]
            if p_empty[i][j] < 1.0 / scaler_init - EPSILON_PRECISION:
                occupied.pop(k)

    init_probs = []
    for i in range(area_num):
        row = []
        for j in range(type_num):
            v = np.zeros(processes[i][j].state_num)
            v[0], v[1] = p_empty[i][j], p_agent[i][j]
            row.append(v)
        init_probs.append(row)

    return Instance(area_num, type_num, read_adjacency_cxx(graph_path, area_num), processes, init_probs,
                    maxtime=maxtime)


def knowledge_set_lines(inst: Instance) -> list[str]:
    """The "knowledgeSets[i][j], C_a1=..." lines main.cpp prints."""
    return [f"knowledgeSets[{i}][{j}], C_a1={ap.c_a1}, C_a2={ap.c_a2}, C_b={ap.c_b}: size={ap.K}{ap.knowledge_str()}"
            for i, row in enumerate(inst.processes) for j, ap in enumerate(row)]


def init_probs_text(inst: Instance) -> str:
    """The initProbs-scaler<h>-seed<seed>.out file main.cpp writes (default ostream precision, 6 digits)."""
    return "".join(f"initProbs[{i}][{j}]: (I=0,prob={inst.init_probs[i][j][0]:.6g}), "
                   f"(I=1,prob={inst.init_probs[i][j][1]:.6g})\n"
                   for i in range(inst.area_num) for j in range(inst.type_num))
