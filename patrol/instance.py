"""A concrete problem instance: graph, per-(area, type) processes, initial distribution."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import AreaProcess

MAXTIME = 10   # stdafx.h


@dataclass
class Instance:
    area_num: int
    type_num: int
    adjacency: np.ndarray                          # (area_num, area_num) 0/1
    processes: list[list[AreaProcess]]             # [i][j]
    init_probs: list[list[np.ndarray]]             # [i][j] -> length stateNum, sums to 1
    neighbourhood: list[list[int]] = field(default_factory=list)   # [i] -> ordered list, self included
    maxtime: int = MAXTIME

    def __post_init__(self):
        # initNeighbourhood (stdafx.cpp:2847)
        self.neighbourhood = [
            [c for c in range(self.area_num) if self.adjacency[r, c] > 0 or c == r]
            for r in range(self.area_num)
        ]

    def state_num(self, i: int, j: int) -> int:
        return self.processes[i][j].state_num

    def var_num(self) -> int:
        """main.cpp's varNum: total number of (t, i, j, s) value-function entries."""
        return self.maxtime * sum(self.state_num(i, j) for i in range(self.area_num) for j in range(self.type_num))


# ---------------------------------------------------------------------------- loading
_KS_RE = re.compile(
    r"knowledgeSets\[(\d+)\]\[(\d+)\], C_a1=(\d+), C_a2=(\d+), C_b=(\d+): size=(\d+)\((\d+), (\d+)\)"
)
_IP_RE = re.compile(r"initProbs\[(\d+)\]\[(\d+)\]: \(I=0,prob=([0-9.eE+-]+)\), \(I=1,prob=([0-9.eE+-]+)\)")


def load_adjacency(path: str | Path) -> np.ndarray:
    rows = [list(map(int, line.split())) for line in Path(path).read_text().splitlines() if line.strip()]
    return np.array(rows, dtype=int)


def load_from_cpp_outputs(run_log: str | Path, init_probs_file: str | Path, graph_file: str | Path) -> Instance:
    """Rebuild the instance the C++ binary actually ran, from its own outputs."""
    params: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
    for line in Path(run_log).read_text().splitlines():
        m = _KS_RE.match(line)
        if m:
            i, j, ca1, ca2, cb, size, a0, b0 = map(int, m.groups())
            if size != 51:
                raise ValueError(f"unexpected knowledge set size {size} at [{i}][{j}]")
            params[(i, j)] = (ca1, ca2, cb, a0, b0)
    if not params:
        raise ValueError(f"no knowledgeSets lines found in {run_log}")
    area_num = 1 + max(i for i, _ in params)
    type_num = 1 + max(j for _, j in params)

    init: dict[tuple[int, int], tuple[float, float]] = {}
    for line in Path(init_probs_file).read_text().splitlines():
        m = _IP_RE.match(line)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            init[(i, j)] = (float(m.group(3)), float(m.group(4)))
    if len(init) != area_num * type_num:
        raise ValueError(f"expected {area_num*type_num} initProbs lines, found {len(init)}")

    adjacency = load_adjacency(graph_file)
    if adjacency.shape != (area_num, area_num):
        raise ValueError(f"graph is {adjacency.shape}, instance has {area_num} areas")

    processes, init_probs = [], []
    for i in range(area_num):
        row_p, row_ip = [], []
        for j in range(type_num):
            ca1, ca2, cb, a0, b0 = params[(i, j)]
            ap = AreaProcess(a0, b0, ca1, ca2, cb)
            row_p.append(ap)
            # main.cpp: state 0 = (knowledge 0, no agent), state 1 = (knowledge 0, agent); all others 0
            v = np.zeros(ap.state_num)
            v[0], v[1] = init[(i, j)]
            row_ip.append(v)
        processes.append(row_p)
        init_probs.append(row_ip)

    return Instance(area_num, type_num, adjacency, processes, init_probs)
