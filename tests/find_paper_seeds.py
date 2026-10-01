"""Find the seeds whose generated instance reproduces the paper's Table II (Figure 4 settings).

Table II gives pi0_{i,j}(agent present) for Cases I (I=6), II (I=10) and III (I=14). main.cpp draws the agent
placement from glibc rand() seeded once per run, and the stream position when the placement loop starts depends
on the area count (the C_a1/C_a2/C_b loop consumes 3*I*J draws first). So one seed cannot give the paper's
settings for all three cases; each case has its own.

Replays only the rand() stream (not the full instance), so a seed costs microseconds.

Usage:  ~/patrol-venv/bin/python tests/find_paper_seeds.py [max_seed]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patrol.cxx_compat import GlibcRand
from patrol.model import EPSILON_PRECISION

SCALER_INIT = 20
TYPE_NUM = 2

# Table II: probability that a sub-area of area i holds a type-j agent at t=1, i.e. pi0(i(s)=1).
TABLE_II = {
    6: {  # Case I
        0: [0.1, 0.25, 0.1, 0.15, 0.2, 0.2],
        1: [0.5, 0.5, 0.65, 0.2, 0.45, 0.7],
    },
    10: {  # Case II
        0: [0.05, 0.1, 0.45, 0.15, 0.1, 0.2, 0.15, 0.4, 0.3, 0.1],
        1: [0.3, 0.6, 0.3, 0.55, 0.45, 0.75, 0.65, 0.6, 0.35, 0.45],
    },
    14: {  # Case III
        0: [0.25, 0.0, 0.15, 0.05, 0.15, 0.2, 0.15, 0.1, 0.1, 0.15, 0.25, 0.2, 0.1, 0.15],
        1: [0.35, 0.35, 0.25, 0.25, 0.35, 0.3, 0.45, 0.2, 0.35, 0.3, 0.35, 0.45, 0.45, 0.6],
    },
}
CASE_NAME = {6: "I", 10: "II", 14: "III"}


def placement(seed: int, area_num: int, type_num: int = TYPE_NUM) -> list[list[float]]:
    """p_agent[i][j] for this seed, replaying main.cpp's draws in order."""
    rng = GlibcRand(seed)
    for _ in range(area_num * type_num):          # C_a1, C_a2, C_b per (i, j)
        rng.rand(); rng.rand(); rng.rand()
    p_agent = [[0.0] * type_num for _ in range(area_num)]
    p_empty = [[1.0] * type_num for _ in range(area_num)]
    for j in range(type_num):
        agent_num = (rng.rand() % max(1, area_num // 2) + 1) * SCALER_INIT
        occupied = list(range(area_num))
        for _ in range(agent_num):
            k = rng.rand() % len(occupied)
            i = occupied[k]
            p_agent[i][j] += 1.0 / SCALER_INIT
            p_empty[i][j] = 1 - p_agent[i][j]
            if p_empty[i][j] < 1.0 / SCALER_INIT - EPSILON_PRECISION:
                occupied.pop(k)
    return p_agent


def matches(p_agent: list[list[float]], target: dict[int, list[float]], tol: float = 1e-9) -> bool:
    for j, col in target.items():
        for i, want in enumerate(col):
            if abs(p_agent[i][j] - want) > tol:
                return False
    return True


def main() -> int:
    max_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    print(f"searching seeds 1..{max_seed} for each case\n")
    found: dict[int, list[int]] = {k: [] for k in TABLE_II}
    for seed in range(1, max_seed + 1):
        for area_num, target in TABLE_II.items():
            if len(found[area_num]) < 3 and matches(placement(seed, area_num), target):
                found[area_num].append(seed)
                print(f"Case {CASE_NAME[area_num]:>3} (I={area_num:>2}): seed {seed}")
        if all(len(v) >= 1 for v in found.values()) and seed > 1000:
            break
    print()
    for area_num in sorted(TABLE_II):
        seeds = found[area_num] or ["NONE FOUND"]
        print(f"Case {CASE_NAME[area_num]:>3} (I={area_num:>2}): {seeds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
