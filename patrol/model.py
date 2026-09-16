"""Single-area process: knowledge states, crime probability, cost, and transitions (flat state s = 2*k + indicator)."""
from __future__ import annotations

from dataclasses import dataclass

# Constants from stdafx.h
MAX_SUM_ALPHA_BETA = 50      # every knowledge state is normalised to alpha + beta = 50
MIN_CRIME_RATE = 1e-2        # floor on p when alpha == 0
EPSILON_PRECISION = 1e-10
NORMALIZE_CRIME_RATE = 100   # cost = p * 100


def normalize(alpha: int, beta: int) -> tuple[int, int]:
    """Knowledge::normalization (stdafx.h): project (alpha, beta) onto alpha + beta = 50."""
    a = int(alpha * 1.0 / (alpha + beta) * MAX_SUM_ALPHA_BETA)
    return a, MAX_SUM_ALPHA_BETA - a


def crime_probability(alpha: int, beta: int) -> float:
    """Knowledge::probability (stdafx.h)."""
    p = alpha * 1.0 / (alpha + beta)
    return MIN_CRIME_RATE if p < EPSILON_PRECISION else p


@dataclass(frozen=True)
class Transition:
    """One outgoing transition: to flat state `s_next` with probability `prob`."""
    s_next: int
    prob: float


class AreaProcess:
    """One (area, agent-type) pair's Markov process."""

    def __init__(self, alpha0: int, beta0: int, c_a1: int, c_a2: int, c_b: int):
        if alpha0 + beta0 != MAX_SUM_ALPHA_BETA:
            raise ValueError(f"alpha0 + beta0 must be {MAX_SUM_ALPHA_BETA}, got {alpha0}+{beta0}")
        self.alpha0, self.beta0 = alpha0, beta0
        self.c_a1, self.c_a2, self.c_b = c_a1, c_a2, c_b

        # Knowledge set, in the C++ order (constructKnowledgeSets)
        self.knowledge: list[tuple[int, int]] = [(alpha0, beta0)]
        for a in range(MAX_SUM_ALPHA_BETA + 1):
            if a != alpha0:
                self.knowledge.append((a, MAX_SUM_ALPHA_BETA - a))
        self.K = len(self.knowledge)            # sizeOfKnowledgeSet (51)
        self.state_num = 2 * self.K             # stateNum (102)

        self.p = [crime_probability(a, b) for (a, b) in self.knowledge]
        self.cost = [pk * NORMALIZE_CRIME_RATE for pk in self.p]      # costRateAAPair, line 202
        self._transitions = {
            (s, move_in): self._next_states(s, move_in)
            for s in range(self.state_num) for move_in in (0, 1)
        }

    # ------------------------------------------------------------------ indexing
    def index_of(self, alpha: int, beta: int) -> int:
        """getKnowledgeIndexGivenAlphaBeta (line 209): normalise then look up."""
        a, _ = normalize(alpha, beta)
        if a == self.alpha0:
            return 0
        return a + 1 if a < self.alpha0 else a

    @staticmethod
    def flat(k: int, indicator: int) -> int:
        return 2 * k + indicator

    @staticmethod
    def unflat(s: int) -> tuple[int, int]:
        return s // 2, s % 2

    # --------------------------------------------------------------- transitions
    def _next_states(self, s: int, move_in: int) -> list[Transition]:
        """nextState (line 425) + transitionProbabilityArea (line 278)."""
        k, _ = self.unflat(s)
        alpha, beta = self.knowledge[k]
        p = self.p[k]
        if move_in:
            k1 = self.index_of(alpha, beta + self.c_b)
            k2 = self.index_of(max(0, alpha - self.c_a1), beta)
            ind = 1
        else:
            k1 = self.index_of(alpha + self.c_a2, beta)
            k2 = k
            ind = 0
        if k1 == k2:
            return [Transition(self.flat(k1, ind), 1.0)]
        return [Transition(self.flat(k1, ind), p), Transition(self.flat(k2, ind), 1.0 - p)]

    def next_states(self, s: int, move_in: int) -> list[Transition]:
        return self._transitions[(s, move_in)]

    # ----------------------------------------------------------------- display
    def knowledge_str(self) -> str:
        """Same text the C++ prints for knowledgeSets[i][j]."""
        return "".join(f"({a}, {b}) " for a, b in self.knowledge)
