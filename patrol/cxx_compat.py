"""Bit-exact re-implementations of glibc rand()/srand() and libstdc++ make_heap/pop_heap."""
from __future__ import annotations

from typing import Callable, TypeVar

RAND_MAX = 2147483647

T = TypeVar("T")


class GlibcRand:
    """glibc random_r.c, TYPE_3 (degree 31, separation 3), as used by rand()/srand()."""

    def __init__(self, seed: int):
        self.srand(seed)

    def srand(self, seed: int) -> None:
        seed &= 0xFFFFFFFF
        if seed == 0:
            seed = 1
        r = [0] * 34
        r[0] = seed
        word = seed
        for i in range(1, 31):
            hi, lo = word // 127773, word % 127773
            word = 16807 * lo - 2836 * hi
            if word < 0:
                word += 2147483647
            r[i] = word
        for i in range(31, 34):
            r[i] = r[i - 31]
        self._r = r
        self._i = 34
        for _ in range(310):            # glibc discards the first 310 outputs
            self._next_raw()

    def _next_raw(self) -> int:
        r = self._r
        i = self._i
        val = (r[i - 31] + r[i - 3]) & 0xFFFFFFFF
        r.append(val)
        self._i += 1
        if len(r) > 4096:               # keep the buffer bounded; only the last 31 entries matter
            del r[:-64]
            self._i = len(r)
        return val

    def rand(self) -> int:
        return self._next_raw() >> 1

    def uniform(self) -> float:
        """rand()*1.0/RAND_MAX, in the same double arithmetic."""
        return self.rand() * 1.0 / RAND_MAX


# ----------------------------------------------------------------------------- libstdc++ heap

def _push_heap(a: list, hole: int, top: int, value, comp: Callable) -> None:
    parent = (hole - 1) // 2
    while hole > top and comp(a[parent], value):
        a[hole] = a[parent]
        hole = parent
        parent = (hole - 1) // 2
    a[hole] = value


def _adjust_heap(a: list, hole: int, length: int, value, comp: Callable) -> None:
    top = hole
    second = hole
    while second < (length - 1) // 2:
        second = 2 * (second + 1)
        if comp(a[second], a[second - 1]):
            second -= 1
        a[hole] = a[second]
        hole = second
    if (length & 1) == 0 and second == (length - 2) // 2:
        second = 2 * (second + 1)
        a[hole] = a[second - 1]
        hole = second - 1
    _push_heap(a, hole, top, value, comp)


def make_heap(a: list, comp: Callable) -> None:
    n = len(a)
    if n < 2:
        return
    parent = (n - 2) // 2
    while True:
        _adjust_heap(a, parent, n, a[parent], comp)
        if parent == 0:
            return
        parent -= 1


def pop_heap(a: list, comp: Callable) -> None:
    """std::pop_heap(a.begin(), a.end(), comp): moves the top to a[-1], re-heaps a[:-1]."""
    n = len(a)
    if n < 2:
        return
    value = a[n - 1]
    a[n - 1] = a[0]
    _adjust_heap(a, 0, n - 1, value, comp)


def heap_sort_cxx(items: list, comp: Callable) -> list:
    """The C++ idiom:  make_heap; while (!empty) { pop_heap; out.push_back(back()); pop_back(); }"""
    a = list(items)
    make_heap(a, comp)
    out = []
    while a:
        pop_heap(a, comp)
        out.append(a.pop())
    return out
