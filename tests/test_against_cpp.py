"""Regression tests: the Python port must reproduce the C++ results exactly."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from patrol.cxx_compat import GlibcRand, heap_sort_cxx          # noqa: E402
from patrol.instance import load_from_cpp_outputs                # noqa: E402
from patrol.lp import dp_value_functions, solve_relaxed          # noqa: E402
from patrol.sim import monte_carlo                               # noqa: E402
from patrol.storefile import read_value_funcs                    # noqa: E402

_INST = None
_SOL = None


def _instance():
    global _INST
    if _INST is None:
        _INST = load_from_cpp_outputs(ROOT / "output.txt",
                                      ROOT / "test33/store/initProbs-scaler1-seed395.out",
                                      ROOT / "graph/adjacent_matrix_10.in")
    return _INST


def _solution():
    global _SOL
    if _SOL is None:
        _SOL = solve_relaxed(_instance())
    return _SOL


def _cpp_rows():
    rows = {}
    for line in (ROOT / "test33/long-term-performance-cost.out").read_text().splitlines():
        f = line.split("\t")
        if f and f[0].strip():
            rows[int(f[0])] = f[:10]
    return rows


# ----------------------------------------------------------------------------- tests
def test_glibc_rand_matches_reference():
    """Compare against the C reference stream (needs gcc)."""
    src = ROOT / "python/tests/rand_ref.c"
    exe = Path("/tmp/rand_ref_test")
    subprocess.run(["gcc", str(src), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines()
    for seed in (495, 595, 1394, 0):
        g = GlibcRand(seed)
        assert f"{seed}: " + " ".join(str(g.rand()) for _ in range(6)) in out
    g = GlibcRand(495)
    for _ in range(999):
        g.rand()
    assert f"495@1000: {g.rand()}" in out
    for _ in range(99000):
        g.rand()
    assert f"495@100001: {g.rand()}" in out


def test_heap_sort_is_a_valid_sort():
    import random
    rnd = random.Random(1)
    for _ in range(50):
        xs = [rnd.randint(0, 20) for _ in range(rnd.randint(0, 40))]
        out = heap_sort_cxx(xs, lambda a, b: a > b)
        assert out == sorted(xs)


def test_knowledge_sets_match_cpp_log():
    inst = _instance()
    import re
    txt = (ROOT / "output.txt").read_text().splitlines()
    n = 0
    for line in txt:
        m = re.match(r"knowledgeSets\[(\d+)\]\[(\d+)\].*size=\d+(.*)$", line)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            assert inst.processes[i][j].knowledge_str().strip() == m.group(3).strip()
            n += 1
    assert n == 20


def test_lp_objective_matches_cplex():
    sol = _solution()
    assert abs(sol.agents[0].objective - 371.653) < 5e-3
    assert abs(sol.agents[1].objective - 3036.31) < 5e-3
    assert abs(sol.lower_bound - 3407.97) < 5e-3


def test_multipliers_match_cplex():
    inst, sol = _instance(), _solution()
    _, mu_c = read_value_funcs(ROOT / "test23/store_cplex/store_valueFuncs-seed395.out", inst)
    for j in range(inst.type_num):
        assert np.abs(sol.agents[j].mu - mu_c[:, :, j]).max() < 1e-8


def test_value_functions_equal_dp_of_cplex_mu():
    inst, sol = _instance(), _solution()
    _, mu_c = read_value_funcs(ROOT / "test23/store_cplex/store_valueFuncs-seed395.out", inst)
    for j in range(inst.type_num):
        V_dp = dp_value_functions(inst, j, mu_c[:, :, j])
        assert max(np.abs(sol.agents[j].V[t] - V_dp[t]).max() for t in range(inst.maxtime)) < 1e-8


def test_simulation_bit_exact_scaler_1():
    inst, sol = _instance(), _solution()
    row = monte_carlo(inst, sol, 1)
    assert row.as_line().split("\t")[:10] == _cpp_rows()[1]


def test_simulation_bit_exact_scaler_2():
    inst, sol = _instance(), _solution()
    row = monte_carlo(inst, sol, 2)
    assert row.as_line().split("\t")[:10] == _cpp_rows()[2]


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:          # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
