"""Regression tests: the Python port must reproduce the C++ results exactly.

The C++ reference (tests/data) comes from tests/cpp_harness/make_reference.py: the shared main.cpp/stdafx.cpp,
built without CPLEX, run on store files that gen.py wrote. Simulation rows are compared at full double precision.

Run with `python3 -m pytest tests/` or, without pytest, `python3 tests/test_against_cpp.py`.
"""
from __future__ import annotations

import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data"
sys.path.insert(0, str(ROOT))

from patrol.cxx_compat import GlibcRand, heap_sort_cxx          # noqa: E402
from patrol.generate import generate_instance, init_probs_text, knowledge_set_lines   # noqa: E402
from patrol.gradients import gradients, slackness                # noqa: E402
from patrol.lp import RelaxedSolution, solve_relaxed             # noqa: E402
from patrol.sim import monte_carlo                               # noqa: E402
from patrol.storefile import read_value_funcs                    # noqa: E402

# "Solution value =" printed by CPLEX in cplexEquivalentProblem (both CPLEX runs agree), one per agent type
CPLEX_OBJECTIVES = {0: 371.653, 1: 3036.31}
SCALERS = (1, 2, 5)
_CACHE: dict = {}


def _instance():
    if "inst" not in _CACHE:
        _CACHE["inst"] = generate_instance(ROOT / "process/test2_alpha.in", ROOT / "process/test2_beta.in",
                                           ROOT / "graph/adjacent_matrix_10.in")
    return _CACHE["inst"]


STORE = DATA / "store_lp-seed395.out.gz"     # gen.py's LP solution, the one the C++ reference rows were run on


def _from_store() -> RelaxedSolution:
    if "sol" not in _CACHE:
        _CACHE["sol"] = RelaxedSolution.from_store(_instance(), STORE)
    return _CACHE["sol"]


def _cpp_rows() -> dict[int, list[float]]:
    rows = {}
    for line in (DATA / "cpp_reference.tsv").read_text().splitlines():
        scaler, *cols = line.rstrip("\t").split("\t")
        rows[int(scaler)] = [float(x) for x in cols]
    return rows


# ----------------------------------------------------------------------------- building blocks
def test_glibc_rand_matches_glibc():
    """glibc's rand() stream (reference values from tests/rand_ref.c built against glibc)."""
    expected = {1: [1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335],
                495: [1897024824, 2055002966, 631967504, 936588255, 1773238871, 619741372],
                595: [1540268827, 555748308, 114335029, 2023060965, 1550303082, 781625404],
                1394: [1209920109, 1736359695, 1870539949, 1523400179, 1102788765, 656086340],
                0: [1804289383, 846930886, 1681692777, 1714636915, 1957747793, 424238335]}
    for seed, values in expected.items():
        g = GlibcRand(seed)
        assert [g.rand() for _ in values] == values, seed
    g = GlibcRand(495)
    stream = [g.rand() for _ in range(100001)]
    assert stream[999] == 2003149886 and stream[100000] == 1913098969


def test_heap_sort_is_a_valid_sort():
    rnd = random.Random(1)
    for _ in range(50):
        xs = [rnd.randint(0, 20) for _ in range(rnd.randint(0, 40))]
        assert heap_sort_cxx(xs, lambda a, b: a > b) == sorted(xs)


# ----------------------------------------------------------------------------- instance and LP
def test_generated_instance_matches_cpp_log():
    """Knowledge sets, C parameters and initial agent placement, against the C++ run's own outputs."""
    inst = _instance()
    log = [line for line in (ROOT / "output.txt").read_text().splitlines() if line.startswith("knowledgeSets")]
    assert knowledge_set_lines(inst) == log
    assert init_probs_text(inst) == (ROOT / "test33/store/initProbs-scaler1-seed395.out").read_text()


def test_lp_objective_and_multipliers():
    inst = _instance()
    sol = solve_relaxed(inst)
    for j, ref in CPLEX_OBJECTIVES.items():
        assert abs(sol.agents[j].objective - ref) < 5e-3
    printed = float(re.search(r"lower_bound = ([0-9.]+)", (ROOT / "output.txt").read_text()).group(1))
    assert abs(sol.lower_bound - printed) < 5e-3
    _, mu = read_value_funcs(STORE, inst)
    for j in range(inst.type_num):
        assert np.abs(sol.agents[j].mu - mu[:, :, j]).max() < 1e-8


# ----------------------------------------------------------------------------- against the C++ binary
def test_slackness_matches_cpp():
    """main.cpp's gradients/slackness (argmin actions of its DP) for the stored mu, bit for bit."""
    assert slackness(_from_store()) == _cpp_rows()[1][9]


def test_simulation_matches_cpp():
    """main.cpp run on gen.py's store file, every column at full double precision."""
    sol = _from_store()
    for scaler in SCALERS:
        row = monte_carlo(_instance(), sol, scaler, slackness=slackness(sol))
        got = [row.greedy_avg, row.greedy_ci, row.greedy_adaptions, row.mai_avg, row.mai_ci, row.mai_adaptions,
               row.greedy_dev, row.mai_dev, row.lower_bound, row.slackness]
        assert got == _cpp_rows()[scaler], (scaler, got)


def _check_authors_build(store: str, reference: str):
    """The authors' own build (Linux, real CPLEX 22.2 and Boost) on an LP solution CPLEX wrote.

    Everything is bit-identical except the confidence intervals, whose t-quantile comes from Boost in the C++ and
    from SciPy here; the two differ in the last bit.
    """
    rows = {}
    for line in (DATA / reference).read_text().splitlines():
        if line and not line.startswith("#"):
            scaler, *cols = line.rstrip("\t").split("\t")
            rows[int(scaler)] = [float(x) for x in cols]
    sol = RelaxedSolution.from_store(_instance(), DATA / store)
    slack = slackness(sol)
    for scaler in SCALERS:
        row = monte_carlo(_instance(), sol, scaler, slackness=slack)
        got = [row.greedy_avg, row.greedy_ci, row.greedy_adaptions, row.mai_avg, row.mai_ci, row.mai_adaptions,
               row.greedy_dev, row.mai_dev, row.lower_bound, row.slackness]
        ref = rows[scaler]
        for k in (0, 2, 3, 5, 6, 7, 8, 9):
            assert got[k] == ref[k], (scaler, k, got[k], ref[k])
        for k in (1, 4):
            assert abs(got[k] - ref[k]) <= 1e-15 * abs(ref[k]), (scaler, k, got[k], ref[k])


def test_simulation_matches_authors_build_on_cplex_solution():
    """CPLEX 22.2 on Linux aarch64 (default settings)."""
    _check_authors_build("store_cplex22-seed395.out.gz", "cpp_reference_cplex.tsv")


def test_simulation_matches_authors_build_on_x86_cplex_solution():
    """CPLEX 22.2 on Linux x86-64 (default settings): the run that wrote output.txt."""
    _check_authors_build("store_cplex22-x86-seed395.out.gz", "cpp_reference_cplex_x86.tsv")


def test_gradients_match_cpp_log():
    """output.txt's "Gradients for t=..." and "slackness =" lines, as main.cpp prints them, from the same LP solution."""
    sol = RelaxedSolution.from_store(_instance(), DATA / "store_cplex22-x86-seed395.out.gz")
    g = gradients(sol)
    got = ["Gradients for t=%d: " % t + "".join(f"{x:g} " for x in g[t].ravel()) for t in range(len(g))]
    got.append(f"slackness = {slackness(sol, g):g}")
    log = (ROOT / "output.txt").read_text().splitlines()
    assert got == [line for line in log if line.startswith(("Gradients for", "slackness"))]


def test_trajectory_files_match_cpp():
    """simulation()'s outFlag output: 2 Monte-Carlo iterations at scaler 2."""
    with tempfile.TemporaryDirectory() as d:
        monte_carlo(_instance(), _from_store(), 2, iter_max=2, trajectory_dir=d)
        got = "".join(f"== {f.name}\n{f.read_text()}" for f in sorted(Path(d).glob("trajectory-*")))
    assert got == (DATA / "cpp_trajectory.txt").read_text()


if __name__ == "__main__":
    import inspect
    import time
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name}  [{time.time() - t0:.1f} s]", flush=True)
            except Exception as e:          # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}", flush=True)
    sys.exit(1 if fails else 0)
