"""Read/write the store_valueFuncs / store_lpPolicy files in the layout main.cpp imports.

The C++ format records only the dimensions, so write_value_funcs also writes a JSON sidecar
(<store>.instance.json, without any .gz) holding the instance's fingerprint, and read_value_funcs refuses a file
whose sidecar names a different instance. Files without a sidecar (e.g. ones the C++ wrote) still load.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from .instance import Instance
from .lp import RelaxedSolution


def _read_text(path: str | Path) -> str:
    path = Path(path)
    return gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()


def sidecar_path(path: str | Path) -> Path:
    """store_valueFuncs-seed395.out(.gz) -> store_valueFuncs-seed395.out.instance.json"""
    path = Path(path)
    base = path.with_suffix("") if path.suffix == ".gz" else path
    return base.with_name(base.name + ".instance.json")


def check_instance(path: str | Path, inst: Instance) -> bool | None:
    """True if the store's sidecar matches inst, None if there is no sidecar; raises ValueError on a mismatch."""
    side = sidecar_path(path)
    if not side.exists():
        return None
    recorded = json.loads(side.read_text()).get("instance_sha256")
    if recorded != inst.fingerprint():
        raise ValueError(f"{path} was written for a different instance ({side.name} records {recorded}, "
                         f"this instance is {inst.fingerprint()}); check --seed, --areas, --graph, --alpha, --beta")
    return True


def write_value_funcs(sol: RelaxedSolution, path: str | Path, value_function: str = "lp") -> None:
    """Header, then V for every (t, i, j), then mu one line per t; plus the instance sidecar."""
    inst = sol.inst
    T, N, J = inst.maxtime, inst.area_num, inst.type_num
    lines = [f"{N} {J} {T} {inst.var_num()}"]
    for t in range(T):
        for i in range(N):
            for j in range(J):
                lines.append(" ".join(repr(float(v)) for v in sol.value_function(t, i, j, value_function)))
    for t in range(T):
        lines.append(" ".join(repr(sol.multiplier(t, i, j)) for i in range(N) for j in range(J)))
    Path(path).write_text("\n".join(lines) + "\n")
    sidecar_path(path).write_text(json.dumps({"instance_sha256": inst.fingerprint(), "areas": N, "agent_types": J,
                                              "maxtime": T, "value_function": value_function}, indent=2) + "\n")


def write_policy(sol: RelaxedSolution, path: str | Path) -> None:
    """The LP's randomized policy. A Python-only extra: main.cpp never reads this file."""
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
    """Parse a C++-format store file (plain or .gz) into (V[t][i][j] arrays, mu (T, N, J)).

    Checks the layout against the instance line by line, and the instance itself against the sidecar if there
    is one (see check_instance)."""
    lines = _read_text(path).splitlines()
    N, J, T, var_num = map(int, lines[0].split())
    if (N, J, T) != (inst.area_num, inst.type_num, inst.maxtime):
        raise ValueError(f"file header {(N, J, T)} != instance {(inst.area_num, inst.type_num, inst.maxtime)}")
    if var_num != inst.var_num():
        raise ValueError(f"{path}: header declares {var_num} value-function entries, the instance has {inst.var_num()}")
    if len(lines) < 1 + T * N * J + T:
        raise ValueError(f"{path}: {len(lines)} lines, expected {1 + T * N * J + T}")
    check_instance(path, inst)
    V = [[[None] * J for _ in range(N)] for _ in range(T)]
    k = 1
    for t in range(T):
        for i in range(N):
            for j in range(J):
                V[t][i][j] = np.array(list(map(float, lines[k].split())))
                if V[t][i][j].size != inst.state_num(i, j):
                    raise ValueError(f"{path}: line {k + 1} (t={t}, i={i}, j={j}) has {V[t][i][j].size} values, "
                                     f"expected {inst.state_num(i, j)}")
                k += 1
    mu = np.zeros((T, N, J))
    for t in range(T):
        vals = list(map(float, lines[k].split()))
        if len(vals) != N * J:
            raise ValueError(f"{path}: multiplier line {k + 1} (t={t}) has {len(vals)} values, expected {N * J}")
        k += 1
        mu[t] = np.array(vals).reshape(N, J)
    if not all(np.isfinite(v).all() for row in V for cell in row for v in cell) or not np.isfinite(mu).all():
        raise ValueError(f"{path}: contains a non-finite value")
    return V, mu
