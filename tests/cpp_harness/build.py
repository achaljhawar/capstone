#!/usr/bin/env python3
"""Build the shared C++ (main.cpp + stdafx.cpp) without CPLEX or Boost, to check the Python port against it.

Usage: python3 tests/cpp_harness/build.py "<code for sharing dir>" <build dir> [--iter-max N] [--out-flag]

Changes made to the sources, none of which touch what the simulation computes:
  * CPLEX: buildModelByRow / cplexEquivalentProblem(_agent) are removed; main.cpp never calls them (it imports
    the LP solution from test23/store/store_valueFuncs-seed<seed>.out).
  * Boost: the one t-quantile main.cpp needs, t_{0.975, ITER_MAX-1}, is computed here with SciPy and inlined.
  * rand(): glibc's generator is force-included (glibc_rand.h), so macOS builds draw the same numbers as Linux.
  * the long-term-performance-cost.out row is written with 17 significant digits instead of 6.
  * --iter-max overrides ITER_MAX; --out-flag turns on simulation()'s per-step trajectory output.
Needs GNU g++ (libstdc++'s heap algorithms; on macOS use Homebrew's, e.g. CXX=g++-15), gmp and mpfr.
Built with -ffp-contract=off so arm64 does not fuse multiply-adds that an x86-64 g++ build would not.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent


def patch(path: Path, pairs: list[tuple[str, str]], regex: bool = False) -> None:
    text = path.read_text()
    for old, new in pairs:
        n = len(re.findall(old, text, flags=re.S)) if regex else text.count(old)
        if n != 1:
            raise SystemExit(f"{path.name}: expected one match for {old!r}, found {n}")
        text = re.sub(old, lambda _: new, text, flags=re.S) if regex else text.replace(old, new)
    path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--iter-max", type=int, default=1000)
    ap.add_argument("--out-flag", action="store_true")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for name in ("main.cpp", "stdafx.cpp", "stdafx.h"):
        shutil.copyfile(args.src / name, out / name)
    for d in ("graph", "process"):
        shutil.rmtree(out / d, ignore_errors=True)
        shutil.copytree(args.src / d, out / d)
    for d in ("test33/store", "test23/store"):
        (out / d).mkdir(parents=True, exist_ok=True)

    patch(out / "stdafx.h", [("#include <ilcplex/ilocplex.h>", ""),
                             ("#include <boost/math/distributions/students_t.hpp>", ""),
                             ("#define ITER_MAX 1000", f"#define ITER_MAX {args.iter_max}")])
    patch(out / "stdafx.cpp", [(r"void buildModelByRow\(.*?(?=int PatrollingProcess::initAreaProcesses)", "")],
          regex=True)
    quantile = repr(float(student_t.ppf(0.975, args.iter_max - 1)))
    main_patches = [("boost::math::students_t dist(ITER_MAX-1);", ""),
                    ("double T=quantile(complement(dist,0.025));", f"double T = {quantile};"),
                    ('    out << scaler << "\\t" << average1', '    out << std::setprecision(17) << scaler << "\\t" << average1')]
    if args.out_flag:
        main_patches += [("pp.simulation(&result1[iter],GREEDY,tmp_seed+seed +BASE_SIMULATION_SEED)",
                          "pp.simulation(&result1[iter],GREEDY,tmp_seed+seed +BASE_SIMULATION_SEED, true)"),
                         ("pp.simulation(&result2[iter],DP_INDEX,tmp_seed+seed+BASE_SIMULATION_SEED)",
                          "pp.simulation(&result2[iter],DP_INDEX,tmp_seed+seed+BASE_SIMULATION_SEED, true)")]
    patch(out / "main.cpp", main_patches)

    cxx, cc = os.environ.get("CXX", "g++"), os.environ.get("CC", "gcc")
    inc = ["-I/opt/homebrew/include"] if Path("/opt/homebrew/include").exists() else []
    lib = ["-L/opt/homebrew/lib"] if Path("/opt/homebrew/lib").exists() else []
    flags = ["-O2", "-ffp-contract=off", "-w"]
    run = lambda cmd: subprocess.run(cmd, cwd=out, check=True)
    run([cc, *flags, "-c", str(HERE / "glibc_rand.c"), "-o", "glibc_rand.o"])
    for name in ("main", "stdafx"):
        run([cxx, "--std=c++11", *flags, "-include", str(HERE / "glibc_rand.h"), *inc, "-c", f"{name}.cpp",
             "-o", f"{name}.o"])
    run([cxx, "main.o", "stdafx.o", "glibc_rand.o", *lib, "-lmpfr", "-lgmp", "-o", "edit"])
    print(f"built {out / 'edit'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
