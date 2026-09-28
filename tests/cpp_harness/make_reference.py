#!/usr/bin/env python3
"""Regenerate tests/data (the C++ reference results the Python port is tested against).

Usage: CXX=g++-15 python3 tests/cpp_harness/make_reference.py "<code for sharing dir>" [--work /tmp/cpp-ref]

1. gen.py writes the store file; it is kept, gzipped, in tests/data/store_lp-seed395.out.gz.
2. The C++ (built by build.py) runs on it for every scaler in SCALERS; its 17-digit rows (the 11 columns of
   long-term-performance-cost.out, starting with the scaler) go to tests/data/cpp_reference.tsv.
3. A second build with ITER_MAX=2 and outFlag on writes the trajectory files for scaler 2; they are concatenated
   into tests/data/cpp_trajectory.txt.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data"
SEED = 395
SCALERS = (1, 2, 5)
TRAJECTORY_SCALER, TRAJECTORY_ITERS = 2, 2


def sh(cmd, cwd=None, quiet=False):
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True, stdout=subprocess.DEVNULL if quiet else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help='the unpacked "code for sharing" directory')
    ap.add_argument("--work", type=Path, default=Path("/tmp/cpp-ref"))
    args = ap.parse_args()

    build, build_traj = args.work / "build", args.work / "build-trajectory"
    sh([sys.executable, ROOT / "tests/cpp_harness/build.py", args.src, build])
    sh([sys.executable, ROOT / "tests/cpp_harness/build.py", args.src, build_traj,
        "--iter-max", TRAJECTORY_ITERS, "--out-flag"])

    DATA.mkdir(parents=True, exist_ok=True)
    store_name = f"store_valueFuncs-seed{SEED}.out"
    sh([sys.executable, ROOT / "gen.py", "--seed", SEED, "--out-dir", args.work / "store"], cwd=ROOT)
    store = args.work / "store" / store_name
    (DATA / f"store_lp-seed{SEED}.out.gz").write_bytes(gzip.compress(store.read_bytes(), mtime=0))
    shutil.copyfile(store, build / "test23/store" / store_name)
    perf = build / "test33/long-term-performance-cost.out"
    perf.unlink(missing_ok=True)
    for scaler in SCALERS:
        sh([build / "edit", scaler], cwd=build, quiet=True)
    rows = [line.rstrip() for line in perf.read_text().splitlines() if line.strip()]
    (DATA / "cpp_reference.tsv").write_text("\n".join(rows) + "\n")

    shutil.copyfile(store, build_traj / "test23/store" / store_name)
    for f in (build_traj / "test33/store").glob("trajectory-*"):
        f.unlink()
    sh([build_traj / "edit", TRAJECTORY_SCALER], cwd=build_traj, quiet=True)
    parts = [f"== {f.name}\n{f.read_text()}" for f in sorted((build_traj / "test33/store").glob("trajectory-*"))]
    (DATA / "cpp_trajectory.txt").write_text("".join(parts))
    print(f"wrote {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
