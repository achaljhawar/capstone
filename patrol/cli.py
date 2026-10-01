"""Command-line options shared by gen.py, simulate.py and validate_lp.py."""
from __future__ import annotations

import argparse

from . import generate
from .instance import Instance, load_from_cpp_outputs
from .lp import VALUE_FUNCTIONS

ROW_BOUND_HELP = ("drop the C++'s -1e5 lower bound on every LP row: about 4x faster, same optimum and mu (up to "
                  'rounding), but a different LP vertex, so a different V off the support')

VALUE_FUNCTION_HELP = (
    'which V the MAI index reads. "lp" (default) is the LP vertex\'s V, what the C++ store file holds, so MAI\'s '
    'numbers depend on which solver and vertex produced it. "dp" is the Bellman recursion on the same multipliers '
    "(the paper's definition, Section III-A eq. 15): defined at every state and identical across solvers.")


def add_value_function_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--value-function", choices=VALUE_FUNCTIONS, default="lp", help=VALUE_FUNCTION_HELP)


def add_instance_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("instance (default: generate it as main.cpp does)")
    g.add_argument("--seed", type=int, default=generate.SEED, help="main.cpp's seed (instance and simulation)")
    g.add_argument("--alpha", default=generate.ALPHA_PATH)
    g.add_argument("--beta", default=generate.BETA_PATH)
    g.add_argument("--graph", default=generate.NEIGHBOURHOOD_PATH)
    g.add_argument("--areas", type=int, default=generate.AREA_NUM, help="AREA_NUM")
    g.add_argument("--from-cpp-log", metavar="LOG",
                   help="instead, rebuild the instance from a C++ run's stdout (knowledgeSets lines); needs --init-probs")
    g.add_argument("--init-probs", help="the C++ run's initProbs-scaler<h>-seed<seed>.out (with --from-cpp-log)")


def instance_from_args(args: argparse.Namespace) -> Instance:
    if args.from_cpp_log:
        if not args.init_probs:
            raise SystemExit("--from-cpp-log needs --init-probs")
        return load_from_cpp_outputs(args.from_cpp_log, args.init_probs, args.graph)
    return generate.generate_instance(args.alpha, args.beta, args.graph, seed=args.seed, area_num=args.areas)
