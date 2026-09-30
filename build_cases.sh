#!/bin/bash
# Build one ./edit_fixed per paper case (Appendix IX topologies, Table I/II settings).
#   Case I   I=6   graph/adjacent_matrix_test1.in  process/test1_alpha.in  test1_beta.in
#   Case II  I=10  graph/adjacent_matrix_10.in     process/test2_alpha.in  test2_beta.in
#   Case III I=14  graph/adjacent_matrix_14.in     process/test2_alpha.in  test2_beta.in
# Produces edit_case1 / edit_case2 / edit_case3, each reading test23_case<N>/store/.
set -e
cd "$(dirname "$0")"

C=/opt/ibm/ILOG/CPLEX_Studio222
CXXFLAGS="--std=c++11 -DIL_STD -O2 -I$C/cplex/include -I$C/concert/include"
LDFLAGS="-L$C/cplex/lib/x86-64_linux/static_pic -L$C/concert/lib/x86-64_linux/static_pic -lilocplex -lconcert -lcplex -lgmp -lmpfr -lm -lpthread -ldl"

build_case () {
  local n=$1 areas=$2 graph=$3 alpha=$4 beta=$5 seed=$6
  local d=build_case$n
  rm -rf "$d"; mkdir -p "$d"
  cp main_fixed.cpp stdafx_fixed.cpp stdafx_fixed.h "$d/"
  sed -i "s|int seed=395|int seed=$seed|" "$d/main_fixed.cpp"
  sed -i "s|#define AREA_NUM\t10|#define AREA_NUM\t$areas|" "$d/stdafx_fixed.h"
  sed -i "s|\"graph/adjacent_matrix_10.in\"|\"$graph\"|" "$d/stdafx_fixed.h"
  sed -i "s|\"process/test2_alpha.in\"|\"$alpha\"|" "$d/stdafx_fixed.h"
  sed -i "s|\"process/test2_beta.in\"|\"$beta\"|" "$d/stdafx_fixed.h"
  sed -i "s|#define PREFIX_INDEX_PATH\t\"test23/\"|#define PREFIX_INDEX_PATH\t\"test23_case$n/\"|" "$d/stdafx_fixed.h"
  sed -i "s|#define PREFIX \t\"test33/\"|#define PREFIX \t\"test33_case$n/\"|" "$d/stdafx_fixed.h"
  grep -E "define (AREA_NUM|_NEIGHBOURHOOD_PATH|_INIT_PROB_ALPHA|_INIT_PROB_BETA|PREFIX)" "$d/stdafx_fixed.h"
  ( cd "$d" && g++ $CXXFLAGS -c main_fixed.cpp && g++ $CXXFLAGS -c stdafx_fixed.cpp \
    && g++ main_fixed.o stdafx_fixed.o $LDFLAGS -o ../edit_case$n )
  mkdir -p test23_case$n/store test33_case$n/store test33_case$n/output
  echo "built edit_case$n (I=$areas)"
}

build_case 1 6  "graph/adjacent_matrix_test1.in" "process/test2_alpha.in" "process/test2_beta.in" 300
build_case 2 10 "graph/adjacent_matrix_10.in"    "process/test2_alpha.in" "process/test2_beta.in" 395
build_case 3 14 "graph/adjacent_matrix_14.in"    "process/test2_alpha.in" "process/test2_beta.in" 51

# After building, generate each case's LP solution with the Python port, including --policy
# (main_fixed.cpp reads store_lpPolicy-seed<seed>.out for the corrected slackness):
#
#   cd python
#   P=~/patrol-venv/bin/python; R=..
#   $P gen.py --areas 6  --seed 300 --policy --graph $R/graph/adjacent_matrix_test1.in \
#       --alpha $R/process/test2_alpha.in --beta $R/process/test2_beta.in --out-dir $R/test23_case1/store
#   $P gen.py --areas 10 --seed 395 --policy --graph $R/graph/adjacent_matrix_10.in ... --out-dir $R/test23_case2/store
#   $P gen.py --areas 14 --seed 51  --policy --graph $R/graph/adjacent_matrix_14.in ... --out-dir $R/test23_case3/store
#
# Then sweep Figure 4's h grid:
#   for h in 1 5 10 15 20 25 30 35 40 45 50; do ./edit_case1 $h < /dev/null > /dev/null; done
