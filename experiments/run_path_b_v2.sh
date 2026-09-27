#!/usr/bin/env bash
set -euo pipefail

# One-command local validation for the Path-B v2 experiment.
# Usage:
#   bash experiments/run_path_b_v2.sh
#
# Optional environment overrides:
#   PYTHON=python
#   PYTEST=pytest
#   SEEDS=5
#   SHUFFLES=100
#   NJOBS=1

PYTHON="${PYTHON:-python}"
PYTEST="${PYTEST:-pytest}"
SEEDS="${SEEDS:-5}"
SHUFFLES="${SHUFFLES:-100}"
NJOBS="${NJOBS:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTDIR="experiments/results/path_b_v2_local"
mkdir -p "$OUTDIR"

echo "== [1/7] Focused engineering tests =="
"$PYTEST" -q   causationentropy/tests/test_path_b_v2_engineering.py   causationentropy/tests/test_path_b_benchmark_smoke.py   | tee "$OUTDIR/01_focused_tests.txt"

echo "== [2/7] Discovery + information regression =="
"$PYTEST" -q   causationentropy/tests/test_discovery.py   causationentropy/tests/core/information/test_conditional_mutual_information.py   | tee "$OUTDIR/02_regression_tests.txt"

echo "== [3/7] Screening frontier + stress diagnostics =="
"$PYTHON" -m experiments.path_b_screen_frontier_v2   --seeds "$SEEDS"   --n-shuffles "$SHUFFLES"   > "$OUTDIR/03_screen_frontier.json"

echo "== [4/7] Common-random-number path diagnostic =="
"$PYTHON" -m experiments.path_b_common_random_numbers \
  --seeds 3 \
  --n-shuffles 20 \
  > "$OUTDIR/04_common_random_numbers.json"

echo "== [5/7] End-to-end Gaussian benchmark =="
for seed in $(seq 0 $((SEEDS - 1))); do
  "$PYTHON" -m experiments.path_b_benchmark_v2     --case gaussian     --seed "$seed"     --retention 0.40     --n-shuffles "$SHUFFLES"     --n-jobs "$NJOBS"     > "$OUTDIR/05_gaussian_seed_${seed}.json"
done

echo "== [6/7] Nonlinear logistic + Poisson smoke =="
"$PYTHON" -m experiments.path_b_benchmark_v2   --case logistic   --seed 0   --retention 0.40   --n-shuffles "$SHUFFLES"   --n-jobs "$NJOBS"   > "$OUTDIR/06_logistic_seed_0.json"

"$PYTHON" -m experiments.path_b_benchmark_v2   --case poisson   --seed 0   --retention 0.40   --n-shuffles "$SHUFFLES"   --n-jobs "$NJOBS"   > "$OUTDIR/06_poisson_seed_0.json"

echo "== [7/7] Summarize =="
"$PYTHON" -m experiments.summarize_path_b_v2 "$OUTDIR"   | tee "$OUTDIR/06_summary.md"

echo
echo "PATH_B_V2_VALIDATION_COMPLETE=$OUTDIR"
