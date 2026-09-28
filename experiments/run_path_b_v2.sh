#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the cross-platform Python runner.
# Examples:
#   bash experiments/run_path_b_v2.sh
#   MODE=full bash experiments/run_path_b_v2.sh
#
# Optional:
#   PYTHON=python3
#   MODE=quick|full
#   OUTDIR=experiments/results/path_b_v2_local
#   SEEDS=5
#   SHUFFLES=100
#   RETENTION=0.40

PYTHON="${PYTHON:-python}"
MODE="${MODE:-quick}"
OUTDIR="${OUTDIR:-experiments/results/path_b_v2_local}"
RETENTION="${RETENTION:-0.40}"

ARGS=(
  -m experiments.run_path_b_v2
  --mode "$MODE"
  --outdir "$OUTDIR"
  --retention "$RETENTION"
)

if [[ -n "${SEEDS:-}" ]]; then
  ARGS+=(--seeds "$SEEDS")
fi

if [[ -n "${SHUFFLES:-}" ]]; then
  ARGS+=(--shuffles "$SHUFFLES")
fi

exec "$PYTHON" "${ARGS[@]}"
