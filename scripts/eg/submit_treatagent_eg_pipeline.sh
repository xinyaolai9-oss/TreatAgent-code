#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BACKBONE="${BACKBONE:-local}"
AGENT_VERSION="${AGENT_VERSION:-eg}"
SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
OUTPUT_DIR="${OUTPUT_DIR:-results/eg_scorer/${SPLIT_PREFIX}_${AGENT_VERSION}}"
SAVE_EVERY="${SAVE_EVERY:-10}"
LOG_FILE="${LOG_FILE:-results/eg_scorer_${SPLIT_PREFIX}_${AGENT_VERSION}_background.log}"
PID_FILE="${PID_FILE:-results/eg_scorer_${SPLIT_PREFIX}_${AGENT_VERSION}.pid}"
CONDA_ENV="${CONDA_ENV:-myenv}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

mkdir -p "$(dirname "$OUTPUT_DIR")" "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "Existing EG pipeline is still running: pid=$old_pid"
    echo "Log: $LOG_FILE"
    exit 0
  fi
fi

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

nohup env \
  BACKBONE="$BACKBONE" \
  AGENT_VERSION="$AGENT_VERSION" \
  SPLIT_PREFIX="$SPLIT_PREFIX" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  SAVE_EVERY="$SAVE_EVERY" \
  bash scripts/eg/run_treatagent_eg_pipeline.sh \
  > "$LOG_FILE" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$PID_FILE"

echo "Submitted TreatAgent-${AGENT_VERSION} EG pipeline for ${SPLIT_PREFIX}"
echo "PID: $pid"
echo "Log: $LOG_FILE"
echo "Output: $OUTPUT_DIR"

