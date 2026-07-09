#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BACKBONE="${BACKBONE:-gpt-4o}"
THRESHOLD="${THRESHOLD:-0.35}"
FULL_PLANNER_BUDGET="${FULL_PLANNER_BUDGET:-5}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RESUME="${RESUME:-0}"
SPLITS="${SPLITS:-drug_disjoint:test temporal_submit:test}"
RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")}"

BASE_LOG_DIR="${BASE_LOG_DIR:-results/frozen_runs/${RUN_TAG}/logs}"
PID_FILE="${PID_FILE:-results/frozen_runs/${RUN_TAG}/main.pids}"
MANIFEST="${MANIFEST:-results/frozen_runs/${RUN_TAG}/main_manifest.tsv}"

if [[ -z "${URL_VALUE:-}" || -z "${API_VALUE:-}" ]]; then
  echo "ERROR: URL_VALUE and API_VALUE must be exported before submitting frozen main runs." >&2
  exit 1
fi

mkdir -p "$(dirname "$PID_FILE")" "$BASE_LOG_DIR"
: > "$PID_FILE"
printf "kind\tsplit_prefix\tsplit_name\tbackbone\tthreshold\tpid\tbackground_log\n" > "$MANIFEST"

for split in $SPLITS; do
  split_prefix="${split%%:*}"
  split_name="${split##*:}"
  background_log="results/frozen_runs/${RUN_TAG}/main_${split_prefix}_${split_name}.background.log"
  pid_path="results/frozen_runs/${RUN_TAG}/main_${split_prefix}_${split_name}.pid"
  log_dir="${BASE_LOG_DIR}/main_${split_prefix}_${split_name}"

  echo "Submitting main run: ${split_prefix}_${split_name}"
  SPLIT_PREFIX="$split_prefix" \
  SPLIT_NAME="$split_name" \
  BACKBONE="$BACKBONE" \
  SAVE_EVERY="$SAVE_EVERY" \
  RESUME="$RESUME" \
  FULL_PLANNER_BUDGET="$FULL_PLANNER_BUDGET" \
  THRESHOLD="$THRESHOLD" \
  TREATAGENT_USE_LLM_PLANNER=1 \
  TREATAGENT_USE_LLM_EXPERTS=1 \
  TREATAGENT_LLM_EXPERTS=DrugKB,DiseaseKB,DTI,ADMET,Clinical \
  TREATAGENT_USE_LLM_JUDGE=1 \
  TREATAGENT_FORCE_ALL_EXPERTS=0 \
  TREATAGENT_USE_DERIVED_ARGUMENT_CLAIMS=1 \
  TREATAGENT_DISABLED_EXPERTS="" \
  LOG_DIR="$log_dir" \
  BACKGROUND_LOG="$background_log" \
  PID_FILE="$pid_path" \
  bash scripts/full/submit_treatagent_full.sh

  pid="$(cat "$pid_path")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "main" "$split_prefix" "$split_name" "$BACKBONE" "$THRESHOLD" "$pid" "$background_log" >> "$MANIFEST"
  printf "%s\t%s\t%s\n" "$pid" "main_${split_prefix}_${split_name}" "$background_log" >> "$PID_FILE"
done

echo "Submitted frozen main runs."
echo "PID table: $PID_FILE"
echo "Manifest: $MANIFEST"

