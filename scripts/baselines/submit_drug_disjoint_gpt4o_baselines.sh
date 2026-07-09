#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_PATH="${DATA_PATH:-data/benchmark/split_inputs/drug_disjoint_test.json}"
BACKBONE="${BACKBONE:-gpt-4o}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RESUME="${RESUME:-1}"
LOG_DIR="${LOG_DIR:-results/baseline_drug_disjoint_logs}"
BACKGROUND_LOG="${BACKGROUND_LOG:-results/baseline_drug_disjoint_gpt4o_background.log}"
PID_FILE="${PID_FILE:-results/baseline_drug_disjoint_gpt4o.pid}"
KNOWLEDGE_CUTOFF_DATE="${KNOWLEDGE_CUTOFF_DATE:-}"

if [[ -z "${URL_VALUE:-}" || -z "${API_VALUE:-}" ]]; then
  echo "ERROR: URL_VALUE and API_VALUE must be exported in this WSL shell before submitting baselines." >&2
  exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: data file not found: $DATA_PATH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$(dirname "$BACKGROUND_LOG")"

run_one() {
  local method="$1"
  local agent_version="${2:-eg}"
  local timestamp
  local log_file
  timestamp="$(date +"%Y%m%d_%H%M%S")"
  log_file="$LOG_DIR/${timestamp}_${BACKBONE}_${method}"
  if [[ "$method" == "multiagent" ]]; then
    log_file="${log_file}_${agent_version}"
  fi
  log_file="${log_file}.log"

  echo "============================================================"
  echo "Starting baseline"
  echo "Backbone: $BACKBONE"
  echo "Method: $method"
  if [[ "$method" == "multiagent" ]]; then
    echo "Agent version: $agent_version"
  fi
  echo "Data: $DATA_PATH"
  echo "Log: $log_file"
  echo "============================================================"

  local args=(
    "--json_path" "$DATA_PATH"
    "--method" "$method"
    "--backbone" "$BACKBONE"
    "--save_every" "$SAVE_EVERY"
  )

  if [[ "$RESUME" == "1" ]]; then
    args+=("--resume")
  fi

  if [[ -n "$KNOWLEDGE_CUTOFF_DATE" ]]; then
    args+=("--knowledge_cutoff_date" "$KNOWLEDGE_CUTOFF_DATE")
  fi

  if [[ "$method" == "multiagent" ]]; then
    args+=("--agent_version" "$agent_version")
  fi

  python -m treatagent.cli "${args[@]}" 2>&1 | tee "$log_file"
}

(
  echo "Baseline batch"
  echo "Root: $ROOT_DIR"
  echo "Backbone: $BACKBONE"
  echo "Data: $DATA_PATH"
  echo "Logs: $LOG_DIR"
  echo

  run_one direct
  run_one cot
  run_one rag

  echo
  echo "All requested baselines completed."
) > "$BACKGROUND_LOG" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"
echo "Submitted baseline batch."
echo "PID: $pid"
echo "Background log: $BACKGROUND_LOG"
echo "PID file: $PID_FILE"

