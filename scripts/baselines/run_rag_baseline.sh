#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
SPLIT_NAME="${SPLIT_NAME:-test}"
DATA_PATH="${DATA_PATH:-data/benchmark/split_inputs/${SPLIT_PREFIX}_${SPLIT_NAME}.json}"
BACKBONE="${BACKBONE:-gpt-4o}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RESUME="${RESUME:-1}"
LOG_DIR="${LOG_DIR:-results/rag_baseline_logs}"
KNOWLEDGE_CUTOFF_DATE="${KNOWLEDGE_CUTOFF_DATE:-}"

mkdir -p "$LOG_DIR"

timestamp="$(date +"%Y%m%d_%H%M%S")"
log_file="$LOG_DIR/${timestamp}_${BACKBONE}_${SPLIT_PREFIX}_${SPLIT_NAME}_rag.log"

echo "RAG baseline"
echo "Root: $ROOT_DIR"
echo "Backbone: $BACKBONE"
echo "Data: $DATA_PATH"
echo "Log: $log_file"

if [[ -z "${URL_VALUE:-}" || -z "${API_VALUE:-}" ]]; then
  echo "ERROR: URL_VALUE and API_VALUE must be exported in this WSL shell before running RAG baseline." >&2
  exit 1
fi

args=(
  "--json_path" "$DATA_PATH"
  "--method" "rag"
  "--backbone" "$BACKBONE"
  "--save_every" "$SAVE_EVERY"
)

if [[ "$RESUME" == "1" ]]; then
  args+=("--resume")
fi

if [[ -n "$KNOWLEDGE_CUTOFF_DATE" ]]; then
  args+=("--knowledge_cutoff_date" "$KNOWLEDGE_CUTOFF_DATE")
fi

python -m treatagent.cli "${args[@]}" 2>&1 | tee "$log_file"

