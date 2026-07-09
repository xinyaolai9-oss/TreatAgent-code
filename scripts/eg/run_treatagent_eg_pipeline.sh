#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BACKBONE="${BACKBONE:-local}"
METHOD="${METHOD:-multiagent}"
AGENT_VERSION="${AGENT_VERSION:-eg}"
SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
SAVE_EVERY="${SAVE_EVERY:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-results/eg_scorer}"

run_split() {
  local split_part="$1"
  local split_name="${SPLIT_PREFIX}_${split_part}"
  local json_path="data/benchmark/split_inputs/${split_name}.json"
  local result_marker="${OUTPUT_DIR}/${split_name}_result_path.txt"
  mkdir -p "$OUTPUT_DIR"

  if [[ ! -f "$json_path" ]]; then
    echo "ERROR: split input not found: $json_path" >&2
    exit 1
  fi

  echo "Running ${METHOD}/${BACKBONE} on ${json_path}"
  python -m treatagent.cli \
    --json_path "$json_path" \
    --method "$METHOD" \
    --backbone "$BACKBONE" \
    --agent_version "$AGENT_VERSION" \
    --save_every "$SAVE_EVERY" \
    --resume

  local latest_result
  latest_result="$(ls -t "results/${BACKBONE}"/results_${METHOD}_${AGENT_VERSION}_*.json | head -n 1)"
  echo "$latest_result" > "$result_marker"
  echo "Latest result for ${split_name}: ${latest_result}"

  python -m experiments.orchestration.build_graph_feature_table \
    "$latest_result" \
    --output_csv "${OUTPUT_DIR}/${split_name}_features.csv" \
    --output_jsonl "${OUTPUT_DIR}/${split_name}_features.jsonl"
}

run_split train
run_split val
run_split test

python -m experiments.orchestration.train_eg_scorer \
  --train_csv "${OUTPUT_DIR}/${SPLIT_PREFIX}_train_features.csv" \
  --val_csv "${OUTPUT_DIR}/${SPLIT_PREFIX}_val_features.csv" \
  --test_csv "${OUTPUT_DIR}/${SPLIT_PREFIX}_test_features.csv" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "treatagent_${AGENT_VERSION}_${BACKBONE}_${SPLIT_PREFIX}"

echo "TreatAgent-${AGENT_VERSION} pipeline completed for ${SPLIT_PREFIX}. Outputs: ${OUTPUT_DIR}"

