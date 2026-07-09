#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
SOURCE_DIR="${SOURCE_DIR:-results/eg_scorer/${SPLIT_PREFIX}_eg}"
OUTPUT_DIR="${OUTPUT_DIR:-results/argument_ablation/${SPLIT_PREFIX}}"
REPORT_NAME="${REPORT_NAME:-argument_ablation_${SPLIT_PREFIX}}"

TRAIN_JSON="${TRAIN_JSON:-$(cat "${SOURCE_DIR}/${SPLIT_PREFIX}_train_result_path.txt")}"
VAL_JSON="${VAL_JSON:-$(cat "${SOURCE_DIR}/${SPLIT_PREFIX}_val_result_path.txt")}"
TEST_JSON="${TEST_JSON:-$(cat "${SOURCE_DIR}/${SPLIT_PREFIX}_test_result_path.txt")}"

python -m experiments.orchestration.argument_ablation \
  --train_json "$TRAIN_JSON" \
  --val_json "$VAL_JSON" \
  --test_json "$TEST_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --report_name "$REPORT_NAME"

echo "TreatAgent-ARG ablation completed for ${SPLIT_PREFIX}. Outputs: ${OUTPUT_DIR}"

