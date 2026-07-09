#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
EG_DIR="${EG_DIR:-results/eg_scorer/${SPLIT_PREFIX}_eg}"
OUTPUT_DIR="${OUTPUT_DIR:-results/eg_scorer/${SPLIT_PREFIX}_dropout_eg}"
MODEL_NAME="${MODEL_NAME:-treatagent_eg_dropout_${SPLIT_PREFIX}}"
RANDOM_REPEATS="${RANDOM_REPEATS:-2}"
MAX_DROP="${MAX_DROP:-2}"
MIN_KEEP="${MIN_KEEP:-2}"
SEED="${SEED:-13}"

TRAIN_RESULT_PATH="${TRAIN_RESULT_PATH:-}"
if [[ -z "$TRAIN_RESULT_PATH" ]]; then
  TRAIN_RESULT_PATH="$(cat "$EG_DIR/${SPLIT_PREFIX}_train_result_path.txt")"
fi

VAL_CSV="${VAL_CSV:-${EG_DIR}/${SPLIT_PREFIX}_val_features.csv}"
TEST_CSV="${TEST_CSV:-${EG_DIR}/${SPLIT_PREFIX}_test_features.csv}"
TRAIN_DROPOUT_CSV="${TRAIN_DROPOUT_CSV:-${OUTPUT_DIR}/${SPLIT_PREFIX}_train_dropout_features.csv}"
TRAIN_DROPOUT_JSONL="${TRAIN_DROPOUT_JSONL:-${OUTPUT_DIR}/${SPLIT_PREFIX}_train_dropout_features.jsonl}"

echo "Evidence-dropout EG scorer"
echo "Root: $ROOT_DIR"
echo "Split: $SPLIT_PREFIX"
echo "Train result JSON: $TRAIN_RESULT_PATH"
echo "Val CSV: $VAL_CSV"
echo "Test CSV: $TEST_CSV"
echo "Output dir: $OUTPUT_DIR"

python -m experiments.orchestration.build_dropout_feature_table \
  "$TRAIN_RESULT_PATH" \
  --output_csv "$TRAIN_DROPOUT_CSV" \
  --output_jsonl "$TRAIN_DROPOUT_JSONL" \
  --random_repeats "$RANDOM_REPEATS" \
  --max_drop "$MAX_DROP" \
  --min_keep "$MIN_KEEP" \
  --seed "$SEED"

python -m experiments.orchestration.train_eg_scorer \
  --train_csv "$TRAIN_DROPOUT_CSV" \
  --val_csv "$VAL_CSV" \
  --test_csv "$TEST_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME"

