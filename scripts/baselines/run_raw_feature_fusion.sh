#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
FEATURE_DIR="${FEATURE_DIR:-results/eg_scorer/${SPLIT_PREFIX}_eg}"
OUTPUT_DIR="${OUTPUT_DIR:-results/raw_feature_fusion/${SPLIT_PREFIX}}"

TRAIN_CSV="${TRAIN_CSV:-${FEATURE_DIR}/${SPLIT_PREFIX}_train_features.csv}"
VAL_CSV="${VAL_CSV:-${FEATURE_DIR}/${SPLIT_PREFIX}_val_features.csv}"
TEST_CSV="${TEST_CSV:-${FEATURE_DIR}/${SPLIT_PREFIX}_test_features.csv}"

echo "Raw Feature Fusion baseline"
echo "Root: $ROOT_DIR"
echo "Split: $SPLIT_PREFIX"
echo "Feature dir: $FEATURE_DIR"
echo "Output dir: $OUTPUT_DIR"

python -m experiments.orchestration.train_raw_feature_fusion \
  --split_prefix "$SPLIT_PREFIX" \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --test_csv "$TEST_CSV" \
  --output_dir "$OUTPUT_DIR"

