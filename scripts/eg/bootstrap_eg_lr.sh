#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
PREDICTIONS="${PREDICTIONS:-results/eg_scorer/${SPLIT_PREFIX}_eg/treatagent_eg_local_${SPLIT_PREFIX}_test_predictions.json}"
OUTPUT_JSON="${OUTPUT_JSON:-results/bootstrap/${SPLIT_PREFIX}_eg_lr_bootstrap_ci.json}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

python -m experiments.orchestration.bootstrap_metrics \
  --predictions "$PREDICTIONS" \
  --probability_key eg_probability \
  --prediction_key eg_prediction \
  --output_json "$OUTPUT_JSON" \
  --n_bootstrap "$N_BOOTSTRAP"

echo "Bootstrap CI completed for ${SPLIT_PREFIX}. Output: ${OUTPUT_JSON}"

