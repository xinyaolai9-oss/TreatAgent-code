#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
OUTPUT_DIR="${OUTPUT_DIR:-results/calibration/${SPLIT_PREFIX}}"
REPORT_NAME="${REPORT_NAME:-calibration_${SPLIT_PREFIX}}"

if [[ "$SPLIT_PREFIX" == "drug_disjoint" ]]; then
  ARG_PRED="${ARG_PRED:-results/argument_graph_scorer/drug_disjoint_arg/treatagent_arg_drug_disjoint_test_predictions.json}"
  RAW_PRED="${RAW_PRED:-results/raw_feature_fusion/drug_disjoint/raw_feature_fusion_lr_drug_disjoint_test_predictions.json}"
  LS_PRED="${LS_PRED:-results/gpt-4o/results_multiagent_ls_20260516_185010.json}"
  python -m experiments.orchestration.calibration_analysis \
    --prediction "TreatAgent-ARG:${ARG_PRED}:argument_probability:argument_prediction" \
    --prediction "Raw-Feature-Fusion:${RAW_PRED}:raw_feature_fusion_probability:raw_feature_fusion_prediction" \
    --prediction "LLM-Synthesis:${LS_PRED}:calibrated_probability:prediction_binary" \
    --output_dir "$OUTPUT_DIR" \
    --report_name "$REPORT_NAME"
else
  ARG_PRED="${ARG_PRED:-results/argument_graph_scorer/${SPLIT_PREFIX}_arg/treatagent_arg_${SPLIT_PREFIX}_test_predictions.json}"
  RAW_PRED="${RAW_PRED:-results/raw_feature_fusion/${SPLIT_PREFIX}/raw_feature_fusion_lr_${SPLIT_PREFIX}_test_predictions.json}"
  python -m experiments.orchestration.calibration_analysis \
    --prediction "TreatAgent-ARG:${ARG_PRED}:argument_probability:argument_prediction" \
    --prediction "Raw-Feature-Fusion:${RAW_PRED}:raw_feature_fusion_probability:raw_feature_fusion_prediction" \
    --output_dir "$OUTPUT_DIR" \
    --report_name "$REPORT_NAME"
fi

echo "Calibration analysis completed for ${SPLIT_PREFIX}. Outputs: ${OUTPUT_DIR}"

