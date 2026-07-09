#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
OUTPUT_DIR="${OUTPUT_DIR:-results/bootstrap/${SPLIT_PREFIX}}"
REPORT_NAME="${REPORT_NAME:-bootstrap_${SPLIT_PREFIX}}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

if [[ "$SPLIT_PREFIX" == "drug_disjoint" ]]; then
  ARG_PRED="${ARG_PRED:-results/argument_graph_scorer/drug_disjoint_arg/treatagent_arg_drug_disjoint_test_predictions.json}"
  RAW_PRED="${RAW_PRED:-results/raw_feature_fusion/drug_disjoint/raw_feature_fusion_lr_drug_disjoint_test_predictions.json}"
  RAG_PRED="${RAG_PRED:-results/gpt-4o/results_rag_20260516_181331.json}"
  LS_PRED="${LS_PRED:-results/gpt-4o/results_multiagent_ls_20260516_185010.json}"
  python -m experiments.orchestration.bootstrap_experiments \
    --prediction "TreatAgent-ARG:${ARG_PRED}:argument_probability:argument_prediction" \
    --prediction "Raw-Feature-Fusion:${RAW_PRED}:raw_feature_fusion_probability:raw_feature_fusion_prediction" \
    --prediction "RAG:${RAG_PRED}::prediction_binary" \
    --prediction "LLM-Synthesis:${LS_PRED}:calibrated_probability:prediction_binary" \
    --primary_method TreatAgent-ARG \
    --output_dir "$OUTPUT_DIR" \
    --report_name "$REPORT_NAME" \
    --n_bootstrap "$N_BOOTSTRAP"
else
  ARG_PRED="${ARG_PRED:-results/argument_graph_scorer/${SPLIT_PREFIX}_arg/treatagent_arg_${SPLIT_PREFIX}_test_predictions.json}"
  RAW_PRED="${RAW_PRED:-results/raw_feature_fusion/${SPLIT_PREFIX}/raw_feature_fusion_lr_${SPLIT_PREFIX}_test_predictions.json}"
  python -m experiments.orchestration.bootstrap_experiments \
    --prediction "TreatAgent-ARG:${ARG_PRED}:argument_probability:argument_prediction" \
    --prediction "Raw-Feature-Fusion:${RAW_PRED}:raw_feature_fusion_probability:raw_feature_fusion_prediction" \
    --primary_method TreatAgent-ARG \
    --output_dir "$OUTPUT_DIR" \
    --report_name "$REPORT_NAME" \
    --n_bootstrap "$N_BOOTSTRAP"
fi

echo "Bootstrap experiments completed for ${SPLIT_PREFIX}. Outputs: ${OUTPUT_DIR}"

