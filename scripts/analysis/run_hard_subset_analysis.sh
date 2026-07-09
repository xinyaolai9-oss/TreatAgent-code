#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hard_subset_analysis/${SPLIT_PREFIX}}"
REPORT_NAME="${REPORT_NAME:-hard_subset_${SPLIT_PREFIX}}"

if [[ "$SPLIT_PREFIX" == "drug_disjoint" ]]; then
  ARG_PRED="${ARG_PRED:-results/argument_graph_scorer/drug_disjoint_arg/treatagent_arg_drug_disjoint_test_predictions.json}"
  RAW_PRED="${RAW_PRED:-results/raw_feature_fusion/drug_disjoint/raw_feature_fusion_lr_drug_disjoint_test_predictions.json}"
  DIRECT_PRED="${DIRECT_PRED:-results/gpt-4o/results_direct_20260516_165038.json}"
  COT_PRED="${COT_PRED:-results/gpt-4o/results_cot_20260516_174307.json}"
  RAG_PRED="${RAG_PRED:-results/gpt-4o/results_rag_20260516_181331.json}"
  LS_PRED="${LS_PRED:-results/gpt-4o/results_multiagent_ls_20260516_185010.json}"
  EXTRA_PREDICTIONS=(
    "Direct:${DIRECT_PRED}::prediction_binary"
    "CoT:${COT_PRED}::prediction_binary"
    "RAG:${RAG_PRED}::prediction_binary"
    "LLM-Synthesis:${LS_PRED}:calibrated_probability:prediction_binary"
  )
else
  ARG_PRED="${ARG_PRED:-results/argument_graph_scorer/${SPLIT_PREFIX}_arg/treatagent_arg_${SPLIT_PREFIX}_test_predictions.json}"
  RAW_PRED="${RAW_PRED:-results/raw_feature_fusion/${SPLIT_PREFIX}/raw_feature_fusion_lr_${SPLIT_PREFIX}_test_predictions.json}"
  EXTRA_PREDICTIONS=()
fi

args=(
  --arg_predictions "$ARG_PRED"
  --prediction "TreatAgent-ARG:${ARG_PRED}:argument_probability:argument_prediction"
  --prediction "Raw-Feature-Fusion:${RAW_PRED}:raw_feature_fusion_probability:raw_feature_fusion_prediction"
)
for prediction in "${EXTRA_PREDICTIONS[@]}"; do
  args+=(--prediction "$prediction")
done
args+=(--output_dir "$OUTPUT_DIR" --report_name "$REPORT_NAME")

python -m experiments.orchestration.hard_subset_analysis "${args[@]}"

echo "Hard subset analysis completed for ${SPLIT_PREFIX}. Outputs: ${OUTPUT_DIR}"

