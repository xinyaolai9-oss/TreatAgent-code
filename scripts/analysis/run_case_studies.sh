#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARG_PRED="${ARG_PRED:-results/argument_graph_scorer/drug_disjoint_arg/treatagent_arg_drug_disjoint_test_predictions.json}"
SOURCE_RESULTS="${SOURCE_RESULTS:-$(cat results/eg_scorer/drug_disjoint_eg/drug_disjoint_test_result_path.txt)}"
PLANNER_RESULTS="${PLANNER_RESULTS:-results/gpt-4o/results_multiagent_full_20260516_225929.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/case_studies/treatagent_arg}"
DOC_PATH="${DOC_PATH:-docs/case_studies/treatagent_arg_case_studies.md}"

python -m experiments.orchestration.case_study \
  --arg_predictions "$ARG_PRED" \
  --source_results "$SOURCE_RESULTS" \
  --planner_results "$PLANNER_RESULTS" \
  --output_dir "$OUTPUT_DIR" \
  --doc_path "$DOC_PATH"

echo "Case studies completed."
echo "Markdown: $DOC_PATH"
echo "Figures: $OUTPUT_DIR"

