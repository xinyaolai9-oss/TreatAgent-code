#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ALL_EXPERT_TEST="${ALL_EXPERT_TEST:-results/local/results_multiagent_eg_20260515_214531.json}"
PLANNER_TEST="${PLANNER_TEST:-results/gpt-4o/results_multiagent_full_20260516_225929.json}"
THRESHOLD="${THRESHOLD:-0.36}"
OUTPUT_DIR="${OUTPUT_DIR:-results/budgeted_planner/drug_disjoint}"
REPORT_NAME="${REPORT_NAME:-budgeted_planner_drug_disjoint}"

python -m experiments.orchestration.budgeted_planner_analysis \
  --all_expert_test "$ALL_EXPERT_TEST" \
  --planner_test "$PLANNER_TEST" \
  --threshold "$THRESHOLD" \
  --output_dir "$OUTPUT_DIR" \
  --report_name "$REPORT_NAME"

echo "Budgeted planner analysis completed."
echo "Report JSON: $OUTPUT_DIR/$REPORT_NAME.json"
echo "Report MD:   $OUTPUT_DIR/$REPORT_NAME.md"

