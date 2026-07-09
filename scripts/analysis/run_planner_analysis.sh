#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PREDICTIONS="${PREDICTIONS:-results/gpt-4o/results_multiagent_full_20260516_225929.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/planner_analysis/drug_disjoint}"
REPORT_NAME="${REPORT_NAME:-planner_drug_disjoint_full}"

python -m experiments.orchestration.planner_analysis \
  --predictions "$PREDICTIONS" \
  --output_dir "$OUTPUT_DIR" \
  --report_name "$REPORT_NAME"

echo "Planner analysis completed. Outputs: ${OUTPUT_DIR}"

