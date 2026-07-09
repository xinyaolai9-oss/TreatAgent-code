#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")}"
SUBMIT_MAIN="${SUBMIT_MAIN:-1}"
SUBMIT_ABLATIONS="${SUBMIT_ABLATIONS:-1}"

mkdir -p "results/frozen_runs/${RUN_TAG}"

echo "Frozen experiment batch"
echo "Run tag: $RUN_TAG"
echo "Submit main: $SUBMIT_MAIN"
echo "Submit ablations: $SUBMIT_ABLATIONS"
echo

if [[ "$SUBMIT_MAIN" == "1" ]]; then
  RUN_TAG="$RUN_TAG" bash scripts/full/submit_frozen_main.sh
fi

if [[ "$SUBMIT_ABLATIONS" == "1" ]]; then
  RUN_TAG="$RUN_TAG" bash scripts/full/submit_core_ablations.sh
fi

echo
echo "All requested frozen jobs submitted."
echo "Run directory: results/frozen_runs/${RUN_TAG}"
echo "Check processes:"
echo "  cat results/frozen_runs/${RUN_TAG}/*.pids"
echo "Follow a log:"
echo "  tail -f results/frozen_runs/${RUN_TAG}/main_drug_disjoint_test.background.log"

