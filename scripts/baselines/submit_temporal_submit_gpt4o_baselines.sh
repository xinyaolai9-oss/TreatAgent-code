#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export DATA_PATH="${DATA_PATH:-data/benchmark/split_inputs/temporal_submit_test.json}"
export BACKBONE="${BACKBONE:-gpt-4o}"
export LOG_DIR="${LOG_DIR:-results/baseline_temporal_submit_logs}"
export BACKGROUND_LOG="${BACKGROUND_LOG:-results/baseline_temporal_submit_gpt4o_background.log}"
export PID_FILE="${PID_FILE:-results/baseline_temporal_submit_gpt4o.pid}"

bash scripts/baselines/submit_drug_disjoint_gpt4o_baselines.sh

