#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")}"
BACKBONE="${BACKBONE:-gpt-4o}"
THRESHOLD="${THRESHOLD:-0.35}"
FULL_PLANNER_BUDGET="${FULL_PLANNER_BUDGET:-5}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RESUME="${RESUME:-0}"

SUBMIT_VAL="${SUBMIT_VAL:-0}"
SUBMIT_MAIN="${SUBMIT_MAIN:-1}"
SUBMIT_ABLATIONS="${SUBMIT_ABLATIONS:-0}"
SUBMIT_REPEATABILITY="${SUBMIT_REPEATABILITY:-0}"

TEST_SPLITS="${TEST_SPLITS:-drug_disjoint:test temporal_submit:test}"
VAL_SPLITS="${VAL_SPLITS:-drug_disjoint:val}"
ABLATIONS="${ABLATIONS:-wo_planner wo_llm_experts wo_evidencegraph wo_llm_judge wo_clinical}"
REPEATABILITY_N="${REPEATABILITY_N:-50}"
REPEATABILITY_RUNS="${REPEATABILITY_RUNS:-3}"

RUN_DIR="results/frozen_runs/${RUN_TAG}"
mkdir -p "$RUN_DIR"

if [[ -z "${URL_VALUE:-}" || -z "${API_VALUE:-}" ]]; then
  echo "ERROR: URL_VALUE and API_VALUE must be exported before submitting experiments." >&2
  exit 1
fi

echo "Remaining TreatAgent experiment batch"
echo "Run tag: $RUN_TAG"
echo "Backbone: $BACKBONE"
echo "Threshold: $THRESHOLD"
echo "Planner budget: $FULL_PLANNER_BUDGET"
echo "Submit validation: $SUBMIT_VAL"
echo "Submit main test: $SUBMIT_MAIN"
echo "Submit ablations: $SUBMIT_ABLATIONS"
echo "Submit repeatability: $SUBMIT_REPEATABILITY"
echo "Run directory: $RUN_DIR"
echo

submit_val_runs() {
  local pid_table="$RUN_DIR/validation.pids"
  local manifest="$RUN_DIR/validation_manifest.tsv"
  : > "$pid_table"
  printf "kind\tsplit_prefix\tsplit_name\tbackbone\tthreshold\tpid\tbackground_log\n" > "$manifest"

  for split in $VAL_SPLITS; do
    local split_prefix="${split%%:*}"
    local split_name="${split##*:}"
    local background_log="$RUN_DIR/validation_${split_prefix}_${split_name}.background.log"
    local pid_path="$RUN_DIR/validation_${split_prefix}_${split_name}.pid"
    local log_dir="$RUN_DIR/logs/validation_${split_prefix}_${split_name}"

    echo "Submitting validation run: ${split_prefix}_${split_name}"
    SPLIT_PREFIX="$split_prefix" \
    SPLIT_NAME="$split_name" \
    BACKBONE="$BACKBONE" \
    SAVE_EVERY="$SAVE_EVERY" \
    RESUME="$RESUME" \
    FULL_PLANNER_BUDGET="$FULL_PLANNER_BUDGET" \
    THRESHOLD="$THRESHOLD" \
    TREATAGENT_USE_LLM_PLANNER=1 \
    TREATAGENT_USE_LLM_EXPERTS=1 \
    TREATAGENT_LLM_EXPERTS=DrugKB,DiseaseKB,DTI,ADMET,Clinical \
    TREATAGENT_USE_LLM_JUDGE=1 \
    TREATAGENT_FORCE_ALL_EXPERTS=0 \
    TREATAGENT_USE_DERIVED_ARGUMENT_CLAIMS=1 \
    TREATAGENT_DISABLED_EXPERTS="" \
    LOG_DIR="$log_dir" \
    BACKGROUND_LOG="$background_log" \
    PID_FILE="$pid_path" \
    bash scripts/full/submit_treatagent_full.sh

    local pid
    pid="$(cat "$pid_path")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "validation" "$split_prefix" "$split_name" "$BACKBONE" "$THRESHOLD" "$pid" "$background_log" >> "$manifest"
    printf "%s\t%s\t%s\n" "$pid" "validation_${split_prefix}_${split_name}" "$background_log" >> "$pid_table"
  done
}

submit_repeatability_runs() {
  local pid_table="$RUN_DIR/repeatability.pids"
  local manifest="$RUN_DIR/repeatability_manifest.tsv"
  : > "$pid_table"
  printf "kind\trun_index\tsplit_prefix\tsplit_name\tbackbone\tthreshold\tpid\tbackground_log\n" > "$manifest"

  for i in $(seq 1 "$REPEATABILITY_RUNS"); do
    local background_log="$RUN_DIR/repeatability_${i}.background.log"
    local pid_path="$RUN_DIR/repeatability_${i}.pid"
    local log_dir="$RUN_DIR/logs/repeatability_${i}"
    echo "Submitting repeatability run ${i}/${REPEATABILITY_RUNS}"
    SMOKE_N="$REPEATABILITY_N" \
    SPLIT_PREFIX=drug_disjoint \
    SPLIT_NAME=val \
    BACKBONE="$BACKBONE" \
    SAVE_EVERY="$SAVE_EVERY" \
    RESUME=0 \
    FULL_PLANNER_BUDGET="$FULL_PLANNER_BUDGET" \
    THRESHOLD="$THRESHOLD" \
    TREATAGENT_USE_LLM_PLANNER=1 \
    TREATAGENT_USE_LLM_EXPERTS=1 \
    TREATAGENT_LLM_EXPERTS=DrugKB,DiseaseKB,DTI,ADMET,Clinical \
    TREATAGENT_USE_LLM_JUDGE=1 \
    TREATAGENT_FORCE_ALL_EXPERTS=0 \
    TREATAGENT_USE_DERIVED_ARGUMENT_CLAIMS=1 \
    TREATAGENT_DISABLED_EXPERTS="" \
    LOG_DIR="$log_dir" \
    BACKGROUND_LOG="$background_log" \
    PID_FILE="$pid_path" \
    bash scripts/full/submit_treatagent_full.sh

    local pid
    pid="$(cat "$pid_path")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "repeatability" "$i" "drug_disjoint" "val_smoke_${REPEATABILITY_N}" "$BACKBONE" "$THRESHOLD" "$pid" "$background_log" >> "$manifest"
    printf "%s\t%s\t%s\n" "$pid" "repeatability_${i}" "$background_log" >> "$pid_table"
  done
}

if [[ "$SUBMIT_VAL" == "1" ]]; then
  submit_val_runs
fi

if [[ "$SUBMIT_MAIN" == "1" ]]; then
  RUN_TAG="$RUN_TAG" \
  BACKBONE="$BACKBONE" \
  THRESHOLD="$THRESHOLD" \
  FULL_PLANNER_BUDGET="$FULL_PLANNER_BUDGET" \
  SAVE_EVERY="$SAVE_EVERY" \
  RESUME="$RESUME" \
  SPLITS="$TEST_SPLITS" \
  bash scripts/full/submit_frozen_main.sh
fi

if [[ "$SUBMIT_ABLATIONS" == "1" ]]; then
  RUN_TAG="$RUN_TAG" \
  BACKBONE="$BACKBONE" \
  THRESHOLD="$THRESHOLD" \
  FULL_PLANNER_BUDGET="$FULL_PLANNER_BUDGET" \
  SAVE_EVERY="$SAVE_EVERY" \
  RESUME="$RESUME" \
  SPLITS="$TEST_SPLITS" \
  ABLATIONS="$ABLATIONS" \
  bash scripts/full/submit_core_ablations.sh
fi

if [[ "$SUBMIT_REPEATABILITY" == "1" ]]; then
  submit_repeatability_runs
fi

cat > "$RUN_DIR/README.md" <<EOF
# Frozen TreatAgent run: ${RUN_TAG}

Backbone: ${BACKBONE}
Threshold: ${THRESHOLD}
Planner budget: ${FULL_PLANNER_BUDGET}

Submitted stages:

- validation: ${SUBMIT_VAL}
- main: ${SUBMIT_MAIN}
- ablations: ${SUBMIT_ABLATIONS}
- repeatability: ${SUBMIT_REPEATABILITY}

Monitor:

\`\`\`bash
cat ${RUN_DIR}/*.pids
tail -f ${RUN_DIR}/main_drug_disjoint_test.background.log
python scripts/full/summarize_frozen_runs.py ${RUN_DIR}
\`\`\`
EOF

echo
echo "Submission complete."
echo "Run directory: $RUN_DIR"
echo "PID files:"
find "$RUN_DIR" -maxdepth 1 -name '*.pids' -print
echo
echo "Summarize later with:"
echo "  python scripts/full/summarize_frozen_runs.py $RUN_DIR"

