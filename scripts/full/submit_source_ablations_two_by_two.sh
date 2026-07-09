#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-final_thr020_source_ablation_2by2}"
BACKBONE="${BACKBONE:-gpt-4o}"
THRESHOLD="${THRESHOLD:-0.20}"
FULL_PLANNER_BUDGET="${FULL_PLANNER_BUDGET:-5}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RESUME="${RESUME:-1}"

SPLITS="${SPLITS:-drug_disjoint:test temporal_submit:test}"
ABLATIONS="${ABLATIONS:-wo_drugkb wo_diseasekb wo_dti wo_admet wo_clinical}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"

RUN_DIR="results/frozen_runs/${RUN_TAG}"
mkdir -p "$RUN_DIR"

if [[ -z "${URL_VALUE:-}" || -z "${API_VALUE:-}" ]]; then
  echo "ERROR: URL_VALUE and API_VALUE must be exported before submitting source ablations." >&2
  exit 1
fi

get_disabled_experts_for_ablation() {
  local ablation="$1"

  case "$ablation" in
    wo_drugkb)
      echo "DrugKB"
      ;;
    wo_diseasekb)
      echo "DiseaseKB"
      ;;
    wo_dti)
      echo "DTI"
      ;;
    wo_admet)
      echo "ADMET"
      ;;
    wo_clinical)
      echo "Clinical"
      ;;
    *)
      echo "ERROR_UNKNOWN_ABLATION"
      ;;
  esac
}

is_job_running() {
  local pid="$1"

  # 1. shell PID still exists
  if ps -p "$pid" > /dev/null 2>&1; then
    return 0
  fi

  # 2. child python process still exists, in case the launcher shell has forked
  if pgrep -P "$pid" > /dev/null 2>&1; then
    return 0
  fi

  return 1
}

wait_for_current_batch() {
  if [[ "${#running_pids[@]}" -eq 0 ]]; then
    return
  fi

  echo
  echo "Waiting for current batch to finish:"
  for idx in "${!running_pids[@]}"; do
    echo "  ${running_pids[$idx]}  ${running_names[$idx]}"
  done

  while true; do
    local any_running=0

    for idx in "${!running_pids[@]}"; do
      local pid="${running_pids[$idx]}"
      local name="${running_names[$idx]}"

      if is_job_running "$pid"; then
        any_running=1
        echo "$(date '+%Y-%m-%d %H:%M:%S')  still running: $pid  $name"
      fi
    done

    if [[ "$any_running" -eq 0 ]]; then
      break
    fi

    sleep "$CHECK_INTERVAL"
  done

  running_pids=()
  running_names=()

  echo "Current batch finished."
  echo
}

submit_one_job() {
  local ablation="$1"
  local disabled_experts="$2"
  local split_prefix="$3"
  local split_name="$4"

  local job_name="${ablation}_${split_prefix}_${split_name}"
  local background_log="$RUN_DIR/${job_name}.background.log"
  local pid_path="$RUN_DIR/${job_name}.pid"
  local log_dir="$RUN_DIR/logs/${job_name}"

  echo "============================================================"
  echo "Submitting source ablation"
  echo "Ablation: $ablation"
  echo "Disabled expert: $disabled_experts"
  echo "Split: ${split_prefix}:${split_name}"
  echo "Backbone: $BACKBONE"
  echo "Threshold: $THRESHOLD"
  echo "Resume: $RESUME"
  echo "Log dir: $log_dir"
  echo "Background log: $background_log"
  echo "============================================================"

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
  TREATAGENT_DISABLED_EXPERTS="$disabled_experts" \
  LOG_DIR="$log_dir" \
  BACKGROUND_LOG="$background_log" \
  PID_FILE="$pid_path" \
  bash scripts/full/submit_treatagent_full.sh

  local pid
  pid="$(cat "$pid_path")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "source_ablation" "$ablation" "$disabled_experts" "$split_prefix" "$split_name" "$BACKBONE" "$THRESHOLD" "$pid" "$background_log" >> "$manifest"

  printf "%s\t%s\t%s\n" "$pid" "$job_name" "$background_log" >> "$pid_table"

  running_pids+=("$pid")
  running_names+=("$job_name")
}

pid_table="$RUN_DIR/source_ablations_2by2.pids"
manifest="$RUN_DIR/source_ablations_2by2_manifest.tsv"

: > "$pid_table"
printf "kind\tablation\tdisabled_experts\tsplit_prefix\tsplit_name\tbackbone\tthreshold\tpid\tbackground_log\n" > "$manifest"

running_pids=()
running_names=()

for ablation in $ABLATIONS; do
  disabled_experts="$(get_disabled_experts_for_ablation "$ablation")"

  if [[ "$disabled_experts" == "ERROR_UNKNOWN_ABLATION" ]]; then
    echo "ERROR: unknown source ablation: $ablation" >&2
    exit 1
  fi

  for split in $SPLITS; do
    split_prefix="${split%%:*}"
    split_name="${split##*:}"

    submit_one_job "$ablation" "$disabled_experts" "$split_prefix" "$split_name"

    if [[ "${#running_pids[@]}" -ge "$MAX_PARALLEL" ]]; then
      wait_for_current_batch
    fi
  done
done

wait_for_current_batch

cat > "$RUN_DIR/README.md" <<EOF
# Source ablation two-by-two run: ${RUN_TAG}

Backbone: ${BACKBONE}
Threshold: ${THRESHOLD}
Planner budget: ${FULL_PLANNER_BUDGET}
Max parallel: ${MAX_PARALLEL}
Resume: ${RESUME}
Check interval: ${CHECK_INTERVAL}

Ablations:

${ABLATIONS}

Splits:

${SPLITS}

Monitor:

\`\`\`bash
cat ${RUN_DIR}/source_ablations_2by2.pids
tail -f ${RUN_DIR}/*.background.log
python scripts/full/summarize_frozen_runs.py ${RUN_DIR}
\`\`\`
EOF

echo
echo "All source ablation batches completed."
echo "Run directory: $RUN_DIR"
echo "PID table: $pid_table"
echo "Manifest: $manifest"