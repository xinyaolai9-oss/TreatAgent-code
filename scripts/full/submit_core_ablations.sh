#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BACKBONE="${BACKBONE:-gpt-4o}"
THRESHOLD="${THRESHOLD:-0.35}"
FULL_PLANNER_BUDGET="${FULL_PLANNER_BUDGET:-5}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RESUME="${RESUME:-0}"
SPLITS="${SPLITS:-drug_disjoint:test temporal_submit:test}"
ABLATIONS="${ABLATIONS:-wo_planner wo_llm_experts wo_evidencegraph wo_llm_judge wo_clinical}"
RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")}"

BASE_LOG_DIR="${BASE_LOG_DIR:-results/frozen_runs/${RUN_TAG}/logs}"
PID_FILE="${PID_FILE:-results/frozen_runs/${RUN_TAG}/core_ablations.pids}"
MANIFEST="${MANIFEST:-results/frozen_runs/${RUN_TAG}/core_ablations_manifest.tsv}"

if [[ -z "${URL_VALUE:-}" || -z "${API_VALUE:-}" ]]; then
  echo "ERROR: URL_VALUE and API_VALUE must be exported before submitting core ablations." >&2
  exit 1
fi

mkdir -p "$(dirname "$PID_FILE")" "$BASE_LOG_DIR"
: > "$PID_FILE"
printf "kind\tsplit_prefix\tsplit_name\tbackbone\tthreshold\tpid\tbackground_log\tsettings\n" > "$MANIFEST"

submit_one() {
  local ablation="$1"
  local split_prefix="$2"
  local split_name="$3"

  local use_planner=1
  local use_experts=1
  local use_judge=1
  local force_all=0
  local use_derived=1
  local disabled=""

  case "$ablation" in
    wo_planner)
      use_planner=0
      force_all=1
      ;;
    wo_llm_experts)
      use_experts=0
      ;;
    wo_evidencegraph)
      use_derived=0
      ;;
    wo_llm_judge)
      use_judge=0
      ;;
    wo_clinical)
      disabled="Clinical"
      ;;
    *)
      echo "ERROR: unknown ablation: $ablation" >&2
      exit 1
      ;;
  esac

  local background_log="results/frozen_runs/${RUN_TAG}/${ablation}_${split_prefix}_${split_name}.background.log"
  local pid_path="results/frozen_runs/${RUN_TAG}/${ablation}_${split_prefix}_${split_name}.pid"
  local log_dir="${BASE_LOG_DIR}/${ablation}_${split_prefix}_${split_name}"

  echo "Submitting core ablation: ${ablation} on ${split_prefix}_${split_name}"
  SPLIT_PREFIX="$split_prefix" \
  SPLIT_NAME="$split_name" \
  BACKBONE="$BACKBONE" \
  SAVE_EVERY="$SAVE_EVERY" \
  RESUME="$RESUME" \
  FULL_PLANNER_BUDGET="$FULL_PLANNER_BUDGET" \
  THRESHOLD="$THRESHOLD" \
  TREATAGENT_USE_LLM_PLANNER="$use_planner" \
  TREATAGENT_USE_LLM_EXPERTS="$use_experts" \
  TREATAGENT_LLM_EXPERTS=DrugKB,DiseaseKB,DTI,ADMET,Clinical \
  TREATAGENT_USE_LLM_JUDGE="$use_judge" \
  TREATAGENT_FORCE_ALL_EXPERTS="$force_all" \
  TREATAGENT_USE_DERIVED_ARGUMENT_CLAIMS="$use_derived" \
  TREATAGENT_DISABLED_EXPERTS="$disabled" \
  LOG_DIR="$log_dir" \
  BACKGROUND_LOG="$background_log" \
  PID_FILE="$pid_path" \
  bash scripts/full/submit_treatagent_full.sh

  local pid
  pid="$(cat "$pid_path")"
  local settings
  settings="planner=${use_planner};llm_experts=${use_experts};judge=${use_judge};force_all=${force_all};derived=${use_derived};disabled=${disabled:-none}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$ablation" "$split_prefix" "$split_name" "$BACKBONE" "$THRESHOLD" "$pid" "$background_log" "$settings" >> "$MANIFEST"
  printf "%s\t%s\t%s\n" "$pid" "${ablation}_${split_prefix}_${split_name}" "$background_log" >> "$PID_FILE"
}

for split in $SPLITS; do
  split_prefix="${split%%:*}"
  split_name="${split##*:}"
  for ablation in $ABLATIONS; do
    submit_one "$ablation" "$split_prefix" "$split_name"
  done
done

echo "Submitted frozen core ablations."
echo "PID table: $PID_FILE"
echo "Manifest: $MANIFEST"

