#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SPLIT_PREFIX="${SPLIT_PREFIX:-drug_disjoint}"
SPLIT_NAME="${SPLIT_NAME:-test}"
DATA_PATH="${DATA_PATH:-data/benchmark/split_inputs/${SPLIT_PREFIX}_${SPLIT_NAME}.json}"
BACKBONE="${BACKBONE:-gpt-4o}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RESUME="${RESUME:-1}"
GENERATE_REPORT="${GENERATE_REPORT:-0}"
USE_MEMORY="${USE_MEMORY:-0}"
KNOWLEDGE_CUTOFF_DATE="${KNOWLEDGE_CUTOFF_DATE:-}"
SMOKE_N="${SMOKE_N:-0}"
FULL_PLANNER_BUDGET="${FULL_PLANNER_BUDGET:-}"
THRESHOLD="${THRESHOLD:-}"
TREATAGENT_USE_LLM_EXPERTS="${TREATAGENT_USE_LLM_EXPERTS:-0}"
TREATAGENT_LLM_EXPERTS="${TREATAGENT_LLM_EXPERTS:-DrugKB,DiseaseKB,DTI,ADMET,Clinical}"
TREATAGENT_USE_LLM_PLANNER="${TREATAGENT_USE_LLM_PLANNER:-1}"
TREATAGENT_USE_LLM_JUDGE="${TREATAGENT_USE_LLM_JUDGE:-1}"
TREATAGENT_FORCE_ALL_EXPERTS="${TREATAGENT_FORCE_ALL_EXPERTS:-0}"
TREATAGENT_USE_DERIVED_ARGUMENT_CLAIMS="${TREATAGENT_USE_DERIVED_ARGUMENT_CLAIMS:-1}"
TREATAGENT_DISABLED_EXPERTS="${TREATAGENT_DISABLED_EXPERTS:-}"

LOG_DIR="${LOG_DIR:-results/treatagent_full_logs}"
BACKGROUND_LOG="${BACKGROUND_LOG:-results/treatagent_full_${BACKBONE}_${SPLIT_PREFIX}_${SPLIT_NAME}_background.log}"
PID_FILE="${PID_FILE:-results/treatagent_full_${BACKBONE}_${SPLIT_PREFIX}_${SPLIT_NAME}.pid}"

if [[ -z "${URL_VALUE:-}" || -z "${API_VALUE:-}" ]]; then
  echo "ERROR: URL_VALUE and API_VALUE must be exported in this WSL shell before submitting TreatAgent-Full." >&2
  exit 1
fi

if [[ "${TREATAGENT_LOCAL_ONLY:-0}" == "1" ]]; then
  echo "ERROR: TREATAGENT_LOCAL_ONLY=1 disables LLM calls. Set TREATAGENT_LOCAL_ONLY=0 for TreatAgent-Full." >&2
  exit 1
fi

export TREATAGENT_USE_LLM_SYNTHESIS="${TREATAGENT_USE_LLM_SYNTHESIS:-1}"
export TREATAGENT_LOCAL_ONLY="${TREATAGENT_LOCAL_ONLY:-0}"
export TREATAGENT_USE_LLM_EXPERTS
export TREATAGENT_LLM_EXPERTS
export TREATAGENT_USE_LLM_PLANNER
export TREATAGENT_USE_LLM_JUDGE
export TREATAGENT_FORCE_ALL_EXPERTS
export TREATAGENT_USE_DERIVED_ARGUMENT_CLAIMS
export TREATAGENT_DISABLED_EXPERTS
if [[ -n "$FULL_PLANNER_BUDGET" ]]; then
  export TREATAGENT_FULL_PLANNER_BUDGET="$FULL_PLANNER_BUDGET"
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: data file not found: $DATA_PATH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$(dirname "$BACKGROUND_LOG")"

RUN_DATA_PATH="$DATA_PATH"
if [[ "$SMOKE_N" != "0" ]]; then
  RUN_DATA_PATH="results/treatagent_full_smoke_${SPLIT_PREFIX}_${SPLIT_NAME}_${SMOKE_N}.json"
  python - "$DATA_PATH" "$RUN_DATA_PATH" "$SMOKE_N" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
n = int(sys.argv[3])
data = json.loads(source.read_text(encoding="utf-8"))
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(data[:n], indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {target} with {min(n, len(data))} samples")
PY
fi

(
  timestamp="$(date +"%Y%m%d_%H%M%S")"
  log_file="$LOG_DIR/${timestamp}_${BACKBONE}_${SPLIT_PREFIX}_${SPLIT_NAME}_full.log"

  echo "TreatAgent-Full submission"
  echo "Root: $ROOT_DIR"
  echo "Backbone: $BACKBONE"
  echo "Split: ${SPLIT_PREFIX}_${SPLIT_NAME}"
  echo "Data: $RUN_DATA_PATH"
  echo "Log: $log_file"
  echo "Resume: $RESUME"
  echo "Save every: $SAVE_EVERY"
  echo "Generate report: $GENERATE_REPORT"
  echo "Use memory: $USE_MEMORY"
  echo "Knowledge cutoff date: ${KNOWLEDGE_CUTOFF_DATE:-none}"
  echo "Full planner budget: ${TREATAGENT_FULL_PLANNER_BUDGET:-default}"
  echo "Threshold: ${THRESHOLD:-default}"
  echo "LLM planner enabled: $TREATAGENT_USE_LLM_PLANNER"
  echo "LLM judge enabled: $TREATAGENT_USE_LLM_JUDGE"
  echo "LLM experts enabled: $TREATAGENT_USE_LLM_EXPERTS"
  echo "LLM experts: $TREATAGENT_LLM_EXPERTS"
  echo "Force all experts: $TREATAGENT_FORCE_ALL_EXPERTS"
  echo "Derived argument claims: $TREATAGENT_USE_DERIVED_ARGUMENT_CLAIMS"
  echo "Disabled experts: ${TREATAGENT_DISABLED_EXPERTS:-none}"
  echo

  args=(
    "--json_path" "$RUN_DATA_PATH"
    "--method" "multiagent"
    "--agent_version" "full"
    "--backbone" "$BACKBONE"
    "--save_every" "$SAVE_EVERY"
  )

  if [[ "$RESUME" == "1" ]]; then
    args+=("--resume")
  fi

  if [[ "$GENERATE_REPORT" == "1" ]]; then
    args+=("--generate_report")
  fi

  if [[ "$USE_MEMORY" == "1" ]]; then
    args+=("--use_memory")
  fi

  if [[ -n "$KNOWLEDGE_CUTOFF_DATE" ]]; then
    args+=("--knowledge_cutoff_date" "$KNOWLEDGE_CUTOFF_DATE")
  fi

  if [[ -n "$THRESHOLD" ]]; then
    args+=("--threshold" "$THRESHOLD")
  fi

  python -m treatagent.cli "${args[@]}" 2>&1 | tee "$log_file"
) > "$BACKGROUND_LOG" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"
echo "Submitted TreatAgent-Full."
echo "PID: $pid"
echo "Background log: $BACKGROUND_LOG"
echo "PID file: $PID_FILE"
