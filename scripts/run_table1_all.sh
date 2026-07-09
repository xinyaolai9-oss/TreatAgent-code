#!/usr/bin/env bash
set -euo pipefail

# Run Table 1 experiments on data/benchmark/inputs.json.
#
# This submits the v1-style comparison:
#   - One-shot learning  -> --method direct
#   - Zero-shot CoT      -> --method cot
#   - RAG prompting      -> --method rag
#   - TreatAgent v2      -> --method multiagent
#
# Usage:
#   bash scripts/run_table1_all.sh
#
# Optional environment overrides:
#   DATA_PATH=data/benchmark/inputs_10.json bash scripts/run_table1_all.sh
#   BACKBONES="gpt-4o gpt5 gemini deepseek-v3 qwen glm-4.5" bash scripts/run_table1_all.sh
#   METHODS="direct cot rag multiagent" bash scripts/run_table1_all.sh
#   RESULTS_ROOT=results/table1_dmx bash scripts/run_table1_all.sh
#   SAVE_EVERY=20 bash scripts/run_table1_all.sh
#   RESUME=1 bash scripts/run_table1_all.sh
#   SKIP_FINISHED=1 bash scripts/run_table1_all.sh
#   GENERATE_REPORT=1 bash scripts/run_table1_all.sh
#   USE_MEMORY=1 bash scripts/run_table1_all.sh
#   AGENT_VERSION=full bash scripts/run_table1_all.sh
#   KNOWLEDGE_CUTOFF_DATE=2024-01-01 bash scripts/run_table1_all.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_PATH="${DATA_PATH:-data/benchmark/inputs.json}"
SAVE_EVERY="${SAVE_EVERY:-10}"
RESUME="${RESUME:-0}"
SKIP_FINISHED="${SKIP_FINISHED:-0}"
GENERATE_REPORT="${GENERATE_REPORT:-0}"
USE_MEMORY="${USE_MEMORY:-0}"
AGENT_VERSION="${AGENT_VERSION:-eg}"
KNOWLEDGE_CUTOFF_DATE="${KNOWLEDGE_CUTOFF_DATE:-}"

LOG_DIR="${LOG_DIR:-results/table1_logs}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
mkdir -p "$LOG_DIR"

if [[ -n "${BACKBONES:-}" ]]; then
  read -r -a BACKBONES <<< "$BACKBONES"
else
  BACKBONES=(
    "gpt-4o"
    "gpt5"
    "gemini"
    "deepseek-v3"
    "qwen"
    "glm-4.5"
  )
fi

if [[ -n "${METHODS:-}" ]]; then
  read -r -a METHODS <<< "$METHODS"
else
  METHODS=(
    "direct"
    "cot"
    "rag"
    "multiagent"
  )
fi

method_label() {
  case "$1" in
    direct) echo "One-shot learning" ;;
    cot) echo "Zero-shot CoT" ;;
    rag) echo "RAG prompting" ;;
    multiagent) echo "TreatAgent v2" ;;
    *) echo "$1" ;;
  esac
}

build_common_args() {
  local args=(
    "--json_path" "$DATA_PATH"
    "--save_every" "$SAVE_EVERY"
  )

  if [[ "$RESUME" == "1" ]]; then
    args+=("--resume")
  fi

  if [[ -n "$KNOWLEDGE_CUTOFF_DATE" ]]; then
    args+=("--knowledge_cutoff_date" "$KNOWLEDGE_CUTOFF_DATE")
  fi

  if ((${#args[@]} > 0)); then
    printf '%s\n' "${args[@]}"
  fi
}

build_multiagent_args() {
  local args=("--agent_version" "$AGENT_VERSION")

  if [[ "$GENERATE_REPORT" == "1" ]]; then
    args+=("--generate_report")
  fi

  if [[ "$USE_MEMORY" == "1" ]]; then
    args+=("--use_memory")
  fi

  if ((${#args[@]} > 0)); then
    printf '%s\n' "${args[@]}"
  fi
}

run_one() {
  local backbone="$1"
  local method="$2"
  local label
  local timestamp
  local log_file
  local result_pattern

  label="$(method_label "$method")"
  if [[ "$method" == "multiagent" ]]; then
    result_pattern="${RESULTS_ROOT}/${backbone}/results_${method}_${AGENT_VERSION}_"*.json
  else
    result_pattern="${RESULTS_ROOT}/${backbone}/results_${method}_"*.json
  fi

  if [[ "$SKIP_FINISHED" == "1" ]] && compgen -G "$result_pattern" >/dev/null; then
    echo "Skipping completed task: $backbone / $method"
    return 0
  fi

  timestamp="$(date +"%Y%m%d_%H%M%S")"
  log_file="$LOG_DIR/${timestamp}_${backbone}_${method}.log"

  echo "============================================================"
  echo "Starting Table 1 task"
  echo "Model:  $backbone"
  echo "Method: $label ($method)"
  if [[ "$method" == "multiagent" ]]; then
    echo "Version: $AGENT_VERSION"
  fi
  echo "Data:   $DATA_PATH"
  echo "Log:    $log_file"
  echo "============================================================"

  local common_args=()
  while IFS= read -r arg; do
    [[ -n "$arg" ]] && common_args+=("$arg")
  done < <(build_common_args)

  local extra_args=()
  if [[ "$method" == "multiagent" ]]; then
    while IFS= read -r arg; do
      [[ -n "$arg" ]] && extra_args+=("$arg")
    done < <(build_multiagent_args)
  fi

  python -m treatagent.cli \
    "${common_args[@]}" \
    --method "$method" \
    --backbone "$backbone" \
    "${extra_args[@]}" \
    2>&1 | tee "$log_file"

  echo "Finished: $backbone / $method"
}

echo "Table 1 experiment batch"
echo "Root: $ROOT_DIR"
echo "Data: $DATA_PATH"
echo "Logs: $LOG_DIR"
echo "Results root: $RESULTS_ROOT"
echo "Skip finished: $SKIP_FINISHED"
echo "Backbones: ${BACKBONES[*]}"
echo "TreatAgent version: $AGENT_VERSION"
echo

if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: data file not found: $DATA_PATH" >&2
  exit 1
fi

for backbone in "${BACKBONES[@]}"; do
  for method in "${METHODS[@]}"; do
    run_one "$backbone" "$method"
  done
done

echo
echo "All Table 1 tasks completed."
echo "Detailed result JSON files are under $RESULTS_ROOT/<backbone>/."
echo "Task logs are under $LOG_DIR/."

