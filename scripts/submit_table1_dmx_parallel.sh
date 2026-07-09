#!/usr/bin/env bash
set -euo pipefail

# Submit Table 1 DMX jobs in parallel, one background job per backbone.
#
# Usage:
#   bash scripts/submit_table1_dmx_parallel.sh
#   BACKBONES="gemini deepseek-v3 qwen glm-4.5" bash scripts/submit_table1_dmx_parallel.sh
#
# The per-backbone jobs still run methods sequentially:
#   direct -> cot -> multiagent
#
# Defaults are resume-friendly:
#   SKIP_FINISHED=1 skips methods that already produced results_<method>_*.json
#   RESUME=1 resumes unfinished methods from checkpoint files when available

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${API_VALUE:-}" ]]; then
  echo "ERROR: API_VALUE is not set in this shell." >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export URL_VALUE="${URL_VALUE:-https://www.dmxapi.cn/v1/chat/completions}"
export TREATAGENT_USE_LLM_SYNTHESIS="${TREATAGENT_USE_LLM_SYNTHESIS:-1}"
export TREATAGENT_LOCAL_ONLY="${TREATAGENT_LOCAL_ONLY:-0}"
export RESUME="${RESUME:-1}"
export SKIP_FINISHED="${SKIP_FINISHED:-1}"
export LOG_DIR="${LOG_DIR:-results/table1_dmx_logs}"
export SAVE_EVERY="${SAVE_EVERY:-10}"

if [[ -n "${BACKBONES:-}" ]]; then
  read -r -a BACKBONE_LIST <<< "$BACKBONES"
else
  BACKBONE_LIST=(
    "gpt-4o"
    "gpt5"
    "gemini"
    "deepseek-v3"
    "qwen"
    "glm-4.5"
  )
fi

mkdir -p results "$LOG_DIR"

pid_file="results/table1_dmx_parallel.pids"
: > "$pid_file"

echo "Submitting Table 1 DMX jobs in parallel"
echo "Backbones: ${BACKBONE_LIST[*]}"
echo "Resume: $RESUME"
echo "Skip finished: $SKIP_FINISHED"
echo "Logs: $LOG_DIR"
echo

for backbone in "${BACKBONE_LIST[@]}"; do
  safe_backbone="${backbone//[^A-Za-z0-9_.-]/_}"
  background_log="results/table1_dmx_background_${safe_backbone}.log"

  (
    export BACKBONES="$backbone"
    bash scripts/run_table1_all.sh
  ) > "$background_log" 2>&1 &

  pid=$!
  printf '%s\t%s\t%s\n' "$pid" "$backbone" "$background_log" >> "$pid_file"
  echo "Submitted $backbone: PID=$pid, log=$background_log"
done

echo
echo "PID list: $pid_file"
echo "Monitor all jobs with:"
echo "  tail -f results/table1_dmx_background_*.log"

