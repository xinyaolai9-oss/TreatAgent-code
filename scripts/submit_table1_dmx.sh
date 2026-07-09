#!/usr/bin/env bash
set -euo pipefail

# Submit the current DMX Table 1 batch in the background.
# Run this from the WSL shell where URL_VALUE/API_VALUE are already exported.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${API_VALUE:-}" ]]; then
  echo "ERROR: API_VALUE is not set in this shell." >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export URL_VALUE="${URL_VALUE:-https://www.dmxapi.cn/v1/chat/completions}"
export TREATAGENT_USE_LLM_SYNTHESIS=1
export TREATAGENT_LOCAL_ONLY=0
export BACKBONES="${BACKBONES:-gpt-4o gpt5 gemini deepseek-v3 qwen glm-4.5}"
export LOG_DIR="${LOG_DIR:-results/table1_dmx_logs}"
unset SKIP_FINISHED

mkdir -p results "$LOG_DIR"

nohup bash scripts/run_table1_all.sh > results/table1_dmx_background.log 2>&1 &
pid=$!

echo "$pid" > results/table1_dmx.pid
echo "Submitted Table 1 DMX batch."
echo "PID: $pid"
echo "Background log: results/table1_dmx_background.log"
echo "Task logs: $LOG_DIR"

