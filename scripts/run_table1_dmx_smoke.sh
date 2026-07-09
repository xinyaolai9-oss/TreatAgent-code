#!/usr/bin/env bash
set -euo pipefail

# Fast DMX API smoke test for the current Table 1 scope.
#
# It creates a tiny 3-sample input file and runs:
#   gpt-4o, gpt5, gemini, deepseek-v3, qwen, glm-4.5
# across:
#   direct, cot, multiagent
#
# Run this before the full Table 1 batch. If any model returns 401/503,
# fix API permissions or MODEL_MAPPING before spending on the full benchmark.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SMOKE_SIZE="${SMOKE_SIZE:-3}"
SMOKE_PATH="${SMOKE_PATH:-results/table1_smoke_inputs_${SMOKE_SIZE}.json}"
mkdir -p "$(dirname "$SMOKE_PATH")"

python - <<PY
import json
from pathlib import Path

source = Path("data/benchmark/inputs.json")
target = Path("$SMOKE_PATH")
data = json.loads(source.read_text(encoding="utf-8"))
target.write_text(json.dumps(data[:$SMOKE_SIZE], indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {target} with {min($SMOKE_SIZE, len(data))} samples")
PY

export DATA_PATH="$SMOKE_PATH"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export SKIP_FINISHED=0
export BACKBONES="${BACKBONES:-gpt-4o gpt5 gemini deepseek-v3 qwen glm-4.5}"
export LOG_DIR="${LOG_DIR:-results/table1_smoke_logs}"

bash scripts/run_table1_all.sh

echo
echo "Smoke test finished. Inspect logs under $LOG_DIR."
echo "Search for API failures with:"
echo "  grep -R \"401\\|503\\|API call failed\\|LLM synthesis disabled\" $LOG_DIR"

