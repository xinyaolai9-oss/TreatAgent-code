#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARCHIVE_DIR="${ARCHIVE_DIR:-results/archive_legacy_395_20260516}"

mkdir -p "$ARCHIVE_DIR/local_legacy_20260419"

shopt -s nullglob

# Old local prototype/smoke/full raw runs. Keep the new EG split runs active:
# results/local/results_multiagent_eg_20260515_*.json
for file in results/local/results_multiagent_20260419_*.json; do
  mv -f "$file" "$ARCHIVE_DIR/local_legacy_20260419/"
done

legacy_paths=(
  "results/gpt-4o"
  "results/gpt5"
  "results/gemini"
  "results/deepseek-v3"
  "results/qwen"
  "results/table1_logs"
  "results/table1_dmx_logs"
  "results/table1_smoke_logs"
  "results/eg_scorer/smoke"
  "results/table1_background.log"
  "results/table1_dmx.pid"
  "results/table1_dmx_background.log"
  "results/table1_dmx_background_deepseek-v3.log"
  "results/table1_dmx_background_gemini.log"
  "results/table1_dmx_background_glm-4.5.log"
  "results/table1_dmx_background_qwen.log"
  "results/table1_dmx_background_resume.log"
  "results/table1_dmx_parallel.pids"
  "results/table1_smoke_inputs_3.json"
  "resultstable1_background.log"
)

for path in "${legacy_paths[@]}"; do
  if [[ -e "$path" ]]; then
    mv -f "$path" "$ARCHIVE_DIR/"
  fi
done

cat > "$ARCHIVE_DIR/ARCHIVE_MANIFEST.txt" <<EOF
Archive created: $(date -Iseconds)
Purpose: legacy 395-sample/Table1/smoke results moved out of active results.
Kept active:
- results/local/results_multiagent_eg_20260515_*.json
- results/eg_scorer/drug_disjoint_eg/

Archived path groups:
- old Table1 backbone directories: gpt-4o, gpt5, gemini, deepseek-v3, qwen
- old Table1/smoke logs and background logs
- old local 20260419 prototype result JSON files
- smoke EG scorer outputs
EOF

echo "Archive: $ARCHIVE_DIR"
find "$ARCHIVE_DIR" -maxdepth 2 -mindepth 1 -printf "%p\n" | sort

