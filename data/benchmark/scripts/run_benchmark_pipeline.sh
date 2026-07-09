#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

python data/benchmark/scripts/build_extracted_with_meta.py
python data/benchmark/scripts/canonicalize_and_dedup.py
python data/benchmark/scripts/make_splits.py
if [[ -f data/benchmark/processed/nctid_trial_dates.json ]]; then
  python data/benchmark/scripts/attach_dates_and_make_temporal_split.py
  python data/benchmark/scripts/attach_dates_and_make_temporal_split.py \
    --date_policy study_first_submit \
    --split_prefix temporal_submit \
    --output_json data/benchmark/processed/pair_level_dataset_with_submit_dates.json \
    --undated_json data/benchmark/processed/pair_level_submit_undated.json \
    --stats_json data/benchmark/processed/temporal_submit_split_stats.json
fi
python data/benchmark/scripts/summarize_benchmark.py
python data/benchmark/scripts/export_cli_inputs.py
python data/benchmark/scripts/analyze_conflict_pairs.py

echo
echo "Benchmark pipeline completed."
echo "Processed files: data/benchmark/processed/"
echo "Split files: data/benchmark/splits/"
echo "TreatAgent CLI input files: data/benchmark/split_inputs/"
