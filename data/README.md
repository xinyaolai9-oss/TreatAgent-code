# Data Layout

This directory stores the benchmark, runtime expert data, knowledge snapshots,
priors, and dataset-specific preparation scripts included with the standalone
TreatAgent release package.

This package is intended to be runnable after cloning when Git LFS files have
been fetched. It includes processed/runtime expert artifacts rather than large
third-party raw dumps.

## Current Structure

- `benchmark/`
  - Benchmark datasets.
  - `raw_data.csv`: original tabular source.
  - `scripts/`: benchmark-specific preparation and pipeline scripts.
  - `legacy/`: old pre-split inputs from the 395-sample setup.
  - `processed/`: cleaned pair-level benchmark artifacts.
  - `splits/`: full split records.
  - `split_inputs/`: TreatAgent CLI input files for each split.

- `clinical/`
  - Lightweight clinical priors used by the local Clinical expert.
  - `disease_success_ratio.json`: disease-level clinical success prior.

- `drugcentral/`
  - DrugKB assets derived from DrugCentral.
  - Includes generated `drugkb.jsonl`, coverage audit files, and DrugKB builder
    scripts.
  - The raw DrugCentral SQL dump is not included because it is a multi-GB
    third-party source file and cannot be pushed to a normal GitHub repository.

- `diseasekb/`
  - DiseaseKB assets derived from MONDO and Open Targets.
  - Includes generated `diseasekb.jsonl` and DiseaseKB builder scripts.
  - Raw MONDO/Open Targets snapshots are not included; rebuild them from the
    original sources if needed.

- `dti/`
  - UniProt sequence cache used by the DTI expert.

## GitHub Large File Note

`data/diseasekb/diseasekb.jsonl` is larger than GitHub's normal per-file limit
for regular Git pushes. This release package includes `.gitattributes` so it can
be tracked with Git LFS.

## Recommended Convention

- Keep reusable raw or curated datasets under `data/`.
- Keep dataset-specific Python preparation scripts under that dataset's `scripts/` subdirectory.
- Keep generic experiment shell/batch scripts under top-level `scripts/`.
- Keep lightweight clinical priors under `data/clinical/`.
- Keep large domain knowledge snapshots under dedicated folders such as `data/drugcentral/` and `data/diseasekb/`.
