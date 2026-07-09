# TreatAgent

TreatAgent is an evidence-driven multi-agent system for drug-disease treatment prediction. The current main path combines local ADMET, DTI, and clinical-prior tools with a dynamic orchestrator, structured evidence graph, calibrated scoring, optional interactive HTML reports, and optional vector-based long-term memory retrieval.

This standalone release package includes the full processed benchmark and the
runtime expert data needed by the local experts:

- `data/benchmark/`: processed benchmark, splits, split inputs and benchmark scripts
- `data/drugcentral/drugkb.jsonl`: processed DrugKB runtime index
- `data/diseasekb/diseasekb.jsonl`: processed DiseaseKB runtime index
- `data/clinical/disease_success_ratio.json`: disease-level clinical prior
- `data/dti/uniprot_sequence_cache.json`: UniProt sequence cache for DTI
- `assets/models/model_MPNN_CNN/`: local DeepPurpose DTI model
- `assets/templates/report_template.html`: optional HTML report template

Before pushing this repository to GitHub, install Git LFS and track the large
runtime data files listed in `.gitattributes`.

## Current Modes

### `multiagent`
- Main production path.
- Runs the orchestrator-based workflow through `python -m treatagent.cli`.
- Supports both hybrid mode and fully local mode.
- Use `--backbone local` to disable LLM synthesis and force fully local inference.
- Supports `--generate_report` and `--use_memory`.

### `direct`
- Single-shot LLM baseline.
- Requires `URL_VALUE` and `API_VALUE` environment variables for the remote API endpoint.

### `cot`
- Chain-of-thought style LLM baseline.
- Also requires `URL_VALUE` and `API_VALUE` environment variables.

## Installation

```bash
pip install -r requirements.txt
```

Core local dependencies:
- `admet_ai`
- `DeepPurpose==0.1.5`
- `chromadb`
- `sentence-transformers`

## Main Workflow

Input:
- `SMILES`
- `disease name`

Loop:
1. Planner inspects current evidence.
2. Planner selects the next expert or stops.
3. Expert returns structured evidence.
4. Evidence is added into the graph.
5. Synthesis produces a raw score and explanation.
6. Calibrator maps the raw score to a probability.
7. Optional report generation writes an HTML dashboard.
8. Optional memory retrieval/storage uses ChromaDB plus sentence embeddings.

## Run

### Standard multi-agent inference

```bash
python -m treatagent.cli --json_path data/benchmark/split_inputs/drug_disjoint_test.json --method multiagent --backbone gpt-4o
```

### Multi-agent with HTML reports

```bash
python -m treatagent.cli --json_path data/benchmark/split_inputs/drug_disjoint_test.json --method multiagent --backbone gpt-4o --generate_report
```

### Fully local multi-agent inference

```bash
python -m treatagent.cli --json_path data/benchmark/split_inputs/drug_disjoint_test.json --method multiagent --backbone local
```

This mode disables LLM synthesis and uses only local experts plus heuristic synthesis.

### Multi-agent with report and long-term memory

```bash
python -m treatagent.cli --json_path data/benchmark/split_inputs/drug_disjoint_test.json --method multiagent --backbone gpt-4o --generate_report --use_memory
```

### Resume from checkpoint

```bash
python -m treatagent.cli --json_path data/benchmark/split_inputs/drug_disjoint_test.json --method multiagent --backbone gpt-4o --generate_report --use_memory --resume
```

## Output Directories

- `results/<model>/results_multiagent.json`: batch prediction outputs
- `reports/`: self-contained HTML dashboards
- `memory_db/`: ChromaDB persistent vector memory
- `assets/models/model_MPNN_CNN/`: local DeepPurpose DTI model directory
- `assets/templates/report_template.html`: HTML report template
- `checkpoints/`: resumable progress checkpoints

## Output Fields

The current `multiagent` result entries may include:
- `prediction_binary`
- `raw_score`
- `calibrated_probability`
- `synthesis_explanation`
- `trajectory`
- `evidence_graph`
- `expert_outputs`
- `report_path`
- `report_summary`
- `memory_similar_cases`
- `stored_case_id`

## Validation Checklist

- [ ] Running with `--generate_report` creates HTML files under `reports/`.
- [ ] Opening a generated report shows the metric cards, ADMET radar chart, DTI gauge, clinical progress bar, evidence table, and synthesis explanation.
- [ ] Running with `--use_memory` creates `memory_db/`.
- [ ] Re-running with `--use_memory` increases stored case count in the vector database.
- [ ] Result JSON contains `report_path` and `memory_similar_cases`.
- [ ] Result JSON no longer depends on `memory_update`.

## Notes

- The current `multiagent` path is local-tool driven and does not need API keys.
- `--backbone local` explicitly forces fully local inference and skips all LLM synthesis calls.
- `direct` and `cot` still require remote API configuration.
- `orchestrator_system.py` is now a compatibility layer that re-exports the enhanced orchestrator API.

## Data

The current benchmark pipeline is submitted through:

```bash
bash data/benchmark/scripts/run_benchmark_pipeline.sh
```

The benchmark preparation steps themselves live under `data/benchmark/scripts/`. The legacy single-disease/single-SMILES extractor is available through:

```bash
python data/benchmark/scripts/extract_dataset_legacy.py
```

Knowledge-base builders live beside their data snapshots:

```bash
python data/drugcentral/build_drugkb_jsonl.py
python data/diseasekb/build_diseasekb_jsonl.py
```

## Acknowledgements

- [Clinical Trial Outcome Prediction](https://github.com/futianfan/clinical-trial-outcome-prediction/)
- [DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose)
- [ADMET-AI](https://github.com/swansonk14/admet_ai)
- [ChEMBL](https://github.com/chembl/chembl_webresource_client)
