## Assets

This directory stores runtime resources that are needed by TreatAgent but are not source code.

### Structure

- `models/`
  - Local pretrained model parameters used at inference time.
  - Current example: `models/model_MPNN_CNN/`
- `templates/`
  - HTML and other render templates.
  - Current example: `templates/report_template.html`
- `tools/`
  - Small standalone analysis utilities kept out of the project root.
  - Current examples: `tools/analyze_thresholds.py`, `tools/bootstrap.py`

### Notes

- Code should prefer paths under `assets/` instead of root-level resource folders.
- Large generated outputs such as reports, checkpoints, and results should not be stored here.
