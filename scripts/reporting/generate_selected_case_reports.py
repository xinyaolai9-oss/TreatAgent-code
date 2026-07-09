#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from treatagent.reporting.generator import InteractiveReportGenerator

DEFAULT_RESULT_FILES = [
    PROJECT_ROOT / "results" / "gpt-4o" / "results_multiagent_full_20260524_132127.json",
    PROJECT_ROOT / "results" / "gpt-4o" / "results_multiagent_full_20260524_154143.json",
]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "tutor_case_studies"

CASE_ROLES = {
    "PAIR-000700": "Early-STOP positive case; online planner stopped after Clinical because disease-level feasibility was sufficient. Use the all-expert EvidenceGraph figure for direct-support visualization.",
    "PAIR-001194": "Mechanism-only positive case; no direct indication, but disease-side target and therapy context support mechanism rescue.",
    "PAIR-000147": "Low-clinical-prior true negative; weak support is outweighed by clinical/ADMET risk signals.",
    "PAIR-000524": "Support-conflict mixed failure case; support evidence overwhelms insufficient conflict penalty.",
    "PAIR-002087": "Missing/weak evidence true negative; sparse support and low clinical feasibility lead to a negative decision.",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported result JSON format: {path}")


def evidence_count(row: dict[str, Any]) -> int:
    return len((row.get("evidence_graph") or {}).get("evidence") or [])


def collect_cases(paths: list[Path], sample_ids: list[str]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {sample_id: [] for sample_id in sample_ids}
    for path in paths:
        if not path.exists():
            continue
        for row in load_rows(path):
            sample_id = str(row.get("sample_id") or "")
            if sample_id in candidates:
                enriched = dict(row)
                enriched["_source_file"] = str(path)
                candidates[sample_id].append(enriched)

    selected = {}
    for sample_id, rows in candidates.items():
        if not rows:
            continue
        selected[sample_id] = sorted(
            rows,
            key=lambda row: (len(row.get("expert_outputs") or {}), evidence_count(row)),
            reverse=True,
        )[0]
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate HTML reports for selected TreatAgent case-study samples.")
    parser.add_argument("--result_json", action="append", type=Path, default=[])
    parser.add_argument("--sample_id", action="append", default=[])
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result_files = args.result_json or DEFAULT_RESULT_FILES
    sample_ids = args.sample_id or list(CASE_ROLES)
    generator = InteractiveReportGenerator(report_dir=str(args.report_dir))
    selected = collect_cases(result_files, sample_ids)

    summaries = []
    for sample_id in sample_ids:
        row = selected.get(sample_id)
        if not row:
            print(f"Missing sample: {sample_id}")
            continue
        report_path, summary = generator.generate_from_result(row, case_role=CASE_ROLES.get(sample_id))
        summary["source_file"] = row.get("_source_file")
        summary["expert_count"] = len(row.get("expert_outputs") or {})
        summary["evidence_count"] = evidence_count(row)
        summaries.append(summary)
        print(f"Wrote {sample_id}: {report_path}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.report_dir / "case_report_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
