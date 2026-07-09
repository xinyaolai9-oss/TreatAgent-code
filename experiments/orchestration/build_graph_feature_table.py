#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from treatagent.orchestration.features import FEATURE_NAMES, feature_row_from_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "feature_tables" / "graph_features.csv"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "results" / "feature_tables" / "graph_features.jsonl"


ID_COLUMNS = [
    "source_file",
    "method",
    "backbone",
    "sample_id",
    "label",
    "prediction_binary",
    "prediction_score",
    "calibrated_probability",
    "raw_score",
    "smiles",
    "disease",
]


def iter_result_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("results_multiagent_*.json")))
        else:
            files.append(path)
    return files


def read_results(path: Path) -> tuple[dict, list[dict]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, payload.get("results") or []


def build_rows(input_paths: list[Path]) -> list[dict]:
    rows = []
    for path in iter_result_files(input_paths):
        payload, results = read_results(path)
        method = payload.get("method")
        backbone = path.parent.name
        for result in results:
            if not result.get("evidence_graph"):
                continue
            row = feature_row_from_result(result)
            row["source_file"] = str(path)
            row["method"] = method or result.get("method")
            row["backbone"] = backbone
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ID_COLUMNS + FEATURE_NAMES
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build EvidenceGraph feature tables from TreatAgent result JSON files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Result JSON files or directories containing result JSON files.")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output_jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = build_rows(args.inputs)
    write_csv(args.output_csv, rows)
    write_jsonl(args.output_jsonl, rows)
    print(f"Wrote {len(rows)} feature rows to {args.output_csv}")
    print(f"Wrote {len(rows)} feature rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()

