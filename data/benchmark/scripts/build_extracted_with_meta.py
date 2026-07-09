#!/usr/bin/env python3
"""Build a metadata-preserving benchmark extraction from raw_data.csv."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data" / "benchmark" / "raw_data.csv"
DEFAULT_OUTPUT = ROOT / "data" / "benchmark" / "processed" / "extracted_with_meta.json"
DEFAULT_STATS = ROOT / "data" / "benchmark" / "processed" / "extraction_stats.json"


BAD_DISEASE_TOKENS = [",", "-", "'", ";", ":", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\"]


def parse_list(value: str):
    if not value:
        return []
    parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return parsed
    return []


def is_clean_single_disease(disease: str) -> bool:
    disease_lower = disease.lower().strip()
    if disease_lower == "healthy":
        return False
    if any(token in disease for token in BAD_DISEASE_TOKENS):
        return False
    if any(char.isdigit() for char in disease):
        return False
    return True


def build(input_csv: Path, output_json: Path, stats_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    stats = {
        "raw_rows": 0,
        "parse_errors": 0,
        "single_disease_single_smiles_rows": 0,
        "filtered_disease_rows": 0,
        "kept_rows": 0,
        "label_counts": {"0": 0, "1": 0},
    }

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_id, row in enumerate(reader):
            stats["raw_rows"] += 1
            try:
                diseases = parse_list(row.get("diseases", ""))
                smiless = parse_list(row.get("smiless", ""))
                drugs = parse_list(row.get("drugs", ""))
                icdcodes = parse_list(row.get("icdcodes", ""))
            except (ValueError, SyntaxError):
                stats["parse_errors"] += 1
                continue

            if len(diseases) != 1 or len(smiless) != 1:
                continue
            stats["single_disease_single_smiles_rows"] += 1

            disease = str(diseases[0]).strip()
            if not is_clean_single_disease(disease):
                stats["filtered_disease_rows"] += 1
                continue

            label = int(row["label"])
            entry = {
                "source_row_id": row_id,
                "nctid": row.get("nctid", "").strip(),
                "status": row.get("status", "").strip(),
                "why_stop": row.get("why_stop", "").strip(),
                "label": label,
                "phase": row.get("phase", "").strip(),
                "disease": disease,
                "icdcodes": icdcodes,
                "drug": str(drugs[0]).strip() if len(drugs) == 1 else None,
                "smiles": str(smiless[0]).strip(),
            }
            rows.append(entry)
            stats["kept_rows"] += 1
            stats["label_counts"][str(label)] += 1

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with stats_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rows)} rows to {output_json}")
    print(f"Wrote stats to {stats_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats_json", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args()
    build(args.input_csv, args.output_json, args.stats_json)


if __name__ == "__main__":
    main()
