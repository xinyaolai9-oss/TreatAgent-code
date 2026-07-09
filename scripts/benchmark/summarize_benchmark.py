#!/usr/bin/env python3
"""Collect benchmark processing statistics into one JSON and Markdown summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "benchmark" / "processed"
DEFAULT_OUTPUT_JSON = PROCESSED / "benchmark_stats.json"
DEFAULT_OUTPUT_MD = PROCESSED / "benchmark_stats.md"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build(output_json: Path, output_md: Path) -> None:
    extraction = read_json(PROCESSED / "extraction_stats.json")
    dedup = read_json(PROCESSED / "dedup_stats.json")
    splits = read_json(PROCESSED / "split_stats.json")
    temporal = read_json(PROCESSED / "temporal_split_stats.json")
    temporal_submit = read_json(PROCESSED / "temporal_submit_split_stats.json")
    stats = {
        "extraction": extraction,
        "dedup": dedup,
        "splits": splits,
        "temporal": temporal,
        "temporal_study_first_submit": temporal_submit,
    }
    output_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Benchmark Statistics",
        "",
        "## Extraction",
        "",
        "| Statistic | Value |",
        "|---|---:|",
    ]
    for key, value in extraction.items():
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Pair-Level Deduplication", "", "| Statistic | Value |", "|---|---:|"])
    for key, value in dedup.items():
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"| {key} | {value} |")

    if splits:
        lines.extend(["", "## Splits", ""])
        for split_type, split_stats in splits.items():
            if split_type == "seed":
                continue
            lines.extend([f"### {split_type}", "", "| Split | Rows | Labels | Unique Drugs | Unique Diseases |", "|---|---:|---|---:|---:|"])
            for split_name in ["train", "val", "test"]:
                row = split_stats.get(split_name, {})
                labels = json.dumps(row.get("label_counts", {}), ensure_ascii=False)
                lines.append(
                    f"| {split_name} | {row.get('rows', 0)} | {labels} | "
                    f"{row.get('unique_drugs', 0)} | {row.get('unique_diseases', 0)} |"
                )
            overlap = split_stats.get("drug_overlap", {})
            lines.extend(["", f"Drug overlap: `{json.dumps(overlap, ensure_ascii=False)}`", ""])

    if temporal:
        lines.extend(["", "## Temporal Split", "", "| Statistic | Value |", "|---|---:|"])
        for key in ["pair_rows", "dated_rows", "undated_rows", "date_source_counts"]:
            value = temporal.get(key)
            if isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f"| {key} | {value} |")
        lines.extend(["", "| Split | Rows | Labels | Date Range |", "|---|---:|---|---|"])
        for split in ["train", "val", "test"]:
            row = temporal.get("temporal", {}).get(split, {})
            labels = json.dumps(row.get("label_counts", {}), ensure_ascii=False)
            date_range = f"{row.get('date_min')} to {row.get('date_max')}"
            lines.append(f"| {split} | {row.get('rows', 0)} | {labels} | {date_range} |")

    if temporal_submit:
        lines.extend(["", "## Temporal Split: Study First Submit Only", "", "| Statistic | Value |", "|---|---:|"])
        for key in ["date_policy", "pair_rows", "dated_rows", "undated_rows", "date_source_counts"]:
            value = temporal_submit.get(key)
            if isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f"| {key} | {value} |")
        lines.extend(["", "| Split | Rows | Labels | Date Range |", "|---|---:|---|---|"])
        for split in ["train", "val", "test"]:
            row = temporal_submit.get("temporal", {}).get(split, {})
            labels = json.dumps(row.get("label_counts", {}), ensure_ascii=False)
            date_range = f"{row.get('date_min')} to {row.get('date_max')}"
            lines.append(f"| {split} | {row.get('rows', 0)} | {labels} | {date_range} |")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()
    build(args.output_json, args.output_md)


if __name__ == "__main__":
    main()
