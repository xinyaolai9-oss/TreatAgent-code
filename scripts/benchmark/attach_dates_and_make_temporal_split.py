#!/usr/bin/env python3
"""Attach trial date metadata to pair-level rows and build temporal splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "benchmark" / "processed"
SPLITS = ROOT / "data" / "benchmark" / "splits"
DEFAULT_PAIRS = PROCESSED / "pair_level_dataset.json"
DEFAULT_DATES = PROCESSED / "nctid_trial_dates.json"
DEFAULT_OUTPUT = PROCESSED / "pair_level_dataset_with_dates.json"
DEFAULT_UNDATED = PROCESSED / "pair_level_undated.json"
DEFAULT_STATS = PROCESSED / "temporal_split_stats.json"


DATE_PRIORITY = [
    "primary_completion_date",
    "completion_date",
    "study_first_submit_date",
    "start_date",
]

DATE_POLICIES = {
    "priority": DATE_PRIORITY,
    "study_first_submit": ["study_first_submit_date"],
}


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def best_trial_date(record: dict, date_fields: list[str]) -> tuple[str | None, str | None]:
    for field in date_fields:
        value = record.get(field)
        if parse_date(value):
            return value[:10], field
    return None, None


def attach_dates(pairs: list[dict], dates: dict, date_fields: list[str]) -> tuple[list[dict], list[dict]]:
    dated = []
    undated = []
    for row in pairs:
        candidates = []
        for nctid in row.get("nctids", []):
            trial = dates.get(nctid)
            if not trial:
                continue
            value, source = best_trial_date(trial, date_fields)
            parsed = parse_date(value)
            if parsed:
                candidates.append((parsed, value, source, nctid))

        enriched = dict(row)
        if not candidates:
            enriched["pair_date"] = None
            enriched["pair_date_source"] = None
            enriched["pair_date_nctid"] = None
            undated.append(enriched)
            continue

        candidates.sort(key=lambda item: item[0])
        _, value, source, nctid = candidates[0]
        enriched["pair_date"] = value
        enriched["pair_date_source"] = source
        enriched["pair_date_nctid"] = nctid
        dated.append(enriched)
    return dated, undated


def temporal_split(rows: list[dict], ratios=(0.7, 0.1, 0.2)) -> dict[str, list[dict]]:
    rows = sorted(rows, key=lambda row: (row["pair_date"], row["pair_id"]))
    n = len(rows)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    return {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {
            "rows": 0,
            "label_counts": {},
            "unique_drugs": 0,
            "unique_diseases": 0,
            "date_min": None,
            "date_max": None,
        }
    return {
        "rows": len(rows),
        "label_counts": dict(Counter(str(row["label"]) for row in rows)),
        "unique_drugs": len({row["canonical_smiles"] for row in rows}),
        "unique_diseases": len({row["normalized_disease"] for row in rows}),
        "date_min": min(row["pair_date"] for row in rows),
        "date_max": max(row["pair_date"] for row in rows),
    }


def build(
    pair_json: Path,
    dates_json: Path,
    output_json: Path,
    undated_json: Path,
    output_dir: Path,
    stats_json: Path,
    split_prefix: str,
    date_policy: str,
) -> None:
    pairs = json.loads(pair_json.read_text(encoding="utf-8"))
    dates = json.loads(dates_json.read_text(encoding="utf-8"))
    date_fields = DATE_POLICIES[date_policy]
    dated, undated = attach_dates(pairs, dates, date_fields)
    splits = temporal_split(dated)

    output_json.write_text(json.dumps(dated, indent=2, ensure_ascii=False), encoding="utf-8")
    undated_json.write_text(json.dumps(undated, indent=2, ensure_ascii=False), encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (output_dir / f"{split_prefix}_{split}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    stats = {
        "date_policy": date_policy,
        "date_fields": date_fields,
        "split_prefix": split_prefix,
        "pair_rows": len(pairs),
        "dated_rows": len(dated),
        "undated_rows": len(undated),
        "date_source_counts": dict(Counter(row["pair_date_source"] for row in dated)),
        "temporal": {split: summarize(rows) for split, rows in splits.items()},
    }
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote dated pairs to {output_json}")
    print(f"Wrote undated pairs to {undated_json}")
    print(f"Wrote temporal splits to {output_dir} with prefix {split_prefix}")
    print(f"Wrote stats to {stats_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_json", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--dates_json", type=Path, default=DEFAULT_DATES)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--undated_json", type=Path, default=DEFAULT_UNDATED)
    parser.add_argument("--output_dir", type=Path, default=SPLITS)
    parser.add_argument("--stats_json", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--split_prefix", default="temporal")
    parser.add_argument("--date_policy", choices=sorted(DATE_POLICIES), default="priority")
    args = parser.parse_args()
    build(
        args.pair_json,
        args.dates_json,
        args.output_json,
        args.undated_json,
        args.output_dir,
        args.stats_json,
        args.split_prefix,
        args.date_policy,
    )


if __name__ == "__main__":
    main()
