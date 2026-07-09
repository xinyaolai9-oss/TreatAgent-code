#!/usr/bin/env python3
"""Summarize conflicting drug-disease labels removed during benchmark deduplication."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "benchmark" / "processed"
DEFAULT_CONFLICTS = PROCESSED / "conflict_pairs.json"
DEFAULT_DATES = PROCESSED / "nctid_trial_dates.json"
DEFAULT_OUTPUT_JSON = PROCESSED / "conflict_analysis.json"
DEFAULT_OUTPUT_MD = PROCESSED / "conflict_analysis.md"
DEFAULT_CASES_JSON = PROCESSED / "conflict_case_candidates.json"

NEGATIVE_STATUSES = {
    "terminated",
    "withdrawn",
    "suspended",
}

POSITIVE_STATUSES = {
    "completed",
    "active, not recruiting",
    "approved for marketing",
}

BROAD_DISEASE_TERMS = {
    "cancer",
    "advanced solid tumors",
    "solid tumors",
    "neoplasms",
    "tumors",
    "inflammation",
    "pain",
}

COMBO_PATTERN = re.compile(
    r"\b(with|plus|and|or|combination|placebo|chemotherapy|radiation)\b|[,+/;]",
    flags=re.IGNORECASE,
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def norm_text(value: str | None) -> str:
    return (value or "").strip().lower()


def is_combo_drug_name(name: str) -> bool:
    return bool(COMBO_PATTERN.search(name or ""))


def label_counts(records: list[dict]) -> Counter:
    return Counter(str(int(record["label"])) for record in records)


def status_by_label(records: list[dict]) -> dict[str, Counter]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[str(int(record["label"]))][norm_text(record.get("status"))] += 1
    return {label: counter for label, counter in grouped.items()}


def date_by_label(records: list[dict], dates: dict) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        nctid = record.get("nctid")
        date_record = dates.get(nctid, {})
        submit_date = date_record.get("study_first_submit_date")
        if submit_date:
            grouped[str(int(record["label"]))].append(submit_date[:10])
    return {label: sorted(values) for label, values in grouped.items()}


def classify_conflict(pair: dict, dates: dict) -> tuple[list[str], str]:
    records = pair["records"]
    statuses = status_by_label(records)
    disease = norm_text(pair.get("normalized_disease"))
    drug_names = [record.get("drug", "") for record in records]

    tags = []
    if any(is_combo_drug_name(name) for name in drug_names):
        tags.append("possible_combination_drug_noise")
    if disease in BROAD_DISEASE_TERMS or "cancer" in disease or "tumor" in disease:
        tags.append("broad_disease_label")

    positive_statuses = set(statuses.get("1", Counter()))
    negative_statuses = set(statuses.get("0", Counter()))
    if positive_statuses & POSITIVE_STATUSES and negative_statuses & NEGATIVE_STATUSES:
        tags.append("status_driven_conflict")
    if positive_statuses & POSITIVE_STATUSES and negative_statuses & POSITIVE_STATUSES:
        tags.append("same_status_opposite_label")

    why_stop_negative = [
        norm_text(record.get("why_stop"))
        for record in records
        if int(record["label"]) == 0 and norm_text(record.get("why_stop"))
    ]
    if why_stop_negative:
        tags.append("negative_has_stop_reason")

    label_dates = date_by_label(records, dates)
    if label_dates.get("0") and label_dates.get("1"):
        if max(label_dates["1"]) < min(label_dates["0"]):
            tags.append("positive_before_negative")
        elif max(label_dates["0"]) < min(label_dates["1"]):
            tags.append("negative_before_positive")
        else:
            tags.append("date_overlap")

    if "possible_combination_drug_noise" in tags or "broad_disease_label" in tags:
        recommendation = "inspect_before_reuse"
    elif "same_status_opposite_label" in tags:
        recommendation = "candidate_conflict_case"
    elif "status_driven_conflict" in tags:
        recommendation = "candidate_temporal_or_status_case"
    else:
        recommendation = "keep_removed"

    return tags, recommendation


def summarize_pair(pair: dict, dates: dict) -> dict:
    records = pair["records"]
    tags, recommendation = classify_conflict(pair, dates)
    return {
        "canonical_smiles": pair["canonical_smiles"],
        "normalized_disease": pair["normalized_disease"],
        "record_count": len(records),
        "trial_count": len({record.get("nctid") for record in records if record.get("nctid")}),
        "label_counts": dict(label_counts(records)),
        "status_by_label": {
            label: dict(counter)
            for label, counter in status_by_label(records).items()
        },
        "phase_counts": dict(Counter(norm_text(record.get("phase")) for record in records)),
        "drug_names": sorted({record.get("drug", "") for record in records if record.get("drug")}),
        "nctids": sorted({record.get("nctid", "") for record in records if record.get("nctid")}),
        "date_by_label": date_by_label(records, dates),
        "tags": tags,
        "recommendation": recommendation,
        "example_records": records[:5],
    }


def build(conflicts_json: Path, dates_json: Path, output_json: Path, output_md: Path, cases_json: Path) -> None:
    conflicts = read_json(conflicts_json)
    dates = read_json(dates_json) if dates_json.exists() else {}

    pair_summaries = [summarize_pair(pair, dates) for pair in conflicts]
    recommendation_counts = Counter(row["recommendation"] for row in pair_summaries)
    tag_counts = Counter(tag for row in pair_summaries for tag in row["tags"])
    disease_counts = Counter(row["normalized_disease"] for row in pair_summaries)
    record_count_distribution = Counter(str(row["record_count"]) for row in pair_summaries)
    trial_count_distribution = Counter(str(row["trial_count"]) for row in pair_summaries)
    status_counts = Counter(
        norm_text(record.get("status"))
        for pair in conflicts
        for record in pair["records"]
    )

    case_candidates = sorted(
        [
            row
            for row in pair_summaries
            if row["recommendation"] in {"candidate_conflict_case", "candidate_temporal_or_status_case"}
        ],
        key=lambda row: (
            row["recommendation"] != "candidate_conflict_case",
            -row["trial_count"],
            row["normalized_disease"],
        ),
    )

    summary = {
        "conflict_pairs": len(conflicts),
        "conflict_records": sum(len(pair["records"]) for pair in conflicts),
        "unique_conflict_drugs": len({pair["canonical_smiles"] for pair in conflicts}),
        "unique_conflict_diseases": len({pair["normalized_disease"] for pair in conflicts}),
        "recommendation_counts": dict(recommendation_counts),
        "tag_counts": dict(tag_counts),
        "top_diseases": dict(disease_counts.most_common(25)),
        "record_count_distribution": dict(record_count_distribution),
        "trial_count_distribution": dict(trial_count_distribution),
        "status_counts": dict(status_counts),
        "case_candidate_count": len(case_candidates),
        "pairs": pair_summaries,
    }

    write_json(output_json, summary)
    write_json(cases_json, case_candidates[:50])

    lines = [
        "# Conflict Pair Analysis",
        "",
        "## Summary",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| conflict_pairs | {summary['conflict_pairs']} |",
        f"| conflict_records | {summary['conflict_records']} |",
        f"| unique_conflict_drugs | {summary['unique_conflict_drugs']} |",
        f"| unique_conflict_diseases | {summary['unique_conflict_diseases']} |",
        f"| case_candidate_count | {summary['case_candidate_count']} |",
        "",
        "## Recommendation Counts",
        "",
        "| Recommendation | Count |",
        "|---|---:|",
    ]
    for key, value in recommendation_counts.most_common():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Tag Counts", "", "| Tag | Count |", "|---|---:|"])
    for key, value in tag_counts.most_common():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Top Conflict Diseases", "", "| Disease | Count |", "|---|---:|"])
    for key, value in disease_counts.most_common(15):
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Candidate Cases", "", "| Disease | Trials | Labels | Tags | Recommendation |", "|---|---:|---|---|---|"])
    for row in case_candidates[:20]:
        labels = json.dumps(row["label_counts"], ensure_ascii=False)
        tags = ", ".join(row["tags"])
        lines.append(
            f"| {row['normalized_disease']} | {row['trial_count']} | {labels} | {tags} | {row['recommendation']} |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    print(f"Wrote {cases_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze removed conflict pairs.")
    parser.add_argument("--conflicts_json", type=Path, default=DEFAULT_CONFLICTS)
    parser.add_argument("--dates_json", type=Path, default=DEFAULT_DATES)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--cases_json", type=Path, default=DEFAULT_CASES_JSON)
    args = parser.parse_args()
    build(args.conflicts_json, args.dates_json, args.output_json, args.output_md, args.cases_json)


if __name__ == "__main__":
    main()
