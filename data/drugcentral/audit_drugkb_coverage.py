#!/usr/bin/env python3
"""Audit DrugKB matching coverage for TreatAgent benchmark inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [
    ROOT / "data" / "benchmark" / "split_inputs" / "drug_disjoint_train.json",
    ROOT / "data" / "benchmark" / "split_inputs" / "drug_disjoint_val.json",
    ROOT / "data" / "benchmark" / "split_inputs" / "drug_disjoint_test.json",
    ROOT / "data" / "benchmark" / "split_inputs" / "temporal_submit_train.json",
    ROOT / "data" / "benchmark" / "split_inputs" / "temporal_submit_val.json",
    ROOT / "data" / "benchmark" / "split_inputs" / "temporal_submit_test.json",
]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "drugcentral" / "coverage"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from treatagent.skills.drug_kb import DrugKBExpert  # noqa: E402


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def audit_file(path: Path, expert: DrugKBExpert) -> Dict[str, Any]:
    rows = load_json(path)
    by_method: Counter[str] = Counter()
    by_label: Dict[str, Counter[str]] = {"0": Counter(), "1": Counter()}
    matched_examples: List[Dict[str, Any]] = []
    unmatched_examples: List[Dict[str, Any]] = []

    unique_smiles = set()
    unique_matched_drugs = set()

    for row in rows:
        smiles = row.get("smiles") or row.get("canonical_smiles") or row.get("example_smiles") or ""
        drug_names = as_list(row.get("drugs") or row.get("drug_names") or row.get("drug"))
        identifiers = as_list(row.get("drug_identifiers") or row.get("identifiers"))
        inchikey = row.get("inchikey") or row.get("drug_inchikey")
        record, matched_by, match_score = expert.lookup(
            smiles,
            drug_names=drug_names,
            identifiers=identifiers,
            inchikey=inchikey,
        )
        by_method[matched_by] += 1
        by_label[str(int(row.get("label", 0)))][matched_by] += 1
        unique_smiles.add(smiles)

        item = {
            "sample_id": row.get("sample_id") or row.get("pair_id"),
            "label": row.get("label"),
            "disease": row.get("disease") or row.get("normalized_disease") or row.get("example_disease"),
            "query_drugs": drug_names,
            "matched_by": matched_by,
            "match_score": match_score,
            "matched_drug": record.get("drug_name") if record else None,
        }
        if record:
            unique_matched_drugs.add(record.get("drugcentral_id") or record.get("drug_name"))
            if len(matched_examples) < 20:
                matched_examples.append(item)
        elif len(unmatched_examples) < 20:
            unmatched_examples.append(item)

    total = len(rows)
    matched = total - by_method.get("unmatched", 0)
    return {
        "input": str(path.relative_to(ROOT)),
        "rows": total,
        "unique_smiles": len(unique_smiles),
        "matched_rows": matched,
        "matched_rate": round(matched / total, 4) if total else 0.0,
        "unique_matched_drugcentral_records": len(unique_matched_drugs),
        "match_methods": dict(sorted(by_method.items())),
        "match_methods_by_label": {label: dict(counter) for label, counter in by_label.items()},
        "matched_examples": matched_examples,
        "unmatched_examples": unmatched_examples,
    }


def render_markdown(results: Iterable[Dict[str, Any]]) -> str:
    lines = [
        "# DrugKB Coverage Audit",
        "",
        "This audit reports how benchmark drug inputs map to the local DrugCentral-derived DrugKB.",
        "",
        "| split | rows | matched | matched rate | unique SMILES | unique matched DrugCentral records | top match methods |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        methods = ", ".join(f"{name}: {count}" for name, count in row["match_methods"].items())
        lines.append(
            f"| {Path(row['input']).name} | {row['rows']} | {row['matched_rows']} | "
            f"{row['matched_rate']:.3f} | {row['unique_smiles']} | "
            f"{row['unique_matched_drugcentral_records']} | {methods} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DrugKB coverage over TreatAgent benchmark split inputs.")
    parser.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    expert = DrugKBExpert()
    results = [audit_file(path.resolve(), expert) for path in args.inputs if path.exists()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "drugkb_coverage_audit.json", results)
    (args.output_dir / "drugkb_coverage_audit.md").write_text(render_markdown(results), encoding="utf-8")

    print(f"Wrote coverage audit to {args.output_dir}")
    for row in results:
        print(f"{Path(row['input']).name}: matched {row['matched_rows']}/{row['rows']} ({row['matched_rate']:.1%})")


if __name__ == "__main__":
    main()
