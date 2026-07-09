#!/usr/bin/env python3
"""Canonicalize SMILES and build a pair-level benchmark dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data" / "benchmark" / "processed" / "extracted_with_meta.json"
DEFAULT_OUTPUT = ROOT / "data" / "benchmark" / "processed" / "pair_level_dataset.json"
DEFAULT_CONFLICTS = ROOT / "data" / "benchmark" / "processed" / "conflict_pairs.json"
DEFAULT_STATS = ROOT / "data" / "benchmark" / "processed" / "dedup_stats.json"


def load_rdkit():
    try:
        from rdkit import Chem  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "RDKit is required for formal canonicalization. "
            "Run this script in the WSL/conda environment with RDKit installed."
        ) from exc
    return Chem


def normalize_disease(name: str) -> str:
    normalized = name.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    aliases = {
        "hiv infections": "hiv infection",
        "alzheimer disease": "alzheimer's disease",
        "neoplasms": "cancer",
    }
    return aliases.get(normalized, normalized)


def canonicalize(smiles: str, chem) -> tuple[str | None, str]:
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "failed"
    return chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), "ok"


def build(input_json: Path, output_json: Path, conflicts_json: Path, stats_json: Path) -> None:
    chem = load_rdkit()
    records = json.loads(input_json.read_text(encoding="utf-8"))

    canonicalized = []
    invalid = []
    for item in records:
        canonical, status = canonicalize(item["smiles"], chem)
        enriched = dict(item)
        enriched["normalized_disease"] = normalize_disease(item["disease"])
        enriched["canonical_smiles"] = canonical
        enriched["canonicalization_status"] = status
        if canonical is None:
            invalid.append(enriched)
            continue
        canonicalized.append(enriched)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in canonicalized:
        grouped[(item["canonical_smiles"], item["normalized_disease"])].append(item)

    pair_rows = []
    conflicts = []
    for pair_id, ((canonical_smiles, normalized_disease), items) in enumerate(sorted(grouped.items())):
        labels = {int(item["label"]) for item in items}
        if len(labels) > 1:
            conflicts.append(
                {
                    "canonical_smiles": canonical_smiles,
                    "normalized_disease": normalized_disease,
                    "labels": sorted(labels),
                    "nctids": sorted({item["nctid"] for item in items if item.get("nctid")}),
                    "records": items,
                }
            )
            continue

        label = labels.pop()
        phases = sorted({item["phase"] for item in items if item.get("phase")})
        statuses = sorted({item["status"] for item in items if item.get("status")})
        drugs = sorted({item["drug"] for item in items if item.get("drug")})
        nctids = sorted({item["nctid"] for item in items if item.get("nctid")})
        row_ids = sorted({item["source_row_id"] for item in items})
        pair_rows.append(
            {
                "pair_id": f"PAIR-{pair_id:06d}",
                "label": label,
                "canonical_smiles": canonical_smiles,
                "normalized_disease": normalized_disease,
                "example_smiles": items[0]["smiles"],
                "example_disease": items[0]["disease"],
                "drugs": drugs,
                "nctids": nctids,
                "phases": phases,
                "statuses": statuses,
                "source_row_ids": row_ids,
                "trial_count": len(nctids),
                "record_count": len(items),
            }
        )

    stats = {
        "input_rows": len(records),
        "valid_canonical_smiles_rows": len(canonicalized),
        "invalid_smiles_rows": len(invalid),
        "unique_pair_candidates": len(grouped),
        "pair_rows": len(pair_rows),
        "conflict_pairs_removed": len(conflicts),
        "label_counts": dict(Counter(str(row["label"]) for row in pair_rows)),
        "unique_drugs": len({row["canonical_smiles"] for row in pair_rows}),
        "unique_diseases": len({row["normalized_disease"] for row in pair_rows}),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(pair_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    conflicts_json.write_text(json.dumps(conflicts, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(pair_rows)} pair rows to {output_json}")
    print(f"Wrote {len(conflicts)} conflict pairs to {conflicts_json}")
    print(f"Wrote stats to {stats_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conflicts_json", type=Path, default=DEFAULT_CONFLICTS)
    parser.add_argument("--stats_json", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args()
    build(args.input_json, args.output_json, args.conflicts_json, args.stats_json)


if __name__ == "__main__":
    main()
