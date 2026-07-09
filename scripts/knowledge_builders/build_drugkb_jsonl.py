#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple


TABLES = {
    "structures",
    "synonyms",
    "approval",
    "identifier",
    "act_table_full",
    "omop_relationship",
    "pharma_class",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "drugcentral"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DrugKB JSONL from a DrugCentral PostgreSQL dump."
    )
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=DATA_DIR / "drugcentral.dump.11012023.sql",
        help="Path to the DrugCentral .sql dump file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "drugkb.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=12,
        help="Maximum number of target/mechanism entries kept per drug.",
    )
    parser.add_argument(
        "--max-indications",
        type=int,
        default=12,
        help="Maximum number of indication entries kept per drug.",
    )
    parser.add_argument(
        "--max-pharma-classes",
        type=int,
        default=8,
        help="Maximum number of pharmacologic classes kept per drug.",
    )
    return parser.parse_args()


def normalize_value(value: str) -> Optional[str]:
    if value == r"\N":
        return None
    return value


def parse_copy_header(line: str) -> Tuple[str, List[str]]:
    match = re.match(r"^COPY public\.(\w+) \((.+)\) FROM stdin;$", line.strip())
    if not match:
        raise ValueError(f"Unrecognized COPY header: {line!r}")
    table_name = match.group(1)
    columns = [item.strip() for item in match.group(2).split(",")]
    return table_name, columns


def to_row(columns: List[str], raw_line: str) -> Dict[str, Optional[str]]:
    parts = raw_line.rstrip("\n").split("\t")
    if len(parts) != len(columns):
        parts = parts[: len(columns) - 1] + ["\t".join(parts[len(columns) - 1 :])]
    return {column: normalize_value(value) for column, value in zip(columns, parts)}


def safe_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def ensure_record(records: Dict[str, Dict[str, object]], struct_id: str) -> Dict[str, object]:
    if struct_id not in records:
        records[struct_id] = {
            "drugcentral_id": struct_id,
            "drug_name": None,
            "synonyms": [],
            "canonical_smiles": None,
            "inchikey": None,
            "inchi": None,
            "cas_reg_no": None,
            "molecular_formula": None,
            "molecular_weight": None,
            "approval": [],
            "identifiers": [],
            "known_indications": [],
            "known_targets": [],
            "known_pathways": [],
            "known_safety_signals": [],
            "pharmacologic_classes": [],
            "source": "DrugCentral",
            "snapshot_date": None,
        }
    return records[struct_id]


def append_unique(container: List[dict], item: dict, key_fields: Iterable[str]) -> None:
    key = tuple(item.get(field) for field in key_fields)
    for existing in container:
        existing_key = tuple(existing.get(field) for field in key_fields)
        if existing_key == key:
            return
    container.append(item)


def finalize_record(
    record: Dict[str, object],
    max_targets: int,
    max_indications: int,
    max_pharma_classes: int,
) -> Dict[str, object]:
    record["synonyms"] = sorted({item for item in record["synonyms"] if item})
    record["approval"] = sorted(
        record["approval"],
        key=lambda item: ((item.get("approval_date") or ""), (item.get("agency") or "")),
    )
    record["known_indications"] = sorted(
        record["known_indications"],
        key=lambda item: (
            0 if (item.get("relationship") or "").lower() == "indication" else 1,
            item.get("disease_name") or "",
        ),
    )[:max_indications]
    record["known_targets"] = sorted(
        record["known_targets"],
        key=lambda item: (
            0 if item.get("moa") else 1,
            item.get("target_gene") or "",
            item.get("target_name") or "",
        ),
    )[:max_targets]
    record["pharmacologic_classes"] = sorted(
        record["pharmacologic_classes"],
        key=lambda item: (
            item.get("type") or "",
            item.get("name") or "",
        ),
    )[:max_pharma_classes]
    if record["drug_name"] is None and record["synonyms"]:
        record["drug_name"] = record["synonyms"][0]
    record["known_pathways"] = []
    record["known_safety_signals"] = []
    return record


def main() -> None:
    args = parse_args()
    dump_path = args.dump_path.resolve()
    output_path = args.output_path.resolve()

    if not dump_path.exists():
        raise FileNotFoundError(f"Dump file not found: {dump_path}")

    snapshot_match = re.search(r"(\d{8})", dump_path.name)
    snapshot_date = snapshot_match.group(1) if snapshot_match else dump_path.stat().st_mtime_ns

    records: Dict[str, Dict[str, object]] = {}
    current_table: Optional[str] = None
    current_columns: List[str] = []
    tables_seen = set()

    with dump_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            if raw_line.startswith("COPY public."):
                table_name, columns = parse_copy_header(raw_line)
                if table_name in TABLES:
                    current_table = table_name
                    current_columns = columns
                    tables_seen.add(table_name)
                else:
                    current_table = None
                    current_columns = []
                continue

            if current_table is None:
                continue

            if raw_line.startswith("\\."):
                current_table = None
                current_columns = []
                continue

            row = to_row(current_columns, raw_line)

            if current_table == "structures":
                struct_id = row["id"]
                if not struct_id:
                    continue
                record = ensure_record(records, struct_id)
                record["drug_name"] = row.get("name") or record["drug_name"]
                record["canonical_smiles"] = row.get("smiles") or record["canonical_smiles"]
                record["inchikey"] = row.get("inchikey") or record["inchikey"]
                record["inchi"] = row.get("inchi") or record["inchi"]
                record["cas_reg_no"] = row.get("cas_reg_no") or record["cas_reg_no"]
                record["molecular_formula"] = row.get("cd_formula") or record["molecular_formula"]
                record["molecular_weight"] = safe_float(row.get("cd_molweight")) or record["molecular_weight"]
                record["snapshot_date"] = snapshot_date

            elif current_table == "synonyms":
                struct_id = row.get("id")
                synonym = row.get("name")
                if not struct_id or not synonym:
                    continue
                record = ensure_record(records, struct_id)
                record["synonyms"].append(synonym)
                if row.get("preferred_name") == "1" and not record["drug_name"]:
                    record["drug_name"] = synonym
                record["snapshot_date"] = snapshot_date

            elif current_table == "approval":
                struct_id = row.get("struct_id")
                if not struct_id:
                    continue
                record = ensure_record(records, struct_id)
                append_unique(
                    record["approval"],
                    {
                        "approval_date": row.get("approval"),
                        "agency": row.get("type"),
                        "applicant": row.get("applicant"),
                        "orphan": row.get("orphan") == "t",
                    },
                    ("approval_date", "agency", "applicant"),
                )
                record["snapshot_date"] = snapshot_date

            elif current_table == "identifier":
                struct_id = row.get("struct_id")
                if not struct_id:
                    continue
                record = ensure_record(records, struct_id)
                append_unique(
                    record["identifiers"],
                    {
                        "identifier": row.get("identifier"),
                        "id_type": row.get("id_type"),
                        "parent_match": row.get("parent_match"),
                    },
                    ("identifier", "id_type"),
                )
                record["snapshot_date"] = snapshot_date

            elif current_table == "omop_relationship":
                struct_id = row.get("struct_id")
                disease_name = row.get("concept_name")
                if not struct_id or not disease_name:
                    continue
                record = ensure_record(records, struct_id)
                append_unique(
                    record["known_indications"],
                    {
                        "disease_name": disease_name,
                        "relationship": row.get("relationship_name"),
                        "umls_cui": row.get("umls_cui"),
                        "snomed_conceptid": row.get("snomed_conceptid"),
                        "source": "DrugCentral OMOP",
                    },
                    ("disease_name", "relationship"),
                )
                record["snapshot_date"] = snapshot_date

            elif current_table == "act_table_full":
                struct_id = row.get("struct_id")
                target_name = row.get("target_name")
                if not struct_id or not target_name:
                    continue
                record = ensure_record(records, struct_id)
                append_unique(
                    record["known_targets"],
                    {
                        "target_name": target_name,
                        "target_gene": row.get("gene"),
                        "accession": row.get("accession"),
                        "target_class": row.get("target_class"),
                        "action_type": row.get("action_type"),
                        "moa": row.get("moa"),
                        "activity_type": row.get("act_type"),
                        "activity_value": safe_float(row.get("act_value")),
                        "activity_unit": row.get("act_unit"),
                        "activity_source": row.get("act_source"),
                        "organism": row.get("organism"),
                    },
                    ("target_name", "target_gene", "action_type", "moa", "activity_type"),
                )
                record["snapshot_date"] = snapshot_date

            elif current_table == "pharma_class":
                struct_id = row.get("struct_id")
                if not struct_id:
                    continue
                record = ensure_record(records, struct_id)
                append_unique(
                    record["pharmacologic_classes"],
                    {
                        "type": row.get("type"),
                        "name": row.get("name"),
                        "class_code": row.get("class_code"),
                        "source": row.get("source"),
                    },
                    ("type", "name", "class_code"),
                )
                record["snapshot_date"] = snapshot_date

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for struct_id in sorted(records, key=lambda item: int(item)):
            record = finalize_record(
                records[struct_id],
                max_targets=args.max_targets,
                max_indications=args.max_indications,
                max_pharma_classes=args.max_pharma_classes,
            )
            if not record["drug_name"] and not record["canonical_smiles"]:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Processed tables: {', '.join(sorted(tables_seen))}")
    print(f"Wrote {written} DrugKB records to: {output_path}")


if __name__ == "__main__":
    main()
