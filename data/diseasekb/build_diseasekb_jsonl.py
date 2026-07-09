#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "diseasekb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DiseaseKB JSONL from local MONDO and Open Targets snapshot files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_DIR / "raw_data",
        help="Directory containing MONDO and Open Targets raw files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "diseasekb.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=12,
        help="Maximum number of disease-associated targets kept per disease.",
    )
    parser.add_argument(
        "--max-drugs",
        type=int,
        default=12,
        help="Maximum number of clinically linked drugs kept per disease.",
    )
    parser.add_argument(
        "--max-pathways",
        type=int,
        default=12,
        help="Maximum number of target-derived pathways kept per disease.",
    )
    parser.add_argument(
        "--max-clinical-target-links",
        type=int,
        default=12,
        help="Maximum number of clinical target links kept per disease.",
    )
    parser.add_argument(
        "--max-diseases",
        type=int,
        default=None,
        help="Optional cap for debugging. When set, only the first N disease records are written.",
    )
    parser.add_argument(
        "--snapshot-date",
        default="26.03",
        help="Snapshot/release tag recorded in the output.",
    )
    return parser.parse_args()


def normalize_id(raw_id: Optional[str]) -> Optional[str]:
    if not raw_id:
        return None
    raw_id = raw_id.strip()
    if raw_id.startswith("http://purl.obolibrary.org/obo/"):
        raw_id = raw_id.rsplit("/", 1)[-1]
    if "_" in raw_id and ":" not in raw_id:
        prefix, remainder = raw_id.split("_", 1)
        if prefix.isupper() or prefix in {"MONDO", "DOID", "EFO", "OTAR"}:
            return f"{prefix}:{remainder}"
    return raw_id


def normalize_label(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text or None


def append_unique(container: List[dict], item: dict, key_fields: Iterable[str]) -> None:
    key = tuple(item.get(field) for field in key_fields)
    for existing in container:
        existing_key = tuple(existing.get(field) for field in key_fields)
        if existing_key == key:
            return
    container.append(item)


def stage_rank(stage: Optional[str]) -> int:
    if not stage:
        return -1
    normalized = str(stage).upper().replace(" ", "").replace("-", "")
    mapping = {
        "APPROVAL": 6,
        "APPROVED": 6,
        "PHASE4": 5,
        "PHASE3": 4,
        "PHASE2": 3,
        "PHASE1": 2,
        "EARLY_PHASE1": 1,
        "UNKNOWN": 0,
    }
    return mapping.get(normalized, 0)


def support_from_stage(stage: Optional[str]) -> float:
    rank = stage_rank(stage)
    if rank <= 0:
        return 0.35
    if rank >= 6:
        return 0.95
    return round(0.35 + rank * 0.1, 4)


def iter_parquet_rows(source: Path, columns: Optional[List[str]] = None) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to build DiseaseKB from Open Targets parquet snapshots.") from exc

    if source.is_dir():
        files = sorted(source.glob("*.parquet"))
    else:
        files = [source]
    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=2048, columns=columns):
            for row in batch.to_pylist():
                yield row


def load_mondo(mondo_path: Path) -> Tuple[Dict[str, dict], Dict[str, Set[str]]]:
    with mondo_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    graph = data["graphs"][0]
    mondo_by_id: Dict[str, dict] = {}
    xref_to_mondo: DefaultDict[str, Set[str]] = defaultdict(set)

    for node in graph.get("nodes", []):
        raw_id = node.get("id")
        mondo_id = normalize_id(raw_id)
        if not mondo_id or not mondo_id.startswith("MONDO:"):
            continue

        meta = node.get("meta") or {}
        if meta.get("deprecated") or str(node.get("lbl", "")).lower().startswith("obsolete "):
            continue

        aliases = set()
        label = normalize_label(node.get("lbl"))
        if label:
            aliases.add(label)
        for synonym in meta.get("synonyms") or []:
            value = normalize_label((synonym or {}).get("val"))
            if value:
                aliases.add(value)

        xrefs = set()
        for item in meta.get("xrefs") or []:
            if isinstance(item, dict):
                value = normalize_id(item.get("val"))
            else:
                value = normalize_id(str(item))
            if value:
                xrefs.add(value)
                xref_to_mondo[value].add(mondo_id)

        mondo_by_id[mondo_id] = {
            "mondo_id": mondo_id,
            "label": label,
            "aliases": sorted(aliases),
            "xrefs": sorted(xrefs),
        }

    return mondo_by_id, xref_to_mondo


def load_diseases(
    disease_path: Path,
    mondo_by_id: Dict[str, dict],
    xref_to_mondo: Dict[str, Set[str]],
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    disease_records: Dict[str, dict] = {}
    disease_name_by_id: Dict[str, str] = {}

    for row in iter_parquet_rows(
        disease_path,
        columns=[
            "id",
            "name",
            "description",
            "dbXRefs",
            "exactSynonyms",
            "relatedSynonyms",
            "narrowSynonyms",
            "broadSynonyms",
            "therapeuticAreas",
        ],
    ):
        disease_id = row.get("id")
        if not disease_id:
            continue

        disease_name = normalize_label(row.get("name"))
        disease_name_by_id[disease_id] = disease_name or disease_id

        aliases = set()
        if disease_name:
            aliases.add(disease_name)
        for field in ("exactSynonyms", "relatedSynonyms", "narrowSynonyms", "broadSynonyms"):
            for item in row.get(field) or []:
                value = normalize_label(item)
                if value:
                    aliases.add(value)

        db_xrefs = set()
        mondo_ids = set()

        raw_self_id = normalize_id(disease_id)
        if raw_self_id:
            db_xrefs.add(raw_self_id)
            if raw_self_id.startswith("MONDO:"):
                mondo_ids.add(raw_self_id)
            mondo_ids.update(xref_to_mondo.get(raw_self_id, set()))

        for item in row.get("dbXRefs") or []:
            value = normalize_id(item)
            if not value:
                continue
            db_xrefs.add(value)
            if value.startswith("MONDO:"):
                mondo_ids.add(value)
            mondo_ids.update(xref_to_mondo.get(value, set()))

        for mondo_id in mondo_ids:
            mondo_record = mondo_by_id.get(mondo_id)
            if not mondo_record:
                continue
            aliases.update(mondo_record["aliases"])
            db_xrefs.update(mondo_record["xrefs"])

        disease_records[disease_id] = {
            "disease_id": disease_id,
            "canonical_disease_id": raw_self_id or disease_id,
            "disease_name": disease_name,
            "aliases": sorted(aliases),
            "description": normalize_label(row.get("description")),
            "db_xrefs": sorted(db_xrefs),
            "mondo_ids": sorted(mondo_ids),
            "therapeutic_area_ids": sorted(
                {item for item in (row.get("therapeuticAreas") or []) if item}
            ),
            "therapy_prior": [],
            "known_drugs": [],
            "known_targets": [],
            "known_pathways": [],
            "clinical_target_links": [],
            "failure_modes": [],
            "special_requirements": [],
            "source": "Open Targets + MONDO",
        }

    for record in disease_records.values():
        record["therapeutic_areas"] = [
            {
                "id": area_id,
                "name": disease_name_by_id.get(area_id) or normalize_label(normalize_id(area_id)),
            }
            for area_id in record.pop("therapeutic_area_ids")
        ]

    return disease_records, disease_name_by_id


def load_clinical_reports(clinical_report_path: Path) -> Dict[str, dict]:
    reports: Dict[str, dict] = {}
    for row in iter_parquet_rows(
        clinical_report_path,
        columns=[
            "id",
            "clinicalStage",
            "source",
            "year",
            "title",
            "url",
            "trialDescription",
            "trialOverallStatus",
            "trialPhase",
        ],
    ):
        report_id = row.get("id")
        if not report_id:
            continue
        reports[report_id] = {
            "clinical_stage": row.get("clinicalStage"),
            "source": row.get("source"),
            "year": row.get("year"),
            "title": normalize_label(row.get("title")),
            "url": row.get("url"),
            "description": normalize_label(row.get("trialDescription")),
            "overall_status": row.get("trialOverallStatus"),
            "trial_phase": row.get("trialPhase"),
        }
    return reports


def load_drug_molecules(drug_molecule_dir: Path) -> Dict[str, dict]:
    drug_by_id: Dict[str, dict] = {}
    for row in iter_parquet_rows(
        drug_molecule_dir,
        columns=[
            "id",
            "name",
            "synonyms",
            "canonicalSmiles",
            "maximumClinicalStage",
            "description",
            "drugType",
        ],
    ):
        drug_id = row.get("id")
        if not drug_id:
            continue
        synonyms = sorted(
            {
                value
                for value in [normalize_label(row.get("name"))] + [normalize_label(item) for item in (row.get("synonyms") or [])]
                if value
            }
        )
        drug_by_id[drug_id] = {
            "drug_id": drug_id,
            "drug_name": normalize_label(row.get("name")) or drug_id,
            "synonyms": synonyms,
            "canonical_smiles": row.get("canonicalSmiles"),
            "max_clinical_stage": row.get("maximumClinicalStage"),
            "drug_type": row.get("drugType"),
            "description": normalize_label(row.get("description")),
        }
    return drug_by_id


def load_drug_mechanisms(drug_moa_dir: Path) -> Dict[str, List[dict]]:
    mechanisms_by_drug: DefaultDict[str, List[dict]] = defaultdict(list)
    for row in iter_parquet_rows(
        drug_moa_dir,
        columns=[
            "actionType",
            "mechanismOfAction",
            "chemblIds",
            "targetName",
            "targetType",
            "targets",
        ],
    ):
        entry = {
            "action_type": row.get("actionType"),
            "mechanism_of_action": normalize_label(row.get("mechanismOfAction")),
            "target_name": normalize_label(row.get("targetName")),
            "target_type": row.get("targetType"),
            "target_ids": [item for item in (row.get("targets") or []) if item],
        }
        for chembl_id in row.get("chemblIds") or []:
            if chembl_id:
                append_unique(
                    mechanisms_by_drug[chembl_id],
                    entry,
                    ("mechanism_of_action", "target_name"),
                )
    return dict(mechanisms_by_drug)


def load_targets(target_dir: Path) -> Dict[str, dict]:
    target_by_id: Dict[str, dict] = {}
    for row in iter_parquet_rows(
        target_dir,
        columns=["id", "approvedSymbol", "approvedName", "pathways", "symbolSynonyms", "nameSynonyms"],
    ):
        target_id = row.get("id")
        if not target_id:
            continue
        aliases = set()
        for field in ("symbolSynonyms", "nameSynonyms"):
            for item in row.get(field) or []:
                label = normalize_label((item or {}).get("label"))
                if label:
                    aliases.add(label)
        pathways = []
        for pathway in row.get("pathways") or []:
            pathway_id = (pathway or {}).get("pathwayId")
            pathway_name = normalize_label((pathway or {}).get("pathway"))
            top_level = normalize_label((pathway or {}).get("topLevelTerm"))
            if pathway_id or pathway_name:
                pathways.append(
                    {
                        "pathway_id": pathway_id,
                        "pathway_name": pathway_name,
                        "top_level_term": top_level,
                    }
                )
        target_by_id[target_id] = {
            "target_id": target_id,
            "target_symbol": normalize_label(row.get("approvedSymbol")) or target_id,
            "target_name": normalize_label(row.get("approvedName")),
            "aliases": sorted(aliases),
            "pathways": pathways,
        }
    return target_by_id


def enrich_with_associations(
    disease_records: Dict[str, dict],
    association_dir: Path,
    target_by_id: Dict[str, dict],
    max_targets: int,
    max_pathways: int,
) -> None:
    associations_by_disease: DefaultDict[str, List[dict]] = defaultdict(list)

    for row in iter_parquet_rows(
        association_dir,
        columns=["diseaseId", "targetId", "associationScore", "evidenceCount", "currentNovelty"],
    ):
        disease_id = row.get("diseaseId")
        target_id = row.get("targetId")
        if disease_id not in disease_records or not target_id:
            continue
        target_info = target_by_id.get(target_id, {})
        entry = {
            "target_id": target_id,
            "target_symbol": target_info.get("target_symbol") or target_id,
            "target_name": target_info.get("target_name"),
            "association_score": float(row.get("associationScore") or 0.0),
            "evidence_count": int(row.get("evidenceCount") or 0),
            "current_novelty": float(row.get("currentNovelty") or 0.0),
            "pathways": target_info.get("pathways") or [],
        }
        associations_by_disease[disease_id].append(entry)

    for disease_id, entries in associations_by_disease.items():
        entries.sort(
            key=lambda item: (
                -(item.get("association_score") or 0.0),
                -(item.get("evidence_count") or 0),
                item.get("target_symbol") or "",
            )
        )
        top_entries = entries[:max_targets]
        disease_records[disease_id]["known_targets"] = [
            {
                "target_id": item["target_id"],
                "target_symbol": item["target_symbol"],
                "target_name": item.get("target_name"),
                "support_score": round(item["association_score"], 6),
                "evidence_count": item["evidence_count"],
                "current_novelty": round(item["current_novelty"], 6),
                "source": "Open Targets association_overall_direct",
            }
            for item in top_entries
        ]

        pathway_items: List[dict] = []
        for item in top_entries:
            for pathway in item.get("pathways") or []:
                append_unique(
                    pathway_items,
                    {
                        "pathway_id": pathway.get("pathway_id"),
                        "pathway_name": pathway.get("pathway_name"),
                        "top_level_term": pathway.get("top_level_term"),
                        "support_score": round(item["association_score"], 6),
                        "target_id": item["target_id"],
                        "target_symbol": item["target_symbol"],
                        "source": "Open Targets target.pathways",
                    },
                    ("pathway_id", "pathway_name", "target_id"),
                )
        pathway_items.sort(
            key=lambda item: (-(item.get("support_score") or 0.0), item.get("pathway_name") or "")
        )
        disease_records[disease_id]["known_pathways"] = pathway_items[:max_pathways]


def enrich_with_clinical_indications(
    disease_records: Dict[str, dict],
    clinical_indication_path: Path,
    clinical_reports: Dict[str, dict],
    drug_by_id: Dict[str, dict],
    drug_mechanisms: Dict[str, List[dict]],
    max_drugs: int,
) -> None:
    by_disease: DefaultDict[str, List[dict]] = defaultdict(list)

    for row in iter_parquet_rows(
        clinical_indication_path,
        columns=["diseaseId", "drugId", "maxClinicalStage", "clinicalReportIds"],
    ):
        disease_id = row.get("diseaseId")
        drug_id = row.get("drugId")
        if disease_id not in disease_records or not drug_id:
            continue
        report_ids = [item for item in (row.get("clinicalReportIds") or []) if item]
        report_sources = sorted(
            {
                report.get("source")
                for report_id in report_ids
                for report in [clinical_reports.get(report_id, {})]
                if report.get("source")
            }
        )
        report_years = sorted(
            {
                report.get("year")
                for report_id in report_ids
                for report in [clinical_reports.get(report_id, {})]
                if report.get("year") is not None
            }
        )
        drug_info = drug_by_id.get(drug_id, {})
        mechanisms = drug_mechanisms.get(drug_id, [])
        entry = {
            "drug_id": drug_id,
            "drug_name": drug_info.get("drug_name") or drug_id,
            "synonyms": drug_info.get("synonyms") or [],
            "max_clinical_stage": row.get("maxClinicalStage") or drug_info.get("max_clinical_stage"),
            "support_score": support_from_stage(row.get("maxClinicalStage") or drug_info.get("max_clinical_stage")),
            "report_count": len(report_ids),
            "report_ids": report_ids,
            "report_sources": report_sources,
            "report_years": report_years,
            "drug_type": drug_info.get("drug_type"),
            "description": drug_info.get("description"),
            "mechanisms_of_action": [
                item
                for item in mechanisms[:4]
                if item.get("mechanism_of_action") or item.get("target_name")
            ],
            "source": "Open Targets clinical_indication",
        }
        by_disease[disease_id].append(entry)

    for disease_id, items in by_disease.items():
        items.sort(
            key=lambda item: (
                -stage_rank(item.get("max_clinical_stage")),
                -(item.get("report_count") or 0),
                item.get("drug_name") or "",
            )
        )
        top_items = items[:max_drugs]
        disease_records[disease_id]["known_drugs"] = top_items
        disease_records[disease_id]["therapy_prior"] = [
            {
                "therapy_name": item["drug_name"],
                "therapy_type": "drug",
                "drug_id": item["drug_id"],
                "support_score": item["support_score"],
                "max_clinical_stage": item.get("max_clinical_stage"),
                "evidence_level": "clinical_indication",
                "report_count": item["report_count"],
                "source": item["source"],
            }
            for item in top_items
        ]


def enrich_with_clinical_targets(
    disease_records: Dict[str, dict],
    clinical_target_path: Path,
    drug_by_id: Dict[str, dict],
    target_by_id: Dict[str, dict],
    max_links: int,
) -> None:
    by_disease: DefaultDict[str, List[dict]] = defaultdict(list)
    for row in iter_parquet_rows(
        clinical_target_path,
        columns=["drugId", "targetId", "diseases", "clinicalReportIds", "maxClinicalStage"],
    ):
        drug_id = row.get("drugId")
        target_id = row.get("targetId")
        if not drug_id or not target_id:
            continue
        for disease_entry in row.get("diseases") or []:
            disease_id = (disease_entry or {}).get("diseaseId")
            if disease_id not in disease_records:
                continue
            drug_info = drug_by_id.get(drug_id, {})
            target_info = target_by_id.get(target_id, {})
            by_disease[disease_id].append(
                {
                    "drug_id": drug_id,
                    "drug_name": drug_info.get("drug_name") or drug_id,
                    "target_id": target_id,
                    "target_symbol": target_info.get("target_symbol") or target_id,
                    "target_name": target_info.get("target_name"),
                    "max_clinical_stage": row.get("maxClinicalStage"),
                    "report_count": len(row.get("clinicalReportIds") or []),
                    "support_score": support_from_stage(row.get("maxClinicalStage")),
                    "source": "Open Targets clinical_target",
                }
            )
    for disease_id, items in by_disease.items():
        deduped: List[dict] = []
        for item in items:
            append_unique(deduped, item, ("drug_id", "target_id"))
        deduped.sort(
            key=lambda item: (
                -stage_rank(item.get("max_clinical_stage")),
                -(item.get("report_count") or 0),
                item.get("drug_name") or "",
                item.get("target_symbol") or "",
            )
        )
        disease_records[disease_id]["clinical_target_links"] = deduped[:max_links]


def build_summary(record: dict) -> str:
    parts: List[str] = []
    if record.get("known_targets"):
        targets = ", ".join(
            item.get("target_symbol") or item.get("target_id")
            for item in record["known_targets"][:3]
        )
        parts.append(f"Top associated targets include {targets}.")
    if record.get("known_drugs"):
        drugs = ", ".join(item.get("drug_name") or item.get("drug_id") for item in record["known_drugs"][:3])
        parts.append(f"Clinically linked drugs include {drugs}.")
    if record.get("known_pathways"):
        pathways = ", ".join(
            item.get("pathway_name") or item.get("pathway_id")
            for item in record["known_pathways"][:2]
        )
        parts.append(f"Relevant pathways include {pathways}.")
    if record.get("description"):
        parts.append(record["description"])
    return " ".join(parts).strip()


def finalize_record(record: dict, snapshot_date: str) -> dict:
    record["aliases"] = sorted({item for item in record.get("aliases", []) if item})
    record["db_xrefs"] = sorted({item for item in record.get("db_xrefs", []) if item})
    record["mondo_ids"] = sorted({item for item in record.get("mondo_ids", []) if item})
    record["summary"] = build_summary(record)
    record["snapshot_date"] = snapshot_date
    return record


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_path = args.output_path.resolve()

    mondo_path = raw_dir / "mondo-base.json"
    opentargets_dir = raw_dir / "opentargets"
    disease_path = opentargets_dir / "disease.parquet"
    clinical_indication_path = opentargets_dir / "clinical_indication.parquet"
    clinical_report_path = opentargets_dir / "clinical_report.parquet"
    clinical_target_path = opentargets_dir / "clinical_target.parquet"
    association_dir = opentargets_dir / "association_overall_direct"
    drug_molecule_dir = opentargets_dir / "drug_molecule"
    drug_moa_dir = opentargets_dir / "drug_mechanism_of_action"
    target_dir = opentargets_dir / "target"

    required_paths = [
        mondo_path,
        disease_path,
        clinical_indication_path,
        clinical_report_path,
        clinical_target_path,
        association_dir,
        drug_molecule_dir,
        drug_moa_dir,
        target_dir,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    mondo_by_id, xref_to_mondo = load_mondo(mondo_path)
    disease_records, _ = load_diseases(disease_path, mondo_by_id, xref_to_mondo)
    clinical_reports = load_clinical_reports(clinical_report_path)
    drug_by_id = load_drug_molecules(drug_molecule_dir)
    drug_mechanisms = load_drug_mechanisms(drug_moa_dir)
    target_by_id = load_targets(target_dir)

    enrich_with_associations(
        disease_records,
        association_dir,
        target_by_id,
        max_targets=args.max_targets,
        max_pathways=args.max_pathways,
    )
    enrich_with_clinical_indications(
        disease_records,
        clinical_indication_path,
        clinical_reports,
        drug_by_id,
        drug_mechanisms,
        max_drugs=args.max_drugs,
    )
    enrich_with_clinical_targets(
        disease_records,
        clinical_target_path,
        drug_by_id,
        target_by_id,
        max_links=args.max_clinical_target_links,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for disease_id in sorted(disease_records):
            record = finalize_record(disease_records[disease_id], snapshot_date=args.snapshot_date)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if args.max_diseases is not None and written >= args.max_diseases:
                break

    print(f"Wrote {written} disease records to {output_path}")


if __name__ == "__main__":
    main()
