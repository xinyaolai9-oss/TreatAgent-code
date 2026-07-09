from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .knowledge_common import (
    DISEASEKB_PATH,
    _build_disease_variants,
    _jaccard_similarity,
    _normalize_disease_name,
    _parse_snapshot_date,
    json,
    Path,
    difflib,
)


class DiseaseKBExpert:
    _loaded = False
    _records: List[Dict[str, Any]] = []
    _name_index: Dict[str, Dict[str, Any]] = {}
    _alias_index: Dict[str, Dict[str, Any]] = {}
    _candidate_names: List[Tuple[str, Dict[str, Any]]] = []

    def __init__(self, diseasekb_path: Optional[Path] = None, knowledge_cutoff_date: Optional[str] = None):
        self.diseasekb_path = Path(diseasekb_path or DISEASEKB_PATH)
        self.knowledge_cutoff_date = _parse_snapshot_date(knowledge_cutoff_date)
        self._ensure_loaded()

    def _record_allowed(self, record: Dict[str, Any]) -> bool:
        if self.knowledge_cutoff_date is None:
            return True
        record_date = _parse_snapshot_date(record.get("snapshot_date"))
        if record_date is None:
            return True
        return record_date <= self.knowledge_cutoff_date

    def _ensure_loaded(self) -> None:
        if DiseaseKBExpert._loaded:
            return

        if not self.diseasekb_path.exists():
            DiseaseKBExpert._loaded = True
            return

        with self.diseasekb_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                DiseaseKBExpert._records.append(record)

                disease_name = record.get("disease_name")
                if disease_name:
                    normalized_name = _normalize_disease_name(disease_name)
                    DiseaseKBExpert._name_index.setdefault(normalized_name, record)
                    DiseaseKBExpert._candidate_names.append((normalized_name, record))

                for alias in record.get("aliases", []):
                    normalized_alias = _normalize_disease_name(alias)
                    if normalized_alias:
                        DiseaseKBExpert._alias_index.setdefault(normalized_alias, record)
                        DiseaseKBExpert._candidate_names.append((normalized_alias, record))
        DiseaseKBExpert._loaded = True

    def lookup(self, disease: str) -> Tuple[Optional[Dict[str, Any]], str, float]:
        for variant in _build_disease_variants(disease):
            exact = DiseaseKBExpert._name_index.get(variant)
            if exact is not None and self._record_allowed(exact):
                return exact, "disease_name", 1.0

            alias = DiseaseKBExpert._alias_index.get(variant)
            if alias is not None and self._record_allowed(alias):
                return alias, "alias", 0.99

        best_record = None
        best_score = 0.0
        for query in _build_disease_variants(disease):
            for candidate_name, record in DiseaseKBExpert._candidate_names:
                if not self._record_allowed(record):
                    continue
                jaccard = _jaccard_similarity(query, candidate_name)
                if jaccard < 0.25:
                    continue
                ratio = difflib.SequenceMatcher(None, query, candidate_name).ratio()
                score = max(jaccard, ratio)
                if query in candidate_name:
                    score += 0.08
                if score > best_score:
                    best_score = score
                    best_record = record

        if best_record is None or best_score < 0.62:
            return None, "unmatched", 0.0
        return best_record, "fuzzy", round(min(best_score, 1.0), 4)

    def analyze(self, disease: str) -> Dict[str, Any]:
        record, matched_by, match_score = self.lookup(disease)
        if record is None:
            return {"expert": "DiseaseKB", "status": "no_data", "evidence": []}

        disease_name = record.get("disease_name") or disease
        disease_id = record.get("canonical_disease_id") or record.get("disease_id")
        evidence: List[Dict[str, Any]] = []

        for item in (record.get("known_targets") or [])[:4]:
            target_symbol = item.get("target_symbol") or item.get("target_id")
            if not target_symbol:
                continue
            score = float(item.get("support_score") or 0.0)
            evidence.append(
                self._evidence(
                    category="disease_target_prior",
                    claim=f"DiseaseKB associates {disease_name} with target {target_symbol}.",
                    value=max(0.0, min(1.0, score)),
                    impact="neutral",
                    confidence=min(0.93, 0.58 + score * 0.32),
                    metadata={"disease_id": disease_id, "matched_by": matched_by, "match_score": match_score, **item},
                )
            )

        for item in (record.get("therapy_prior") or [])[:4]:
            therapy_name = item.get("therapy_name") or item.get("drug_id")
            if not therapy_name:
                continue
            support_score = float(item.get("support_score") or 0.0)
            evidence.append(
                self._evidence(
                    category="therapy_prior",
                    claim=f"DiseaseKB records {therapy_name} as a clinically relevant therapy prior for {disease_name}.",
                    value=max(0.0, min(1.0, support_score)),
                    impact="neutral",
                    confidence=min(0.95, 0.6 + support_score * 0.3),
                    metadata={"disease_id": disease_id, "matched_by": matched_by, "match_score": match_score, **item},
                )
            )

        for item in (record.get("known_pathways") or [])[:3]:
            pathway_name = item.get("pathway_name") or item.get("pathway_id")
            if not pathway_name:
                continue
            support_score = float(item.get("support_score") or 0.0)
            evidence.append(
                self._evidence(
                    category="pathway_prior",
                    claim=f"DiseaseKB highlights {pathway_name} as a relevant pathway context for {disease_name}.",
                    value=max(0.0, min(1.0, support_score)),
                    impact="neutral",
                    confidence=min(0.9, 0.56 + support_score * 0.28),
                    metadata={"disease_id": disease_id, "matched_by": matched_by, "match_score": match_score, **item},
                )
            )

        for item in (record.get("clinical_target_links") or [])[:2]:
            drug_name = item.get("drug_name") or item.get("drug_id")
            target_symbol = item.get("target_symbol") or item.get("target_id")
            if not drug_name or not target_symbol:
                continue
            support_score = float(item.get("support_score") or 0.0)
            evidence.append(
                self._evidence(
                    category="clinical_target_bridge",
                    claim=f"DiseaseKB links clinically used drug {drug_name} to target {target_symbol} in {disease_name}.",
                    value=max(0.0, min(1.0, support_score)),
                    impact="neutral",
                    confidence=min(0.94, 0.62 + support_score * 0.25),
                    metadata={"disease_id": disease_id, "matched_by": matched_by, "match_score": match_score, **item},
                )
            )

        return {
            "expert": "DiseaseKB",
            "status": "ok" if evidence else "partial",
            "raw_data": {
                "disease_id": disease_id,
                "disease_name": disease_name,
                "matched_by": matched_by,
                "match_score": match_score,
                "description": record.get("description"),
                "summary": record.get("summary"),
                "known_target_count": len(record.get("known_targets") or []),
                "known_drug_count": len(record.get("known_drugs") or []),
            },
            "record": {
                "disease_id": record.get("disease_id"),
                "canonical_disease_id": record.get("canonical_disease_id"),
                "snapshot_date": record.get("snapshot_date"),
            },
            "evidence": evidence,
        }

    def _evidence(
        self,
        category: str,
        claim: str,
        value: float,
        impact: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "expert": "DiseaseKB",
            "category": category,
            "claim": claim,
            "value": round(max(0.0, min(1.0, value)), 4),
            "impact": impact,
            "confidence": round(max(0.5, min(0.95, confidence)), 4),
            "source": "Local DiseaseKB snapshot",
            "metadata": metadata or {},
        }
