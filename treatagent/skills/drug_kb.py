from __future__ import annotations

import re
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .knowledge_common import (
    DRUGKB_PATH,
    _jaccard_similarity,
    _normalize_smiles,
    _parse_snapshot_date,
    json,
    Path,
    difflib,
)


class DrugKBExpert:
    _loaded = False
    _records: List[Dict[str, Any]] = []
    _exact_smiles_index: Dict[str, Dict[str, Any]] = {}
    _relaxed_smiles_index: Dict[str, Dict[str, Any]] = {}
    _inchikey_index: Dict[str, Dict[str, Any]] = {}
    _identifier_index: Dict[str, Dict[str, Any]] = {}
    _name_index: Dict[str, Dict[str, Any]] = {}
    _name_token_index: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    _normalized_names: List[Tuple[str, Dict[str, Any]]] = []
    _normalized_smiles: List[Tuple[str, Dict[str, Any]]] = []

    def __init__(self, drugkb_path: Optional[Path] = None, knowledge_cutoff_date: Optional[str] = None):
        self.drugkb_path = Path(drugkb_path or DRUGKB_PATH)
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
        if DrugKBExpert._loaded:
            return

        if not self.drugkb_path.exists():
            DrugKBExpert._loaded = True
            return

        with self.drugkb_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                DrugKBExpert._records.append(record)
                self._index_record_aliases(record)
                canonical_smiles = record.get("canonical_smiles")
                if canonical_smiles:
                    DrugKBExpert._exact_smiles_index.setdefault(canonical_smiles, record)
                    normalized_smiles = _normalize_smiles(canonical_smiles)
                    DrugKBExpert._relaxed_smiles_index.setdefault(normalized_smiles, record)
                    DrugKBExpert._normalized_smiles.append((normalized_smiles, record))
        DrugKBExpert._loaded = True

    def _index_record_aliases(self, record: Dict[str, Any]) -> None:
        inchikey = _normalize_identifier(record.get("inchikey"))
        if inchikey:
            DrugKBExpert._inchikey_index.setdefault(inchikey, record)

        for item in record.get("identifiers") or []:
            identifier = _normalize_identifier(item.get("identifier"))
            id_type = _normalize_identifier(item.get("id_type"))
            if not identifier:
                continue
            DrugKBExpert._identifier_index.setdefault(identifier, record)
            if id_type:
                DrugKBExpert._identifier_index.setdefault(f"{id_type}:{identifier}", record)

        names = [record.get("drug_name"), *(record.get("synonyms") or [])]
        for name in names:
            for variant in _drug_name_variants(name):
                if not variant:
                    continue
                DrugKBExpert._name_index.setdefault(variant, record)
                DrugKBExpert._normalized_names.append((variant, record))
                for token in variant.split():
                    if len(token) >= 4:
                        DrugKBExpert._name_token_index.setdefault(token, []).append((variant, record))

    def lookup(
        self,
        smiles: str,
        drug_names: Optional[Iterable[str]] = None,
        identifiers: Optional[Iterable[Any]] = None,
        inchikey: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str, float]:
        exact = DrugKBExpert._exact_smiles_index.get(smiles)
        if exact is not None and self._record_allowed(exact):
            return exact, "canonical_smiles", 1.0

        normalized = _normalize_smiles(smiles)
        relaxed = DrugKBExpert._relaxed_smiles_index.get(normalized)
        if relaxed is not None and self._record_allowed(relaxed):
            return relaxed, "relaxed_smiles", 0.98

        inchikey_match = self._lookup_inchikey(inchikey)
        if inchikey_match is not None:
            return inchikey_match

        identifier_match = self._lookup_identifiers(identifiers)
        if identifier_match is not None:
            return identifier_match

        name_match = self._lookup_names(drug_names)
        if name_match is not None:
            return name_match

        if _env_flag("TREATAGENT_DRUGKB_ENABLE_NEAREST", False):
            candidate = self._nearest_smiles_match(normalized)
            if candidate is not None:
                return candidate

        return None, "unmatched", 0.0

    def _lookup_inchikey(self, inchikey: Optional[str]) -> Optional[Tuple[Dict[str, Any], str, float]]:
        normalized = _normalize_identifier(inchikey)
        if not normalized:
            return None
        record = DrugKBExpert._inchikey_index.get(normalized)
        if record is not None and self._record_allowed(record):
            return record, "inchikey", 1.0
        return None

    def _lookup_identifiers(self, identifiers: Optional[Iterable[Any]]) -> Optional[Tuple[Dict[str, Any], str, float]]:
        if not identifiers:
            return None
        for raw_item in identifiers:
            id_type = ""
            identifier = ""
            if isinstance(raw_item, dict):
                id_type = _normalize_identifier(raw_item.get("id_type") or raw_item.get("type") or raw_item.get("source"))
                identifier = _normalize_identifier(raw_item.get("identifier") or raw_item.get("id") or raw_item.get("value"))
            else:
                identifier = _normalize_identifier(raw_item)
            candidates = [identifier]
            if id_type and identifier:
                candidates.insert(0, f"{id_type}:{identifier}")
            for key in candidates:
                record = DrugKBExpert._identifier_index.get(key)
                if record is not None and self._record_allowed(record):
                    return record, "identifier", 0.99
        return None

    def _lookup_names(self, drug_names: Optional[Iterable[str]]) -> Optional[Tuple[Dict[str, Any], str, float]]:
        if not drug_names:
            return None

        variants: List[str] = []
        for name in drug_names:
            variants.extend(_drug_name_variants(name))

        for variant in variants:
            record = DrugKBExpert._name_index.get(variant)
            if record is not None and self._record_allowed(record):
                return record, "drug_name", 0.95

        best_record = None
        best_score = 0.0
        best_method = "drug_name_fuzzy"
        for variant in variants:
            if len(variant) < 4:
                continue
            candidate_pool: List[Tuple[str, Dict[str, Any]]] = []
            seen = set()
            for token in variant.split():
                if len(token) < 4:
                    continue
                for candidate_name, record in DrugKBExpert._name_token_index.get(token, []):
                    key = (candidate_name, id(record))
                    if key not in seen:
                        candidate_pool.append((candidate_name, record))
                        seen.add(key)

            for candidate_name, record in candidate_pool:
                if not self._record_allowed(record):
                    continue
                if len(candidate_name) < 4:
                    continue
                score = difflib.SequenceMatcher(None, variant, candidate_name).ratio()
                if candidate_name in variant or variant in candidate_name:
                    score = max(score, 0.88)
                    best_method = "drug_name_contains"
                if score > best_score:
                    best_score = score
                    best_record = record

        if best_record is None or best_score < 0.88:
            return None
        return best_record, best_method, round(min(best_score, 0.94), 4)

    def _nearest_smiles_match(self, normalized_smiles: str) -> Optional[Tuple[Dict[str, Any], str, float]]:
        if not normalized_smiles:
            return None

        best_record = None
        best_score = 0.0
        query_length = len(normalized_smiles)
        for candidate_smiles, record in DrugKBExpert._normalized_smiles:
            if not self._record_allowed(record):
                continue
            if abs(len(candidate_smiles) - query_length) > max(8, query_length * 0.35):
                continue
            score = difflib.SequenceMatcher(None, normalized_smiles, candidate_smiles).ratio()
            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None or best_score < 0.72:
            return None
        return best_record, "nearest_smiles", round(best_score, 4)

    def analyze(
        self,
        smiles: str,
        disease: str,
        drug_names: Optional[Iterable[str]] = None,
        identifiers: Optional[Iterable[Any]] = None,
        inchikey: Optional[str] = None,
    ) -> Dict[str, Any]:
        record, matched_by, match_score = self.lookup(
            smiles,
            drug_names=drug_names,
            identifiers=identifiers,
            inchikey=inchikey,
        )
        if record is None:
            return {
                "expert": "DrugKB",
                "status": "no_data",
                "raw_data": {
                    "query_drug_names": list(drug_names or []),
                    "query_identifiers": list(identifiers or []),
                    "query_inchikey": inchikey,
                    "matched_by": "unmatched",
                    "match_score": 0.0,
                },
                "evidence": [],
            }

        evidence: List[Dict[str, Any]] = []
        drug_name = record.get("drug_name") or "This molecule"
        approvals = record.get("approval", [])
        indications = record.get("known_indications", [])
        targets = record.get("known_targets", [])
        classes = record.get("pharmacologic_classes", [])

        if approvals:
            latest_approval = approvals[-1]
            evidence.append(
                self._evidence(
                    category="drug_identity",
                    claim=f"DrugCentral records {drug_name} as an approved or clinically annotated molecule.",
                    value=0.72 if matched_by != "nearest_smiles" else 0.63,
                    impact="neutral",
                    confidence=0.78 if matched_by != "nearest_smiles" else 0.68,
                    metadata={
                        "approval_date": latest_approval.get("approval_date"),
                        "agency": latest_approval.get("agency"),
                        "matched_by": matched_by,
                        "match_score": match_score,
                    },
                )
            )

        best_indication = None
        best_similarity = 0.0
        for item in indications:
            disease_name = item.get("disease_name") or ""
            similarity = _jaccard_similarity(disease, disease_name)
            if similarity > best_similarity:
                best_similarity = similarity
                best_indication = item

        if best_indication is not None and best_similarity >= 0.25:
            relationship = (best_indication.get("relationship") or "indication").lower()
            if relationship == "contraindication":
                evidence.append(
                    self._evidence(
                        category="drug_history",
                        claim=(
                            f"DrugCentral links {drug_name} to a contraindication profile overlapping "
                            f"with {best_indication.get('disease_name')}."
                        ),
                        value=min(1.0, 0.55 + best_similarity * 0.35),
                        impact="risk",
                        confidence=min(0.92, 0.62 + best_similarity * 0.28),
                        metadata={
                            **best_indication,
                            "matched_by": matched_by,
                            "match_score": match_score,
                            "indication_similarity": round(best_similarity, 4),
                        },
                    )
                )
            else:
                evidence.append(
                    self._evidence(
                        category="drug_history",
                        claim=(
                            f"DrugCentral reports a disease indication overlap between {drug_name} "
                            f"and {best_indication.get('disease_name')}."
                        ),
                        value=min(1.0, 0.6 + best_similarity * 0.3),
                        impact="supportive",
                        confidence=min(0.94, 0.65 + best_similarity * 0.25),
                        metadata={
                            **best_indication,
                            "matched_by": matched_by,
                            "match_score": match_score,
                            "indication_similarity": round(best_similarity, 4),
                        },
                    )
                )

        for item in targets[:3]:
            target_name = item.get("target_gene") or item.get("target_name")
            if not target_name:
                continue
            evidence.append(
                self._evidence(
                    category="mechanism_prior",
                    claim=f"DrugCentral annotates {drug_name} with target or mechanism prior on {target_name}.",
                    value=(0.62 if item.get("moa") else 0.55) - (0.07 if matched_by == "nearest_smiles" else 0.0),
                    impact="neutral",
                    confidence=(0.68 if item.get("moa") else 0.6) - (0.08 if matched_by == "nearest_smiles" else 0.0),
                    metadata={**item, "matched_by": matched_by, "match_score": match_score},
                )
            )

        for item in classes[:2]:
            name = item.get("name")
            if not name:
                continue
            evidence.append(
                self._evidence(
                    category="drug_class",
                    claim=f"DrugCentral classifies {drug_name} under pharmacologic class {name}.",
                    value=0.55,
                    impact="neutral",
                    confidence=0.58,
                    metadata={**item, "matched_by": matched_by, "match_score": match_score},
                )
            )

        return {
            "expert": "DrugKB",
            "status": "ok" if evidence else "partial",
            "raw_data": {
                "drug_name": record.get("drug_name"),
                "query_drug_names": list(drug_names or []),
                "query_identifiers": list(identifiers or []),
                "query_inchikey": inchikey,
                "matched_by": matched_by,
                "match_score": match_score,
                "identifiers": record.get("identifiers", [])[:10],
                "approvals": approvals[:2],
                "best_indication_overlap": best_indication,
                "best_indication_similarity": round(best_similarity, 4),
            },
            "record": {
                "drugcentral_id": record.get("drugcentral_id"),
                "drug_name": record.get("drug_name"),
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
            "expert": "DrugKB",
            "category": category,
            "claim": claim,
            "value": round(max(0.0, min(1.0, value)), 4),
            "impact": impact,
            "confidence": round(max(0.5, min(0.95, confidence)), 4),
            "source": "DrugCentral snapshot",
            "metadata": metadata or {},
        }


def _normalize_identifier(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value).strip().upper())


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_FORMULATION_TOKENS = {
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "injection",
    "injectable",
    "solution",
    "suspension",
    "ointment",
    "cream",
    "gel",
    "patch",
    "vaginal",
    "oral",
    "intravenous",
    "intravitreal",
    "extended",
    "release",
    "delayed",
    "mg",
    "mcg",
    "ml",
}


def _normalize_drug_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = text.replace("®", "").replace("™", "")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|%)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in _FORMULATION_TOKENS]
    return " ".join(tokens).strip()


def _drug_name_variants(value: Any) -> List[str]:
    normalized = _normalize_drug_name(value)
    if not normalized:
        return []

    variants = {normalized}
    split_pattern = r"\b(?:and|with|plus|coadministered|combined)\b|\+|/|,"
    for part in re.split(split_pattern, normalized):
        part = part.strip()
        if len(part) >= 4:
            variants.add(part)

    salt_suffixes = [
        " hydrochloride",
        " hydrobromide",
        " sulfate",
        " phosphate",
        " sodium",
        " potassium",
        " acetate",
        " mesylate",
        " maleate",
        " fumarate",
    ]
    for item in list(variants):
        for suffix in salt_suffixes:
            if item.endswith(suffix) and len(item) > len(suffix) + 3:
                variants.add(item[: -len(suffix)].strip())

    return sorted(variants)


