from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _clip01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _first_present(metadata: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


@dataclass
class TypedEvidenceTuple:
    subject: str
    relation: str
    object: str
    score: float
    direction: str
    confidence: float
    source: str
    match_type: str
    timestamp: Optional[str] = None
    reliability: float = 1.0
    expert: str = ""
    category: str = ""
    claim: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_status: str = "found"
    semantic_role: str = ""
    argument_direction: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    grounding_requirement: str = ""
    derived_from: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "score": round(self.score, 4),
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "match_type": self.match_type,
            "timestamp": self.timestamp,
            "reliability": round(self.reliability, 4),
            "expert": self.expert,
            "category": self.category,
            "claim": self.claim,
            "metadata": self.metadata,
            "retrieval_status": self.retrieval_status,
            "semantic_role": self.semantic_role,
            "argument_direction": self.argument_direction or self.direction,
            "provenance": self.provenance,
            "grounding_requirement": self.grounding_requirement,
            "derived_from": self.derived_from,
        }


SOURCE_RELIABILITY = {
    "DrugKB": 0.9,
    "DiseaseKB": 0.85,
    "DTI": 0.6,
    "ADMET": 0.6,
    "Clinical": 0.45,
}

MATCH_RELIABILITY = {
    "canonical_smiles": 1.0,
    "relaxed_smiles": 0.95,
    "disease_name": 1.0,
    "alias": 0.95,
    "fuzzy": 0.7,
    "nearest_smiles": 0.65,
    "predicted": 0.6,
    "model_prediction": 0.6,
    "disease_prior": 0.55,
    "unknown": 0.5,
}

SPECIFICITY_RELIABILITY = {
    "drug_identity": 0.8,
    "drug_history": 0.95,
    "mechanism_prior": 0.75,
    "drug_class": 0.55,
    "disease_target_prior": 0.75,
    "therapy_prior": 0.8,
    "pathway_prior": 0.65,
    "clinical_target_bridge": 0.82,
    "mechanism": 0.65,
    "absorption": 0.55,
    "distribution": 0.5,
    "toxicity": 0.75,
    "clinical_prior": 0.45,
}


def infer_match_type(expert: str, category: str, metadata: Dict[str, Any]) -> str:
    matched_by = metadata.get("matched_by")
    if matched_by:
        return str(matched_by)
    if expert in {"DTI", "ADMET"}:
        return "model_prediction"
    if expert == "Clinical":
        return "disease_prior"
    return "unknown"


def infer_relation_object(
    expert: str,
    category: str,
    impact: str,
    metadata: Dict[str, Any],
    drug: str,
    disease: str,
) -> tuple[str, str, str]:
    if expert == "DrugKB":
        if category == "drug_identity":
            return drug, "has_identity", str(_first_present(metadata, "drug_name", "drugcentral_id", "agency") or "known_drug")
        if category == "drug_history":
            relation = "contraindicated_for" if impact == "risk" else "has_indication"
            return drug, relation, str(_first_present(metadata, "disease_name", "disease_id") or disease)
        if category == "mechanism_prior":
            return drug, "targets", str(_first_present(metadata, "target_gene", "target_name", "target_id") or "known_target")
        if category == "drug_class":
            return drug, "has_class", str(_first_present(metadata, "name", "class_id") or "pharmacologic_class")

    if expert == "DiseaseKB":
        if category == "disease_target_prior":
            return disease, "associated_with_target", str(_first_present(metadata, "target_symbol", "target_id") or "known_target")
        if category == "therapy_prior":
            return disease, "has_known_therapy", str(_first_present(metadata, "therapy_name", "drug_id") or "known_therapy")
        if category == "pathway_prior":
            return disease, "associated_with_pathway", str(_first_present(metadata, "pathway_name", "pathway_id") or "known_pathway")
        if category == "clinical_target_bridge":
            return disease, "therapy_targets", str(_first_present(metadata, "target_symbol", "target_id") or "clinical_target")

    if expert == "DTI":
        return drug, "predicted_mechanistic_support_for", disease

    if expert == "ADMET":
        if category == "toxicity":
            relation = "has_toxicity_risk" if impact == "risk" else "has_admet_context"
            return drug, relation, str(_first_present(metadata, "endpoint") or "toxicity")
        return drug, f"has_admet_{category}", disease

    if expert == "Clinical":
        return disease, "has_clinical_success_prior", disease

    return drug, f"has_{category}_evidence", disease


def compute_reliability(expert: str, category: str, match_type: str) -> float:
    source = SOURCE_RELIABILITY.get(expert, 0.5)
    match = MATCH_RELIABILITY.get(match_type, MATCH_RELIABILITY["unknown"])
    specificity = SPECIFICITY_RELIABILITY.get(category, 0.55)
    return _clip01(source * match * specificity, default=0.0)


def infer_semantic_role(expert: str, category: str, impact: str, metadata: Dict[str, Any]) -> str:
    if expert == "DrugKB":
        if category == "drug_identity":
            return "drug_identity"
        if category == "drug_history":
            if impact == "risk" or str(metadata.get("relationship") or "").lower() == "contraindication":
                return "contraindication"
            similarity = _clip01(
                metadata.get("indication_similarity")
                or metadata.get("disease_similarity")
                or metadata.get("similarity")
                or metadata.get("best_similarity"),
                0.0,
            )
            if similarity >= 0.55:
                return "direct_indication"
            return "related_indication"
        if category == "mechanism_prior":
            return "drug_target"
        if category == "drug_class":
            return "drug_class"
    if expert == "DiseaseKB":
        if category == "disease_target_prior":
            return "disease_target"
        if category == "therapy_prior":
            return "known_therapy"
        if category == "pathway_prior":
            return "disease_pathway"
        if category == "clinical_target_bridge":
            return "clinical_target_context"
    if expert == "DTI":
        return "mechanism_plausibility"
    if expert == "ADMET":
        return "safety_risk" if impact == "risk" else "developability_context"
    if expert == "Clinical":
        return "clinical_prior"
    return category or "background_context"


def infer_argument_direction(
    expert: str,
    category: str,
    impact: str,
    semantic_role: str,
    metadata: Dict[str, Any],
) -> tuple[str, str]:
    """Map retrieved facts to argument direction.

    Source-level biomedical facts are not treatment support by default. DrugKB
    targets/classes and DiseaseKB targets/pathways remain neutral until the
    EvidenceGraph grounds them through cross-source links.
    """
    if semantic_role == "contraindication":
        return "conflict", "Contraindication or warning evidence is directly relevant as treatment conflict."
    if semantic_role == "direct_indication":
        return "support", "Direct indication overlap is treatment-relevant support."
    if semantic_role == "related_indication":
        return "neutral", "Related indication evidence is contextual unless strengthened by disease matching or mechanism links."
    if semantic_role in {"drug_identity", "drug_target", "drug_class"}:
        return "neutral", "Drug-side facts require disease-side grounding before they support treatment."
    if semantic_role in {"disease_target", "disease_pathway", "known_therapy", "clinical_target_context"}:
        return "neutral", "Disease-side context requires drug-side grounding before it supports treatment."
    if expert == "DTI":
        if impact == "risk":
            return "conflict", "Weak or contradictory mechanism prediction can reduce treatment plausibility."
        return "support", "Drug-target interaction evidence directly informs mechanism plausibility."
    if expert == "ADMET":
        if impact == "risk":
            return "conflict", "Safety or developability risk can conflict with treatment suitability."
        return "neutral", "Favorable ADMET is contextual feasibility information, not treatment efficacy support."
    if expert == "Clinical":
        if impact == "risk":
            return "conflict", "Low clinical feasibility weakens translational plausibility."
        return "support", "Disease-level clinical feasibility supports translational plausibility."
    return "neutral", "Retrieved context is not direct treatment support without query-specific grounding."


def legacy_evidence_to_tuple(evidence: Any, drug: str, disease: str) -> TypedEvidenceTuple:
    if hasattr(evidence, "to_dict"):
        data = evidence.to_dict()
    else:
        data = dict(evidence)

    expert = str(data.get("expert") or "")
    category = str(data.get("category") or "")
    impact = str(data.get("impact") or "").lower()
    metadata = dict(data.get("metadata") or {})
    match_type = infer_match_type(expert, category, metadata)
    subject, relation, object_value = infer_relation_object(
        expert=expert,
        category=category,
        impact=impact,
        metadata=metadata,
        drug=drug,
        disease=disease,
    )
    semantic_role = infer_semantic_role(expert, category, impact, metadata)
    direction, grounding_requirement = infer_argument_direction(
        expert=expert,
        category=category,
        impact=impact,
        semantic_role=semantic_role,
        metadata=metadata,
    )
    score = _clip01(data.get("value"), default=0.0)
    reliability = compute_reliability(expert, category, match_type)
    legacy_confidence = _clip01(data.get("confidence"), default=0.0)
    confidence = _clip01(score * reliability if score else legacy_confidence * reliability, default=0.0)
    timestamp = metadata.get("snapshot_date") or metadata.get("approval_date")

    return TypedEvidenceTuple(
        subject=str(subject),
        relation=str(relation),
        object=str(object_value),
        score=score,
        direction=direction,
        confidence=confidence,
        source=str(data.get("source") or expert),
        match_type=match_type,
        timestamp=str(timestamp) if timestamp else None,
        reliability=reliability,
        expert=expert,
        category=category,
        claim=str(data.get("claim") or ""),
        metadata=metadata,
        retrieval_status="found",
        semantic_role=semantic_role,
        argument_direction=direction,
        provenance={
            "source": str(data.get("source") or expert),
            "match_type": match_type,
            "snapshot_date": metadata.get("snapshot_date"),
            "record_id": metadata.get("drugcentral_id")
            or metadata.get("disease_id")
            or metadata.get("target_id")
            or metadata.get("pathway_id"),
        },
        grounding_requirement=grounding_requirement,
        derived_from=[],
    )


def expand_typed_evidence_tuple(item: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Create fine-grained deterministic evidence views from one typed tuple."""
    base = dict(item)
    metadata = dict(base.get("metadata") or {})
    expanded = [base]
    expert = str(base.get("expert") or "")
    category = str(base.get("category") or "")
    score = _clip01(base.get("score"), 0.0)
    reliability = _clip01(base.get("reliability"), 0.0)
    confidence = _clip01(base.get("confidence"), score * reliability)

    def add_view(
        relation: str,
        object_value: Any,
        view_category: str,
        view_score: float | None = None,
        view_reliability_scale: float = 0.85,
    ) -> None:
        if object_value in (None, ""):
            return
        view = dict(base)
        view["relation"] = relation
        view["object"] = str(object_value)
        view["category"] = view_category
        view["score"] = round(_clip01(score if view_score is None else view_score), 4)
        view["reliability"] = round(_clip01(reliability * view_reliability_scale), 4)
        view["confidence"] = round(_clip01(confidence * view_reliability_scale), 4)
        view["claim"] = f"{expert} {view_category} evidence: {object_value}"
        view_metadata = dict(metadata)
        view_metadata["derived_from_category"] = category
        view_metadata["derived_evidence_view"] = True
        view["metadata"] = view_metadata
        expanded.append(view)

    if expert == "ADMET":
        endpoint = metadata.get("endpoint")
        raw_risk = metadata.get("raw_risk_score")
        add_view("has_admet_endpoint", endpoint or category, f"admet_endpoint_{endpoint or category}", score)
        if raw_risk is not None:
            add_view("has_admet_raw_risk", endpoint or "toxicity", "admet_raw_risk", _clip01(raw_risk), 0.75)
    elif expert == "DTI":
        target = metadata.get("target_symbol") or metadata.get("target_id") or metadata.get("target_name")
        add_view("has_predicted_target_signal", target or "disease_target_panel", "dti_target_signal", score)
    elif expert == "DrugKB":
        add_view("drugkb_target_signal", metadata.get("target_gene") or metadata.get("target_name") or metadata.get("target_id"), "drugkb_target_signal", score)
        add_view("drugkb_indication_signal", metadata.get("disease_name") or metadata.get("disease_id"), "drugkb_indication_signal", score)
        add_view("drugkb_identity_signal", metadata.get("drug_name") or metadata.get("drugcentral_id"), "drugkb_identity_signal", score)
    elif expert == "DiseaseKB":
        add_view("diseasekb_target_signal", metadata.get("target_symbol") or metadata.get("target_id") or metadata.get("target_name"), "diseasekb_target_signal", score)
        add_view("diseasekb_pathway_signal", metadata.get("pathway_name") or metadata.get("pathway_id"), "diseasekb_pathway_signal", score)
        add_view("diseasekb_therapy_signal", metadata.get("therapy_name") or metadata.get("drug_id"), "diseasekb_therapy_signal", score)
    elif expert == "Clinical":
        add_view("clinical_prior_signal", metadata.get("disease_name") or base.get("object"), "clinical_prior_signal", score)

    match_type = base.get("match_type")
    if match_type:
        add_view("match_quality_signal", match_type, f"match_quality_{match_type}", score, 0.7)
    source = base.get("source")
    if source:
        add_view("source_quality_signal", source, f"source_quality_{expert or source}", score, 0.7)
    return expanded


def expand_typed_evidence(typed: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    expanded: list[Dict[str, Any]] = []
    for item in typed:
        expanded.extend(expand_typed_evidence_tuple(item))
    return expanded
