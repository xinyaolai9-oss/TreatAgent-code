from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from treatagent.orchestration.evidence import legacy_evidence_to_tuple


CORE_EXPERTS = ["DrugKB", "DiseaseKB", "DTI", "ADMET", "Clinical"]
BASE_GRAPH_FEATURE_NAMES = [
    "direct_indication_match",
    "indication_similarity",
    "target_overlap",
    "weighted_target_score",
    "pathway_consistency",
    "dti_strength",
    "admet_safety_score",
    "toxicity_conflict_score",
    "clinical_prior",
    "evidence_count",
    "evidence_coverage",
    "source_diversity",
    "support_count",
    "conflict_count",
    "support_score",
    "conflict_score",
    "support_conflict_ratio",
    "reliability_mean",
    "reliability_max",
    "confidence_mean",
    "confidence_max",
    "missing_evidence_count",
    "agent_failure_count",
    "drugkb_present",
    "diseasekb_present",
    "dti_present",
    "admet_present",
    "clinical_present",
]

INTERACTION_FEATURE_NAMES = [
    "drugkb_support_score",
    "diseasekb_support_score",
    "dti_support_score",
    "admet_support_score",
    "admet_conflict_score",
    "clinical_support_score",
    "drugkb_support_x_dti_support",
    "diseasekb_support_x_dti_support",
    "drugkb_support_x_clinical_prior",
    "dti_support_x_clinical_prior",
    "admet_conflict_x_low_clinical_prior",
    "admet_conflict_x_dti_support",
    "drugkb_support_x_admet_safety",
    "missing_clinical_x_dti_support",
    "no_direct_indication_x_dti_support",
    "support_x_source_diversity",
    "conflict_x_low_reliability",
    "cns_disease_flag",
    "admet_distribution_support_score",
    "admet_distribution_conflict_score",
    "admet_refined_support_score",
    "admet_refined_conflict_score",
    "admet_bbb_cns_support",
    "admet_bbb_non_cns_noise",
    "clinical_prior_raw_score",
    "clinical_prior_centered",
    "clinical_prior_high_bin",
    "clinical_prior_low_bin",
    "clinical_prior_x_drugkb_support",
    "clinical_prior_x_diseasekb_support",
    "clinical_prior_x_dti_support",
    "clinical_prior_x_admet_refined_support",
    "low_clinical_prior_x_admet_refined_conflict",
]

FEATURE_NAMES = BASE_GRAPH_FEATURE_NAMES + INTERACTION_FEATURE_NAMES


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    if not values:
        return default
    return sum(values) / len(values)


def _max(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    if not values:
        return default
    return max(values)


def _weighted_mean(items: Iterable[dict], default: float = 0.0) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for item in items:
        reliability = _safe_float(item.get("reliability"), 0.0)
        score = _safe_float(item.get("score"), 0.0)
        weighted_sum += score * reliability
        weight_sum += reliability
    if weight_sum <= 0:
        return default
    return weighted_sum / weight_sum


def _normalize_typed_evidence(evidence_graph: dict) -> List[dict]:
    typed = evidence_graph.get("typed_evidence") or []
    if typed:
        return [dict(item) for item in typed]

    drug = evidence_graph.get("drug") or ""
    disease = evidence_graph.get("disease") or ""
    legacy = evidence_graph.get("evidence") or []
    converted = []
    for item in legacy:
        converted.append(legacy_evidence_to_tuple(item, drug, disease).to_dict())
    return converted


def _is_cns_disease(disease: str) -> float:
    text = str(disease or "").lower()
    keywords = [
        "brain",
        "central nervous",
        "cns",
        "neuro",
        "epilep",
        "seizure",
        "alzheimer",
        "parkinson",
        "multiple sclerosis",
        "bipolar",
        "schizophrenia",
        "depression",
        "anxiety",
        "migraine",
        "glioma",
        "meningitis",
        "spinal",
        "huntington",
        "amyotrophic",
        "dementia",
        "autism",
        "stroke",
    ]
    return 1.0 if any(keyword in text for keyword in keywords) else 0.0


def _expert_status_counts(expert_outputs: Optional[Dict[str, Dict[str, Any]]]) -> Counter:
    counts: Counter = Counter()
    if not expert_outputs:
        return counts
    for expert, output in expert_outputs.items():
        status = str((output or {}).get("status") or "missing").lower()
        counts[f"{expert}:{status}"] += 1
    return counts


def _target_overlap(typed: List[dict]) -> float:
    drug_targets = {
        str(item.get("object")).lower()
        for item in typed
        if item.get("expert") == "DrugKB" and item.get("relation") == "targets" and item.get("object")
    }
    disease_targets = {
        str(item.get("object")).lower()
        for item in typed
        if item.get("expert") == "DiseaseKB"
        and item.get("relation") in {"associated_with_target", "therapy_targets"}
        and item.get("object")
    }
    if not drug_targets or not disease_targets:
        return 0.0
    return len(drug_targets & disease_targets) / len(drug_targets | disease_targets)


def _score_by_category(typed: List[dict], category: str, direction: Optional[str] = None) -> float:
    items = [item for item in typed if item.get("category") == category]
    if direction:
        items = [item for item in items if item.get("direction") == direction]
    return _weighted_mean(items)


def _max_by_category(typed: List[dict], category: str, direction: Optional[str] = None) -> float:
    items = [item for item in typed if item.get("category") == category]
    if direction:
        items = [item for item in items if item.get("direction") == direction]
    return _max((_safe_float(item.get("score"), 0.0) for item in items))


def extract_graph_features(
    evidence_graph: Dict[str, Any],
    expert_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, float]:
    typed = _normalize_typed_evidence(evidence_graph)
    support_items = [item for item in typed if item.get("direction") == "support"]
    conflict_items = [item for item in typed if item.get("direction") == "conflict"]
    experts_present = {str(item.get("expert")) for item in typed if item.get("expert")}
    sources_present = {str(item.get("source")) for item in typed if item.get("source")}
    categories_present = {str(item.get("category")) for item in typed if item.get("category")}

    support_score = sum(_safe_float(item.get("confidence"), 0.0) for item in support_items)
    conflict_score = sum(_safe_float(item.get("confidence"), 0.0) for item in conflict_items)
    reliability_values = [_safe_float(item.get("reliability"), 0.0) for item in typed]
    confidence_values = [_safe_float(item.get("confidence"), 0.0) for item in typed]

    status_counts = _expert_status_counts(expert_outputs)
    failed_or_partial = 0
    for expert in CORE_EXPERTS:
        has_success = (
            status_counts.get(f"{expert}:ok", 0)
            or status_counts.get(f"{expert}:success", 0)
            or status_counts.get(f"{expert}:partial", 0)
        )
        has_failure = (
            status_counts.get(f"{expert}:no_data", 0)
            or status_counts.get(f"{expert}:failed", 0)
            or status_counts.get(f"{expert}:error", 0)
            or status_counts.get(f"{expert}:unknown", 0)
        )
        if has_failure or (expert_outputs is not None and not has_success and expert not in experts_present):
            failed_or_partial += 1

    missing_evidence_count = sum(1 for expert in CORE_EXPERTS if expert not in experts_present)
    indication_items = [
        item for item in typed if item.get("expert") == "DrugKB" and item.get("relation") == "has_indication"
    ]
    indication_similarity = _max(
        _safe_float((item.get("metadata") or {}).get("match_score"), 0.0) for item in indication_items
    )

    features = {
        "direct_indication_match": 1.0 if indication_items else 0.0,
        "indication_similarity": indication_similarity,
        "target_overlap": _target_overlap(typed),
        "weighted_target_score": _weighted_mean(
            item for item in typed if item.get("relation") in {"targets", "associated_with_target", "therapy_targets"}
        ),
        "pathway_consistency": _score_by_category(typed, "pathway_prior", "support"),
        "dti_strength": _max_by_category(typed, "mechanism", "support"),
        "admet_safety_score": _weighted_mean(
            item for item in typed if item.get("expert") == "ADMET" and item.get("direction") == "support"
        ),
        "toxicity_conflict_score": _score_by_category(typed, "toxicity", "conflict"),
        "clinical_prior": _score_by_category(typed, "clinical_prior", "support"),
        "evidence_count": float(len(typed)),
        "evidence_coverage": min(1.0, len(categories_present) / 12.0),
        "source_diversity": min(1.0, len(sources_present) / 5.0),
        "support_count": float(len(support_items)),
        "conflict_count": float(len(conflict_items)),
        "support_score": support_score,
        "conflict_score": conflict_score,
        "support_conflict_ratio": support_score / (support_score + conflict_score + 1e-6),
        "reliability_mean": _mean(reliability_values),
        "reliability_max": _max(reliability_values),
        "confidence_mean": _mean(confidence_values),
        "confidence_max": _max(confidence_values),
        "missing_evidence_count": float(missing_evidence_count),
        "agent_failure_count": float(failed_or_partial),
        "drugkb_present": 1.0 if "DrugKB" in experts_present else 0.0,
        "diseasekb_present": 1.0 if "DiseaseKB" in experts_present else 0.0,
        "dti_present": 1.0 if "DTI" in experts_present else 0.0,
        "admet_present": 1.0 if "ADMET" in experts_present else 0.0,
        "clinical_present": 1.0 if "Clinical" in experts_present else 0.0,
    }

    drugkb_support = sum(
        _safe_float(item.get("confidence"), 0.0)
        for item in typed
        if item.get("expert") == "DrugKB" and item.get("direction") == "support"
    )
    diseasekb_support = sum(
        _safe_float(item.get("confidence"), 0.0)
        for item in typed
        if item.get("expert") == "DiseaseKB" and item.get("direction") == "support"
    )
    dti_support = sum(
        _safe_float(item.get("confidence"), 0.0)
        for item in typed
        if item.get("expert") == "DTI" and item.get("direction") == "support"
    )
    admet_support = sum(
        _safe_float(item.get("confidence"), 0.0)
        for item in typed
        if item.get("expert") == "ADMET" and item.get("direction") == "support"
    )
    admet_conflict = sum(
        _safe_float(item.get("confidence"), 0.0)
        for item in typed
        if item.get("expert") == "ADMET" and item.get("direction") == "conflict"
    )
    clinical_support = sum(
        _safe_float(item.get("confidence"), 0.0)
        for item in typed
        if item.get("expert") == "Clinical" and item.get("direction") == "support"
    )
    low_clinical_prior = 1.0 - min(1.0, clinical_support)
    low_reliability = 1.0 - features["reliability_mean"]
    no_direct_indication = 1.0 - features["direct_indication_match"]
    cns_disease_flag = _is_cns_disease(str(evidence_graph.get("disease") or ""))
    admet_distribution_support = sum(
        _safe_float(item.get("confidence"), 0.0)
        for item in typed
        if item.get("expert") == "ADMET"
        and item.get("category") == "distribution"
        and item.get("direction") == "support"
    )
    admet_distribution_conflict = sum(
        _safe_float(item.get("confidence"), 0.0)
        for item in typed
        if item.get("expert") == "ADMET"
        and item.get("category") == "distribution"
        and item.get("direction") == "conflict"
    )
    admet_non_distribution_support = max(0.0, admet_support - admet_distribution_support)
    admet_non_distribution_conflict = max(0.0, admet_conflict - admet_distribution_conflict)
    admet_refined_support = admet_non_distribution_support + (admet_distribution_support * cns_disease_flag)
    admet_refined_conflict = admet_non_distribution_conflict + (admet_distribution_conflict * cns_disease_flag)
    clinical_prior_values = [
        _safe_float(item.get("score"), 0.0)
        for item in typed
        if item.get("expert") == "Clinical" and item.get("category") == "clinical_prior"
    ]
    clinical_prior_raw = _mean(clinical_prior_values)
    clinical_prior_centered = clinical_prior_raw - 0.5 if clinical_prior_values else 0.0
    clinical_prior_high = max(0.0, clinical_prior_raw - 0.6)
    clinical_prior_low = max(0.0, 0.45 - clinical_prior_raw) if clinical_prior_values else 0.0
    features.update(
        {
            "drugkb_support_score": drugkb_support,
            "diseasekb_support_score": diseasekb_support,
            "dti_support_score": dti_support,
            "admet_support_score": admet_support,
            "admet_conflict_score": admet_conflict,
            "clinical_support_score": clinical_support,
            "drugkb_support_x_dti_support": drugkb_support * dti_support,
            "diseasekb_support_x_dti_support": diseasekb_support * dti_support,
            "drugkb_support_x_clinical_prior": drugkb_support * clinical_support,
            "dti_support_x_clinical_prior": dti_support * clinical_support,
            "admet_conflict_x_low_clinical_prior": admet_conflict * low_clinical_prior,
            "admet_conflict_x_dti_support": admet_conflict * dti_support,
            "drugkb_support_x_admet_safety": drugkb_support * admet_support,
            "missing_clinical_x_dti_support": (1.0 - features["clinical_present"]) * dti_support,
            "no_direct_indication_x_dti_support": no_direct_indication * dti_support,
            "support_x_source_diversity": support_score * features["source_diversity"],
            "conflict_x_low_reliability": conflict_score * low_reliability,
            "cns_disease_flag": cns_disease_flag,
            "admet_distribution_support_score": admet_distribution_support,
            "admet_distribution_conflict_score": admet_distribution_conflict,
            "admet_refined_support_score": admet_refined_support,
            "admet_refined_conflict_score": admet_refined_conflict,
            "admet_bbb_cns_support": admet_distribution_support * cns_disease_flag,
            "admet_bbb_non_cns_noise": (admet_distribution_support + admet_distribution_conflict) * (1.0 - cns_disease_flag),
            "clinical_prior_raw_score": clinical_prior_raw,
            "clinical_prior_centered": clinical_prior_centered,
            "clinical_prior_high_bin": clinical_prior_high,
            "clinical_prior_low_bin": clinical_prior_low,
            "clinical_prior_x_drugkb_support": clinical_prior_raw * drugkb_support,
            "clinical_prior_x_diseasekb_support": clinical_prior_raw * diseasekb_support,
            "clinical_prior_x_dti_support": clinical_prior_raw * dti_support,
            "clinical_prior_x_admet_refined_support": clinical_prior_raw * admet_refined_support,
            "low_clinical_prior_x_admet_refined_conflict": max(0.0, 0.5 - clinical_prior_raw) * admet_refined_conflict,
        }
    )

    return {name: round(float(features.get(name, 0.0)), 6) for name in FEATURE_NAMES}


def feature_row_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "sample_id": result.get("sample_id"),
        "label": result.get("label"),
        "prediction_binary": result.get("prediction_binary"),
        "prediction_score": result.get("prediction_score"),
        "calibrated_probability": result.get("calibrated_probability"),
        "raw_score": result.get("raw_score"),
        "smiles": result.get("smiles"),
        "disease": result.get("disease"),
    }
    row.update(extract_graph_features(result.get("evidence_graph") or {}, result.get("expert_outputs") or {}))
    return row
