from __future__ import annotations

import math
from typing import Any, Sequence

from treatagent.orchestration.features import CORE_EXPERTS


ARG_FACTOR_NAMES = [
    "direct_support",
    "mechanism_support",
    "knowledge_support",
    "clinical_feasibility",
    "safety_support",
    "safety_conflict",
    "mechanism_conflict",
    "knowledge_conflict",
    "coverage",
    "source_diversity",
    "missing_penalty",
    "cross_source_consistency",
    "admet_bbb_non_cns_noise",
    "support_strength",
    "conflict_strength",
    "raw_argument_score",
]


CNS_KEYWORDS = [
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _clip01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _noisy_or(values: Sequence[float]) -> float:
    product = 1.0
    for value in values:
        product *= 1.0 - _clip01(value)
    return 1.0 - product


def _is_cns_disease(disease: str) -> bool:
    text = str(disease or "").lower()
    return any(keyword in text for keyword in CNS_KEYWORDS)


def _typed_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
    graph = result.get("evidence_graph") or {}
    typed = [dict(item) for item in graph.get("typed_evidence") or []]
    return typed


def _evidence_strength(item: dict[str, Any], use_reliability: bool = True) -> float:
    score = _clip01(item.get("score"), 0.0)
    if not use_reliability:
        confidence = _clip01(item.get("confidence"), score)
        return max(confidence, score)
    reliability = _clip01(item.get("reliability"), 0.0)
    confidence = _clip01(item.get("confidence"), score * reliability)
    return max(confidence, score * reliability)


def _items(
    typed: Sequence[dict[str, Any]],
    *,
    direction: str | None = None,
    expert: str | None = None,
    categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    output = []
    for item in typed:
        if direction is not None and item.get("direction") != direction:
            continue
        if expert is not None and item.get("expert") != expert:
            continue
        if categories is not None and item.get("category") not in categories:
            continue
        output.append(item)
    return output


def _strengths(items: Sequence[dict[str, Any]], scale: float = 1.0, use_reliability: bool = True) -> list[float]:
    return [_clip01(_evidence_strength(item, use_reliability=use_reliability) * scale) for item in items]


def argument_factors_from_result(result: dict[str, Any], use_reliability: bool = True) -> dict[str, Any]:
    typed = _typed_evidence(result)
    disease = str(result.get("disease") or (result.get("evidence_graph") or {}).get("disease") or "")
    cns = _is_cns_disease(disease)
    experts_present = {str(item.get("expert")) for item in typed if item.get("expert")}

    direct_support_items = _items(
        typed,
        direction="support",
        expert="DrugKB",
        categories={"drug_history", "drugkb_indication_signal"},
    )
    mechanism_support_items = _items(
        typed,
        direction="support",
        categories={
            "mechanism",
            "mechanism_prior",
            "disease_target_prior",
            "clinical_target_bridge",
            "dti_target_signal",
            "drugkb_target_signal",
            "diseasekb_target_signal",
        },
    )
    knowledge_support_items = _items(
        typed,
        direction="support",
        categories={
            "therapy_prior",
            "pathway_prior",
            "drug_class",
            "drug_identity",
            "diseasekb_therapy_signal",
            "diseasekb_pathway_signal",
            "drugkb_identity_signal",
        },
    )
    clinical_items = _items(typed, expert="Clinical", categories={"clinical_prior", "clinical_prior_signal"})
    clinical_scores = [_clip01(item.get("score"), 0.0) for item in clinical_items]
    clinical_raw = sum(clinical_scores) / len(clinical_scores) if clinical_scores else 0.5
    clinical_feasibility = _clip01((clinical_raw - 0.25) / 0.6)

    admet_support_items = []
    admet_conflict_items = []
    bbb_noise_values = []
    for item in _items(typed, expert="ADMET"):
        category = str(item.get("category") or "")
        strength = _evidence_strength(item, use_reliability=use_reliability)
        is_distribution = category in {"distribution", "admet_endpoint_distribution"}
        if is_distribution and not cns:
            bbb_noise_values.append(strength)
            continue
        if item.get("direction") == "support":
            admet_support_items.append(item)
        elif item.get("direction") == "conflict":
            admet_conflict_items.append(item)

    direct_support = _noisy_or(_strengths(direct_support_items, 1.35, use_reliability=use_reliability))
    mechanism_support = _noisy_or(_strengths(mechanism_support_items, 1.05, use_reliability=use_reliability))
    knowledge_support = _noisy_or(_strengths(knowledge_support_items, 0.9, use_reliability=use_reliability))
    safety_support = _noisy_or(_strengths(admet_support_items, 0.8, use_reliability=use_reliability))
    safety_conflict = _noisy_or(_strengths(admet_conflict_items, 1.2, use_reliability=use_reliability))
    mechanism_conflict = _noisy_or(
        _strengths(
            _items(typed, direction="conflict", categories={"mechanism", "dti_target_signal"}),
            1.0,
            use_reliability=use_reliability,
        )
    )
    knowledge_conflict = _noisy_or(
        _strengths(_items(typed, direction="conflict", expert="DrugKB"), 0.9, use_reliability=use_reliability)
    )
    bbb_noise = _noisy_or(bbb_noise_values)

    support_sources = {
        str(item.get("expert"))
        for item in typed
        if item.get("direction") == "support" and item.get("expert")
    }
    conflict_sources = {
        str(item.get("expert"))
        for item in typed
        if item.get("direction") == "conflict" and item.get("expert")
    }
    coverage = len(experts_present & set(CORE_EXPERTS)) / len(CORE_EXPERTS)
    source_diversity = len(support_sources | conflict_sources) / len(CORE_EXPERTS)
    missing_penalty = 1.0 - coverage
    cross_source_consistency = _clip01((len(support_sources) / len(CORE_EXPERTS)) * (1.0 - min(0.75, len(conflict_sources) / len(CORE_EXPERTS))))

    support_strength = _clip01(
        0.55 * clinical_feasibility
        + 0.35 * direct_support
        + 0.10 * cross_source_consistency
    )
    conflict_strength = _clip01(
        0.50 * safety_conflict
        + 0.25 * mechanism_conflict
        + 0.15 * knowledge_conflict
        + 0.10 * bbb_noise
    )
    raw_argument_score = _clip01(
        0.18
        + 0.45 * clinical_feasibility
        + 0.35 * direct_support
        + 0.08 * cross_source_consistency
        - 0.10 * conflict_strength
    )

    factors = {
        "direct_support": direct_support,
        "mechanism_support": mechanism_support,
        "knowledge_support": knowledge_support,
        "clinical_feasibility": clinical_feasibility,
        "safety_support": safety_support,
        "safety_conflict": safety_conflict,
        "mechanism_conflict": mechanism_conflict,
        "knowledge_conflict": knowledge_conflict,
        "coverage": coverage,
        "source_diversity": source_diversity,
        "missing_penalty": missing_penalty,
        "cross_source_consistency": cross_source_consistency,
        "admet_bbb_non_cns_noise": bbb_noise,
        "support_strength": support_strength,
        "conflict_strength": conflict_strength,
        "raw_argument_score": raw_argument_score,
    }
    return {
        "sample_id": result.get("sample_id"),
        "label": result.get("label"),
        "smiles": result.get("smiles"),
        "disease": disease,
        "factors": {name: round(float(factors[name]), 6) for name in ARG_FACTOR_NAMES},
        "top_support_arguments": _top_arguments(typed, "support"),
        "top_conflict_arguments": _top_arguments(typed, "conflict"),
    }


def _top_arguments(typed: Sequence[dict[str, Any]], direction: str, limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(
        [item for item in typed if item.get("direction") == direction],
        key=_evidence_strength,
        reverse=True,
    )[:limit]
    return [
        {
            "expert": item.get("expert"),
            "category": item.get("category"),
            "relation": item.get("relation"),
            "object": item.get("object"),
            "strength": round(_evidence_strength(item), 6),
            "claim": item.get("claim"),
        }
        for item in ranked
    ]
