import json
from pathlib import Path

from .knowledge_common import _build_disease_variants, _jaccard_similarity, _normalize_disease_name, difflib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUCCESS_RATIO_PATH = PROJECT_ROOT / "data" / "clinical" / "disease_success_ratio.json"

with SUCCESS_RATIO_PATH.open("r", encoding="utf-8") as handle:
    disease_success_ratio = json.load(handle)

_normalized_success_ratio = {
    _normalize_disease_name(name): value
    for name, value in disease_success_ratio.items()
    if _normalize_disease_name(name)
}


def get_disease_success_prior(disease_name):
    if disease_name in disease_success_ratio:
        return {
            "success_rate": float(disease_success_ratio[disease_name]),
            "matched_disease": disease_name,
            "matched_by": "exact",
            "match_score": 1.0,
        }
    for variant in _build_disease_variants(disease_name):
        if variant in _normalized_success_ratio:
            return {
                "success_rate": float(_normalized_success_ratio[variant]),
                "matched_disease": variant,
                "matched_by": "normalized",
                "match_score": 0.98,
            }

    best_key = None
    best_score = 0.0
    for variant in _build_disease_variants(disease_name):
        for candidate in _normalized_success_ratio:
            jaccard = _jaccard_similarity(variant, candidate)
            if jaccard < 0.35:
                continue
            ratio = difflib.SequenceMatcher(None, variant, candidate).ratio()
            score = max(jaccard, ratio)
            if variant in candidate or candidate in variant:
                score += 0.08
            if score > best_score:
                best_score = score
                best_key = candidate
    if best_key is not None and best_score >= 0.72:
        return {
            "success_rate": float(_normalized_success_ratio[best_key]),
            "matched_disease": best_key,
            "matched_by": "fuzzy",
            "match_score": round(min(best_score, 1.0), 4),
        }
    return None


def get_disease_risk(disease_name):
    prior = get_disease_success_prior(disease_name)
    if prior is None:
        return None
    return prior["success_rate"]
