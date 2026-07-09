#!/usr/bin/env python3
import difflib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRUGKB_PATH = PROJECT_ROOT / "data" / "drugcentral" / "drugkb.jsonl"
DISEASEKB_PATH = PROJECT_ROOT / "data" / "diseasekb" / "diseasekb.jsonl"

DISEASE_STOPWORDS = {
    "disease",
    "disorder",
    "syndrome",
    "condition",
    "adult",
    "children",
    "childhood",
    "primary",
    "secondary",
    "metastatic",
    "advanced",
    "acute",
    "chronic",
    "refractory",
    "resistant",
    "hormone",
    "therapy",
    "treatment",
}

DISEASE_VARIANT_REPLACEMENTS = {
    "hormone refractory": "castration resistant",
    "hormone-refractory": "castration-resistant",
    "castrate resistant": "castration resistant",
    "castrate-resistant": "castration-resistant",
    "non small cell": "non-small cell",
    "triple negative": "triple-negative",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token and token not in DISEASE_STOPWORDS]


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def _normalize_smiles(smiles: str) -> str:
    return re.sub(r"[@/\\\\]", "", smiles.strip())


def _normalize_disease_name(text: str) -> str:
    normalized = text.lower().strip()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    for source, target in DISEASE_VARIANT_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized.strip()


def _parse_snapshot_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None

    for parser in (
        lambda item: datetime.fromisoformat(item.replace("Z", "+00:00")).date(),
        lambda item: datetime.strptime(item, "%Y-%m-%d").date(),
        lambda item: datetime.strptime(item, "%Y/%m/%d").date(),
        lambda item: datetime.strptime(item, "%Y%m%d").date(),
    ):
        try:
            return parser(text)
        except ValueError:
            continue
    return None


def _build_disease_variants(disease: str) -> List[str]:
    variants = []
    normalized = _normalize_disease_name(disease)
    if normalized:
        variants.append(normalized)

    tokens = [token for token in normalized.split() if token not in DISEASE_STOPWORDS]
    if tokens:
        variants.append(" ".join(tokens))

    for modifier in ["metastatic", "advanced", "refractory", "resistant", "hormone", "secondary", "primary"]:
        collapsed = " ".join(token for token in tokens if token != modifier)
        if collapsed:
            variants.append(collapsed)

    if normalized.endswith(" cancer"):
        variants.append(normalized.replace(" cancer", " carcinoma"))
        variants.append(normalized.replace(" cancer", " neoplasm"))
    if normalized.endswith(" carcinoma"):
        variants.append(normalized.replace(" carcinoma", " cancer"))

    seen = set()
    unique_variants = []
    for item in variants:
        item = re.sub(r"\s+", " ", item).strip()
        if item and item not in seen:
            unique_variants.append(item)
            seen.add(item)
    return unique_variants


