from __future__ import annotations

from pathlib import Path
from typing import Sequence

from experiments.orchestration.eg_scorer import EGScorer, train_eg_scorer


RAW_FEATURE_FUSION_FEATURES = [
    "direct_indication_match",
    "indication_similarity",
    "target_overlap",
    "weighted_target_score",
    "pathway_consistency",
    "dti_strength",
    "admet_safety_score",
    "toxicity_conflict_score",
    "clinical_prior",
    "drugkb_present",
    "diseasekb_present",
    "dti_present",
    "admet_present",
    "clinical_present",
]


def train_raw_feature_fusion_scorer(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[int],
    val_x: Sequence[Sequence[float]],
    val_y: Sequence[int],
    random_state: int = 13,
) -> tuple[EGScorer, dict]:
    """Train the non-graph raw feature fusion baseline.

    This baseline intentionally uses only source-level scalar features from the
    evidence agents. It excludes EvidenceGraph aggregation, support/conflict
    structure, reliability, confidence, coverage, and missing-evidence features.
    """

    return train_eg_scorer(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        feature_names=RAW_FEATURE_FUSION_FEATURES,
        random_state=random_state,
    )


def default_model_name(split_prefix: str) -> str:
    return f"raw_feature_fusion_lr_{split_prefix}"


def default_output_dir(project_root: Path, split_prefix: str) -> Path:
    return project_root / "results" / "raw_feature_fusion" / split_prefix

