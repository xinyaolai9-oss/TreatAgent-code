from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.orchestration.eg_scorer import (
    choose_threshold,
    classification_metrics,
    evaluate_scorer,
)
from treatagent.orchestration.evidence import expand_typed_evidence
from treatagent.orchestration.features import BASE_GRAPH_FEATURE_NAMES, CORE_EXPERTS, FEATURE_NAMES, extract_graph_features


DIRECTIONS = ["support", "conflict", "neutral"]
EXPERTS = CORE_EXPERTS
MATCH_TYPES = [
    "canonical_smiles",
    "relaxed_smiles",
    "disease_name",
    "alias",
    "fuzzy",
    "nearest_smiles",
    "predicted",
    "model_prediction",
    "disease_prior",
    "unknown",
]
CATEGORIES = [
    "drug_identity",
    "drug_history",
    "mechanism_prior",
    "drug_class",
    "disease_target_prior",
    "therapy_prior",
    "pathway_prior",
    "clinical_target_bridge",
    "mechanism",
    "absorption",
    "distribution",
    "toxicity",
    "clinical_prior",
    "admet_endpoint_absorption",
    "admet_endpoint_distribution",
    "admet_endpoint_toxicity",
    "admet_raw_risk",
    "dti_target_signal",
    "drugkb_target_signal",
    "drugkb_indication_signal",
    "drugkb_identity_signal",
    "diseasekb_target_signal",
    "diseasekb_pathway_signal",
    "diseasekb_therapy_signal",
    "clinical_prior_signal",
]


BASE_FEATURE_NAMES = [
    "support_att_score",
    "support_att_confidence",
    "support_att_reliability",
    "support_weight_sum",
    "support_count",
    "support_entropy",
    "conflict_att_score",
    "conflict_att_confidence",
    "conflict_att_reliability",
    "conflict_weight_sum",
    "conflict_count",
    "conflict_entropy",
    "neutral_att_score",
    "neutral_att_confidence",
    "neutral_att_reliability",
    "neutral_weight_sum",
    "neutral_count",
    "neutral_entropy",
    "support_conflict_score_delta",
    "support_conflict_confidence_delta",
    "support_conflict_reliability_delta",
    "support_conflict_ratio",
    "evidence_count",
    "source_diversity",
    "missing_evidence_count",
    "agent_failure_count",
]

EXPERT_FEATURE_NAMES = [
    f"{direction}_{expert.lower()}_weight" for direction in DIRECTIONS for expert in EXPERTS
] + [f"{expert.lower()}_present" for expert in EXPERTS]

MATCH_FEATURE_NAMES = [
    f"{direction}_match_{match_type}_weight" for direction in DIRECTIONS for match_type in MATCH_TYPES
]

CATEGORY_FEATURE_NAMES = [
    f"{direction}_category_{category}_weight" for direction in DIRECTIONS for category in CATEGORIES
]

GRAPH_FEATURE_NAMES = [f"graph_{name}" for name in FEATURE_NAMES]

RAEG_FEATURE_NAMES = (
    BASE_FEATURE_NAMES
    + EXPERT_FEATURE_NAMES
    + MATCH_FEATURE_NAMES
    + CATEGORY_FEATURE_NAMES
    + GRAPH_FEATURE_NAMES
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _clip01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _normalize_direction(value: Any) -> str:
    text = str(value or "neutral").lower()
    if text in {"support", "conflict"}:
        return text
    return "neutral"


def _normalize_expert(value: Any) -> str:
    text = str(value or "")
    return text if text in EXPERTS else "unknown"


def _normalize_match_type(value: Any) -> str:
    text = str(value or "unknown")
    return text if text in MATCH_TYPES else "unknown"


def _normalize_category(value: Any) -> str:
    text = str(value or "")
    return text if text in CATEGORIES else ""


def _typed_evidence_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    graph = result.get("evidence_graph") or {}
    typed = graph.get("typed_evidence") or []
    return expand_typed_evidence([dict(item) for item in typed])


def _attention_weights(items: list[dict[str, Any]]) -> list[float]:
    if not items:
        return []
    logits = []
    for item in items:
        reliability = max(_clip01(item.get("reliability"), 0.0), 1e-6)
        score = _clip01(item.get("score"), 0.0)
        confidence = _clip01(item.get("confidence"), score * reliability)
        logits.append(math.log(reliability) + 0.5 * score + 0.25 * confidence)
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    if total <= 0:
        return [1.0 / len(items) for _ in items]
    return [value / total for value in exp_values]


def _entropy(weights: Sequence[float]) -> float:
    if not weights:
        return 0.0
    return -sum(weight * math.log(weight + 1e-12) for weight in weights) / math.log(len(weights) + 1e-12)


def _direction_pool(direction: str, typed: list[dict[str, Any]]) -> dict[str, float]:
    items = [item for item in typed if _normalize_direction(item.get("direction")) == direction]
    weights = _attention_weights(items)
    result = {
        f"{direction}_att_score": 0.0,
        f"{direction}_att_confidence": 0.0,
        f"{direction}_att_reliability": 0.0,
        f"{direction}_weight_sum": float(sum(_clip01(item.get("reliability"), 0.0) for item in items)),
        f"{direction}_count": float(len(items)),
        f"{direction}_entropy": _entropy(weights),
    }
    for item, weight in zip(items, weights):
        score = _clip01(item.get("score"), 0.0)
        reliability = _clip01(item.get("reliability"), 0.0)
        confidence = _clip01(item.get("confidence"), score * reliability)
        result[f"{direction}_att_score"] += weight * score
        result[f"{direction}_att_confidence"] += weight * confidence
        result[f"{direction}_att_reliability"] += weight * reliability
    return result


def _categorical_weight_features(direction: str, typed: list[dict[str, Any]]) -> dict[str, float]:
    items = [item for item in typed if _normalize_direction(item.get("direction")) == direction]
    weights = _attention_weights(items)
    result = {}
    for expert in EXPERTS:
        result[f"{direction}_{expert.lower()}_weight"] = 0.0
    for match_type in MATCH_TYPES:
        result[f"{direction}_match_{match_type}_weight"] = 0.0
    for category in CATEGORIES:
        result[f"{direction}_category_{category}_weight"] = 0.0

    for item, weight in zip(items, weights):
        expert = _normalize_expert(item.get("expert"))
        if expert in EXPERTS:
            result[f"{direction}_{expert.lower()}_weight"] += weight
        match_type = _normalize_match_type(item.get("match_type"))
        result[f"{direction}_match_{match_type}_weight"] += weight
        category = _normalize_category(item.get("category"))
        if category:
            result[f"{direction}_category_{category}_weight"] += weight
    return result


def raeg_feature_row_from_result(result: dict[str, Any]) -> dict[str, Any]:
    typed = _typed_evidence_from_result(result)
    row: dict[str, Any] = {
        "sample_id": result.get("sample_id"),
        "label": result.get("label"),
        "smiles": result.get("smiles"),
        "disease": result.get("disease"),
    }
    features = {name: 0.0 for name in RAEG_FEATURE_NAMES}
    for direction in DIRECTIONS:
        features.update(_direction_pool(direction, typed))
        features.update(_categorical_weight_features(direction, typed))

    experts_present = {_normalize_expert(item.get("expert")) for item in typed}
    for expert in EXPERTS:
        features[f"{expert.lower()}_present"] = 1.0 if expert in experts_present else 0.0

    graph_features = extract_graph_features(result.get("evidence_graph") or {}, result.get("expert_outputs") or {})
    features["support_conflict_score_delta"] = (
        features["support_att_score"] - features["conflict_att_score"]
    )
    features["support_conflict_confidence_delta"] = (
        features["support_att_confidence"] - features["conflict_att_confidence"]
    )
    features["support_conflict_reliability_delta"] = (
        features["support_att_reliability"] - features["conflict_att_reliability"]
    )
    support_weight = features["support_weight_sum"]
    conflict_weight = features["conflict_weight_sum"]
    features["support_conflict_ratio"] = support_weight / (support_weight + conflict_weight + 1e-6)
    features["evidence_count"] = float(len(typed))
    features["source_diversity"] = graph_features.get("source_diversity", 0.0)
    features["missing_evidence_count"] = graph_features.get("missing_evidence_count", 0.0)
    features["agent_failure_count"] = graph_features.get("agent_failure_count", 0.0)
    for name in FEATURE_NAMES:
        features[f"graph_{name}"] = graph_features.get(name, 0.0)

    row.update({name: round(float(features.get(name, 0.0)), 6) for name in RAEG_FEATURE_NAMES})
    return row


def raeg_attention_summary(result: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    typed = _typed_evidence_from_result(result)
    summary: dict[str, Any] = {}
    for direction in DIRECTIONS:
        items = [item for item in typed if _normalize_direction(item.get("direction")) == direction]
        weights = _attention_weights(items)
        ranked = sorted(
            zip(items, weights),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
        summary[direction] = [
            {
                "attention": round(float(weight), 6),
                "expert": item.get("expert"),
                "category": item.get("category"),
                "relation": item.get("relation"),
                "object": item.get("object"),
                "score": item.get("score"),
                "reliability": item.get("reliability"),
                "confidence": item.get("confidence"),
                "claim": item.get("claim"),
            }
            for item, weight in ranked
        ]
    return summary


def read_result_json(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, payload.get("results") or []


def read_raeg_result_features(path: Path) -> tuple[list[dict[str, Any]], list[list[float]], list[int]]:
    _, results = read_result_json(path)
    rows: list[dict[str, Any]] = []
    x: list[list[float]] = []
    y: list[int] = []
    for result in results:
        if result.get("label") in (None, ""):
            continue
        row = raeg_feature_row_from_result(result)
        rows.append(row)
        x.append([_safe_float(row.get(name), 0.0) for name in RAEG_FEATURE_NAMES])
        y.append(int(float(result.get("label"))))
    return rows, x, y


def matrix_from_rows(rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> list[list[float]]:
    return [[_safe_float(row.get(name), 0.0) for name in feature_names] for row in rows]


@dataclass
class RAEGScorer:
    model: Pipeline
    calibrator: LogisticRegression | None
    feature_names: list[str]
    threshold: float

    def raw_predict_proba(self, x: Iterable[Sequence[float]]) -> list[float]:
        probabilities = self.model.predict_proba(list(x))[:, 1]
        return [float(value) for value in probabilities]

    def predict_proba(self, x: Iterable[Sequence[float]]) -> list[float]:
        raw = self.raw_predict_proba(x)
        if self.calibrator is None:
            return raw
        calibrated = self.calibrator.predict_proba([[value] for value in raw])[:, 1]
        return [float(value) for value in calibrated]

    def predict(self, x: Iterable[Sequence[float]]) -> list[int]:
        return [1 if prob >= self.threshold else 0 for prob in self.predict_proba(x)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "model": self.model,
                    "calibrator": self.calibrator,
                    "feature_names": self.feature_names,
                    "threshold": self.threshold,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "RAEGScorer":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(
            model=payload["model"],
            calibrator=payload.get("calibrator"),
            feature_names=list(payload["feature_names"]),
            threshold=float(payload["threshold"]),
        )


def train_raeg_scorer(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[int],
    val_x: Sequence[Sequence[float]],
    val_y: Sequence[int],
    random_state: int = 13,
) -> tuple[RAEGScorer, dict[str, Any]]:
    all_feature_names = list(RAEG_FEATURE_NAMES)
    graph_names = [f"graph_{name}" for name in BASE_GRAPH_FEATURE_NAMES]
    graph_interaction_names = list(GRAPH_FEATURE_NAMES)
    refined_feature_suffixes = [
        "cns_disease_flag",
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
        "drugkb_support_x_dti_support",
        "diseasekb_support_x_dti_support",
        "admet_conflict_x_low_clinical_prior",
        "admet_conflict_x_dti_support",
        "no_direct_indication_x_dti_support",
    ]
    graph_refined_names = graph_names + [f"graph_{name}" for name in refined_feature_suffixes]
    raeg_only_names = [name for name in all_feature_names if name not in graph_names]

    def select_matrix(matrix: Sequence[Sequence[float]], selected_names: Sequence[str]) -> list[list[float]]:
        indices = [all_feature_names.index(name) for name in selected_names]
        return [[float(row[index]) for index in indices] for row in matrix]

    def make_lr(c: float = 1.0) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=c,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def make_mlp(hidden_layer_sizes: tuple[int, ...], alpha: float) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation="relu",
                        alpha=alpha,
                        batch_size=64,
                        learning_rate_init=0.001,
                        max_iter=800,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    candidates = [
        ("graph_lr", make_lr(1.0), graph_names),
        ("graph_interaction_lr", make_lr(1.0), graph_interaction_names),
        ("graph_refined_lr", make_lr(1.0), graph_refined_names),
        ("graph_refined_lr_c03", make_lr(0.3), graph_refined_names),
        ("raeg_lr", make_lr(1.0), raeg_only_names),
        ("raeg_graph_lr_c03", make_lr(0.3), all_feature_names),
        ("raeg_graph_lr_c10", make_lr(1.0), all_feature_names),
        ("raeg_graph_mlp_16", make_mlp((16,), 0.08), all_feature_names),
        ("raeg_graph_mlp_24", make_mlp((24,), 0.05), all_feature_names),
        ("raeg_mlp_16", make_mlp((16,), 0.08), raeg_only_names),
        (
            "raeg_graph_gbdt",
            GradientBoostingClassifier(
                n_estimators=80,
                learning_rate=0.03,
                max_depth=2,
                subsample=0.85,
                random_state=random_state,
            ),
            all_feature_names,
        ),
        (
            "raeg_graph_histgb",
            HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.03,
                max_leaf_nodes=8,
                l2_regularization=0.2,
                random_state=random_state,
            ),
            all_feature_names,
        ),
        (
            "raeg_graph_extratrees",
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            all_feature_names,
        ),
        (
            "graph_extratrees",
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            graph_names,
        ),
        (
            "raeg_graph_randomforest",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=5,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
            all_feature_names,
        ),
    ]

    best: dict[str, Any] | None = None
    graph_lr_candidate: dict[str, Any] | None = None
    trained_candidates: list[dict[str, Any]] = []
    candidate_reports: list[dict[str, Any]] = []
    for candidate_name, model, selected_names in candidates:
        selected_train_x = select_matrix(train_x, selected_names)
        selected_val_x = select_matrix(val_x, selected_names)
        model.fit(selected_train_x, train_y)
        val_raw_prob = [float(value) for value in model.predict_proba(selected_val_x)[:, 1]]

        calibrator: LogisticRegression | None = None
        val_prob = val_raw_prob
        if len(set(val_y)) > 1:
            calibration_candidate = LogisticRegression(max_iter=1000, random_state=random_state)
            calibration_candidate.fit([[value] for value in val_raw_prob], val_y)
            candidate_prob = [
                float(value) for value in calibration_candidate.predict_proba([[p] for p in val_raw_prob])[:, 1]
            ]
            raw_metrics_at_half = classification_metrics(val_y, val_raw_prob, 0.5)
            candidate_metrics_at_half = classification_metrics(val_y, candidate_prob, 0.5)
            if (candidate_metrics_at_half["brier"], candidate_metrics_at_half["ece"]) < (
                raw_metrics_at_half["brier"],
                raw_metrics_at_half["ece"],
            ):
                calibrator = calibration_candidate
                val_prob = candidate_prob

        threshold, val_metrics = choose_threshold(val_y, val_prob)
        report = {
            "candidate": candidate_name,
            "feature_count": len(selected_names),
            "feature_group": "graph" if selected_names == graph_names else "raeg" if selected_names == raeg_only_names else "raeg_graph",
            "calibrator_selected": calibrator is not None,
            "raw_validation": classification_metrics(val_y, val_raw_prob, 0.5),
            "validation": val_metrics,
        }
        candidate_reports.append(report)
        selection_key = (
            val_metrics["f1"],
            val_metrics["accuracy"],
            val_metrics.get("auprc") or 0.0,
            -(val_metrics["brier"]),
            -(val_metrics["ece"]),
        )
        if best is None or selection_key > best["selection_key"]:
            best = {
                "selection_key": selection_key,
                "candidate": candidate_name,
                "model": model,
                "calibrator": calibrator,
                "feature_names": list(selected_names),
                "threshold": threshold,
                "validation": val_metrics,
                "raw_validation": classification_metrics(val_y, val_raw_prob, 0.5),
                "calibrator_selected": calibrator is not None,
            }
        trained_candidates.append(
            {
                "selection_key": selection_key,
                "candidate": candidate_name,
                "model": model,
                "calibrator": calibrator,
                "feature_names": list(selected_names),
                "threshold": threshold,
                "validation": val_metrics,
                "raw_validation": classification_metrics(val_y, val_raw_prob, 0.5),
                "calibrator_selected": calibrator is not None,
            }
        )
        if candidate_name == "graph_lr":
            graph_lr_candidate = {
                "selection_key": selection_key,
                "candidate": candidate_name,
                "model": model,
                "calibrator": calibrator,
                "feature_names": list(selected_names),
                "threshold": threshold,
                "validation": val_metrics,
                "raw_validation": classification_metrics(val_y, val_raw_prob, 0.5),
                "calibrator_selected": calibrator is not None,
            }

    assert best is not None
    assert graph_lr_candidate is not None
    graph_metrics = graph_lr_candidate["validation"]
    eligible_candidates: list[dict[str, Any]] = [graph_lr_candidate]
    for candidate in trained_candidates:
        metrics = candidate["validation"]
        if candidate["candidate"] == "graph_lr":
            continue
        f1_delta = metrics["f1"] - graph_metrics["f1"]
        auprc_delta = (metrics.get("auprc") or 0.0) - (graph_metrics.get("auprc") or 0.0)
        auroc_delta = (metrics.get("auroc") or 0.0) - (graph_metrics.get("auroc") or 0.0)
        brier_delta = metrics["brier"] - graph_metrics["brier"]
        ece_delta = metrics["ece"] - graph_metrics["ece"]
        is_nonlinear = any(token in candidate["candidate"] for token in ["mlp", "gbdt", "histgb", "forest", "trees"])
        strongly_better_f1 = f1_delta >= (0.03 if is_nonlinear else 0.02) and brier_delta <= 0.0 and ece_delta <= 0.0
        better_ranking_or_calibration = (
            not is_nonlinear
            and f1_delta >= 0.0
            and brier_delta <= 0.005
            and ece_delta <= 0.015
            and (auprc_delta >= 0.01 or auroc_delta >= 0.01 or brier_delta <= -0.005)
        )
        if strongly_better_f1 or better_ranking_or_calibration:
            candidate["guardrail_reason"] = {
                "f1_delta": round(f1_delta, 6),
                "auprc_delta": round(auprc_delta, 6),
                "auroc_delta": round(auroc_delta, 6),
                "brier_delta": round(brier_delta, 6),
                "ece_delta": round(ece_delta, 6),
            }
            eligible_candidates.append(candidate)

    best = max(
        eligible_candidates,
        key=lambda candidate: (
            candidate["validation"].get("auprc") or 0.0,
            candidate["validation"].get("auroc") or 0.0,
            candidate["validation"]["f1"],
            -(candidate["validation"]["brier"]),
            -(candidate["validation"]["ece"]),
        ),
    )

    scorer = RAEGScorer(
        model=best["model"],
        calibrator=best["calibrator"],
        feature_names=list(best["feature_names"]),
        threshold=float(best["threshold"]),
    )
    return scorer, {
        "selected_candidate": best["candidate"],
        "selection_guardrail": {
            "fallback_candidate": "graph_lr",
            "min_f1_gain_over_graph_lr": 0.02,
            "max_brier_degradation": 0.01,
            "max_ece_degradation": 0.02,
            "ranking_or_calibration_path": "Allowed when F1 is non-decreasing, Brier/ECE do not degrade, and AUROC/AUPRC or Brier improves on validation.",
        },
        "validation": best["validation"],
        "raw_validation": best["raw_validation"],
        "calibrator_selected": best["calibrator_selected"],
        "candidate_reports": candidate_reports,
    }


def evaluate_raeg_scorer(scorer: RAEGScorer, x: Sequence[Sequence[float]], y: Sequence[int], split: str) -> dict:
    probabilities = scorer.predict_proba(x)
    metrics = classification_metrics(y, probabilities, scorer.threshold)
    return {
        "split": split,
        "rows": len(y),
        "positive": int(sum(y)),
        "negative": int(len(y) - sum(y)),
        "metrics": metrics,
    }


def write_raeg_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    probabilities: Sequence[float],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = []
    for row, result, probability in zip(rows, results, probabilities):
        output.append(
            {
                "sample_id": row.get("sample_id"),
                "label": row.get("label"),
                "smiles": row.get("smiles"),
                "disease": row.get("disease"),
                "raeg_probability": round(float(probability), 6),
                "raeg_prediction": 1 if probability >= threshold else 0,
                "attention_summary": raeg_attention_summary(result),
            }
        )
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def write_raeg_feature_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "label", "smiles", "disease"] + RAEG_FEATURE_NAMES
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

