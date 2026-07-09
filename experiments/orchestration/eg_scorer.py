from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from treatagent.orchestration.features import FEATURE_NAMES


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_feature_csv(path: Path, feature_names: Sequence[str] = FEATURE_NAMES) -> tuple[list[dict], list[list[float]], list[int]]:
    rows: list[dict] = []
    x: list[list[float]] = []
    y: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("label") in (None, ""):
                continue
            rows.append(row)
            x.append([_safe_float(row.get(name), 0.0) for name in feature_names])
            y.append(_safe_int(row.get("label"), 0))
    return rows, x, y


def expected_calibration_error(y_true: Sequence[int], y_prob: Sequence[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0
    total = len(y_true)
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = [lower <= p <= upper for p in y_prob]
        else:
            mask = [lower <= p < upper for p in y_prob]
        count = sum(mask)
        if count == 0:
            continue
        bin_true = [label for label, keep in zip(y_true, mask) if keep]
        bin_prob = [prob for prob, keep in zip(y_prob, mask) if keep]
        accuracy = sum(bin_true) / count
        confidence = sum(bin_prob) / count
        ece += (count / total) * abs(accuracy - confidence)
    return ece


def classification_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> dict:
    y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
    metrics = {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    if len(set(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()}


def choose_threshold(y_true: Sequence[int], y_prob: Sequence[float]) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = classification_metrics(y_true, y_prob, best_threshold)
    for step in range(1, 100):
        threshold = step / 100
        metrics = classification_metrics(y_true, y_prob, threshold)
        if (metrics["f1"], metrics["accuracy"]) > (best_metrics["f1"], best_metrics["accuracy"]):
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


@dataclass
class EGScorer:
    model: Pipeline
    feature_names: List[str]
    threshold: float

    def predict_proba(self, x: Iterable[Sequence[float]]) -> list[float]:
        probabilities = self.model.predict_proba(list(x))[:, 1]
        return [float(value) for value in probabilities]

    def predict(self, x: Iterable[Sequence[float]]) -> list[int]:
        return [1 if prob >= self.threshold else 0 for prob in self.predict_proba(x)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "threshold": self.threshold,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "EGScorer":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(
            model=payload["model"],
            feature_names=list(payload["feature_names"]),
            threshold=float(payload["threshold"]),
        )


def train_eg_scorer(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[int],
    val_x: Sequence[Sequence[float]],
    val_y: Sequence[int],
    feature_names: Sequence[str] = FEATURE_NAMES,
    random_state: int = 13,
) -> tuple[EGScorer, dict]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    val_prob = [float(value) for value in model.predict_proba(val_x)[:, 1]]
    threshold, val_metrics = choose_threshold(val_y, val_prob)
    scorer = EGScorer(model=model, feature_names=list(feature_names), threshold=threshold)
    return scorer, {"validation": val_metrics}


def evaluate_scorer(scorer: EGScorer, x: Sequence[Sequence[float]], y: Sequence[int], split: str) -> dict:
    probabilities = scorer.predict_proba(x)
    metrics = classification_metrics(y, probabilities, scorer.threshold)
    return {
        "split": split,
        "rows": len(y),
        "positive": int(sum(y)),
        "negative": int(len(y) - sum(y)),
        "metrics": metrics,
    }


def write_predictions(
    path: Path,
    rows: list[dict],
    probabilities: Sequence[float],
    threshold: float,
    probability_key: str = "eg_probability",
    prediction_key: str = "eg_prediction",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row, probability in zip(rows, probabilities):
        enriched = dict(row)
        enriched[probability_key] = round(float(probability), 6)
        enriched[prediction_key] = 1 if probability >= threshold else 0
        output_rows.append(enriched)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output_rows, handle, indent=2, ensure_ascii=False)

