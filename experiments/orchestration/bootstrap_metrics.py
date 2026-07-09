#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from experiments.orchestration.eg_scorer import classification_metrics


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_prediction_file(path: Path, probability_key: str, prediction_key: str | None = None) -> tuple[list[int], list[float], float]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    y_true = [_safe_int(row.get("label"), 0) for row in rows]
    y_prob = [_safe_float(row.get(probability_key), 0.0) for row in rows]
    if prediction_key:
        predictions = [_safe_int(row.get(prediction_key), 0) for row in rows]
        positives = [prob for prob, pred in zip(y_prob, predictions) if pred == 1]
        negatives = [prob for prob, pred in zip(y_prob, predictions) if pred == 0]
        if positives and negatives:
            threshold = (min(positives) + max(negatives)) / 2.0
        else:
            threshold = 0.5
    else:
        threshold = 0.5
    return y_true, y_prob, threshold


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(
    y_true: list[int],
    y_prob: list[float],
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 13,
) -> dict:
    rng = random.Random(seed)
    n = len(y_true)
    metric_samples: dict[str, list[float]] = {
        key: []
        for key in ["accuracy", "f1", "precision", "recall", "brier", "ece", "auroc", "auprc"]
    }
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_y = [y_true[index] for index in indices]
        sample_prob = [y_prob[index] for index in indices]
        metrics = classification_metrics(sample_y, sample_prob, threshold)
        for key in metric_samples:
            value = metrics.get(key)
            if isinstance(value, float):
                metric_samples[key].append(value)

    base_metrics = classification_metrics(y_true, y_prob, threshold)
    return {
        key: {
            "point": base_metrics.get(key),
            "mean": round(sum(values) / len(values), 6) if values else None,
            "ci95_low": round(percentile(values, 0.025), 6) if values else None,
            "ci95_high": round(percentile(values, 0.975), 6) if values else None,
        }
        for key, values in metric_samples.items()
    }


def paired_bootstrap_delta(
    first: tuple[list[int], list[float], float],
    second: tuple[list[int], list[float], float],
    n_bootstrap: int = 1000,
    seed: int = 13,
) -> dict:
    y_true, first_prob, first_threshold = first
    second_y, second_prob, second_threshold = second
    if y_true != second_y:
        raise ValueError("Prediction files must have aligned labels for paired bootstrap.")
    rng = random.Random(seed)
    n = len(y_true)
    deltas: dict[str, list[float]] = {key: [] for key in ["accuracy", "f1", "brier", "ece", "auroc", "auprc"]}
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_y = [y_true[index] for index in indices]
        first_sample = [first_prob[index] for index in indices]
        second_sample = [second_prob[index] for index in indices]
        first_metrics = classification_metrics(sample_y, first_sample, first_threshold)
        second_metrics = classification_metrics(sample_y, second_sample, second_threshold)
        for key in deltas:
            if isinstance(first_metrics.get(key), float) and isinstance(second_metrics.get(key), float):
                deltas[key].append(first_metrics[key] - second_metrics[key])
    return {
        key: {
            "mean_delta": round(sum(values) / len(values), 6) if values else None,
            "ci95_low": round(percentile(values, 0.025), 6) if values else None,
            "ci95_high": round(percentile(values, 0.975), 6) if values else None,
            "p_delta_le_0": round(sum(1 for value in values if value <= 0) / len(values), 6) if values else None,
        }
        for key, values in deltas.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for TreatAgent prediction JSON files.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--probability_key", required=True)
    parser.add_argument("--prediction_key")
    parser.add_argument("--compare_predictions", type=Path)
    parser.add_argument("--compare_probability_key")
    parser.add_argument("--compare_prediction_key")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    primary = read_prediction_file(args.predictions, args.probability_key, args.prediction_key)
    report = {
        "predictions": str(args.predictions),
        "probability_key": args.probability_key,
        "n_bootstrap": args.n_bootstrap,
        "metrics_ci": bootstrap_ci(*primary, n_bootstrap=args.n_bootstrap, seed=args.seed),
    }
    if args.compare_predictions:
        if not args.compare_probability_key:
            raise ValueError("--compare_probability_key is required when --compare_predictions is used.")
        comparison = read_prediction_file(
            args.compare_predictions,
            args.compare_probability_key,
            args.compare_prediction_key,
        )
        report["comparison"] = {
            "compare_predictions": str(args.compare_predictions),
            "compare_probability_key": args.compare_probability_key,
            "delta_primary_minus_compare": paired_bootstrap_delta(
                primary,
                comparison,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            ),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

