#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import List

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(values: List[float], name: str) -> None:
    if not values:
        print(f"{name}: no data")
        return

    sorted_values = sorted(values)
    print(f"{name}:")
    print(f"  count: {len(values)}")
    print(f"  min:   {min(values):.4f}")
    print(f"  p25:   {percentile(sorted_values, 0.25):.4f}")
    print(f"  p50:   {median(values):.4f}")
    print(f"  mean:  {mean(values):.4f}")
    print(f"  p75:   {percentile(sorted_values, 0.75):.4f}")
    print(f"  max:   {max(values):.4f}")
    print(f"  top repeated values: {Counter(round(v, 4) for v in values).most_common(10)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze threshold sensitivity for TreatAgent multiagent results.")
    parser.add_argument("--results_path", type=str, default="results/gpt-4o/results_multiagent.json")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    args = parser.parse_args()

    results_path = Path(args.results_path)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])

    calibrated = [float(item["calibrated_probability"]) for item in results if "calibrated_probability" in item]
    raw_scores = [float(item["raw_score"]) for item in results if "raw_score" in item]
    labels = [int(item["label"]) for item in results if "calibrated_probability" in item and "label" in item]

    print(f"Results file: {results_path}")
    print(f"Samples with calibrated_probability: {len(calibrated)}")
    print()

    summarize(raw_scores, "raw_score")
    print()
    summarize(calibrated, "calibrated_probability")
    print()

    print("threshold analysis:")
    best_f1 = None
    best_threshold = None
    for threshold in args.thresholds:
        predictions = [1 if value >= threshold else 0 for value in calibrated]
        predicted_positive = sum(predictions)
        ratio = (predicted_positive / len(calibrated)) if calibrated else 0.0
        accuracy = accuracy_score(labels, predictions) if labels else 0.0
        precision = precision_score(labels, predictions, zero_division=0) if labels else 0.0
        recall = recall_score(labels, predictions, zero_division=0) if labels else 0.0
        f1 = f1_score(labels, predictions, zero_division=0) if labels else 0.0

        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        print(
            f"  threshold={threshold:.2f} -> positive={predicted_positive}/{len(calibrated)} "
            f"({ratio:.4%}), accuracy={accuracy:.4f}, precision={precision:.4f}, "
            f"recall={recall:.4f}, f1={f1:.4f}"
        )

    if best_threshold is not None:
        print()
        print(f"best f1 threshold: {best_threshold:.2f} (f1={best_f1:.4f})")


if __name__ == "__main__":
    main()
