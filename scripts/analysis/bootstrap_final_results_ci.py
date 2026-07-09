#!/usr/bin/env python3
"""Bootstrap confidence intervals for final TreatAgent results.

The script uses frozen per-sample predictions in results/final_results/baselines
and computes non-parametric 95% confidence intervals over test-set samples.
No model/API calls are made.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "final_results"
BOOTSTRAP_N = int(os.getenv("BOOTSTRAP_N", "1000"))
SEED = int(os.getenv("BOOTSTRAP_SEED", "13"))
ALPHA = 0.05


RUNS = [
    ("Drug-disjoint", "Direct", "baselines/results_direct_dd_test.json"),
    ("Drug-disjoint", "CoT", "baselines/results_cot_dd_test.json"),
    ("Drug-disjoint", "RAG", "baselines/results_rag_dd_test.json"),
    ("Drug-disjoint", "TreatAgent", "baselines/results_multiagent_dd_test.json"),
    ("Temporal-submit", "Direct", "baselines/results_direct_ts_test.json"),
    ("Temporal-submit", "CoT", "baselines/results_cot_ts_test.json"),
    ("Temporal-submit", "RAG", "baselines/results_rag_ts_test.json"),
    ("Temporal-submit", "TreatAgent", "baselines/results_multiagent_ts_test.json"),
]


METRICS_FOR_FIGURE3 = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auroc",
    "auprc",
    "missed_positive_rate",
]


def _score_from_row(row: dict) -> float:
    """Return the score used for ranking metrics.

    TreatAgent stores the constrained LLM judge probability separately. When it
    is missing, fall back to prediction_score, matching the final result audit.
    Baselines usually store binary prediction_score.
    """
    value = row.get("llm_judge_probability")
    if value is None:
        value = row.get("judge_probability")
    if value is None:
        value = row.get("prediction_score")
    if value is None:
        value = row.get("prediction_binary", 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(row.get("prediction_binary", 0))


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload["results"]
    y_true = np.array([int(r["label"]) for r in rows], dtype=int)
    y_pred = np.array([int(r.get("prediction_binary", 0)) for r in rows], dtype=int)
    y_score = np.array([_score_from_row(r) for r in rows], dtype=float)
    return y_true, y_pred, y_score


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return math.nan
    return float(average_precision_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": recall,
        "auroc": safe_auroc(y_true, y_score),
        "auprc": safe_auprc(y_true, y_score),
        "missed_positive_rate": float(1.0 - recall),
    }


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return math.nan, math.nan
    lo, hi = np.percentile(finite, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(lo), float(hi)


def bootstrap_one(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
                  rng: np.random.Generator) -> dict[str, tuple[float, float, float, float]]:
    n = len(y_true)
    observed = compute_metrics(y_true, y_pred, y_score)
    samples = {metric: np.empty(BOOTSTRAP_N, dtype=float) for metric in METRICS_FOR_FIGURE3}

    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        vals = compute_metrics(y_true[idx], y_pred[idx], y_score[idx])
        for metric in METRICS_FOR_FIGURE3:
            samples[metric][i] = vals[metric]

    summary = {}
    for metric in METRICS_FOR_FIGURE3:
        lo, hi = percentile_ci(samples[metric])
        summary[metric] = (observed[metric], float(np.nanmean(samples[metric])), lo, hi)
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict], path: Path) -> None:
    selected = [r for r in rows if r["metric"] in {"f1", "recall", "precision", "missed_positive_rate"}]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Bootstrap 95% confidence intervals\n\n")
        handle.write(f"Bootstrap resamples: {BOOTSTRAP_N}; seed: {SEED}; interval: percentile 95% CI.\n\n")
        handle.write("| Split | Method | Metric | Observed | 95% CI |\n")
        handle.write("|---|---|---|---:|---:|\n")
        for r in selected:
            handle.write(
                f"| {r['split']} | {r['method']} | {r['metric']} | "
                f"{float(r['observed']):.3f} | "
                f"[{float(r['ci_lower']):.3f}, {float(r['ci_upper']):.3f}] |\n"
            )


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for split, method, rel_path in RUNS:
        y_true, y_pred, y_score = load_arrays(RESULT_DIR / rel_path)
        summary = bootstrap_one(y_true, y_pred, y_score, rng)
        for metric, (observed, boot_mean, ci_lower, ci_upper) in summary.items():
            rows.append({
                "split": split,
                "method": method,
                "metric": metric,
                "observed": round(observed, 6),
                "bootstrap_mean": round(boot_mean, 6),
                "ci_lower": round(ci_lower, 6),
                "ci_upper": round(ci_upper, 6),
                "n": len(y_true),
                "n_bootstrap": BOOTSTRAP_N,
                "seed": SEED,
                "source_json": rel_path,
            })

    write_csv(rows, RESULT_DIR / "bootstrap_ci.csv")
    write_md(rows, RESULT_DIR / "bootstrap_ci.md")
    print(f"Wrote {RESULT_DIR / 'bootstrap_ci.csv'}")
    print(f"Wrote {RESULT_DIR / 'bootstrap_ci.md'}")


if __name__ == "__main__":
    main()
