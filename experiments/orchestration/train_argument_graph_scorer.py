#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from treatagent.orchestration.argument_graph_scorer import (
    ARG_FACTOR_NAMES,
    argument_factors_from_result,
)
from experiments.orchestration.eg_scorer import choose_threshold, classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "argument_graph_scorer"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def read_argument_results(path: Path) -> tuple[list[dict[str, Any]], list[float], list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    scores = []
    labels = []
    for result in payload.get("results") or []:
        if result.get("label") in (None, ""):
            continue
        row = argument_factors_from_result(result)
        rows.append(row)
        scores.append(float(row["factors"]["raw_argument_score"]))
        labels.append(int(float(result.get("label"))))
    return rows, scores, labels


@dataclass
class ArgumentGraphScorer:
    threshold: float

    def predict_proba(self, scores: Sequence[float]) -> list[float]:
        return [_clip01(score) for score in scores]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"threshold": self.threshold, "calibration": "none"}, handle)


def train_argument_graph_scorer(
    train_scores: Sequence[float],
    train_y: Sequence[int],
    val_scores: Sequence[float],
    val_y: Sequence[int],
) -> tuple[ArgumentGraphScorer, dict[str, Any]]:
    val_prob = [_clip01(score) for score in val_scores]
    threshold, metrics = choose_threshold(val_y, val_prob)
    scorer = ArgumentGraphScorer(threshold=float(threshold))
    return scorer, {
        "selected_calibration": "none",
        "validation": metrics,
        "candidate_reports": [
            {
                "calibration": "none",
                "threshold": round(threshold, 4),
                "validation": metrics,
            }
        ],
    }


def evaluate_argument_graph_scorer(
    scorer: ArgumentGraphScorer,
    scores: Sequence[float],
    labels: Sequence[int],
    split: str,
) -> dict[str, Any]:
    probabilities = scorer.predict_proba(scores)
    return {
        "split": split,
        "rows": len(labels),
        "positive": int(sum(labels)),
        "negative": int(len(labels) - sum(labels)),
        "metrics": classification_metrics(labels, probabilities, scorer.threshold),
    }


def write_argument_predictions(
    path: Path,
    rows: Sequence[dict[str, Any]],
    probabilities: Sequence[float],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = []
    for row, probability in zip(rows, probabilities):
        output.append(
            {
                "sample_id": row.get("sample_id"),
                "label": row.get("label"),
                "smiles": row.get("smiles"),
                "disease": row.get("disease"),
                "argument_probability": round(float(probability), 6),
                "argument_prediction": 1 if probability >= threshold else 0,
                "argument_factors": row.get("factors"),
                "top_support_arguments": row.get("top_support_arguments"),
                "top_conflict_arguments": row.get("top_conflict_arguments"),
            }
        )
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate TreatAgent-ARG from TreatAgent result JSON files.")
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--val_json", type=Path, required=True)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", default="treatagent_arg")
    return parser


def train_from_result_jsons(
    train_json: Path,
    val_json: Path,
    test_json: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = "treatagent_arg",
) -> dict:
    train_rows, train_scores, train_y = read_argument_results(train_json)
    val_rows, val_scores, val_y = read_argument_results(val_json)
    test_rows, test_scores, test_y = read_argument_results(test_json)
    scorer, training_info = train_argument_graph_scorer(train_scores, train_y, val_scores, val_y)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.pkl"
    scorer.save(model_path)
    val_prob = scorer.predict_proba(val_scores)
    test_prob = scorer.predict_proba(test_scores)
    write_argument_predictions(output_dir / f"{model_name}_val_predictions.json", val_rows, val_prob, scorer.threshold)
    write_argument_predictions(output_dir / f"{model_name}_test_predictions.json", test_rows, test_prob, scorer.threshold)

    report = {
        "model": "TreatAgent-ARG reliability-aware argument graph reasoner",
        "reasoning": "Fixed support/conflict argument aggregation; no learned classifier or learned calibration. The validation set is used only to select the decision threshold.",
        "factor_names": ARG_FACTOR_NAMES,
        "model_path": str(model_path),
        "selected_calibration": "none",
        "threshold": round(scorer.threshold, 4),
        "train": {
            "rows": len(train_y),
            "positive": int(sum(train_y)),
            "negative": int(len(train_y) - sum(train_y)),
        },
        "validation_threshold_selection": training_info["validation"],
        "calibration_candidates": training_info["candidate_reports"],
        "validation": evaluate_argument_graph_scorer(scorer, val_scores, val_y, "val"),
        "test": evaluate_argument_graph_scorer(scorer, test_scores, test_y, "test"),
        "inputs": {
            "train_json": str(train_json),
            "val_json": str(val_json),
            "test_json": str(test_json),
        },
    }
    report_path = output_dir / f"{model_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = train_from_result_jsons(
        train_json=args.train_json,
        val_json=args.val_json,
        test_json=args.test_json,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )
    print(f"Wrote model to {report['model_path']}")
    print(f"Wrote report to {Path(args.output_dir) / f'{args.model_name}_report.json'}")
    print(
        json.dumps(
            {
                "threshold": report["threshold"],
                "calibration": report["selected_calibration"],
                "val": report["validation"]["metrics"],
                "test": report["test"]["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

