#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from experiments.orchestration.argument_ablation import score_from_factors
from treatagent.orchestration.argument_graph_scorer import argument_factors_from_result
from experiments.orchestration.build_dropout_feature_table import mask_result_experts
from experiments.orchestration.eg_scorer import choose_threshold, classification_metrics
from treatagent.orchestration.features import CORE_EXPERTS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "argument_source_ablation"


def read_rows(
    path: Path,
    *,
    masked_experts: Sequence[str],
) -> tuple[list[dict[str, Any]], list[float], list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    scores = []
    labels = []
    for result in payload.get("results") or []:
        if result.get("label") in (None, ""):
            continue
        masked = mask_result_experts(result, masked_experts)
        row = argument_factors_from_result(masked)
        score = score_from_factors(row["factors"], variant="full")
        row["source_ablation_score"] = round(score, 6)
        row["masked_experts"] = list(masked_experts)
        rows.append(row)
        scores.append(score)
        labels.append(int(float(result.get("label"))))
    return rows, scores, labels


def evaluate_scores(labels: Sequence[int], scores: Sequence[float], threshold: float, split: str) -> dict[str, Any]:
    return {
        "split": split,
        "rows": len(labels),
        "positive": int(sum(labels)),
        "negative": int(len(labels) - sum(labels)),
        "metrics": classification_metrics(labels, scores, threshold),
    }


def write_predictions(path: Path, rows: Sequence[dict[str, Any]], scores: Sequence[float], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = []
    for row, score in zip(rows, scores):
        output.append(
            {
                "sample_id": row.get("sample_id"),
                "label": row.get("label"),
                "smiles": row.get("smiles"),
                "disease": row.get("disease"),
                "masked_experts": row.get("masked_experts"),
                "source_ablation_score": round(float(score), 6),
                "source_ablation_prediction": 1 if score >= threshold else 0,
                "argument_factors": row.get("factors"),
                "top_support_arguments": row.get("top_support_arguments"),
                "top_conflict_arguments": row.get("top_conflict_arguments"),
            }
        )
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def run_argument_source_ablation(
    train_json: Path,
    val_json: Path,
    test_json: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_name: str = "argument_source_ablation_report",
    experts: Sequence[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    experts = list(experts or CORE_EXPERTS)
    settings = [("full", [])] + [(f"wo_{expert}", [expert]) for expert in experts]
    report = {
        "model": "TreatAgent-ARG source/expert ablation",
        "reasoning": "Each setting masks one expert source, recomputes fixed no-classifier ARG scores, selects threshold on validation, and evaluates on test.",
        "inputs": {
            "train_json": str(train_json),
            "val_json": str(val_json),
            "test_json": str(test_json),
        },
        "settings": {},
    }

    for setting_name, masked_experts in settings:
        setting_dir = output_dir / setting_name
        train_rows, train_scores, train_y = read_rows(train_json, masked_experts=masked_experts)
        val_rows, val_scores, val_y = read_rows(val_json, masked_experts=masked_experts)
        test_rows, test_scores, test_y = read_rows(test_json, masked_experts=masked_experts)
        threshold, validation_metrics = choose_threshold(val_y, val_scores)

        write_predictions(
            setting_dir / f"{report_name}_{setting_name}_test_predictions.json",
            test_rows,
            test_scores,
            threshold,
        )
        report["settings"][setting_name] = {
            "masked_experts": masked_experts,
            "threshold": round(float(threshold), 4),
            "train": {
                "rows": len(train_y),
                "positive": int(sum(train_y)),
                "negative": int(len(train_y) - sum(train_y)),
            },
            "validation_threshold_selection": validation_metrics,
            "validation": evaluate_scores(val_y, val_scores, threshold, "val"),
            "test": evaluate_scores(test_y, test_scores, threshold, "test"),
        }

    full_metrics = report["settings"]["full"]["test"]["metrics"]
    for setting in report["settings"].values():
        metrics = setting["test"]["metrics"]
        setting["delta_vs_full"] = {
            key: round(metrics[key] - full_metrics[key], 6)
            for key in ["accuracy", "f1", "precision", "recall", "brier", "ece", "auroc", "auprc"]
            if metrics.get(key) is not None and full_metrics.get(key) is not None
        }

    report_path = output_dir / f"{report_name}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run source/expert ablation for the no-classifier TreatAgent-ARG scorer.")
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--val_json", type=Path, required=True)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_name", default="argument_source_ablation_report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_argument_source_ablation(
        train_json=args.train_json,
        val_json=args.val_json,
        test_json=args.test_json,
        output_dir=args.output_dir,
        report_name=args.report_name,
    )
    print(json.dumps({name: value["test"]["metrics"] for name, value in report["settings"].items()}, indent=2))
    print(f"Wrote report to {args.output_dir / f'{args.report_name}.json'}")


if __name__ == "__main__":
    main()

