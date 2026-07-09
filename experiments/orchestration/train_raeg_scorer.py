#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.orchestration.raeg_scorer import (
    RAEG_FEATURE_NAMES,
    evaluate_raeg_scorer,
    matrix_from_rows,
    read_raeg_result_features,
    read_result_json,
    train_raeg_scorer,
    write_raeg_feature_csv,
    write_raeg_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "raeg_scorer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate TreatAgent-EG-RAEG from result JSON files with typed evidence."
    )
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--val_json", type=Path, required=True)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", default="treatagent_eg_raeg")
    return parser


def train_from_result_jsons(
    train_json: Path,
    val_json: Path,
    test_json: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = "treatagent_eg_raeg",
) -> dict:
    train_rows, train_x, train_y = read_raeg_result_features(train_json)
    val_rows, val_x, val_y = read_raeg_result_features(val_json)
    test_rows, test_x, test_y = read_raeg_result_features(test_json)
    _, val_results = read_result_json(val_json)
    _, test_results = read_result_json(test_json)

    scorer, training_info = train_raeg_scorer(train_x, train_y, val_x, val_y)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.pkl"
    scorer.save(model_path)

    selected_val_x = matrix_from_rows(val_rows, scorer.feature_names)
    selected_test_x = matrix_from_rows(test_rows, scorer.feature_names)
    val_prob = scorer.predict_proba(selected_val_x)
    test_prob = scorer.predict_proba(selected_test_x)

    write_raeg_feature_csv(output_dir / f"{model_name}_train_features.csv", train_rows)
    write_raeg_feature_csv(output_dir / f"{model_name}_val_features.csv", val_rows)
    write_raeg_feature_csv(output_dir / f"{model_name}_test_features.csv", test_rows)
    write_raeg_predictions(
        output_dir / f"{model_name}_val_predictions.json",
        val_rows,
        val_results,
        val_prob,
        scorer.threshold,
    )
    write_raeg_predictions(
        output_dir / f"{model_name}_test_predictions.json",
        test_rows,
        test_results,
        test_prob,
        scorer.threshold,
    )

    report = {
        "model": "Reliability-gated support-conflict EvidenceGraph scorer",
        "base_estimator": "Validation-selected candidate among Graph-LR, RAEG-LR, RAEG+Graph-LR, and RAEG+Graph-MLP",
        "calibration": "Validation-selected Platt scaling; disabled when validation Brier/ECE does not improve",
        "model_path": str(model_path),
        "all_feature_names": RAEG_FEATURE_NAMES,
        "selected_feature_names": scorer.feature_names,
        "selected_candidate": training_info["selected_candidate"],
        "selection_guardrail": training_info["selection_guardrail"],
        "threshold": round(scorer.threshold, 4),
        "train": {
            "rows": len(train_y),
            "positive": int(sum(train_y)),
            "negative": int(len(train_y) - sum(train_y)),
        },
        "validation_threshold_selection": training_info["validation"],
        "raw_validation_before_calibration": training_info["raw_validation"],
        "calibrator_selected": training_info["calibrator_selected"],
        "candidate_reports": training_info["candidate_reports"],
        "validation": evaluate_raeg_scorer(scorer, selected_val_x, val_y, "val"),
        "test": evaluate_raeg_scorer(scorer, selected_test_x, test_y, "test"),
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
                "val": report["validation"]["metrics"],
                "test": report["test"]["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

