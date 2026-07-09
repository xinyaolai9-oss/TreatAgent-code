#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.orchestration.eg_scorer import evaluate_scorer, read_feature_csv, write_predictions
from experiments.orchestration.raw_feature_fusion import (
    RAW_FEATURE_FUSION_FEATURES,
    default_model_name,
    default_output_dir,
    train_raw_feature_fusion_scorer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the raw feature fusion baseline.")
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--val_csv", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--split_prefix", default="drug_disjoint")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--model_name")
    return parser


def train_from_feature_tables(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    split_prefix: str = "drug_disjoint",
    output_dir: Path | None = None,
    model_name: str | None = None,
) -> dict:
    output_dir = output_dir or default_output_dir(PROJECT_ROOT, split_prefix)
    model_name = model_name or default_model_name(split_prefix)

    train_rows, train_x, train_y = read_feature_csv(train_csv, RAW_FEATURE_FUSION_FEATURES)
    val_rows, val_x, val_y = read_feature_csv(val_csv, RAW_FEATURE_FUSION_FEATURES)
    test_rows, test_x, test_y = read_feature_csv(test_csv, RAW_FEATURE_FUSION_FEATURES)

    scorer, training_info = train_raw_feature_fusion_scorer(train_x, train_y, val_x, val_y)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.pkl"
    scorer.save(model_path)

    val_prob = scorer.predict_proba(val_x)
    test_prob = scorer.predict_proba(test_x)

    write_predictions(
        output_dir / f"{model_name}_val_predictions.json",
        val_rows,
        val_prob,
        scorer.threshold,
        probability_key="raw_feature_fusion_probability",
        prediction_key="raw_feature_fusion_prediction",
    )
    write_predictions(
        output_dir / f"{model_name}_test_predictions.json",
        test_rows,
        test_prob,
        scorer.threshold,
        probability_key="raw_feature_fusion_probability",
        prediction_key="raw_feature_fusion_prediction",
    )

    report = {
        "model": "RawFeatureFusion(LogisticRegression(class_weight=balanced))",
        "model_path": str(model_path),
        "feature_names": RAW_FEATURE_FUSION_FEATURES,
        "excluded_feature_groups": [
            "EvidenceGraph support/conflict aggregation",
            "reliability and confidence statistics",
            "evidence coverage and source diversity",
            "missing evidence and agent failure counts",
        ],
        "threshold": round(scorer.threshold, 4),
        "train": {
            "rows": len(train_y),
            "positive": int(sum(train_y)),
            "negative": int(len(train_y) - sum(train_y)),
        },
        "validation_threshold_selection": training_info["validation"],
        "validation": evaluate_scorer(scorer, val_x, val_y, "val"),
        "test": evaluate_scorer(scorer, test_x, test_y, "test"),
        "inputs": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
        },
    }

    report_path = output_dir / f"{model_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = train_from_feature_tables(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        split_prefix=args.split_prefix,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )

    print(f"Wrote model to {report['model_path']}")
    print(f"Wrote report to {Path(report['model_path']).with_name(Path(report['model_path']).stem + '_report.json')}")
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

