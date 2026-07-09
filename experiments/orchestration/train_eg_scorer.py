#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.orchestration.eg_scorer import (
    evaluate_scorer,
    read_feature_csv,
    train_eg_scorer,
    write_predictions,
)
from treatagent.orchestration.features import FEATURE_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "eg_scorer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate TreatAgent-EG scorer from graph feature tables.")
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--val_csv", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", default="treatagent_eg_lr")
    return parser


def train_from_feature_tables(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = "treatagent_eg_lr",
) -> dict:
    train_rows, train_x, train_y = read_feature_csv(train_csv)
    val_rows, val_x, val_y = read_feature_csv(val_csv)
    test_rows, test_x, test_y = read_feature_csv(test_csv)

    scorer, training_info = train_eg_scorer(train_x, train_y, val_x, val_y)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.pkl"
    scorer.save(model_path)

    val_prob = scorer.predict_proba(val_x)
    test_prob = scorer.predict_proba(test_x)

    write_predictions(output_dir / f"{model_name}_val_predictions.json", val_rows, val_prob, scorer.threshold)
    write_predictions(output_dir / f"{model_name}_test_predictions.json", test_rows, test_prob, scorer.threshold)

    report = {
        "model": "LogisticRegression(class_weight=balanced)",
        "model_path": str(model_path),
        "feature_names": FEATURE_NAMES,
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

