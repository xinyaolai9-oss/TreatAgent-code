#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.orchestration.eg_scorer import EGScorer, evaluate_scorer, read_feature_csv, write_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "eg_scorer" / "applied"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a trained EG scorer to an existing feature CSV.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", default="eg_scorer_applied")
    parser.add_argument("--split", default="test")
    return parser


def apply_scorer(
    model_path: Path,
    input_csv: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_name: str = "eg_scorer_applied",
    split: str = "test",
) -> dict:
    scorer = EGScorer.load(model_path)
    rows, x, y = read_feature_csv(input_csv, scorer.feature_names)
    probabilities = scorer.predict_proba(x)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        output_dir / f"{output_name}_predictions.json",
        rows,
        probabilities,
        scorer.threshold,
        probability_key="applied_eg_probability",
        prediction_key="applied_eg_prediction",
    )
    report = {
        "model_path": str(model_path),
        "input_csv": str(input_csv),
        "feature_names": scorer.feature_names,
        "threshold": round(scorer.threshold, 4),
        "evaluation": evaluate_scorer(scorer, x, y, split),
    }
    report_path = output_dir / f"{output_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = apply_scorer(
        model_path=args.model_path,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_name=args.output_name,
        split=args.split,
    )
    print(f"Wrote report to {args.output_dir / f'{args.output_name}_report.json'}")
    print(json.dumps(report["evaluation"], indent=2))


if __name__ == "__main__":
    main()

