#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.orchestration.build_dropout_feature_table import mask_result_experts
from experiments.orchestration.build_graph_feature_table import ID_COLUMNS, read_results
from experiments.orchestration.eg_scorer import evaluate_scorer, train_eg_scorer, write_predictions
from treatagent.orchestration.features import CORE_EXPERTS, FEATURE_NAMES, feature_row_from_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "source_ablation"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def rows_from_result_json(path: Path, masked_experts: list[str]) -> list[dict]:
    payload, results = read_results(path)
    rows = []
    for result in results:
        if not result.get("evidence_graph"):
            continue
        masked = mask_result_experts(result, masked_experts)
        row = feature_row_from_result(masked)
        row["source_file"] = str(path)
        row["method"] = payload.get("method") or result.get("method")
        row["backbone"] = path.parent.name
        rows.append(row)
    return rows


def xy_from_rows(rows: list[dict]) -> tuple[list[list[float]], list[int]]:
    x = [[_safe_float(row.get(name), 0.0) for name in FEATURE_NAMES] for row in rows if row.get("label") not in (None, "")]
    y = [_safe_int(row.get("label"), 0) for row in rows if row.get("label") not in (None, "")]
    return x, y


def write_feature_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ID_COLUMNS + ["masked_experts"] + FEATURE_NAMES
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_source_ablation(
    train_json: Path,
    val_json: Path,
    test_json: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_prefix: str = "eg_lr_source_ablation",
    experts: list[str] | None = None,
) -> dict:
    experts = experts or CORE_EXPERTS
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = [("full", [])] + [(f"wo_{expert}", [expert]) for expert in experts]
    report = {
        "model": "EG-LR retrained under source ablation",
        "feature_names": FEATURE_NAMES,
        "inputs": {
            "train_json": str(train_json),
            "val_json": str(val_json),
            "test_json": str(test_json),
        },
        "settings": {},
    }

    for setting_name, masked_experts in settings:
        setting_dir = output_dir / setting_name
        train_rows = rows_from_result_json(train_json, masked_experts)
        val_rows = rows_from_result_json(val_json, masked_experts)
        test_rows = rows_from_result_json(test_json, masked_experts)
        for row in train_rows + val_rows + test_rows:
            row["masked_experts"] = ",".join(masked_experts)
        write_feature_csv(setting_dir / f"{model_prefix}_{setting_name}_train_features.csv", train_rows)
        write_feature_csv(setting_dir / f"{model_prefix}_{setting_name}_val_features.csv", val_rows)
        write_feature_csv(setting_dir / f"{model_prefix}_{setting_name}_test_features.csv", test_rows)

        train_x, train_y = xy_from_rows(train_rows)
        val_x, val_y = xy_from_rows(val_rows)
        test_x, test_y = xy_from_rows(test_rows)
        scorer, training_info = train_eg_scorer(train_x, train_y, val_x, val_y)
        model_path = setting_dir / f"{model_prefix}_{setting_name}.pkl"
        scorer.save(model_path)
        test_prob = scorer.predict_proba(test_x)
        write_predictions(
            setting_dir / f"{model_prefix}_{setting_name}_test_predictions.json",
            test_rows,
            test_prob,
            scorer.threshold,
            probability_key="ablation_probability",
            prediction_key="ablation_prediction",
        )
        report["settings"][setting_name] = {
            "masked_experts": masked_experts,
            "model_path": str(model_path),
            "threshold": round(scorer.threshold, 4),
            "train": {
                "rows": len(train_y),
                "positive": int(sum(train_y)),
                "negative": int(len(train_y) - sum(train_y)),
            },
            "validation_threshold_selection": training_info["validation"],
            "validation": evaluate_scorer(scorer, val_x, val_y, "val"),
            "test": evaluate_scorer(scorer, test_x, test_y, "test"),
        }

    full_metrics = report["settings"]["full"]["test"]["metrics"]
    for setting_name, payload in report["settings"].items():
        metrics = payload["test"]["metrics"]
        payload["delta_vs_full"] = {
            key: round(metrics[key] - full_metrics[key], 6)
            for key in ["accuracy", "f1", "precision", "recall", "brier", "ece", "auroc", "auprc"]
            if metrics.get(key) is not None and full_metrics.get(key) is not None
        }

    report_path = output_dir / f"{model_prefix}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run source ablation by masking each evidence expert and retraining EG-LR.")
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--val_json", type=Path, required=True)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_prefix", default="eg_lr_source_ablation")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_source_ablation(
        train_json=args.train_json,
        val_json=args.val_json,
        test_json=args.test_json,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
    )
    print(json.dumps({name: value["test"]["metrics"] for name, value in report["settings"].items()}, indent=2))
    print(f"Wrote report to {args.output_dir / f'{args.model_prefix}_report.json'}")


if __name__ == "__main__":
    main()

