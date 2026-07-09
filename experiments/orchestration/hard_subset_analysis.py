#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from experiments.orchestration.eg_scorer import classification_metrics
from experiments.orchestration.prediction_io import infer_threshold, parse_method_spec, prediction_table, safe_float


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hard_subset_analysis"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_arg_rows(path: Path) -> dict[str, dict[str, Any]]:
    table = prediction_table(path, "argument_probability", "argument_prediction")
    return {sample_id: value["row"] for sample_id, value in table.items()}


def factor(row: dict[str, Any], name: str) -> float:
    return safe_float((row.get("argument_factors") or {}).get(name), 0.0)


def build_subset_masks(arg_rows: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    sample_ids = list(arg_rows)
    conflict_values = [factor(row, "conflict_strength") for row in arg_rows.values()]
    admet_values = [max(factor(row, "safety_conflict"), factor(row, "admet_bbb_non_cns_noise")) for row in arg_rows.values()]
    mechanism_values = [factor(row, "mechanism_support") for row in arg_rows.values()]

    high_conflict_threshold = max(0.05, percentile(conflict_values, 0.75))
    admet_risk_threshold = max(0.05, percentile(admet_values, 0.75))
    mechanism_threshold = max(0.5, percentile(mechanism_values, 0.50))

    definitions: dict[str, Callable[[dict[str, Any]], bool]] = {
        "all": lambda row: True,
        "no_direct_indication": lambda row: factor(row, "direct_support") <= 1e-9,
        "high_conflict": lambda row: factor(row, "conflict_strength") >= high_conflict_threshold,
        "low_evidence_coverage": lambda row: factor(row, "coverage") <= 0.6,
        "low_clinical_prior": lambda row: factor(row, "clinical_feasibility") <= 0.4,
        "admet_risk": lambda row: max(factor(row, "safety_conflict"), factor(row, "admet_bbb_non_cns_noise")) >= admet_risk_threshold,
        "mechanism_only_support": lambda row: factor(row, "direct_support") <= 1e-9
        and factor(row, "mechanism_support") >= mechanism_threshold,
    }
    masks = {
        name: {sample_id for sample_id in sample_ids if predicate(arg_rows[sample_id])}
        for name, predicate in definitions.items()
    }
    return {name: ids for name, ids in masks.items() if ids}


def metrics_for_subset(method_table: dict[str, dict[str, Any]], subset_ids: set[str], threshold: float) -> dict[str, Any]:
    aligned = [method_table[sample_id] for sample_id in subset_ids if sample_id in method_table]
    y_true = [row["label"] for row in aligned]
    y_prob = [row["probability"] for row in aligned]
    if not y_true:
        return {"rows": 0}
    metrics = classification_metrics(y_true, y_prob, threshold)
    metrics["rows"] = len(y_true)
    metrics["positive"] = int(sum(y_true))
    metrics["negative"] = int(len(y_true) - sum(y_true))
    return metrics


def run_hard_subset_analysis(
    arg_predictions: Path,
    method_specs: list[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_name: str = "hard_subset_report",
) -> dict[str, Any]:
    arg_rows = load_arg_rows(arg_predictions)
    masks = build_subset_masks(arg_rows)
    method_tables = {}
    method_thresholds = {}
    for spec in method_specs:
        parsed = parse_method_spec(spec)
        method_tables[parsed["name"]] = prediction_table(
            Path(parsed["path"]),
            parsed["probability_key"],
            parsed["prediction_key"],
        )
        method_thresholds[parsed["name"]] = infer_threshold(method_tables[parsed["name"]])

    report = {
        "arg_predictions": str(arg_predictions),
        "subsets": {
            name: {
                "rows": len(ids),
                "positive": int(sum(safe_float(arg_rows[sample_id].get("label"), 0.0) for sample_id in ids)),
                "negative": len(ids) - int(sum(safe_float(arg_rows[sample_id].get("label"), 0.0) for sample_id in ids)),
            }
            for name, ids in masks.items()
        },
        "methods": {},
        "method_thresholds": method_thresholds,
    }
    for method_name, table in method_tables.items():
        report["methods"][method_name] = {
            subset_name: metrics_for_subset(table, subset_ids, method_thresholds[method_name])
            for subset_name, subset_ids in masks.items()
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{report_name}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate methods on ARG-defined hard evidence subsets.")
    parser.add_argument("--arg_predictions", type=Path, required=True)
    parser.add_argument("--prediction", action="append", default=[], help="name:path[:probability_key[:prediction_key]]")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_name", default="hard_subset_report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_hard_subset_analysis(
        arg_predictions=args.arg_predictions,
        method_specs=args.prediction,
        output_dir=args.output_dir,
        report_name=args.report_name,
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote report to {args.output_dir / f'{args.report_name}.json'}")


if __name__ == "__main__":
    main()

