#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.orchestration.eg_scorer import classification_metrics
from experiments.orchestration.prediction_io import infer_threshold, parse_method_spec, prediction_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "calibration"


def reliability_bins(table: dict, n_bins: int = 10) -> list[dict]:
    rows = list(table.values())
    bins = []
    for bin_index in range(n_bins):
        low = bin_index / n_bins
        high = (bin_index + 1) / n_bins
        selected = [
            row
            for row in rows
            if (low <= row["probability"] < high) or (bin_index == n_bins - 1 and row["probability"] <= high)
        ]
        if not selected:
            bins.append({"bin": bin_index, "low": low, "high": high, "count": 0})
            continue
        avg_confidence = sum(row["probability"] for row in selected) / len(selected)
        accuracy = sum(1 for row in selected if int(row["probability"] >= 0.5) == row["label"]) / len(selected)
        positive_rate = sum(row["label"] for row in selected) / len(selected)
        bins.append(
            {
                "bin": bin_index,
                "low": round(low, 3),
                "high": round(high, 3),
                "count": len(selected),
                "avg_confidence": round(avg_confidence, 6),
                "accuracy": round(accuracy, 6),
                "positive_rate": round(positive_rate, 6),
            }
        )
    return bins


def threshold_sweep(table: dict) -> list[dict]:
    rows = list(table.values())
    y_true = [row["label"] for row in rows]
    y_prob = [row["probability"] for row in rows]
    output = []
    for index in range(0, 101):
        threshold = index / 100
        metrics = classification_metrics(y_true, y_prob, threshold)
        output.append(
            {
                "threshold": round(threshold, 2),
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_calibration_analysis(
    method_specs: list[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_name: str = "calibration_report",
    n_bins: int = 10,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"n_bins": n_bins, "methods": {}}
    for spec in method_specs:
        parsed = parse_method_spec(spec)
        table = prediction_table(Path(parsed["path"]), parsed["probability_key"], parsed["prediction_key"])
        rows = list(table.values())
        threshold = infer_threshold(table)
        metrics = classification_metrics(
            [row["label"] for row in rows],
            [row["probability"] for row in rows],
            threshold,
        )
        bins = reliability_bins(table, n_bins=n_bins)
        sweep = threshold_sweep(table)
        method_dir = output_dir / parsed["name"]
        write_csv(method_dir / f"{report_name}_{parsed['name']}_reliability_bins.csv", bins)
        write_csv(method_dir / f"{report_name}_{parsed['name']}_threshold_sweep.csv", sweep)
        report["methods"][parsed["name"]] = {
            "input": parsed,
            "selected_threshold": round(threshold, 6),
            "metrics": metrics,
            "reliability_bins_csv": str(method_dir / f"{report_name}_{parsed['name']}_reliability_bins.csv"),
            "threshold_sweep_csv": str(method_dir / f"{report_name}_{parsed['name']}_threshold_sweep.csv"),
        }
    report_path = output_dir / f"{report_name}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate calibration bins and threshold sweep tables.")
    parser.add_argument("--prediction", action="append", required=True, help="name:path[:probability_key[:prediction_key]]")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_name", default="calibration_report")
    parser.add_argument("--n_bins", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_calibration_analysis(
        method_specs=args.prediction,
        output_dir=args.output_dir,
        report_name=args.report_name,
        n_bins=args.n_bins,
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote report to {args.output_dir / f'{args.report_name}.json'}")


if __name__ == "__main__":
    main()

