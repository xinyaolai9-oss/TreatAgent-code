#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.orchestration.bootstrap_metrics import bootstrap_ci, paired_bootstrap_delta
from experiments.orchestration.prediction_io import infer_threshold, parse_method_spec, prediction_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "bootstrap"


def table_to_tuple(table: dict) -> tuple[list[int], list[float], float]:
    ordered_ids = sorted(table)
    return (
        [table[sample_id]["label"] for sample_id in ordered_ids],
        [table[sample_id]["probability"] for sample_id in ordered_ids],
        infer_threshold(table),
    )


def aligned_tuples(primary: dict, other: dict) -> tuple[tuple[list[int], list[float], float], tuple[list[int], list[float], float]]:
    common = sorted(set(primary) & set(other))
    return (
        (
            [primary[sample_id]["label"] for sample_id in common],
            [primary[sample_id]["probability"] for sample_id in common],
            infer_threshold(primary),
        ),
        (
            [other[sample_id]["label"] for sample_id in common],
            [other[sample_id]["probability"] for sample_id in common],
            infer_threshold(other),
        ),
    )


def run_bootstrap_experiments(
    method_specs: list[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_name: str = "bootstrap_report",
    primary_method: str = "TreatAgent-ARG",
    n_bootstrap: int = 1000,
    seed: int = 13,
) -> dict:
    method_tables = {}
    method_inputs = {}
    for spec in method_specs:
        parsed = parse_method_spec(spec)
        method_inputs[parsed["name"]] = parsed
        method_tables[parsed["name"]] = prediction_table(
            Path(parsed["path"]),
            parsed["probability_key"],
            parsed["prediction_key"],
        )

    report = {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "inputs": method_inputs,
        "metrics_ci": {},
        "paired_delta_vs_primary": {},
    }
    for method_name, table in method_tables.items():
        report["metrics_ci"][method_name] = bootstrap_ci(
            *table_to_tuple(table),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    if primary_method in method_tables:
        primary_table = method_tables[primary_method]
        for method_name, table in method_tables.items():
            if method_name == primary_method:
                continue
            primary_tuple, other_tuple = aligned_tuples(primary_table, table)
            report["paired_delta_vs_primary"][f"{primary_method}_minus_{method_name}"] = paired_bootstrap_delta(
                primary_tuple,
                other_tuple,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{report_name}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap CIs and paired deltas for multiple prediction files.")
    parser.add_argument("--prediction", action="append", required=True, help="name:path[:probability_key[:prediction_key]]")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_name", default="bootstrap_report")
    parser.add_argument("--primary_method", default="TreatAgent-ARG")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_bootstrap_experiments(
        method_specs=args.prediction,
        output_dir=args.output_dir,
        report_name=args.report_name,
        primary_method=args.primary_method,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote report to {args.output_dir / f'{args.report_name}.json'}")


if __name__ == "__main__":
    main()

