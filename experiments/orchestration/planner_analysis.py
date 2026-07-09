#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.orchestration.eg_scorer import classification_metrics
from experiments.orchestration.prediction_io import load_prediction_rows, safe_float, safe_int


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "planner_analysis"
EXPERTS = ["DrugKB", "DiseaseKB", "DTI", "ADMET", "Clinical"]


def trajectory_experts(row: dict[str, Any]) -> list[str]:
    trajectory = row.get("trajectory") or []
    experts = []
    for item in trajectory:
        if isinstance(item, dict):
            name = item.get("expert") or item.get("agent") or item.get("action")
        else:
            name = str(item)
        if name in EXPERTS or name == "STOP":
            experts.append(str(name))
    if experts:
        return experts

    expert_outputs = row.get("expert_outputs") or {}
    return [expert for expert in EXPERTS if expert_outputs.get(expert)]


def run_planner_analysis(path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR, report_name: str = "planner_report") -> dict:
    rows = load_prediction_rows(path)
    y_true = [safe_int(row.get("label"), 0) for row in rows]
    y_prob = [
        safe_float(row.get("calibrated_probability"), safe_float(row.get("prediction_binary"), 0.0))
        for row in rows
    ]
    metrics = classification_metrics(y_true, y_prob, 0.5) if rows else {}

    trajectory_counter: Counter[str] = Counter()
    expert_counter: Counter[str] = Counter()
    stop_count = 0
    total_calls = 0
    planner_enabled = 0
    explanation_enabled = 0
    synthesis_enabled = 0
    for row in rows:
        planner_enabled += int(bool(row.get("llm_planner_enabled")))
        explanation_enabled += int(bool(row.get("llm_explanation_enabled")))
        synthesis_enabled += int(bool(row.get("llm_synthesis_enabled")))
        experts = trajectory_experts(row)
        trajectory_counter[" -> ".join(experts)] += 1
        for expert in experts:
            if expert == "STOP":
                stop_count += 1
            elif expert in EXPERTS:
                expert_counter[expert] += 1
                total_calls += 1

    max_calls = len(rows) * len(EXPERTS)
    report = {
        "input": str(path),
        "rows": len(rows),
        "metrics": metrics,
        "planner_enabled": planner_enabled,
        "llm_explanation_enabled": explanation_enabled,
        "llm_synthesis_enabled": synthesis_enabled,
        "total_expert_calls": total_calls,
        "max_static_all_expert_calls": max_calls,
        "average_expert_calls": round(total_calls / len(rows), 6) if rows else 0.0,
        "expert_call_reduction": round(1.0 - total_calls / max_calls, 6) if max_calls else 0.0,
        "stop_count": stop_count,
        "trajectory_diversity": len(trajectory_counter),
        "expert_calls": dict(expert_counter),
        "top_trajectories": [
            {"trajectory": trajectory, "count": count}
            for trajectory, count in trajectory_counter.most_common(20)
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{report_name}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze TreatAgent planner trajectories and expert-call cost.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_name", default="planner_report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_planner_analysis(args.predictions, args.output_dir, args.report_name)
    print(json.dumps(report, indent=2))
    print(f"Wrote report to {args.output_dir / f'{args.report_name}.json'}")


if __name__ == "__main__":
    main()

