#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from treatagent.orchestration.argument_graph_scorer import argument_factors_from_result
from experiments.orchestration.eg_scorer import classification_metrics
from experiments.orchestration.prediction_io import load_prediction_rows, safe_float, safe_int


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "budgeted_planner"
EXPERTS = ["DrugKB", "DiseaseKB", "DTI", "ADMET", "Clinical"]
STATIC_ORDERS = {
    "static_default": ["DrugKB", "DiseaseKB", "DTI", "Clinical", "ADMET"],
    "static_clinical_first": ["Clinical", "DrugKB", "DiseaseKB", "DTI", "ADMET"],
    "static_mechanism_first": ["DTI", "DiseaseKB", "DrugKB", "Clinical", "ADMET"],
}


def result_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id") or row.get("id") or idx): row for idx, row in enumerate(rows)}


def get_typed(result: dict[str, Any]) -> list[dict[str, Any]]:
    graph = result.get("evidence_graph") or {}
    return [dict(item) for item in graph.get("typed_evidence") or []]


def mask_result_to_experts(result: dict[str, Any], experts: set[str]) -> dict[str, Any]:
    masked = deepcopy(result)
    graph = dict(masked.get("evidence_graph") or {})
    graph["typed_evidence"] = [item for item in get_typed(masked) if str(item.get("expert")) in experts]
    masked["evidence_graph"] = graph
    return masked


def trajectory_experts(result: dict[str, Any]) -> list[str]:
    output = []
    for item in result.get("trajectory") or []:
        if not isinstance(item, dict):
            action = str(item)
        else:
            planner = item.get("planner_output") or {}
            action = planner.get("next_action") or planner.get("selected_action") or item.get("expert") or item.get("action")
        if action in EXPERTS or action == "STOP":
            output.append(str(action))
    if output:
        return output
    present = []
    for item in get_typed(result):
        expert = str(item.get("expert") or "")
        if expert in EXPERTS and expert not in present:
            present.append(expert)
    return present


def unique_prefix(actions: list[str], budget: int) -> list[str]:
    experts = []
    for action in actions:
        if action == "STOP":
            break
        if action in EXPERTS and action not in experts:
            experts.append(action)
        if len(experts) >= budget:
            break
    return experts


def value_planner_v2_action(result: dict[str, Any], selected: list[str], threshold: float) -> str:
    factors = argument_factors_from_result(mask_result_to_experts(result, set(selected))).get("factors") or {}
    current_score = safe_float(factors.get("raw_argument_score"), 0.0)
    decision_margin = abs(current_score - threshold)
    conflict_strength = max(
        safe_float(factors.get("conflict_strength"), 0.0),
        safe_float(factors.get("safety_conflict"), 0.0),
        safe_float(factors.get("mechanism_conflict"), 0.0),
        safe_float(factors.get("knowledge_conflict"), 0.0),
    )
    support_strength = max(
        safe_float(factors.get("direct_support"), 0.0),
        safe_float(factors.get("mechanism_support"), 0.0),
        safe_float(factors.get("knowledge_support"), 0.0),
    )
    clinical = safe_float(factors.get("clinical_feasibility"), 0.0)
    remaining = [expert for expert in EXPERTS if expert not in selected]

    if "Clinical" in remaining:
        return "Clinical"

    if conflict_strength < 0.08 and (decision_margin >= 0.18 or current_score <= 0.14):
        return "STOP"

    if conflict_strength >= 0.12:
        for expert in ["DiseaseKB", "DrugKB", "DTI", "ADMET"]:
            if expert in remaining:
                return expert

    if clinical >= 0.72:
        for expert in ["DrugKB", "DiseaseKB", "DTI"]:
            if expert in remaining:
                return expert
        if support_strength >= 0.45 and "ADMET" in remaining:
            return "ADMET"

    if clinical <= 0.25:
        for expert in ["DTI", "DrugKB", "DiseaseKB"]:
            if expert in remaining:
                return expert
        if support_strength >= 0.50 and "ADMET" in remaining:
            return "ADMET"

    if decision_margin < 0.18:
        for expert in ["DTI", "DrugKB", "DiseaseKB"]:
            if expert in remaining:
                return expert

    if support_strength >= 0.45 and "ADMET" in remaining:
        return "ADMET"

    for expert in ["DrugKB", "DiseaseKB", "DTI", "ADMET"]:
        if expert in remaining:
            return expert
    return "STOP"


def value_planner_v2_selection(result: dict[str, Any], budget: int, threshold: float) -> list[str]:
    selected: list[str] = []
    for _ in range(budget):
        action = value_planner_v2_action(result, selected, threshold)
        if action == "STOP":
            break
        if action in EXPERTS and action not in selected:
            selected.append(action)
    return selected


def score_rows(rows: list[dict[str, Any]], threshold: float) -> tuple[list[int], list[float], list[int]]:
    y_true = []
    y_score = []
    y_pred = []
    for row in rows:
        label = safe_int(row.get("label"), 0)
        scored = argument_factors_from_result(row)
        score = safe_float((scored.get("factors") or {}).get("raw_argument_score"), 0.0)
        y_true.append(label)
        y_score.append(score)
        y_pred.append(1 if score >= threshold else 0)
    return y_true, y_score, y_pred


def evaluate_policy(
    name: str,
    rows: list[dict[str, Any]],
    selected_experts: list[list[str]],
    threshold: float,
) -> dict[str, Any]:
    masked_rows = [mask_result_to_experts(row, set(experts)) for row, experts in zip(rows, selected_experts)]
    y_true, y_score, _ = score_rows(masked_rows, threshold)
    calls = [len(set(experts)) for experts in selected_experts]
    trajectory_counter = Counter(" -> ".join(experts) if experts else "NONE" for experts in selected_experts)
    expert_counter = Counter(expert for experts in selected_experts for expert in set(experts))
    metrics = classification_metrics(y_true, y_score, threshold)
    return {
        "name": name,
        "rows": len(rows),
        "threshold": round(threshold, 6),
        "metrics": metrics,
        "total_expert_calls": int(sum(calls)),
        "average_expert_calls": round(sum(calls) / len(calls), 6) if calls else 0.0,
        "call_reduction_vs_all5": round(1.0 - (sum(calls) / (len(rows) * len(EXPERTS))), 6) if rows else 0.0,
        "expert_calls": dict(expert_counter),
        "trajectory_diversity": len(trajectory_counter),
        "top_trajectories": [
            {"trajectory": trajectory, "count": count}
            for trajectory, count in trajectory_counter.most_common(10)
        ],
    }


def write_policy_predictions(
    name: str,
    rows: list[dict[str, Any]],
    selected_experts: list[list[str]],
    threshold: float,
    output_dir: Path,
    report_name: str,
) -> Path:
    predictions = []
    for row, experts in zip(rows, selected_experts):
        masked = mask_result_to_experts(row, set(experts))
        scored = argument_factors_from_result(masked)
        factors = scored.get("factors") or {}
        score = safe_float(factors.get("raw_argument_score"), 0.0)
        pred = 1 if score >= threshold else 0
        output = dict(row)
        output["budgeted_policy"] = name
        output["selected_experts"] = list(experts)
        output["selected_expert_count"] = len(set(experts))
        output["budgeted_probability"] = round(score, 6)
        output["budgeted_prediction"] = pred
        output["argument_factors"] = factors
        predictions.append(output)
    path = output_dir / f"{report_name}_{name}_predictions.json"
    path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_budgeted_analysis(
    all_expert_test: Path,
    planner_test: Path | None,
    threshold: float,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_name: str = "budgeted_planner_drug_disjoint",
) -> dict[str, Any]:
    all_rows = load_prediction_rows(all_expert_test)
    planner_rows = load_prediction_rows(planner_test) if planner_test else []
    all_by_id = result_by_sample(all_rows)
    if planner_rows:
        aligned_planner = [row for row in planner_rows if str(row.get("sample_id")) in all_by_id]
        aligned_all = [all_by_id[str(row.get("sample_id"))] for row in aligned_planner]
    else:
        aligned_planner = []
        aligned_all = all_rows

    policies: list[dict[str, Any]] = []
    policy_prediction_paths: dict[str, str] = {}

    for order_name, order in STATIC_ORDERS.items():
        for budget in range(1, len(EXPERTS) + 1):
            selected = [order[:budget] for _ in aligned_all]
            policies.append(evaluate_policy(f"{order_name}_budget{budget}", aligned_all, selected, threshold))

    if aligned_planner:
        for budget in range(1, len(EXPERTS) + 1):
            selected = [unique_prefix(trajectory_experts(row), budget) for row in aligned_planner]
            policies.append(evaluate_policy(f"llm_planner_prefix_budget{budget}", aligned_planner, selected, threshold))

    for budget in range(1, len(EXPERTS) + 1):
        selected = [value_planner_v2_selection(row, budget, threshold) for row in aligned_all]
        policy_name = f"value_planner_v2_budget{budget}"
        policies.append(evaluate_policy(policy_name, aligned_all, selected, threshold))

    summary = {
        "inputs": {
            "all_expert_test": str(all_expert_test),
            "planner_test": str(planner_test) if planner_test else None,
        },
        "threshold_policy": "fixed full-ARG validation threshold",
        "threshold": threshold,
        "rows": len(aligned_all),
        "policies": policies,
        "policy_prediction_paths": policy_prediction_paths,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for budget in range(1, len(EXPERTS) + 1):
        policy_name = f"value_planner_v2_budget{budget}"
        selected = [value_planner_v2_selection(row, budget, threshold) for row in aligned_all]
        pred_path = write_policy_predictions(policy_name, aligned_all, selected, threshold, output_dir, report_name)
        policy_prediction_paths[policy_name] = str(pred_path)

    report_path = output_dir / f"{report_name}.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, output_dir / f"{report_name}.md")
    return summary


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Budgeted Planner Analysis",
        "",
        f"Rows: {report['rows']}",
        f"Threshold: {report['threshold']:.4f} ({report['threshold_policy']})",
        "",
        "| Policy | Avg calls | Call reduction | Acc | F1 | Precision | Recall | AUROC | AUPRC | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in report["policies"]:
        m = policy["metrics"]
        lines.append(
            "| {name} | {calls:.2f} | {red:.2%} | {acc:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} | {auroc:.4f} | {auprc:.4f} | {ece:.4f} |".format(
                name=policy["name"],
                calls=policy["average_expert_calls"],
                red=policy["call_reduction_vs_all5"],
                acc=m.get("accuracy", 0.0),
                f1=m.get("f1", 0.0),
                prec=m.get("precision", 0.0),
                rec=m.get("recall", 0.0),
                auroc=m.get("auroc", 0.0),
                auprc=m.get("auprc", 0.0),
                ece=m.get("ece", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline budgeted planner analysis with ARG scoring.")
    parser.add_argument("--all_expert_test", type=Path, required=True)
    parser.add_argument("--planner_test", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.36)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_name", default="budgeted_planner_drug_disjoint")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_budgeted_analysis(
        all_expert_test=args.all_expert_test,
        planner_test=args.planner_test,
        threshold=args.threshold,
        output_dir=args.output_dir,
        report_name=args.report_name,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

