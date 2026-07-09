#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "results" / "argument_source_ablation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "expert_relevance_analysis"

EXPERTS = ["DrugKB", "DiseaseKB", "DTI", "ADMET", "Clinical"]
SPLITS = ["drug_disjoint", "temporal_submit"]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list prediction JSON: {path}")
    rows = {}
    for index, row in enumerate(payload):
        sample_id = str(row.get("sample_id") or index)
        rows[sample_id] = row
    return rows


def prediction_path(input_root: Path, split: str, setting: str) -> Path:
    filename = f"argument_source_ablation_{split}_{setting}_test_predictions.json"
    return input_root / split / setting / filename


def factor(row: dict[str, Any], name: str) -> float:
    return safe_float((row.get("argument_factors") or {}).get(name), 0.0)


def score(row: dict[str, Any]) -> float:
    return safe_float(row.get("source_ablation_score"), 0.0)


def pred(row: dict[str, Any]) -> int:
    return safe_int(row.get("source_ablation_prediction"), 0)


def label(row: dict[str, Any]) -> int:
    return safe_int(row.get("label"), 0)


def metric_dict(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    if not selected:
        return {
            "n": 0,
            "positive": 0,
            "accuracy": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "auroc": None,
            "auprc": None,
        }
    y_true = [label(row) for row in selected]
    y_pred = [pred(row) for row in selected]
    y_score = [score(row) for row in selected]
    metrics = {
        "n": len(selected),
        "positive": int(sum(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auroc": None,
        "auprc": None,
    }
    if len(set(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_score)
        metrics["auprc"] = average_precision_score(y_true, y_score)
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()}


def build_subsets(full_rows: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    rows = list(full_rows.values())
    conflict_values = [factor(row, "conflict_strength") for row in rows]
    admet_values = [max(factor(row, "safety_conflict"), factor(row, "admet_bbb_non_cns_noise")) for row in rows]
    mechanism_values = [factor(row, "mechanism_support") for row in rows]
    score_values = [score(row) for row in rows]

    high_conflict_threshold = max(0.05, percentile(conflict_values, 0.75))
    admet_risk_threshold = max(0.05, percentile(admet_values, 0.75))
    mechanism_threshold = max(0.5, percentile(mechanism_values, 0.50))
    score_threshold = percentile(score_values, 0.50)

    definitions = {
        "all": lambda row: True,
        "no_direct_indication": lambda row: factor(row, "direct_support") <= 1e-9,
        "low_clinical_prior": lambda row: factor(row, "clinical_feasibility") <= 0.4,
        "admet_risk": lambda row: max(factor(row, "safety_conflict"), factor(row, "admet_bbb_non_cns_noise"))
        >= admet_risk_threshold,
        "mechanism_only_support": lambda row: factor(row, "direct_support") <= 1e-9
        and factor(row, "mechanism_support") >= mechanism_threshold,
        "high_conflict": lambda row: factor(row, "conflict_strength") >= high_conflict_threshold,
        "near_threshold": lambda row: abs(score(row) - score_threshold) <= 0.08,
    }
    subsets = {
        name: {sample_id for sample_id, row in full_rows.items() if predicate(row)}
        for name, predicate in definitions.items()
    }
    return {name: ids for name, ids in subsets.items() if ids}


def join_rows(
    full_rows: dict[str, dict[str, Any]],
    ablated_rows: dict[str, dict[str, Any]],
    subset_ids: set[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (full_rows[sample_id], ablated_rows[sample_id])
        for sample_id in sorted(subset_ids)
        if sample_id in full_rows and sample_id in ablated_rows
    ]


def flip_summary(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    if not pairs:
        return {
            "n": 0,
            "flip_count": 0,
            "flip_rate": None,
            "expert_corrected": 0,
            "expert_broken": 0,
            "full_positive_to_ablated_negative": 0,
            "full_negative_to_ablated_positive": 0,
            "mean_score_shift": None,
            "mean_abs_score_shift": None,
        }
    shifts = [score(full) - score(ablated) for full, ablated in pairs]
    flips = [(full, ablated) for full, ablated in pairs if pred(full) != pred(ablated)]
    corrected = [1 for full, ablated in pairs if pred(full) == label(full) and pred(ablated) != label(full)]
    broken = [1 for full, ablated in pairs if pred(full) != label(full) and pred(ablated) == label(full)]
    pos_to_neg = [1 for full, ablated in pairs if pred(full) == 1 and pred(ablated) == 0]
    neg_to_pos = [1 for full, ablated in pairs if pred(full) == 0 and pred(ablated) == 1]
    return {
        "n": len(pairs),
        "flip_count": len(flips),
        "flip_rate": round(len(flips) / len(pairs), 6),
        "expert_corrected": int(sum(corrected)),
        "expert_broken": int(sum(broken)),
        "full_positive_to_ablated_negative": int(sum(pos_to_neg)),
        "full_negative_to_ablated_positive": int(sum(neg_to_pos)),
        "mean_score_shift": round(mean(shifts), 6),
        "mean_abs_score_shift": round(mean(abs(value) for value in shifts), 6),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_delta(full_metrics: dict[str, Any], ablated_metrics: dict[str, Any], key: str) -> float | None:
    if full_metrics.get(key) is None or ablated_metrics.get(key) is None:
        return None
    return round(float(full_metrics[key]) - float(ablated_metrics[key]), 6)


def abbreviate_claims(arguments: list[dict[str, Any]], expert: str | None = None, limit: int = 3) -> list[str]:
    claims = []
    for argument in arguments or []:
        if expert and str(argument.get("expert")) != expert:
            continue
        claim = str(argument.get("claim") or argument.get("category") or "")
        if claim:
            claims.append(claim)
        if len(claims) >= limit:
            break
    return claims


def case_candidates(
    full_rows: dict[str, dict[str, Any]],
    ablated_rows: dict[str, dict[str, Any]],
    subset_ids: set[str],
    expert: str,
    prefer_negative_shift: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for full, ablated in join_rows(full_rows, ablated_rows, subset_ids):
        shift = score(full) - score(ablated)
        if prefer_negative_shift:
            rank_shift = -shift
        else:
            rank_shift = shift
        corrected = pred(full) == label(full) and pred(ablated) != label(full)
        rows.append(
            {
                "sample_id": full.get("sample_id"),
                "disease": full.get("disease"),
                "smiles": full.get("smiles"),
                "label": label(full),
                "full_score": round(score(full), 6),
                "ablated_score": round(score(ablated), 6),
                "score_shift_full_minus_ablated": round(shift, 6),
                "full_prediction": pred(full),
                "ablated_prediction": pred(ablated),
                "expert_corrected": corrected,
                "top_support_claims": abbreviate_claims(full.get("top_support_arguments") or [], expert),
                "top_conflict_claims": abbreviate_claims(full.get("top_conflict_arguments") or [], expert),
                "_rank": (1 if corrected else 0, rank_shift),
            }
        )
    return sorted(rows, key=lambda row: row["_rank"], reverse=True)


def select_cases(
    split: str,
    full_rows: dict[str, dict[str, Any]],
    ablation_tables: dict[str, dict[str, dict[str, Any]]],
    subsets: dict[str, set[str]],
) -> list[dict[str, Any]]:
    specs = [
        ("DrugKB", "no_direct_indication", False, "DrugKB support outside direct indication"),
        ("DiseaseKB", "mechanism_only_support", False, "Disease-level mechanism bridge"),
        ("DTI", "mechanism_only_support", False, "DTI / mechanism rescue"),
        ("ADMET", "admet_risk", True, "ADMET safety conflict"),
        ("Clinical", "low_clinical_prior", False, "Clinical prior contribution"),
    ]
    selected = []
    for expert, subset_name, prefer_negative_shift, rationale in specs:
        setting = f"wo_{expert}"
        candidates = case_candidates(
            full_rows,
            ablation_tables[setting],
            subsets.get(subset_name, set()),
            expert,
            prefer_negative_shift=prefer_negative_shift,
        )
        if not candidates:
            continue
        case = candidates[0]
        case.pop("_rank", None)
        case["split"] = split
        case["expert"] = expert
        case["subset"] = subset_name
        case["rationale"] = rationale
        selected.append(case)
    return selected


def analyze_split(input_root: Path, split: str) -> dict[str, Any]:
    full_rows = load_rows(prediction_path(input_root, split, "full"))
    ablation_tables = {
        f"wo_{expert}": load_rows(prediction_path(input_root, split, f"wo_{expert}"))
        for expert in EXPERTS
    }
    subsets = build_subsets(full_rows)

    subset_rows = []
    flip_rows = []
    for subset_name, subset_ids in subsets.items():
        full_metrics = metric_dict(full_rows[sample_id] for sample_id in subset_ids if sample_id in full_rows)
        for expert in EXPERTS:
            setting = f"wo_{expert}"
            ablated_rows = ablation_tables[setting]
            ablated_metrics = metric_dict(ablated_rows[sample_id] for sample_id in subset_ids if sample_id in ablated_rows)
            subset_rows.append(
                {
                    "split": split,
                    "subset": subset_name,
                    "expert_removed": expert,
                    "n": full_metrics["n"],
                    "positive": full_metrics["positive"],
                    "full_f1": full_metrics["f1"],
                    "ablated_f1": ablated_metrics["f1"],
                    "delta_f1_full_minus_ablated": row_delta(full_metrics, ablated_metrics, "f1"),
                    "full_auroc": full_metrics["auroc"],
                    "ablated_auroc": ablated_metrics["auroc"],
                    "delta_auroc_full_minus_ablated": row_delta(full_metrics, ablated_metrics, "auroc"),
                    "full_accuracy": full_metrics["accuracy"],
                    "ablated_accuracy": ablated_metrics["accuracy"],
                    "delta_accuracy_full_minus_ablated": row_delta(full_metrics, ablated_metrics, "accuracy"),
                }
            )
            flips = flip_summary(join_rows(full_rows, ablated_rows, subset_ids))
            flip_rows.append(
                {
                    "split": split,
                    "subset": subset_name,
                    "expert_removed": expert,
                    **flips,
                }
            )

    cases = select_cases(split, full_rows, ablation_tables, subsets)
    return {
        "split": split,
        "subsets": {name: len(ids) for name, ids in subsets.items()},
        "subset_ablation": subset_rows,
        "flip_shift": flip_rows,
        "cases": cases,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.4f}"
    return str(value)


def best_expert_rows(subset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expert_to_subset = {
        "DrugKB": {"no_direct_indication", "mechanism_only_support"},
        "DiseaseKB": {"mechanism_only_support", "no_direct_indication"},
        "DTI": {"mechanism_only_support", "low_clinical_prior", "near_threshold"},
        "ADMET": {"admet_risk"},
        "Clinical": {"low_clinical_prior", "all"},
    }
    selected = []
    for row in subset_rows:
        if row["subset"] in expert_to_subset.get(row["expert_removed"], set()):
            selected.append(row)
    return selected


def write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    subset_rows = analysis["subset_ablation"]
    flip_rows = analysis["flip_shift"]
    cases = analysis["cases"]

    focused_rows = best_expert_rows(subset_rows)
    lines = [
        "# Expert Relevance Analysis",
        "",
        "This analysis uses the no-classifier TreatAgent-ARG source ablation predictions. It asks where each expert matters, rather than only comparing global averages.",
        "",
        "## A. Expert-relevant subset ablation",
        "",
        "| Split | Relevant subset | Removed expert | N | Full F1 | Ablated F1 | Delta F1 | Full AUROC | Ablated AUROC | Delta AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in focused_rows:
        lines.append(
            "| {split} | {subset} | {expert_removed} | {n} | {full_f1} | {ablated_f1} | {delta_f1} | {full_auroc} | {ablated_auroc} | {delta_auroc} |".format(
                split=row["split"],
                subset=row["subset"],
                expert_removed=row["expert_removed"],
                n=row["n"],
                full_f1=fmt(row["full_f1"]),
                ablated_f1=fmt(row["ablated_f1"]),
                delta_f1=fmt(row["delta_f1_full_minus_ablated"]),
                full_auroc=fmt(row["full_auroc"]),
                ablated_auroc=fmt(row["ablated_auroc"]),
                delta_auroc=fmt(row["delta_auroc_full_minus_ablated"]),
            )
        )

    lines.extend(
        [
            "",
            "## B. Decision flip and score-shift analysis",
            "",
            "| Split | Relevant subset | Removed expert | N | Flips | Corrected by expert | Broken by expert | Mean score shift |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    focused_keys = {(row["split"], row["subset"], row["expert_removed"]) for row in focused_rows}
    for row in flip_rows:
        if (row["split"], row["subset"], row["expert_removed"]) not in focused_keys:
            continue
        lines.append(
            "| {split} | {subset} | {expert_removed} | {n} | {flip_count} | {expert_corrected} | {expert_broken} | {mean_score_shift} |".format(
                split=row["split"],
                subset=row["subset"],
                expert_removed=row["expert_removed"],
                n=row["n"],
                flip_count=row["flip_count"],
                expert_corrected=row["expert_corrected"],
                expert_broken=row["expert_broken"],
                mean_score_shift=fmt(row["mean_score_shift"]),
            )
        )

    lines.extend(
        [
            "",
            "## C. Expert-specific cases",
            "",
            "| Split | Expert | Subset | Sample | Disease | Label | Full / w-o expert score | Full / w-o pred | Evidence signal |",
            "|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for case in cases:
        claims = case["top_conflict_claims"] or case["top_support_claims"]
        evidence = claims[0] if claims else ""
        lines.append(
            "| {split} | {expert} | {subset} | {sample_id} | {disease} | {label} | {full_score} / {ablated_score} | {full_prediction} / {ablated_prediction} | {evidence} |".format(
                split=case["split"],
                expert=case["expert"],
                subset=case["subset"],
                sample_id=case["sample_id"],
                disease=str(case["disease"]).replace("|", "/"),
                label=case["label"],
                full_score=fmt(case["full_score"]),
                ablated_score=fmt(case["ablated_score"]),
                full_prediction=case["full_prediction"],
                ablated_prediction=case["ablated_prediction"],
                evidence=evidence.replace("|", "/"),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_root: Path, output_dir: Path, splits: list[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports = [analyze_split(input_root, split) for split in splits]
    analysis = {
        "input_root": str(input_root),
        "splits": splits,
        "subset_ablation": [row for report in split_reports for row in report["subset_ablation"]],
        "flip_shift": [row for report in split_reports for row in report["flip_shift"]],
        "cases": [case for report in split_reports for case in report["cases"]],
        "subset_sizes": {report["split"]: report["subsets"] for report in split_reports},
    }
    write_csv(output_dir / "expert_relevant_subset_ablation.csv", analysis["subset_ablation"])
    write_csv(output_dir / "expert_decision_flip_score_shift.csv", analysis["flip_shift"])
    (output_dir / "expert_specific_cases.json").write_text(
        json.dumps(analysis["cases"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "expert_relevance_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(output_dir / "expert_relevance_summary.md", analysis)
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze where individual TreatAgent experts matter.")
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", action="append", choices=SPLITS, default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    splits = args.split or SPLITS
    analysis = run(args.input_root, args.output_dir, splits)
    print(f"Wrote expert relevance analysis to {args.output_dir}")
    print(json.dumps(analysis["subset_sizes"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
