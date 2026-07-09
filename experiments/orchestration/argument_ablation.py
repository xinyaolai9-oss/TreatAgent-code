#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from treatagent.orchestration.argument_graph_scorer import (
    argument_factors_from_result,
)
from experiments.orchestration.eg_scorer import choose_threshold, classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "argument_ablation"


ABLATION_SETTINGS = [
    {
        "name": "full",
        "description": "Full TreatAgent-ARG rule.",
        "variant": "full",
        "use_reliability": True,
    },
    {
        "name": "wo_direct_support",
        "description": "Remove DrugKB direct indication/direct support contribution.",
        "variant": "wo_direct_support",
        "use_reliability": True,
    },
    {
        "name": "wo_clinical_feasibility",
        "description": "Remove clinical feasibility prior contribution.",
        "variant": "wo_clinical_feasibility",
        "use_reliability": True,
    },
    {
        "name": "wo_cross_source_consistency",
        "description": "Remove cross-source consistency bonus.",
        "variant": "wo_cross_source_consistency",
        "use_reliability": True,
    },
    {
        "name": "wo_conflict_penalty",
        "description": "Remove safety/mechanism/knowledge conflict penalty.",
        "variant": "wo_conflict_penalty",
        "use_reliability": True,
    },
    {
        "name": "wo_admet_noise_handling",
        "description": "Remove non-CNS BBB noise penalty from ADMET handling.",
        "variant": "wo_admet_noise_handling",
        "use_reliability": True,
    },
    {
        "name": "wo_reliability_weighting",
        "description": "Recompute evidence strengths without multiplying by reliability.",
        "variant": "full",
        "use_reliability": False,
    },
    {
        "name": "wo_evidence_graph_structure",
        "description": "Collapse typed support/conflict factors into an unstructured evidence summary score.",
        "variant": "wo_evidence_graph_structure",
        "use_reliability": True,
    },
]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_from_factors(factors: dict[str, Any], variant: str = "full") -> float:
    direct = float(factors.get("direct_support", 0.0))
    mechanism = float(factors.get("mechanism_support", 0.0))
    knowledge = float(factors.get("knowledge_support", 0.0))
    clinical = float(factors.get("clinical_feasibility", 0.0))
    safety_support = float(factors.get("safety_support", 0.0))
    consistency = float(factors.get("cross_source_consistency", 0.0))
    safety_conflict = float(factors.get("safety_conflict", 0.0))
    mechanism_conflict = float(factors.get("mechanism_conflict", 0.0))
    knowledge_conflict = float(factors.get("knowledge_conflict", 0.0))
    bbb_noise = float(factors.get("admet_bbb_non_cns_noise", 0.0))

    if variant == "wo_evidence_graph_structure":
        support_summary = (direct + mechanism + knowledge + clinical + safety_support) / 5.0
        conflict_summary = (safety_conflict + mechanism_conflict + knowledge_conflict + bbb_noise) / 4.0
        return _clip01(0.18 + 0.65 * support_summary - 0.25 * conflict_summary)

    if variant == "wo_direct_support":
        direct = 0.0
    elif variant == "wo_clinical_feasibility":
        clinical = 0.0
    elif variant == "wo_cross_source_consistency":
        consistency = 0.0

    if variant == "wo_conflict_penalty":
        conflict_strength = 0.0
    else:
        if variant == "wo_admet_noise_handling":
            bbb_noise = 0.0
        conflict_strength = _clip01(
            0.50 * safety_conflict
            + 0.25 * mechanism_conflict
            + 0.15 * knowledge_conflict
            + 0.10 * bbb_noise
        )

    return _clip01(
        0.18
        + 0.45 * clinical
        + 0.35 * direct
        + 0.08 * consistency
        - 0.10 * conflict_strength
    )


def read_rows(path: Path, *, variant: str, use_reliability: bool) -> tuple[list[dict[str, Any]], list[float], list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    scores = []
    labels = []
    for result in payload.get("results") or []:
        if result.get("label") in (None, ""):
            continue
        row = argument_factors_from_result(result, use_reliability=use_reliability)
        score = score_from_factors(row["factors"], variant=variant)
        row["ablation_score"] = round(score, 6)
        rows.append(row)
        scores.append(score)
        labels.append(int(float(result.get("label"))))
    return rows, scores, labels


def evaluate_scores(labels: Sequence[int], scores: Sequence[float], threshold: float, split: str) -> dict[str, Any]:
    return {
        "split": split,
        "rows": len(labels),
        "positive": int(sum(labels)),
        "negative": int(len(labels) - sum(labels)),
        "metrics": classification_metrics(labels, scores, threshold),
    }


def write_predictions(path: Path, rows: Sequence[dict[str, Any]], scores: Sequence[float], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for row, score in zip(rows, scores):
        payload.append(
            {
                "sample_id": row.get("sample_id"),
                "label": row.get("label"),
                "smiles": row.get("smiles"),
                "disease": row.get("disease"),
                "ablation_score": round(float(score), 6),
                "ablation_prediction": 1 if score >= threshold else 0,
                "argument_factors": row.get("factors"),
                "top_support_arguments": row.get("top_support_arguments"),
                "top_conflict_arguments": row.get("top_conflict_arguments"),
            }
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_argument_ablation(
    train_json: Path,
    val_json: Path,
    test_json: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_name: str = "argument_ablation_report",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "TreatAgent-ARG component ablation",
        "reasoning": "Each setting uses a fixed no-classifier argument score. The validation split is used only to select the decision threshold.",
        "inputs": {
            "train_json": str(train_json),
            "val_json": str(val_json),
            "test_json": str(test_json),
        },
        "settings": {},
    }

    for setting in ABLATION_SETTINGS:
        name = setting["name"]
        variant = setting["variant"]
        use_reliability = bool(setting["use_reliability"])
        setting_dir = output_dir / name

        train_rows, train_scores, train_y = read_rows(train_json, variant=variant, use_reliability=use_reliability)
        val_rows, val_scores, val_y = read_rows(val_json, variant=variant, use_reliability=use_reliability)
        test_rows, test_scores, test_y = read_rows(test_json, variant=variant, use_reliability=use_reliability)
        threshold, validation_metrics = choose_threshold(val_y, val_scores)

        write_predictions(setting_dir / f"{report_name}_{name}_test_predictions.json", test_rows, test_scores, threshold)
        report["settings"][name] = {
            "description": setting["description"],
            "variant": variant,
            "use_reliability": use_reliability,
            "threshold": round(float(threshold), 4),
            "train": {
                "rows": len(train_y),
                "positive": int(sum(train_y)),
                "negative": int(len(train_y) - sum(train_y)),
            },
            "validation_threshold_selection": validation_metrics,
            "validation": evaluate_scores(val_y, val_scores, threshold, "val"),
            "test": evaluate_scores(test_y, test_scores, threshold, "test"),
        }

    full_metrics = report["settings"]["full"]["test"]["metrics"]
    for setting in report["settings"].values():
        metrics = setting["test"]["metrics"]
        setting["delta_vs_full"] = {
            key: round(metrics[key] - full_metrics[key], 6)
            for key in ["accuracy", "f1", "precision", "recall", "brier", "ece", "auroc", "auprc"]
            if metrics.get(key) is not None and full_metrics.get(key) is not None
        }

    report_path = output_dir / f"{report_name}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run component ablation for the no-classifier TreatAgent-ARG scorer.")
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--val_json", type=Path, required=True)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_name", default="argument_ablation_report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_argument_ablation(
        train_json=args.train_json,
        val_json=args.val_json,
        test_json=args.test_json,
        output_dir=args.output_dir,
        report_name=args.report_name,
    )
    print(json.dumps({name: value["test"]["metrics"] for name, value in report["settings"].items()}, indent=2))
    print(f"Wrote report to {args.output_dir / f'{args.report_name}.json'}")


if __name__ == "__main__":
    main()

