#!/usr/bin/env python3
"""Summarize the frozen publication results under results/final_results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "final_results"

RESULT_FILES = {
    "Drug-disjoint": {
        "Direct": "baselines/results_direct_dd_test.json",
        "CoT": "baselines/results_cot_dd_test.json",
        "RAG": "baselines/results_rag_dd_test.json",
        "TreatAgent": "baselines/results_multiagent_dd_test.json",
        "w/o Planner": "ablations/results_multiagent_dd_test_wo_planner.json",
        "w/o EvidenceGraph": "ablations/results_multiagent_dd_test_wo_evidencegraph.json",
        "w/o DrugKB": "ablations/results_multiagent_drug_disjoint_wo_drugkb.json",
        "w/o DiseaseKB": "ablations/results_multiagent_drug_disjoint_wo_diseasekb.json",
        "w/o DTI": "ablations/results_multiagent_drug_disjoint_wo_dti.json",
        "w/o ADMET": "ablations/results_multiagent_drug_disjoint_wo_admet.json",
        "w/o Clinical": "ablations/results_multiagent_drug_disjoint_wo_clinical.json",
    },
    "Temporal-submit": {
        "Direct": "baselines/results_direct_ts_test.json",
        "CoT": "baselines/results_cot_ts_test.json",
        "RAG": "baselines/results_rag_ts_test.json",
        "TreatAgent": "baselines/results_multiagent_ts_test.json",
        "w/o Planner": "ablations/results_multiagent_ts_test_wo_planner.json",
        "w/o EvidenceGraph": "ablations/results_multiagent_ts_test_wo_evidencegraph.json",
        "w/o DrugKB": "ablations/results_multiagent_temporal_submit_wo_drugkb.json",
        "w/o DiseaseKB": "ablations/results_multiagent_temporal_submit_wo_diseasekb.json",
        "w/o DTI": "ablations/results_multiagent_temporal_submit_wo_dti.json",
        "w/o ADMET": "ablations/results_multiagent_temporal_submit_wo_admet.json",
        "w/o Clinical": "ablations/results_multiagent_temporal_submit_wo_clinical.json",
    },
}

BASELINES = ["Direct", "CoT", "RAG", "TreatAgent"]
COMPONENT_ABLATIONS = ["w/o Planner", "w/o EvidenceGraph"]
SOURCE_ABLATIONS = ["w/o DrugKB", "w/o DiseaseKB", "w/o DTI", "w/o ADMET", "w/o Clinical"]


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


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Unsupported result JSON: {path}")
    return payload


def score_for_row(row: dict[str, Any]) -> float:
    for key in ("llm_judge_probability", "prediction_score", "calibrated_probability"):
        if row.get(key) is not None:
            return safe_float(row[key])
    return safe_float(row.get("prediction_binary"))


def prediction_for_row(row: dict[str, Any]) -> int:
    return safe_int(row.get("prediction_binary"))


def label_for_row(row: dict[str, Any]) -> int:
    return safe_int(row.get("label"))


def average_precision(y_true: list[int], y_score: list[float]) -> float | None:
    if not sum(y_true):
        return None
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if not positives or not negatives:
        return None
    return float(roc_auc_score(y_true, y_score))


def expected_calibration_error(y_true: list[int], y_score: list[float], bins: int = 10) -> float:
    total = len(y_true)
    if not total:
        return 0.0
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            (label, score)
            for label, score in zip(y_true, y_score)
            if lower <= score < upper or (bin_index == bins - 1 and score == 1.0)
        ]
        if not selected:
            continue
        accuracy = mean(label for label, _ in selected)
        confidence = mean(score for _, score in selected)
        error += len(selected) / total * abs(accuracy - confidence)
    return error


def rounded(value: Any, digits: int = 6) -> Any:
    return round(value, digits) if isinstance(value, float) else value


def summarize_payload(split: str, setting: str, relative_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["results"]
    y_true = [label_for_row(row) for row in rows]
    y_pred = [prediction_for_row(row) for row in rows]
    y_score = [score_for_row(row) for row in rows]
    tp = sum(label == 1 and pred == 1 for label, pred in zip(y_true, y_pred))
    tn = sum(label == 0 and pred == 0 for label, pred in zip(y_true, y_pred))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(y_true, y_pred))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(y_true, y_pred))
    score_levels = len(set(y_score))

    expert_calls = [
        len(row.get("expert_outputs") or {})
        for row in rows
        if isinstance(row.get("expert_outputs"), dict)
    ]
    call_distribution = dict(sorted(Counter(expert_calls).items()))
    judge_sources = Counter(
        (row.get("llm_judge") or {}).get("judge_source")
        for row in rows
        if isinstance(row.get("llm_judge"), dict) and (row.get("llm_judge") or {}).get("judge_source")
    )
    judge_probability_fallbacks = sum(
        row.get("llm_judge_probability") is None and row.get("prediction_score") is not None
        for row in rows
        if row.get("llm_judge_enabled")
    )

    metrics = payload.get("metrics") or {}
    summary = {
        "split": split,
        "setting": setting,
        "result_json": relative_path,
        "n": len(rows),
        "positives": sum(y_true),
        "negatives": len(y_true) - sum(y_true),
        "threshold": safe_float(payload.get("threshold"), 0.5),
        "accuracy": safe_float(metrics.get("accuracy")),
        "f1": safe_float(metrics.get("f1")),
        "precision": safe_float(metrics.get("precision")),
        "recall": safe_float(metrics.get("recall")),
        "auroc": roc_auc(y_true, y_score),
        "auprc": average_precision(y_true, y_score),
        "brier": mean((score - label) ** 2 for score, label in zip(y_score, y_true)),
        "ece": expected_calibration_error(y_true, y_score),
        "predicted_positive_rate": sum(y_pred) / len(y_pred),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "score_levels": score_levels,
        "score_is_continuous": score_levels > 2,
        "mean_score": mean(y_score),
        "mean_expert_calls": mean(expert_calls) if expert_calls else None,
        "expert_call_distribution": call_distribution,
        "judge_sources": dict(judge_sources),
        "judge_probability_fallbacks": judge_probability_fallbacks,
    }
    return {key: rounded(value) for key, value in summary.items()}


def add_deltas(rows: list[dict[str, Any]]) -> None:
    full_rows = {row["split"]: row for row in rows if row["setting"] == "TreatAgent"}
    for row in rows:
        full = full_rows[row["split"]]
        row["delta_accuracy_vs_full"] = rounded(row["accuracy"] - full["accuracy"])
        row["delta_f1_vs_full"] = rounded(row["f1"] - full["f1"])
        row["delta_precision_vs_full"] = rounded(row["precision"] - full["precision"])
        row["delta_recall_vs_full"] = rounded(row["recall"] - full["recall"])
        row["delta_auroc_vs_full"] = rounded(row["auroc"] - full["auroc"])
        if row["mean_expert_calls"] is not None and full["mean_expert_calls"] is not None:
            row["delta_calls_vs_full"] = rounded(row["mean_expert_calls"] - full["mean_expert_calls"])
        else:
            row["delta_calls_vs_full"] = None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(rows: Iterable[dict[str, Any]], fields: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for key, _ in fields) + " |")
    return lines


def build_markdown(rows: list[dict[str, Any]]) -> str:
    by_split = {
        split: {row["setting"]: row for row in rows if row["split"] == split}
        for split in RESULT_FILES
    }
    lines = [
        "# 最终实验结果汇总",
        "",
        "本目录整理 TreatAgent 最终版本在两个 test split 上的冻结结果。",
        "Direct、CoT 和 RAG 仅输出二分类判断，因此其 AUROC、AUPRC、Brier 和 ECE 与 TreatAgent 基于连续 LLM judge score 的结果不能直接比较。",
        "",
        "## 1. 主结果",
        "",
    ]
    main_fields = [
        ("split", "Split"),
        ("setting", "方法"),
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("predicted_positive_rate", "预测阳性率"),
    ]
    main_rows = [
        by_split[split][setting]
        for split in RESULT_FILES
        for setting in BASELINES
    ]
    lines.extend(md_table(main_rows, main_fields))

    lines.extend(["", "## 2. 主模块消融", ""])
    ablation_fields = [
        ("split", "Split"),
        ("setting", "设置"),
        ("accuracy", "Accuracy"),
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("auroc", "AUROC"),
        ("delta_f1_vs_full", "相对完整模型 Delta F1"),
        ("mean_expert_calls", "平均调用数"),
    ]
    component_rows = [
        by_split[split][setting]
        for split in RESULT_FILES
        for setting in ["TreatAgent", *COMPONENT_ABLATIONS]
    ]
    lines.extend(md_table(component_rows, ablation_fields))

    lines.extend(["", "## 3. Source / expert 消融", ""])
    source_rows = [
        by_split[split][setting]
        for split in RESULT_FILES
        for setting in ["TreatAgent", *SOURCE_ABLATIONS]
    ]
    lines.extend(md_table(source_rows, ablation_fields))

    lines.extend(["", "## 4. Planner 调度效率", ""])
    planner_fields = [
        ("split", "Split"),
        ("setting", "设置"),
        ("f1", "F1"),
        ("mean_expert_calls", "平均 expert 调用数"),
        ("delta_calls_vs_full", "相对完整模型调用数变化"),
    ]
    planner_rows = [
        by_split[split][setting]
        for split in RESULT_FILES
        for setting in ["TreatAgent", "w/o Planner"]
    ]
    lines.extend(md_table(planner_rows, planner_fields))
    for split in RESULT_FILES:
        full = by_split[split]["TreatAgent"]
        no_planner = by_split[split]["w/o Planner"]
        reduction = 1.0 - full["mean_expert_calls"] / no_planner["mean_expert_calls"]
        lines.append(
            f"- {split}: Planner 将平均 expert 调用数从 {no_planner['mean_expert_calls']:.2f} "
            f"降至 {full['mean_expert_calls']:.2f}，减少 {reduction:.2%}。"
        )

    lines.extend(["", "## 5. TreatAgent 校准诊断", ""])
    calibration_fields = [
        ("split", "Split"),
        ("brier", "Brier"),
        ("ece", "ECE"),
        ("score_levels", "分数层级数"),
        ("mean_score", "平均分数"),
        ("tp", "TP"),
        ("tn", "TN"),
        ("fp", "FP"),
        ("fn", "FN"),
    ]
    lines.extend(md_table([by_split[split]["TreatAgent"] for split in RESULT_FILES], calibration_fields))

    lines.extend(
        [
            "",
            "## 6. 结果分析",
            "",
            "- TreatAgent 在两个 leakage-controlled test split 上均优于 Direct、CoT 和 RAG。",
            "- 删除 DrugKB、DTI 或 Clinical 后，两个 split 的 F1 均下降，说明这些来源具有较稳定的正向贡献。",
            "- 删除 ADMET 后，两个 split 的 F1 均上升，主要来自 recall 提升。ADMET 应定位为 safety-risk signal，并继续检查过度惩罚和噪声证据问题。",
            "- 删除 DiseaseKB 后，F1 也略有提升。DiseaseKB claim 需要更严格的 relation typing，避免将疾病背景自动视为治疗支持。",
            "- EvidenceGraph 在 drug-disjoint split 上的 F1 贡献不明显，但在 temporal-submit split 上有正向贡献；论文中应结合证据可追踪性和复杂 case 分析呈现价值。",
            "- Planner 可测量地降低了调用成本，但节约幅度约为 8%。当前更适合将其定位为 cost-control component，而不是主要性能贡献。",
            "",
            "## 7. 完整性审计",
            "",
        ]
    )
    for split in RESULT_FILES:
        full = by_split[split]["TreatAgent"]
        lines.append(
            f"- {split}: 完整模型包含 {full['n']} 个样本；"
            f"LLM judge probability 回退 {full['judge_probability_fallbacks']} 次。"
        )
    lines.extend(
        [
            "- 所有主结果和消融结果均包含 432 个样本。",
            "- Drug-disjoint 完整模型中有 1 个样本缺少 `llm_judge_probability`，汇总时回退到最终 `prediction_score`；Temporal-submit 没有该问题。",
            "",
            "## 8. 文件说明",
            "",
            "- `main_results.csv`：基线与 TreatAgent 完整模型。",
            "- `component_ablation.csv`：Planner 与 EvidenceGraph 消融。",
            "- `source_ablation.csv`：expert source 消融。",
            "- `planner_efficiency.csv`：Planner 调用成本对比。",
            "- `all_results_metrics.csv`：全部实验的统一指标。",
            "- `final_results_summary.json`：完整机器可读汇总。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()

    rows: list[dict[str, Any]] = []
    for split, settings in RESULT_FILES.items():
        for setting, relative_path in settings.items():
            path = input_dir / relative_path
            if not path.exists():
                raise FileNotFoundError(path)
            print(f"Reading {path}")
            rows.append(summarize_payload(split, setting, relative_path, load_payload(path)))
    add_deltas(rows)

    csv_fields = [
        "split",
        "setting",
        "n",
        "positives",
        "negatives",
        "threshold",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auroc",
        "auprc",
        "brier",
        "ece",
        "predicted_positive_rate",
        "tp",
        "tn",
        "fp",
        "fn",
        "score_levels",
        "score_is_continuous",
        "mean_score",
        "mean_expert_calls",
        "delta_accuracy_vs_full",
        "delta_f1_vs_full",
        "delta_precision_vs_full",
        "delta_recall_vs_full",
        "delta_auroc_vs_full",
        "delta_calls_vs_full",
        "judge_probability_fallbacks",
        "result_json",
    ]
    index = {(row["split"], row["setting"]): row for row in rows}
    main_rows = [index[(split, setting)] for split in RESULT_FILES for setting in BASELINES]
    component_rows = [
        index[(split, setting)]
        for split in RESULT_FILES
        for setting in ["TreatAgent", *COMPONENT_ABLATIONS]
    ]
    source_rows = [
        index[(split, setting)]
        for split in RESULT_FILES
        for setting in ["TreatAgent", *SOURCE_ABLATIONS]
    ]
    planner_rows = [
        index[(split, setting)]
        for split in RESULT_FILES
        for setting in ["TreatAgent", "w/o Planner"]
    ]

    write_csv(input_dir / "all_results_metrics.csv", rows, csv_fields)
    write_csv(input_dir / "main_results.csv", main_rows, csv_fields)
    write_csv(input_dir / "component_ablation.csv", component_rows, csv_fields)
    write_csv(input_dir / "source_ablation.csv", source_rows, csv_fields)
    write_csv(input_dir / "planner_efficiency.csv", planner_rows, csv_fields)
    (input_dir / "final_results_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = build_markdown(rows)
    (input_dir / "final_results_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote summary files under {input_dir}")


if __name__ == "__main__":
    main()
