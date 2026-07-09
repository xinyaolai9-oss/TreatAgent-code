#!/usr/bin/env python3
"""Build hard-subset diagnostics and case-study recommendations from frozen results.

This script is deliberately analysis-only: it reads results/final_results and
never invokes TreatAgent, an LLM API, or any external service.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results" / "final_results"

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

MAIN_METHODS = ["Direct", "CoT", "RAG", "TreatAgent"]

SUBSET_DEFINITIONS: dict[str, tuple[str, Callable[[dict[str, Any]], bool]]] = {
    "direct_indication": (
        "direct_support > 0",
        lambda row: factor(row, "direct_support") > 0,
    ),
    "no_direct_indication": (
        "direct_support = 0",
        lambda row: factor(row, "direct_support") <= 1e-9,
    ),
    "mechanism_supported_without_direct": (
        "direct_support = 0 and mechanism_support >= 0.35",
        lambda row: factor(row, "direct_support") <= 1e-9 and factor(row, "mechanism_support") >= 0.35,
    ),
    "high_safety_conflict": (
        "safety_conflict >= 0.25",
        lambda row: factor(row, "safety_conflict") >= 0.25,
    ),
    "support_conflict_coexist": (
        "support_strength >= 0.30 and conflict_strength >= 0.15",
        lambda row: factor(row, "support_strength") >= 0.30 and factor(row, "conflict_strength") >= 0.15,
    ),
    "low_clinical_prior": (
        "clinical_feasibility < 0.35",
        lambda row: factor(row, "clinical_feasibility") < 0.35,
    ),
    "planner_early_stop": (
        "expert calls < 5",
        lambda row: expert_calls(row) < 5,
    ),
}

SUBSET_ABLATION_FOCUS = {
    "mechanism_supported_without_direct": ["TreatAgent", "w/o DTI", "w/o DrugKB", "w/o EvidenceGraph"],
    "high_safety_conflict": ["TreatAgent", "w/o ADMET", "w/o EvidenceGraph"],
    "support_conflict_coexist": ["TreatAgent", "w/o ADMET", "w/o EvidenceGraph"],
    "low_clinical_prior": ["TreatAgent", "w/o Clinical", "w/o DTI"],
    "planner_early_stop": ["TreatAgent", "w/o Planner"],
}

CASE_PLAN = [
    {
        "placement": "Main Figure 4A",
        "split": "Drug-disjoint",
        "sample_id": "PAIR-000828",
        "case_type": "Common disease / direct support positive",
        "reason": "High-confidence sanity check with recognizable drug and disease; direct indication and mechanistic context are easy to audit.",
    },
    {
        "placement": "Main Figure 4B",
        "split": "Drug-disjoint",
        "sample_id": "PAIR-001194",
        "case_type": "Target-grounded positive",
        "reason": "CASR target convergence shows how DrugKB, DiseaseKB and DTI are organized into a disease-relevant argument.",
    },
    {
        "placement": "Main Figure 4C",
        "split": "Drug-disjoint",
        "sample_id": "PAIR-001938",
        "case_type": "Negative boundary",
        "reason": "Cancer-related surface plausibility is rejected because direct evidence and mechanistic grounding are missing while ADMET risks accumulate.",
    },
    {
        "placement": "Supplementary Case S1",
        "split": "Drug-disjoint",
        "sample_id": "PAIR-000700",
        "case_type": "Rare disease / early STOP positive",
        "reason": "Direct indication and high clinical feasibility allow the Planner to stop after two expert calls.",
    },
    {
        "placement": "Supplementary Case S2",
        "split": "Drug-disjoint",
        "sample_id": "PAIR-001005",
        "case_type": "Popular compound / insufficient evidence negative",
        "reason": "Curcumin is not prioritized for bipolar disorder despite apparent biological plausibility because evidence remains indirect and safety conflicts are present.",
    },
    {
        "placement": "Supplementary Case S3",
        "split": "Drug-disjoint",
        "sample_id": "PAIR-000535",
        "case_type": "Failure boundary / label complexity",
        "reason": "Hydroxychloroquine-related SLE false positive illustrates tension between biomedical plausibility and clinical-trial-derived labels.",
    },
    {
        "placement": "Supplementary Case S4",
        "split": "Drug-disjoint",
        "sample_id": "PAIR-000921",
        "case_type": "Failure boundary / safety over-penalization",
        "reason": "Sofosbuvir/velpatasvir-related HCV false negative illustrates the current ADMET over-penalization boundary.",
    },
]


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


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Unsupported result JSON: {path}")
    return payload["results"]


def by_sample_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id")): row for row in rows}


def factor(row: dict[str, Any], name: str) -> float:
    return safe_float((row.get("argument_factors") or {}).get(name), 0.0)


def expert_calls(row: dict[str, Any]) -> int:
    outputs = row.get("expert_outputs") or {}
    return len(outputs) if isinstance(outputs, dict) else 0


def prediction(row: dict[str, Any]) -> int:
    return safe_int(row.get("prediction_binary"))


def label(row: dict[str, Any]) -> int:
    return safe_int(row.get("label"))


def judge_score(row: dict[str, Any]) -> float:
    if row.get("llm_judge_probability") is not None:
        return safe_float(row["llm_judge_probability"])
    return safe_float(row.get("prediction_score"))


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [label(row) for row in rows]
    y_pred = [prediction(row) for row in rows]
    if not rows:
        return {"n": 0, "positive": 0, "negative": 0, "accuracy": None, "f1": None, "precision": None, "recall": None}
    return {
        "n": len(rows),
        "positive": sum(y_true),
        "negative": len(y_true) - sum(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def md_table(rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label_text for _, label_text in fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_fmt(row.get(key)) for key, _ in fields) + " |")
    return lines


def compact_claims(row: dict[str, Any], direction: str, limit: int = 3) -> str:
    graph = (row.get("evidence_graph") or {}).get("argument_graph") or {}
    key = "support_claims" if direction == "support" else "conflict_claims"
    claims = [str(item.get("claim") or "") for item in graph.get(key) or [] if item.get("claim")]
    return " / ".join(claims[:limit])


def planner_trace(row: dict[str, Any]) -> str:
    trace = []
    for item in row.get("trajectory") or []:
        planner = item.get("planner_output") or {}
        action = planner.get("next_action")
        if action:
            trace.append(str(action))
    return " -> ".join(trace)


def build_analysis(result_dir: Path) -> dict[str, Any]:
    tables = {
        split: {
            method: by_sample_id(load_rows(result_dir / relative_path))
            for method, relative_path in settings.items()
        }
        for split, settings in RESULT_FILES.items()
    }

    subset_metrics: list[dict[str, Any]] = []
    subset_ablation: list[dict[str, Any]] = []
    subset_stats: list[dict[str, Any]] = []
    for split, split_tables in tables.items():
        full = split_tables["TreatAgent"]
        masks = {
            name: {sample_id for sample_id, row in full.items() if predicate(row)}
            for name, (_, predicate) in SUBSET_DEFINITIONS.items()
        }
        for subset, ids in masks.items():
            selected = [full[sample_id] for sample_id in ids]
            subset_stats.append(
                {
                    "split": split,
                    "subset": subset,
                    "definition": SUBSET_DEFINITIONS[subset][0],
                    **metrics(selected),
                }
            )
            for method in MAIN_METHODS:
                rows = [split_tables[method][sample_id] for sample_id in ids if sample_id in split_tables[method]]
                subset_metrics.append({"split": split, "subset": subset, "method": method, **metrics(rows)})
            if subset in SUBSET_ABLATION_FOCUS:
                full_metric = metrics(selected)
                for setting in SUBSET_ABLATION_FOCUS[subset]:
                    rows = [split_tables[setting][sample_id] for sample_id in ids if sample_id in split_tables[setting]]
                    row_metric = metrics(rows)
                    subset_ablation.append(
                        {
                            "split": split,
                            "subset": subset,
                            "setting": setting,
                            **row_metric,
                            "delta_f1_vs_full": round(safe_float(row_metric.get("f1")) - safe_float(full_metric.get("f1")), 6),
                        }
                    )

    cases: list[dict[str, Any]] = []
    for plan in CASE_PLAN:
        row = tables[plan["split"]]["TreatAgent"].get(plan["sample_id"])
        if not row:
            raise ValueError(f"Case not found: {plan}")
        judge = row.get("llm_judge") or {}
        cases.append(
            {
                **plan,
                "drug": ", ".join(row.get("drug_names") or []) or str(row.get("smiles") or ""),
                "disease": row.get("disease"),
                "label": label(row),
                "prediction": prediction(row),
                "judge_score": round(judge_score(row), 6),
                "expert_calls": expert_calls(row),
                "planner_trace": planner_trace(row),
                "direct_support": round(factor(row, "direct_support"), 6),
                "mechanism_support": round(factor(row, "mechanism_support"), 6),
                "clinical_feasibility": round(factor(row, "clinical_feasibility"), 6),
                "safety_conflict": round(factor(row, "safety_conflict"), 6),
                "support_strength": round(factor(row, "support_strength"), 6),
                "conflict_strength": round(factor(row, "conflict_strength"), 6),
                "evidence_grade": judge.get("evidence_grade"),
                "judge_reason": judge.get("grade_reason"),
                "support_claims": compact_claims(row, "support"),
                "conflict_claims": compact_claims(row, "conflict"),
            }
        )
    return {
        "subset_definitions": {name: definition for name, (definition, _) in SUBSET_DEFINITIONS.items()},
        "subset_stats": subset_stats,
        "subset_metrics": subset_metrics,
        "subset_ablation": subset_ablation,
        "case_plan": cases,
    }


def build_markdown(analysis: dict[str, Any]) -> str:
    subset_stats = analysis["subset_stats"]
    subset_metrics = analysis["subset_metrics"]
    subset_ablation = analysis["subset_ablation"]
    cases = analysis["case_plan"]
    lines = [
        "# 冻结结果的 subset 与 case-study 分析",
        "",
        "本分析只读取 `results/final_results/`，不重新运行模型，不调用 API，也不修改框架。",
        "Subset 使用归一化 argument factors 上的固定、可解释阈值定义。它们用于解释模型行为和边界，不应表述为预注册的 confirmatory endpoint。",
        "",
        "## 1. 推荐叙事",
        "",
        "TreatAgent 面向 clinical-trial-derived drug repurposing evidence triage：在直接适应症证据稀缺、机制证据分散且安全信号可能冲突的情况下，系统通过 Planner 按需调用专家，将多源事实组织为可审计的 Argument EvidenceGraph，再由 constrained LLM judge 输出候选优先级与解释。",
        "",
        "主结果支撑三个结论：",
        "",
        "1. TreatAgent 在 drug-disjoint 与 temporal-submit 设置中均优于 Direct、CoT 和 RAG。",
        "2. TreatAgent 的优势主要体现在缺少 direct indication、但存在可整合机制信号的复杂证据状态。",
        "3. Planner 可以减少部分 expert 调用；ADMET 与 EvidenceGraph 更适合作为风险审查和可审计性层，而不是包装成对所有样本都提高 F1 的模块。",
        "",
        "## 2. Subset 定义与规模",
        "",
    ]
    lines.extend(
        md_table(
            subset_stats,
            [
                ("split", "Split"),
                ("subset", "Subset"),
                ("definition", "定义"),
                ("n", "N"),
                ("positive", "Positive"),
                ("negative", "Negative"),
            ],
        )
    )
    lines.extend(["", "## 3. Baseline subset 表现", ""])
    lines.extend(
        md_table(
            subset_metrics,
            [
                ("split", "Split"),
                ("subset", "Subset"),
                ("method", "方法"),
                ("n", "N"),
                ("accuracy", "Accuracy"),
                ("f1", "F1"),
                ("precision", "Precision"),
                ("recall", "Recall"),
            ],
        )
    )
    lines.extend(["", "## 4. 相关模块诊断", ""])
    lines.extend(
        md_table(
            subset_ablation,
            [
                ("split", "Split"),
                ("subset", "Subset"),
                ("setting", "设置"),
                ("n", "N"),
                ("accuracy", "Accuracy"),
                ("f1", "F1"),
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("delta_f1_vs_full", "相对完整模型 Delta F1"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 5. Case-study 安排",
            "",
            "主文 Figure 4 建议只保留三个横向小面板。补充材料再展开 rare disease、热门化合物和 failure boundary。",
            "",
        ]
    )
    lines.extend(
        md_table(
            cases,
            [
                ("placement", "位置"),
                ("sample_id", "Sample"),
                ("drug", "Drug"),
                ("disease", "Disease"),
                ("label", "Label"),
                ("prediction", "Pred"),
                ("judge_score", "Score"),
                ("expert_calls", "Calls"),
                ("case_type", "案例类型"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 6. 写作边界",
            "",
            "- 不要写成“所有 expert 均提高平均性能”。最终消融不支持这个结论。",
            "- 可以写 DrugKB、DTI 和 Clinical 对两个 split 的整体 F1 有稳定贡献。",
            "- 可以写 DTI 对 `mechanism_supported_without_direct` subset 的价值更明显。",
            "- 可以写 Planner 在 `planner_early_stop` subset 中减少调用，并保留较高预测表现。",
            "- ADMET 应写成 safety-risk review layer：它能排除部分风险候选，但当前也存在 over-penalization boundary。",
            "- EvidenceGraph 应写成 auditable evidence organization layer：它在 temporal-submit 上改善 F1 和 AUROC，但 drug-disjoint 上主要体现 precision-risk trade-off 与可解释性。",
            "- `PAIR-000535` 和 `PAIR-000921` 必须明确标注为 failure cases，用于讨论 trial-derived label complexity 与 safety over-penalization。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    analysis = build_analysis(result_dir)
    write_csv(result_dir / "subset_stats.csv", analysis["subset_stats"])
    write_csv(result_dir / "subset_metrics.csv", analysis["subset_metrics"])
    write_csv(result_dir / "subset_ablation_diagnostics.csv", analysis["subset_ablation"])
    write_csv(result_dir / "case_study_candidates.csv", analysis["case_plan"])
    (result_dir / "subset_case_study_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = build_markdown(analysis)
    (result_dir / "subset_case_study_analysis.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote subset and case-study analysis under {result_dir}")


if __name__ == "__main__":
    main()
