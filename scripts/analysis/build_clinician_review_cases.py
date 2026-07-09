import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "clinician_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINES = {
    "drug_disjoint_test": {
        "treatagent": ROOT / "results/final_results/baselines/results_multiagent_dd_test.json",
        "rag": ROOT / "results/final_results/baselines/results_rag_dd_test.json",
        "direct": ROOT / "results/final_results/baselines/results_direct_dd_test.json",
        "cot": ROOT / "results/final_results/baselines/results_cot_dd_test.json",
        "metadata": ROOT / "data/benchmark/splits/drug_disjoint_test.json",
    },
    "temporal_submit_test": {
        "treatagent": ROOT / "results/final_results/baselines/results_multiagent_ts_test.json",
        "rag": ROOT / "results/final_results/baselines/results_rag_ts_test.json",
        "direct": ROOT / "results/final_results/baselines/results_direct_ts_test.json",
        "cot": ROOT / "results/final_results/baselines/results_cot_ts_test.json",
        "metadata": ROOT / "data/benchmark/splits/temporal_submit_test.json",
    },
}


# 16 primary + 8 backup. The total set is balanced at 6 TP / 6 TN / 6 FP / 6 FN.
CASE_PLAN = [
    # Primary TP
    ("primary", "drug_disjoint_test", "PAIR-000828", "Direct support"),
    ("primary", "drug_disjoint_test", "PAIR-000526", "Repurposing / indication expansion"),
    ("primary", "drug_disjoint_test", "PAIR-000641", "Neuroimmune mechanism support"),
    ("primary", "temporal_submit_test", "PAIR-001194", "Mechanism-grounded endocrine support"),
    # Primary TN
    ("primary", "drug_disjoint_test", "PAIR-001938", "Negative oncology boundary"),
    ("primary", "drug_disjoint_test", "PAIR-001005", "Popular compound / insufficient evidence"),
    ("primary", "temporal_submit_test", "PAIR-001461", "Cardiovascular safety/conflict boundary"),
    ("primary", "temporal_submit_test", "PAIR-001824", "Infectious disease missing-evidence boundary"),
    # Primary FP
    ("primary", "drug_disjoint_test", "PAIR-000535", "Biomedical plausibility / label-complexity boundary"),
    ("primary", "drug_disjoint_test", "PAIR-000501", "Pulmonary vascular plausibility boundary"),
    ("primary", "temporal_submit_test", "PAIR-001855", "Oncology target plausibility boundary"),
    ("primary", "temporal_submit_test", "PAIR-000241", "Cardiovascular plausibility boundary"),
    # Primary FN
    ("primary", "temporal_submit_test", "PAIR-000164", "Missed cardiovascular positive"),
    ("primary", "temporal_submit_test", "PAIR-000240", "Missed pulmonary vascular positive"),
    ("primary", "temporal_submit_test", "PAIR-000496", "Missed oncology positive"),
    ("primary", "drug_disjoint_test", "PAIR-000737", "Missed neurology positive"),
    # Backup TP
    ("backup", "drug_disjoint_test", "PAIR-000700", "Rare disease direct evidence"),
    ("backup", "temporal_submit_test", "PAIR-001952", "Infectious disease support"),
    # Backup TN
    ("backup", "drug_disjoint_test", "PAIR-002079", "Oncology insufficient evidence"),
    ("backup", "temporal_submit_test", "PAIR-001089", "Neurology negative boundary"),
    # Backup FP
    ("backup", "drug_disjoint_test", "PAIR-000502", "Neuropsychiatry plausibility boundary"),
    ("backup", "temporal_submit_test", "PAIR-002042", "Inflammatory disease plausibility boundary"),
    # Backup FN
    ("backup", "drug_disjoint_test", "PAIR-001959", "Missed neurology positive"),
    ("backup", "drug_disjoint_test", "PAIR-001268", "Missed oncology positive"),
]


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_results(path):
    obj = load_json(path)
    return {r["sample_id"]: r for r in obj.get("results", [])}


def load_meta(path):
    return {r.get("pair_id", r.get("sample_id")): r for r in load_json(path)}


def safe_join(items, sep=" / "):
    return sep.join(str(x) for x in items if x)


def first_drug_name(r, meta=None):
    names = r.get("drug_names") or []
    if names:
        name = names[0]
        name = re.sub(r"^versus\s+", "", name, flags=re.I).strip()
        return name
    if meta and meta.get("drugs"):
        return meta["drugs"][0]
    return ""


def confusion(label, pred):
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    return "FN"


def decision_from_pred(pred):
    return "prioritize" if pred == 1 else "do not prioritize"


def area_and_specialty(disease):
    d = (disease or "").lower()
    if any(x in d for x in ["melanoma", "carcinoma", "cancer", "tumor", "neoplasm", "lymphoma", "leukemia", "mycosis", "glioblastoma"]):
        return "Oncology", "oncologist"
    if any(x in d for x in ["migraine", "multiple sclerosis", "epilepsy", "schizophrenia", "bipolar", "depression", "parkinson", "alzheimer", "neuropath", "seizure"]):
        return "Neurology / psychiatry", "neurologist or psychiatrist"
    if any(x in d for x in ["pulmonary hypertension", "hypertension", "heart", "coronary", "stroke", "vascular"]):
        return "Cardiovascular / pulmonary vascular", "cardiologist or pulmonologist"
    if any(x in d for x in ["hepatitis", "infection", "virus", "hiv", "influenza", "bacterial"]):
        return "Infectious disease", "infectious disease specialist"
    if any(x in d for x in ["lupus", "rheumatoid", "psoriasis", "colitis", "crohn", "arthritis", "inflammatory"]):
        return "Immunology / inflammatory", "rheumatologist, dermatologist, or gastroenterologist"
    if any(x in d for x in ["diabetes", "hyperparathyroidism", "obesity", "metabolic", "endocrine"]):
        return "Metabolic / endocrine", "endocrinologist"
    return "Other", "disease-area specialist"


def summarize_claims(claims, max_items=3):
    texts = []
    for c in claims or []:
        if isinstance(c, dict):
            txt = c.get("claim") or c.get("text") or c.get("reason")
        else:
            txt = str(c)
        if txt:
            texts.append(txt)
    return " | ".join(texts[:max_items])


def evidence_summaries(r):
    judge = r.get("llm_judge") or {}
    graph = r.get("evidence_graph") or {}
    ag = graph.get("argument_graph") or {}
    support = summarize_claims(ag.get("support_claims")) or safe_join(judge.get("key_support") or [])
    conflict = summarize_claims(ag.get("conflict_claims")) or safe_join(judge.get("key_conflict") or [])
    missing = summarize_claims(ag.get("missing_evidence")) or safe_join(judge.get("missing_evidence") or [])
    if not conflict:
        conflict = "No dominant conflict was reported in the structured evidence record."
    if not missing:
        missing = "No dominant missing-evidence item was reported in the structured evidence record."
    return support, conflict, missing


def evidence_pattern(r):
    factors = r.get("argument_factors") or {}
    judge = r.get("llm_judge") or {}
    support = factors.get("support_strength") or 0
    conflict = factors.get("conflict_strength") or 0
    missing_grade = judge.get("direct_evidence_grade") == "none" or judge.get("mechanistic_grounding_grade") in {"none", None}
    if support >= 0.45 and conflict >= 0.18:
        return "mixed support-conflict"
    if conflict >= 0.18 or judge.get("safety_conflict_grade") in {"significant", "severe"}:
        return "conflict-rich"
    if support >= 0.45:
        return "support-rich"
    if missing_grade:
        return "missing-evidence"
    return "mixed / uncertain"


def baseline_summary(r):
    if not r:
        return {"prediction": "", "score": "", "rationale": "", "available": False}
    return {
        "prediction": r.get("prediction_binary"),
        "score": r.get("prediction_score", r.get("calibrated_probability", "")),
        "rationale": r.get("rag_response") or r.get("synthesis_explanation") or r.get("reasoning") or r.get("explanation") or "",
        "available": True,
    }


def source_notes(meta):
    if not meta:
        return ""
    notes = []
    if meta.get("nctids"):
        notes.append("NCT IDs: " + safe_join(meta.get("nctids"), ", "))
    if meta.get("phases"):
        notes.append("trial phase(s): " + safe_join(meta.get("phases"), ", "))
    if meta.get("statuses"):
        notes.append("trial status: " + safe_join(meta.get("statuses"), ", "))
    if meta.get("pair_date"):
        notes.append(f"pair date: {meta.get('pair_date')} ({meta.get('pair_date_source')})")
    return "; ".join(notes)


all_data = {}
for split, cfg in BASELINES.items():
    all_data[split] = {
        "treatagent": load_results(cfg["treatagent"]),
        "rag": load_results(cfg["rag"]),
        "direct": load_results(cfg["direct"]),
        "cot": load_results(cfg["cot"]),
        "metadata": load_meta(cfg["metadata"]),
    }

rows = []
records = []
for selection_set, split, case_id, selection_reason in CASE_PLAN:
    data = all_data[split]
    r = data["treatagent"][case_id]
    meta = data["metadata"].get(case_id, {})
    drug = first_drug_name(r, meta)
    disease = r.get("disease")
    label = r.get("label")
    pred = r.get("prediction_binary")
    ctype = confusion(label, pred)
    area, specialty = area_and_specialty(disease)
    judge = r.get("llm_judge") or {}
    factors = r.get("argument_factors") or {}
    support, conflict, missing = evidence_summaries(r)
    pattern = evidence_pattern(r)
    rag = baseline_summary(data["rag"].get(case_id))
    direct = baseline_summary(data["direct"].get(case_id))
    cot = baseline_summary(data["cot"].get(case_id))
    baseline_available = rag["available"] and direct["available"] and cot["available"]
    trace = [t.get("selected_skill") for t in (r.get("trajectory") or []) if t.get("selected_skill")]
    rec = {
        "selection_set": selection_set,
        "case_id": case_id,
        "drug": drug,
        "drug_names_raw": safe_join(r.get("drug_names") or meta.get("drugs") or [], "; "),
        "disease": disease,
        "split": split,
        "benchmark_label": label,
        "treatagent_prediction": pred,
        "treatagent_decision": decision_from_pred(pred),
        "treatagent_score": r.get("llm_judge_probability"),
        "confusion_type": ctype,
        "disease_area": area,
        "suggested_specialty": specialty,
        "cross_specialty_review_scope": "evidence clarity, traceability, support/conflict/missing presentation, and usefulness for prioritization review",
        "specialist_review_scope": "clinical relevance, omitted key evidence, disease-specific plausibility, and whether prioritization is clinically reasonable for further investigation",
        "evidence_pattern": pattern,
        "evidence_grade": judge.get("evidence_grade"),
        "direct_evidence_grade": judge.get("direct_evidence_grade"),
        "mechanistic_grounding_grade": judge.get("mechanistic_grounding_grade"),
        "clinical_feasibility_grade": judge.get("clinical_feasibility_grade"),
        "safety_conflict_grade": judge.get("safety_conflict_grade"),
        "short_rationale": judge.get("grade_reason") or judge.get("reasoning_summary") or r.get("synthesis_explanation", ""),
        "support_summary": support,
        "conflict_summary": conflict,
        "missing_summary": missing,
        "source_provenance": source_notes(meta),
        "arg_score": factors.get("raw_argument_score"),
        "argument_probability": r.get("argument_probability"),
        "planner_trace": " -> ".join(trace),
        "expert_calls": len(trace),
        "rag_prediction": rag["prediction"],
        "rag_score": rag["score"],
        "rag_rationale": rag["rationale"],
        "direct_llm_prediction": direct["prediction"],
        "direct_llm_score": direct["score"],
        "direct_llm_rationale": direct["rationale"],
        "cot_prediction": cot["prediction"],
        "cot_score": cot["score"],
        "cot_rationale": cot["rationale"],
        "baseline_available": baseline_available,
        "selection_reason": selection_reason,
    }
    rows.append(rec)
    records.append({"record": rec, "treatagent": r, "rag": data["rag"].get(case_id), "direct": data["direct"].get(case_id), "cot": data["cot"].get(case_id), "metadata": meta})


csv_fields = [
    "selection_set", "case_id", "drug", "disease", "split", "benchmark_label", "treatagent_prediction",
    "treatagent_score", "confusion_type", "disease_area", "suggested_specialty", "evidence_pattern",
    "support_summary", "conflict_summary", "missing_summary", "rag_prediction", "rag_score",
    "direct_llm_prediction", "cot_prediction", "baseline_available", "selection_reason",
]
with (OUT_DIR / "selected_cases.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in csv_fields})


def md_table(rows_, columns):
    out = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows_:
        vals = []
        for col in columns:
            v = str(row.get(col, "")).replace("\n", " ").replace("|", "/")
            if len(v) > 160:
                v = v[:157] + "..."
            vals.append(v)
        out.append("|" + "|".join(vals) + "|")
    return "\n".join(out)


primary = [r for r in rows if r["selection_set"] == "primary"]
backup = [r for r in rows if r["selection_set"] == "backup"]
all_selected = rows

availability = f"""# Clinician expert evaluation case selection report

## Data availability report

可用信息如下：

| 信息类型 | 文件来源 | 备注 |
|---|---|---|
| benchmark label / trial-derived proxy label | `data/benchmark/splits/*_test.json`; `results/final_results/baselines/results_multiagent_*_test.json` | `label` 字段，表示 TOP-derived trial outcome proxy，不是直接临床疗效真值。 |
| drug name / disease name | `results/final_results/baselines/results_*_test.json`; `data/benchmark/splits/*_test.json` | `drug_names` 来自结果文件；split metadata 中保留 `drugs`、`canonical_smiles`、`nctids`。 |
| split 类型 | 文件名和 `split` 映射 | 当前使用 `drug_disjoint_test` 和 `temporal_submit_test`。 |
| TreatAgent prediction / score | `results/final_results/baselines/results_multiagent_dd_test.json`; `results/final_results/baselines/results_multiagent_ts_test.json` | 使用 `prediction_binary` 和 `llm_judge_probability`。 |
| TP/TN/FP/FN | 由 `label` 与 `prediction_binary` 计算 | 仅用于 case selection report 和 selected_cases.csv，不写入 blind packet。 |
| TreatAgent evidence summary / Evidence State | TreatAgent result JSON | `llm_judge`, `trajectory`, `expert_outputs`, `evidence_graph.argument_graph` 可用。 |
| support / conflict / missing evidence | TreatAgent result JSON | 来自 `argument_graph.support_claims`, `conflict_claims`, `missing_evidence`，不足时回退到 LLM judge key lists。 |
| EvidenceGraph / ARG score | TreatAgent result JSON | `argument_factors.raw_argument_score` 和 `argument_probability` 可用。 |
| RAG baseline output | `results/final_results/baselines/results_rag_dd_test.json`; `results/final_results/baselines/results_rag_ts_test.json` | `prediction_binary`, `prediction_score`, `rag_context`, `rag_response`, `synthesis_explanation` 可用。 |
| Direct LLM baseline output | `results/final_results/baselines/results_direct_*_test.json` | 只有 prediction/score；未保存 rationale。 |
| CoT baseline output | `results/final_results/baselines/results_cot_*_test.json` | 只有 prediction/score；未保存 rationale。 |
| 已用 case study | `results/final_results/case_study_candidates.csv`; `docs/test_case_study_options_for_advisor.md` | 用于避免和正文展示完全混淆，同时保留可解释性较好的代表案例。 |

缺失信息：

- Direct LLM 和 CoT baseline 没有保存 rationale，因此不能生成解释型 Version C。
- RAG baseline 有 rationale/evidence，可用于 baseline evidence summary。
- benchmark label 是 trial-derived proxy，不适合作为医生盲评中的“正确答案”展示。
- 部分原始 TOP intervention 名称含 combination/comparator 信息；本 case set 已尽量避免明显组合疗法或 mapping 不一致样本。

## Selection logic

本 case set 面向 clinician expert review，而不是模型排行榜。筛选原则：

- 每个 case 必须能在现有 held-out test result 中定位。
- 主集 16 个 case：TP/TN/FP/FN 各 4 个。
- 备选集 8 个 case：TP/TN/FP/FN 各 2 个。
- 总计 24 个 case：TP/TN/FP/FN 各 6 个。
- 覆盖 drug-disjoint 和 temporal-submit 两种 split。
- 覆盖 neurology/neuropsychiatry、oncology、cardiovascular/pulmonary vascular、infectious disease、immunology/inflammatory、metabolic/endocrine 等领域。
- 包含 support-rich、conflict-rich、missing-evidence、mixed support-conflict 等 evidence state。
- 排除明显无法解释的空药名、明显组合/安慰剂/治疗方案类名称、过短 evidence record 或难以进行医生评估的样本。

## Primary case set distribution

- total primary cases: {len(primary)}
- split distribution: {dict(Counter(r["split"] for r in primary))}
- confusion distribution: {dict(Counter(r["confusion_type"] for r in primary))}
- disease-area distribution: {dict(Counter(r["disease_area"] for r in primary))}
- evidence-pattern distribution: {dict(Counter(r["evidence_pattern"] for r in primary))}

## Backup case distribution

- total backup cases: {len(backup)}
- split distribution: {dict(Counter(r["split"] for r in backup))}
- confusion distribution: {dict(Counter(r["confusion_type"] for r in backup))}
- disease-area distribution: {dict(Counter(r["disease_area"] for r in backup))}
- evidence-pattern distribution: {dict(Counter(r["evidence_pattern"] for r in backup))}

## Primary cases

{md_table(primary, ["case_id", "drug", "disease", "split", "confusion_type", "disease_area", "suggested_specialty", "evidence_pattern", "selection_reason"])}

## Backup cases

{md_table(backup, ["case_id", "drug", "disease", "split", "confusion_type", "disease_area", "suggested_specialty", "evidence_pattern", "selection_reason"])}

## Why these cases fit clinician expert evaluation

这些 case 不是为了证明药物可以临床使用，而是为了让临床专家判断：structured evidence record 是否比 decision-only 或 baseline evidence 更容易审查 drug repurposing candidate。主集同时包含正确预测和错误预测，避免只展示系统表现好的案例。错误预测 case 对医生评估尤其重要，因为它们能检验 evidence report 是否暴露了不确定性、证据缺口和潜在冲突。

## Cross-specialty versus specialty-dependent review

可跨科室评价的内容：

- evidence 是否清楚分成 support、conflict、missing；
- 是否能追踪到来源或专家模块；
- decision-only 与 structured evidence 哪个更利于审查；
- 是否明确表达“prioritize for further investigation”，而不是临床用药建议。

需要专科医生评价的内容：

- disease-specific clinical relevance；
- 是否遗漏关键治疗证据；
- safety 或 mechanism conflict 的临床重要性；
- prioritization 是否符合该疾病领域的候选筛选逻辑。

## Main risks

- TOP-derived label 是 trial outcome proxy，医生评估不能被设计成判断“临床疗效真值”。
- Direct/CoT 缺少 rationale，无法与 structured evidence 做完整解释型对照。
- RAG baseline 有 evidence/rationale，但其格式与 TreatAgent structured record 不完全一致。
- 个别样本可能仍含原始 trial intervention naming noise；医生包中保留 drug/disease 和 evidence，不暴露 benchmark label。
"""
(OUT_DIR / "case_selection_report.md").write_text(availability, encoding="utf-8")


blind_lines = ["# Blind clinician review packet", ""]
blind_lines.append("This packet is intended for expert review of drug repurposing candidate prioritization for further investigation. It does not provide clinical treatment recommendations, trial outcome labels, correctness categories, or system names.")
for i, row in enumerate(rows, 1):
    blind_lines += [
        "",
        f"## Case {i:02d}: {row['case_id']}",
        "",
        f"- Drug: {row['drug']}",
        f"- Disease: {row['disease']}",
        "",
        "### Version A: decision-only",
        "",
        f"- Decision: {row['treatagent_decision']}",
        f"- Confidence score: {row['treatagent_score']}",
        "",
        "### Version B: structured evidence",
        "",
        f"- Decision: {row['treatagent_decision']}",
        f"- Confidence score: {row['treatagent_score']}",
        f"- Supporting evidence: {row['support_summary']}",
        f"- Conflicting or risk evidence: {row['conflict_summary']}",
        f"- Missing or uncertain evidence: {row['missing_summary']}",
        f"- Source/provenance notes: {row['source_provenance'] or 'Structured source notes are available from the evidence record; no additional trial metadata shown in blind packet.'}",
    ]
    if row["rag_prediction"] != "":
        baseline_decision = decision_from_pred(int(row["rag_prediction"]))
        rationale = row["rag_rationale"] or "No baseline rationale was stored for this case."
        blind_lines += [
            "",
            "### Version C: baseline evidence summary",
            "",
            f"- Decision: {baseline_decision}",
            f"- Baseline rationale/evidence: {rationale}",
        ]
(OUT_DIR / "blind_review_packet.md").write_text("\n".join(blind_lines), encoding="utf-8")


missing_lines = ["# Baseline missing report", ""]
missing_lines.append("## Summary")
missing_lines.append("")
missing_lines.append("- RAG baseline output is available for all selected cases.")
missing_lines.append("- Direct LLM and CoT predictions are available for all selected cases.")
missing_lines.append("- Direct LLM and CoT rationale fields are not present in the stored baseline JSON files.")
missing_lines.append("")
missing_lines.append("## Case-level baseline availability")
missing_lines.append("")
missing_lines.append(md_table(rows, ["selection_set", "case_id", "split", "rag_prediction", "direct_llm_prediction", "cot_prediction", "baseline_available"]))
missing_lines.append("")
missing_lines.append("## Missing baseline rationale")
missing_lines.append("")
missing_lines.append("Direct LLM and CoT output files contain `prediction_binary` and `prediction_score`, but no free-text rationale or evidence field. If explanation-level blind comparison is required, rerun Direct and CoT baselines with response logging enabled.")
missing_lines.append("")
missing_lines.append("Recommended rerun scripts for reproducing baseline predictions:")
missing_lines.append("")
missing_lines.append("```bash")
missing_lines.append("BACKBONE=gpt-4o bash scripts/baselines/submit_drug_disjoint_gpt4o_baselines.sh")
missing_lines.append("BACKBONE=gpt-4o bash scripts/baselines/submit_temporal_submit_gpt4o_baselines.sh")
missing_lines.append("```")
missing_lines.append("")
missing_lines.append("These scripts reproduce Direct, CoT, and RAG predictions. The current Direct and CoT result schema does not store raw model rationale. If explanation-level blind comparison is required, extend `treatagent.cli` / baseline response saving to store the raw model response, parsed prediction, and short rationale for Direct and CoT.")
(OUT_DIR / "baseline_missing_report.md").write_text("\n".join(missing_lines), encoding="utf-8")


print("Wrote:")
print(OUT_DIR / "case_selection_report.md")
print(OUT_DIR / "selected_cases.csv")
print(OUT_DIR / "blind_review_packet.md")
print(OUT_DIR / "baseline_missing_report.md")
print("Primary confusion:", dict(Counter(r["confusion_type"] for r in primary)))
print("Primary split:", dict(Counter(r["split"] for r in primary)))
print("Primary area:", dict(Counter(r["disease_area"] for r in primary)))
print("All confusion:", dict(Counter(r["confusion_type"] for r in all_selected)))
print("All split:", dict(Counter(r["split"] for r in all_selected)))
print("All area:", dict(Counter(r["disease_area"] for r in all_selected)))
