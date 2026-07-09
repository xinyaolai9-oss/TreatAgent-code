#!/usr/bin/env python3
"""Build a WJX/QuestionStar import-ready clinician review questionnaire.

The questionnaire is intentionally method-blinded. It does not expose
benchmark labels, confusion types, method names, trial IDs, or trial status.
An internal A/B mapping CSV is generated separately for analysis.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "clinician_review"
SELECTED_CASES = OUT_DIR / "selected_cases.csv"
RAG_DD = ROOT / "results" / "final_results" / "baselines" / "results_rag_dd_test.json"
RAG_TS = ROOT / "results" / "final_results" / "baselines" / "results_rag_ts_test.json"
TA_DD = ROOT / "results" / "final_results" / "baselines" / "results_multiagent_dd_test.json"
TA_TS = ROOT / "results" / "final_results" / "baselines" / "results_multiagent_ts_test.json"

DOCX_OUT = OUT_DIR / "wjx_import_questionnaire.docx"
MD_OUT = OUT_DIR / "wjx_import_questionnaire.md"
CASE_OUT = OUT_DIR / "wjx_selected_cases.csv"
MAP_OUT = OUT_DIR / "wjx_ab_mapping.csv"
TRACE_OUT = OUT_DIR / "wjx_structured_evidence_trace.csv"


ERROR_CASE_PRIORITY = [
    "PAIR-000535",  # hydroxychloroquine - SLE
    "PAIR-000501",  # macitentan - pulmonary arterial hypertension
    "PAIR-000164",  # aspirin - acute coronary syndrome
    "PAIR-000496",  # atezolizumab - bladder cancer
]

CORRECT_CASE_ORDER = [
    "PAIR-000526",  # sildenafil - pulmonary hypertension
    "PAIR-000641",  # siponimod - MS
    "PAIR-000828",  # sumatriptan - migraine
    "PAIR-001194",  # evocalcet - secondary hyperparathyroidism
    "PAIR-001005",  # curcumin - bipolar disorder
    "PAIR-001461",  # febuxostat - coronary artery disease
    "PAIR-001824",  # ruxolitinib - HIV infection
    "PAIR-001006",  # zafirlukast - breast cancer; replaces API-error case
    "PAIR-000700",  # mechlorethamine - mycosis fungoides
    "PAIR-001952",  # dolutegravir - HIV
    "PAIR-001089",  # tozadenant - Parkinson disease
    "PAIR-002079",  # niclosamide - colon cancer
]

DISPLAY_NAMES = {
    "sildenafil citrate": "Sildenafil",
    "baf312": "Siponimod (BAF312)",
    "sumatriptan": "Sumatriptan",
    "khk7580": "Evocalcet (KHK7580)",
    "curcumin": "Curcumin",
    "febuxostat": "Febuxostat",
    "ruxolitinib": "Ruxolitinib",
    "0.04% mechlorethamine gel": "Mechlorethamine",
    "dolutegravir": "Dolutegravir",
    "tozadenant": "Tozadenant",
    "niclosamide": "Niclosamide",
    "hydroxychloroquine": "Hydroxychloroquine",
    "macitentan": "Macitentan",
    "aspirin": "Aspirin",
    "atezolizumab": "Atezolizumab",
    "zafirlukast": "Zafirlukast",
}

DISEASE_DISPLAY_NAMES = {
    "relapsing remitting multiple sclerosis": "Relapsing-remitting multiple sclerosis",
    "secondary hyperparathyroidism": "Secondary hyperparathyroidism",
    "hiv infection": "HIV infection",
    "hiv": "HIV",
    "idiopathic parkinson disease": "Idiopathic Parkinson disease",
    "systemic lupus erythematosus": "Systemic lupus erythematosus",
    "acute coronary syndrome": "Acute coronary syndrome",
}

EXTRA_ROWS = {
    "PAIR-001006": {
        "selection_set": "wjx_extra",
        "case_id": "PAIR-001006",
        "drug": "zafirlukast",
        "disease": "breast cancer",
        "split": "drug_disjoint_test",
        "benchmark_label": "0",
        "treatagent_prediction": "0",
        "treatagent_score": "0.11",
        "confusion_type": "TN",
        "disease_area": "Oncology",
        "suggested_specialty": "oncologist or clinical pharmacologist",
        "evidence_pattern": "conflict-rich / missing evidence",
        "support_summary": "Limited indirect disease-target context was observed, but no direct drug-side indication support was derived for breast cancer.",
        "conflict_summary": "Predicted drug-induced liver injury risk and hERG blocking risk indicate safety and developability concerns.",
        "missing_summary": "No direct indication support and no strong disease-specific mechanism grounding were identified.",
        "rag_prediction": "0",
        "rag_score": "0",
        "direct_llm_prediction": "",
        "cot_prediction": "",
        "baseline_available": "True",
        "selection_reason": "Negative oncology boundary case replacing a baseline API-error case",
    }
}


def display_drug_name(name: str) -> str:
    return DISPLAY_NAMES.get(name.strip().lower(), name.strip())


def display_disease_name(name: str) -> str:
    clean = name.strip()
    return DISEASE_DISPLAY_NAMES.get(clean.lower(), clean[:1].upper() + clean[1:])


def disease_matches_claim(disease: str, claim: str) -> bool:
    generic = {
        "disease",
        "disorder",
        "infection",
        "cancer",
        "carcinoma",
        "syndrome",
        "stage",
        "acute",
        "chronic",
        "arterial",
        "relapsing",
        "remitting",
    }
    disease_tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", disease.lower())
        if len(tok) > 2 and tok not in generic
    }
    if not disease_tokens:
        disease_tokens = {tok for tok in re.findall(r"[a-z0-9]+", disease.lower()) if len(tok) > 2}
    claim_tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
    if not disease_tokens:
        return False
    return len(disease_tokens & claim_tokens) / len(disease_tokens) >= 0.67


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rag_map(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {row["sample_id"]: row for row in data.get("results", [])}


def load_result_map(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {row["sample_id"]: row for row in data.get("results", [])}


def clean_text(text: str | None) -> str:
    if not text:
        return "No clear evidence summary was available in the available evidence."
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if re.search(r"api call failed|^error\s*:", text, flags=re.I):
        return "No alternative evidence summary was available for this case."
    text = re.sub(r"\bTreatAgent\b", "the structured evidence system", text, flags=re.I)
    text = re.sub(r"\bRAG\b", "the evidence summary", text, flags=re.I)
    text = re.sub(r"\bbenchmark label\b", "reference label", text, flags=re.I)
    text = re.sub(r"\bTP\b|\bTN\b|\bFP\b|\bFN\b", "", text)
    text = re.sub(r"\bNCT\d+\b", "trial record", text)
    text = re.sub(r"\b(label|ground truth)\s*=\s*[01]\b", "", text, flags=re.I)
    text = re.sub(r"\bin the dataset\b", "in the available evidence", text, flags=re.I)
    text = re.sub(r"\bdataset\b", "available evidence", text, flags=re.I)
    text = re.sub(r"\befficacy\b", "clinical support", text, flags=re.I)
    text = re.sub(r"\bmodel\b", "system", text, flags=re.I)
    text = re.sub(r"\bcurrent output\b", "available evidence", text, flags=re.I)
    text = re.sub(r"\bcould effectively treat\b", "may warrant prioritization for further investigation in", text, flags=re.I)
    text = re.sub(r"\bcan effectively treat\b", "may warrant prioritization for further investigation in", text, flags=re.I)
    text = re.sub(r"\bcould treat\b", "may warrant prioritization for further investigation in", text, flags=re.I)
    text = re.sub(r"\bcan treat\b", "may warrant prioritization for further investigation in", text, flags=re.I)
    text = re.sub(r"\bability to treat\b", "evidence supporting prioritization for", text, flags=re.I)
    text = re.sub(r"\btherapeutic potential\b", "prioritization potential", text, flags=re.I)
    text = re.sub(r"\btherapeutic relevance\b", "prioritization relevance", text, flags=re.I)
    text = re.sub(r"\binteracts effectively with\b", "shows sufficiently strong interaction evidence for", text, flags=re.I)
    text = re.sub(r"\btherapeutic role\b", "prioritization role", text, flags=re.I)
    text = re.sub(r"\btherapeutic effect\b", "prioritization support", text, flags=re.I)
    text = re.sub(r"\btherapeutic targets\b", "disease-relevant targets", text, flags=re.I)
    text = re.sub(r"\btherapeutic target\b", "disease-relevant target", text, flags=re.I)
    text = re.sub(r"\btherapeutic mechanism\b", "mechanistic context", text, flags=re.I)
    text = re.sub(r"\btherapeutic plausibility\b", "prioritization plausibility", text, flags=re.I)
    text = re.sub(r"\btherapeutic claims\b", "prioritization claims", text, flags=re.I)
    text = re.sub(r"\btherapeutic option\b", "candidate for further investigation", text, flags=re.I)
    text = re.sub(r"\beffective or approved for treating\b", "sufficiently supported for prioritization in", text, flags=re.I)
    text = re.sub(r"\beffective for treating\b", "sufficiently supported for prioritization in", text, flags=re.I)
    text = re.sub(r"\befficacy of this drug in treating\b", "prioritization support for", text, flags=re.I)
    text = re.sub(r"\btreatment of\b", "prioritization for further investigation in", text, flags=re.I)
    text = re.sub(r"\bclinically relevant therapies\b", "clinically relevant comparator therapies", text, flags=re.I)
    text = re.sub(r"\bclinical support of this drug in prioritizing for further investigation in\b", "evidence supporting prioritization for", text, flags=re.I)
    text = re.sub(r"\bclinical support in\b", "prioritization support in", text, flags=re.I)
    text = re.sub(r"\bclinical support for\b", "prioritization support for", text, flags=re.I)
    text = re.sub(r"\bclinical support\b", "prioritization support", text, flags=re.I)
    text = re.sub(r"\bsupports its use for\b", "supports prioritization for further investigation in", text, flags=re.I)
    text = re.sub(r"\bHIV treatment\b", "HIV clinical context", text, flags=re.I)
    text = re.sub(r"\bRRMS treatment\b", "RRMS clinical context", text, flags=re.I)
    text = re.sub(r"\btreatments for\b", "standard clinical options in", text, flags=re.I)
    text = re.sub(r"\btreatment evidence\b", "drug-specific prioritization evidence", text, flags=re.I)
    text = re.sub(r"\btreating\b", "prioritizing for further investigation in", text, flags=re.I)
    text = re.sub(r"\btreat\b", "support prioritization for", text, flags=re.I)
    text = re.sub(r"\bdrug-specific drug-specific\b", "drug-specific", text, flags=re.I)
    text = re.sub(r"\bprioritization relevance for prioritizing for further investigation in\b", "prioritization relevance for", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ;")
    return text or "No clear evidence summary was available in the available evidence."


def strip_rag_response(text: str | None) -> str:
    text = clean_text(text)
    # Keep clinician-facing evidence content, not the machine-readable final answer.
    text = re.sub(r"ANSWER\s*:\s*[01].*$", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"^ANALYSIS\s*:\s*", "", text, flags=re.I).strip()
    text = re.sub(r"^Based on the retrieved evidence,?\s*", "", text, flags=re.I)
    return clean_text(text)


def decision_from_prediction(pred: str) -> str:
    return "Prioritize for further investigation" if str(pred) == "1" else "Do not prioritize for further investigation"


def source_label(evidence: dict[str, Any]) -> str:
    expert = evidence.get("expert", "")
    category = evidence.get("category", "")
    role = evidence.get("semantic_role", "")
    if expert == "DrugKB":
        return "Drug knowledge"
    if expert == "DiseaseKB":
        return "Disease biology"
    if expert == "DTI":
        return "Mechanism evidence"
    if expert == "ADMET" or "safety" in role or "toxicity" in category:
        return "Safety/developability"
    if expert == "Clinical" or "clinical" in role:
        return "Clinical feasibility"
    if "missing" in role:
        return "Evidence gap"
    return "Evidence"


def bullet_text(label: str, claim: str) -> str:
    return f"- {label}: {clean_text(claim)}"


def evidence_priority(evidence: dict[str, Any]) -> tuple[int, float]:
    claim = str(evidence.get("claim", "")).lower()
    category = str(evidence.get("category", "")).lower()
    expert = str(evidence.get("expert", ""))
    confidence = float(evidence.get("confidence") or evidence.get("reliability") or 0)
    if expert == "DrugKB" and ("indication" in claim or "overlap" in claim):
        return (0, -confidence)
    if expert == "DrugKB" and ("mechanism" in category or "target" in claim):
        return (1, -confidence)
    if expert == "DTI" or "mechanism" in category or "target" in claim:
        return (2, -confidence)
    if expert == "Clinical":
        return (3, -confidence)
    if expert == "ADMET":
        return (4, -confidence)
    return (5, -confidence)


def norm_for_dedupe(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def append_unique_bullet(
    target: list[str],
    seen: set[str],
    bullet: str,
    min_unique_chars: int = 30,
) -> bool:
    key = norm_for_dedupe(bullet)
    if not key:
        return False
    for prior in seen:
        shorter, longer = sorted([key, prior], key=len)
        if len(shorter) >= min_unique_chars and shorter in longer:
            return False
    seen.add(key)
    target.append(bullet)
    return True


def trace_append(
    trace_rows: list[dict[str, str]],
    case_no: str,
    case_id: str,
    section: str,
    bullet_no: int,
    bullet: str,
    source_type: str,
    source_index: int | str,
    evidence: dict[str, Any],
) -> None:
    trace_rows.append({
        "case_no": case_no,
        "case_id": case_id,
        "section": section,
        "bullet_no": str(bullet_no),
        "bullet_text": bullet,
        "source_type": source_type,
        "source_index": str(source_index),
        "expert": str(evidence.get("expert", "")),
        "category": str(evidence.get("category", "")),
        "direction": str(evidence.get("direction") or evidence.get("impact") or evidence.get("semantic_role") or ""),
        "source": str(evidence.get("source", "")),
        "source_claim": str(evidence.get("claim", "")),
    })


def structured_summary(
    row: dict[str, str],
    ta_maps: dict[str, dict[str, dict[str, Any]]],
    trace_rows: list[dict[str, str]],
    case_no: str,
) -> str:
    split = row["split"]
    ta_key = "dd" if split.startswith("drug_disjoint") else "ts"
    result = ta_maps[ta_key].get(row["case_id"], {})
    evidence_graph = result.get("evidence_graph") or {}
    evidence_items = evidence_graph.get("evidence") or []
    derived_items = evidence_graph.get("derived_argument_claims") or []

    support_candidates: list[tuple[int, dict[str, Any]]] = []
    conflict_candidates: list[tuple[str, int, dict[str, Any]]] = []
    missing_candidates: list[tuple[str, int, dict[str, Any]]] = []

    for idx, ev in enumerate(evidence_items):
        impact = str(ev.get("impact", "")).lower()
        category = str(ev.get("category", "")).lower()
        claim = str(ev.get("claim", "")).lower()
        expert = str(ev.get("expert", ""))
        raw_claim = str(ev.get("claim", ""))
        is_off_target_indication = (
            expert == "DrugKB"
            and ("indication" in claim or category == "drug_history")
            and not disease_matches_claim(row.get("disease", ""), raw_claim)
        )
        clinical_is_context = expert == "Clinical" and "not drug-specific" in claim
        if impact == "supportive" and expert != "ADMET":
            if not is_off_target_indication:
                support_candidates.append((idx, ev))
        elif expert == "DrugKB" and category in {"mechanism_prior", "drug_class"}:
            support_candidates.append((idx, ev))
        elif expert == "DrugKB" and category == "drug_history" and not is_off_target_indication:
            support_candidates.append((idx, ev))
        elif impact == "risk" or expert == "ADMET" and ("toxicity" in category or "risk" in claim):
            if expert == "Clinical" and clinical_is_context:
                support_candidates.append((idx, ev))
            else:
                conflict_candidates.append(("evidence", idx, ev))

    for idx, claim in enumerate(derived_items):
        direction = str(claim.get("direction", "")).lower()
        role = str(claim.get("semantic_role", "")).lower()
        if direction == "conflict" or "conflict" in role or "risk" in role:
            conflict_candidates.append(("derived_argument_claims", idx, claim))
        elif direction == "missing" or "missing" in role or "gap" in role:
            missing_candidates.append(("derived_argument_claims", idx, claim))
        elif direction == "support":
            support_candidates.append((idx, claim))

    support_candidates = sorted(support_candidates, key=lambda item: evidence_priority(item[1]))
    conflict_candidates = sorted(
        conflict_candidates,
        key=lambda item: (
            0 if "safety" in str(item[2].get("semantic_role", "")).lower() or item[2].get("expert") == "ADMET" else 1,
            -float(item[2].get("confidence") or item[2].get("reliability") or 0),
        ),
    )
    missing_candidates = sorted(
        missing_candidates,
        key=lambda item: (
            0 if "direct" in str(item[2].get("semantic_role", "")).lower() or "direct" in str(item[2].get("claim", "")).lower() else 1,
            -float(item[2].get("confidence") or item[2].get("reliability") or 0),
        ),
    )

    sections: dict[str, list[str]] = {
        "Supporting or contextual evidence": [],
        "Conflict or risk evidence": [],
        "Missing evidence or uncertainty": [],
    }
    seen_by_section: dict[str, set[str]] = {key: set() for key in sections}

    max_support_items = 5
    max_conflict_items = 4
    max_missing_items = 4

    for src_idx, ev in support_candidates:
        bullet = bullet_text(source_label(ev), ev.get("claim", ""))
        if append_unique_bullet(sections["Supporting or contextual evidence"], seen_by_section["Supporting or contextual evidence"], bullet):
            trace_append(trace_rows, case_no, row["case_id"], "support", len(sections["Supporting or contextual evidence"]), bullet, "evidence", src_idx, ev)
        if len(sections["Supporting or contextual evidence"]) >= max_support_items:
            break

    for src_type, src_idx, ev in conflict_candidates:
        label = source_label(ev)
        if src_type == "derived_argument_claims" and not label:
            label = "Conflict"
        bullet = bullet_text(label, ev.get("claim", ""))
        if append_unique_bullet(sections["Conflict or risk evidence"], seen_by_section["Conflict or risk evidence"], bullet):
            trace_append(trace_rows, case_no, row["case_id"], "conflict", len(sections["Conflict or risk evidence"]), bullet, src_type, src_idx, ev)
        if len(sections["Conflict or risk evidence"]) >= max_conflict_items:
            break

    for src_type, src_idx, ev in missing_candidates:
        bullet = bullet_text(source_label(ev), ev.get("claim", ""))
        if append_unique_bullet(sections["Missing evidence or uncertainty"], seen_by_section["Missing evidence or uncertainty"], bullet):
            trace_append(trace_rows, case_no, row["case_id"], "missing", len(sections["Missing evidence or uncertainty"]), bullet, src_type, src_idx, ev)
        if len(sections["Missing evidence or uncertainty"]) >= max_missing_items:
            break

    if not sections["Supporting or contextual evidence"]:
        fallback = clean_text(row.get("support_summary"))
        bullet = f"- Evidence: {fallback}"
        sections["Supporting or contextual evidence"].append(bullet)
        trace_append(trace_rows, case_no, row["case_id"], "support", 1, bullet, "selected_cases_fallback", "support_summary", {"claim": row.get("support_summary", "")})
    if not sections["Conflict or risk evidence"]:
        fallback = clean_text(row.get("conflict_summary"))
        bullet = f"- Evidence: {fallback}"
        sections["Conflict or risk evidence"].append(bullet)
        trace_append(trace_rows, case_no, row["case_id"], "conflict", 1, bullet, "selected_cases_fallback", "conflict_summary", {"claim": row.get("conflict_summary", "")})
    if not sections["Missing evidence or uncertainty"]:
        fallback = clean_text(row.get("missing_summary"))
        bullet = f"- Evidence gap: {fallback}"
        sections["Missing evidence or uncertainty"].append(bullet)
        trace_append(trace_rows, case_no, row["case_id"], "missing", 1, bullet, "selected_cases_fallback", "missing_summary", {"claim": row.get("missing_summary", "")})

    return "\n\n".join(
        f"{section}:\n" + "\n".join(bullets)
        for section, bullets in sections.items()
    )


def baseline_summary(row: dict[str, str], rag_maps: dict[str, dict[str, dict[str, Any]]]) -> str:
    split = row["split"]
    rag_key = "dd" if split.startswith("drug_disjoint") else "ts"
    rag_row = rag_maps[rag_key].get(row["case_id"], {})
    response = strip_rag_response(rag_row.get("rag_response") or rag_row.get("synthesis_explanation"))
    context = clean_text(rag_row.get("rag_context"))
    if response and response != "No clear evidence summary was available in the available evidence.":
        return response
    if context:
        return f"Retrieved context: {context}"
    return "No alternative evidence summary was available for this case."


def choose_wjx_cases(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build fixed 16-case set: 12 correct predictions + 4 errors."""
    case_by_id = {r["case_id"]: r for r in rows}
    case_by_id.update(EXTRA_ROWS)
    selected_correct = [case_by_id[cid] for cid in CORRECT_CASE_ORDER if cid in case_by_id]
    if len(selected_correct) != 12:
        raise RuntimeError(f"Expected 12 correct cases, got {len(selected_correct)}")

    errors = [r for r in rows if r["confusion_type"] in {"FP", "FN"}]

    error_by_id = {r["case_id"]: r for r in errors}
    selected_errors = [error_by_id[cid] for cid in ERROR_CASE_PRIORITY if cid in error_by_id]
    if len(selected_errors) < 4:
        selected_ids = {r["case_id"] for r in selected_errors}
        selected_errors.extend([r for r in errors if r["case_id"] not in selected_ids][: 4 - len(selected_errors)])

    selected = selected_correct + selected_errors[:4]
    return selected


QUESTION_BLOCK = [
    ("Q1", "Based only on the decision and confidence score, should this candidate be prioritized for further investigation?", ["Yes", "No", "Insufficient information"]),
    ("Q2", "How confident are you in this initial judgment?", ["1 Not confident", "2 Slightly confident", "3 Moderately confident", "4 Confident", "5 Very confident"]),
    ("Q3", "After reviewing the evidence summaries, should this candidate be prioritized for further investigation?", ["Yes", "No", "Insufficient information"]),
    ("Q4", "How confident are you in this final judgment?", ["1 Not confident", "2 Slightly confident", "3 Moderately confident", "4 Confident", "5 Very confident"]),
    ("Q5", "Which evidence summary is more useful for reviewing this candidate?", ["Summary A", "Summary B", "No clear difference"]),
    ("Q6", "Which summary makes the decision more traceable?", ["Summary A", "Summary B", "No clear difference"]),
    ("Q7", "Which summary better identifies conflicts, risks, or missing evidence?", ["Summary A", "Summary B", "No clear difference"]),
    ("Q8", "Which summary better supports deciding the next research step?", ["Summary A", "Summary B", "No clear difference"]),
]


def add_question_md(lines: list[str], label: str, question: str, options: list[str]) -> None:
    lines.append(f"{label}. {question} [单选题]")
    for idx, opt in enumerate(options):
        lines.append(f"{chr(65 + idx)}. {opt}")
    lines.append("")


def w_run(text: str, *, bold: bool = False, size: int = 22) -> str:
    """Return one WordprocessingML run. Size is half-points."""
    text = escape(text)
    bold_xml = "<w:b/>" if bold else ""
    return (
        "<w:r><w:rPr>"
        "<w:rFonts w:ascii=\"Arial\" w:hAnsi=\"Arial\" w:eastAsia=\"Arial\"/>"
        f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
        f"{bold_xml}</w:rPr><w:t xml:space=\"preserve\">{text}</w:t></w:r>"
    )


def w_para(text: str = "", *, bold: bool = False, size: int = 22, style: str | None = None) -> str:
    style_xml = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
    return f"<w:p>{style_xml}{w_run(text, bold=bold, size=size)}</w:p>"


def w_page_break() -> str:
    return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"


def write_docx(path: Path, paragraph_xml: list[str]) -> Path:
    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        "<w:body>"
        + "".join(paragraph_xml)
        + "<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/></w:sectPr>"
        "</w:body></w:document>"
    )
    styles_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        "<w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\">"
        "<w:name w:val=\"Normal\"/><w:qFormat/><w:rPr>"
        "<w:rFonts w:ascii=\"Arial\" w:hAnsi=\"Arial\" w:eastAsia=\"Arial\"/>"
        "<w:sz w:val=\"22\"/><w:szCs w:val=\"22\"/></w:rPr>"
        "<w:pPr><w:spacing w:after=\"160\" w:line=\"276\" w:lineRule=\"auto\"/></w:pPr>"
        "</w:style>"
        "<w:style w:type=\"paragraph\" w:styleId=\"Heading1\">"
        "<w:name w:val=\"heading 1\"/><w:basedOn w:val=\"Normal\"/><w:qFormat/>"
        "<w:pPr><w:spacing w:before=\"240\" w:after=\"120\"/></w:pPr>"
        "<w:rPr><w:b/><w:rFonts w:ascii=\"Arial\" w:hAnsi=\"Arial\" w:eastAsia=\"Arial\"/>"
        "<w:sz w:val=\"32\"/><w:szCs w:val=\"32\"/></w:rPr></w:style>"
        "</w:styles>"
    )
    content_types = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "<Override PartName=\"/word/styles.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/>"
        "</Types>"
    )
    root_rels = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"word/document.xml\"/></Relationships>"
    )
    word_rels = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" "
        "Target=\"styles.xml\"/></Relationships>"
    )
    candidates = [path, path.with_name(f"{path.stem}_updated{path.suffix}")]
    candidates.extend(path.with_name(f"{path.stem}_updated_{i}{path.suffix}") for i in range(2, 10))
    last_error: PermissionError | None = None
    for output_path in candidates:
        try:
            zf = zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED)
            break
        except PermissionError as exc:
            last_error = exc
    else:
        raise last_error or PermissionError(path)
    with zf as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/_rels/document.xml.rels", word_rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/styles.xml", styles_xml)
    return output_path


def build_questionnaire(
    cases: list[dict[str, str]],
    rag_maps: dict[str, dict[str, dict[str, Any]]],
    ta_maps: dict[str, dict[str, dict[str, Any]]],
) -> None:
    doc_xml: list[str] = []
    doc_xml.append(w_para("Clinician Expert Review Questionnaire", bold=True, size=40))
    doc_xml.append(w_para("Purpose: evaluate whether evidence summaries help clinicians review drug--disease candidates for further investigation. This questionnaire is not intended to provide clinical treatment recommendations."))
    doc_xml.append(w_para("Instructions: please judge whether each candidate should be prioritized for further investigation. The questionnaire only contains candidate information and evidence summaries."))

    md_lines = [
        "# Clinician Expert Review Questionnaire",
        "",
        "Purpose: evaluate whether evidence summaries help clinicians review drug--disease candidates for further investigation. This questionnaire is not intended to provide clinical treatment recommendations.",
        "",
        "Instructions: please judge whether each candidate should be prioritized for further investigation. The questionnaire only contains candidate information and evidence summaries.",
        "",
    ]

    mapping_rows: list[dict[str, str]] = []
    trace_rows: list[dict[str, str]] = []

    for i, row in enumerate(cases, start=1):
        case_no = f"C{i:02d}"
        drug = display_drug_name(row["drug"])
        disease = display_disease_name(row["disease"])
        decision = decision_from_prediction(row["treatagent_prediction"])
        confidence = row["treatagent_score"]

        structured = structured_summary(row, ta_maps, trace_rows, case_no)
        baseline = baseline_summary(row, rag_maps)
        if i % 2 == 1:
            summary_a, summary_b = structured, baseline
            source_a, source_b = "structured_evidence", "baseline_evidence"
        else:
            summary_a, summary_b = baseline, structured
            source_a, source_b = "baseline_evidence", "structured_evidence"

        if i > 1:
            doc_xml.append(w_page_break())
        doc_xml.append(w_para(f"Case {i:02d}: {drug} - {disease}", bold=True, size=32, style="Heading1"))
        doc_xml.append(w_para("Part 1. Decision-only information", bold=True))
        doc_xml.append(w_para(f"Drug: {drug}"))
        doc_xml.append(w_para(f"Disease: {disease}"))
        doc_xml.append(w_para(f"Decision: {decision}"))
        doc_xml.append(w_para(f"Confidence score: {confidence}"))

        md_lines.extend([
            f"## Case {i:02d}: {drug} - {disease}",
            "",
            "Part 1. Decision-only information",
            f"Drug: {drug}",
            f"Disease: {disease}",
            f"Decision: {decision}",
            f"Confidence score: {confidence}",
            "",
        ])

        doc_xml.append(w_para(f"{case_no}_Q1. {QUESTION_BLOCK[0][1]} [单选题]", bold=True))
        for idx, opt in enumerate(QUESTION_BLOCK[0][2]):
            doc_xml.append(w_para(f"{chr(65 + idx)}. {opt}"))
        doc_xml.append(w_para(f"{case_no}_Q2. {QUESTION_BLOCK[1][1]} [单选题]", bold=True))
        for idx, opt in enumerate(QUESTION_BLOCK[1][2]):
            doc_xml.append(w_para(f"{chr(65 + idx)}. {opt}"))
        add_question_md(md_lines, f"{case_no}_Q1", QUESTION_BLOCK[0][1], QUESTION_BLOCK[0][2])
        add_question_md(md_lines, f"{case_no}_Q2", QUESTION_BLOCK[1][1], QUESTION_BLOCK[1][2])

        doc_xml.append(w_para("Part 2. Evidence Summary A", bold=True))
        for para in summary_a.split("\n\n"):
            doc_xml.append(w_para(para))
        doc_xml.append(w_para("Part 3. Evidence Summary B", bold=True))
        for para in summary_b.split("\n\n"):
            doc_xml.append(w_para(para))
        md_lines.extend([
            "Part 2. Evidence Summary A",
            summary_a,
            "",
            "Part 3. Evidence Summary B",
            summary_b,
            "",
        ])

        for q_label, q_text, q_opts in QUESTION_BLOCK[2:]:
            doc_xml.append(w_para(f"{case_no}_{q_label}. {q_text} [单选题]", bold=True))
            for idx, opt in enumerate(q_opts):
                doc_xml.append(w_para(f"{chr(65 + idx)}. {opt}"))
            add_question_md(md_lines, f"{case_no}_{q_label}", q_text, q_opts)

        doc_xml.append(w_para(f"{case_no}_Q9. Free-text comments [填空题]", bold=True))
        md_lines.append(f"{case_no}_Q9. Free-text comments [填空题]")
        md_lines.append("")

        mapping_rows.append({
            "case_no": case_no,
            "case_id": row["case_id"],
            "drug": drug,
            "disease": disease,
            "split": row["split"],
            "hidden_benchmark_label": row["benchmark_label"],
            "hidden_confusion_type": row["confusion_type"],
            "summary_a_source": source_a,
            "summary_b_source": source_b,
        })

    doc_xml.append(w_page_break())
    doc_xml.append(w_para("Overall questions", bold=True, size=32, style="Heading1"))
    md_lines.extend(["## Overall questions", ""])

    overall_questions = [
        ("O1", "Overall, how useful were the structured evidence summaries for candidate review?", ["1 Not useful", "2 Slightly useful", "3 Moderately useful", "4 Useful", "5 Very useful"]),
        ("O2", "Which information types were most useful? [多选题]", ["Support evidence", "Conflict or risk evidence", "Missing evidence", "Confidence score", "Source or provenance notes", "None of the above"]),
        ("O3", "Did the evidence summaries help identify cases requiring specialist review?", ["Yes", "No", "Unclear"]),
        ("O4", "Would you use this type of evidence record to triage drug-repurposing candidates for further investigation?", ["Yes", "No", "Depends on the case"]),
    ]
    for label, question, opts in overall_questions:
        q_type = "多选题" if "多选题" in question else "单选题"
        clean_q = question.replace(" [多选题]", "")
        doc_xml.append(w_para(f"{label}. {clean_q} [{q_type}]", bold=True))
        md_lines.append(f"{label}. {clean_q} [{q_type}]")
        for idx, opt in enumerate(opts):
            doc_xml.append(w_para(f"{chr(65 + idx)}. {opt}"))
            md_lines.append(f"{chr(65 + idx)}. {opt}")
        md_lines.append("")

    doc_xml.append(w_para("O5. Overall comments [填空题]", bold=True))
    md_lines.append("O5. Overall comments [填空题]")
    md_lines.append("")

    written_docx = write_docx(DOCX_OUT, doc_xml)
    MD_OUT.write_text("\n".join(md_lines), encoding="utf-8")

    fieldnames = list(mapping_rows[0].keys())
    with MAP_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)

    trace_fields = [
        "case_no",
        "case_id",
        "section",
        "bullet_no",
        "bullet_text",
        "source_type",
        "source_index",
        "expert",
        "category",
        "direction",
        "source",
        "source_claim",
    ]
    with TRACE_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trace_fields)
        writer.writeheader()
        writer.writerows(trace_rows)

    case_fields = [
        "case_no",
        "selection_set",
        "case_id",
        "drug",
        "disease",
        "split",
        "benchmark_label",
        "treatagent_prediction",
        "treatagent_score",
        "confusion_type",
        "disease_area",
        "suggested_specialty",
        "evidence_pattern",
        "baseline_available",
    ]
    with CASE_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=case_fields)
        writer.writeheader()
        for i, row in enumerate(cases, start=1):
            out = {k: row.get(k, "") for k in case_fields if k != "case_no"}
            out["case_no"] = f"C{i:02d}"
            out["drug"] = display_drug_name(row.get("drug", ""))
            out["disease"] = display_disease_name(row.get("disease", ""))
            writer.writerow(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_csv(SELECTED_CASES)
    cases = choose_wjx_cases(rows)
    rag_maps = {"dd": load_rag_map(RAG_DD), "ts": load_rag_map(RAG_TS)}
    ta_maps = {"dd": load_result_map(TA_DD), "ts": load_result_map(TA_TS)}
    build_questionnaire(cases, rag_maps, ta_maps)
    if DOCX_OUT.exists():
        print(f"Wrote or preserved {DOCX_OUT.relative_to(ROOT)}")
    updated = DOCX_OUT.with_name(f"{DOCX_OUT.stem}_updated{DOCX_OUT.suffix}")
    if updated.exists():
        print(f"Wrote {updated.relative_to(ROOT)}")
    print(f"Wrote {MD_OUT.relative_to(ROOT)}")
    print(f"Wrote {CASE_OUT.relative_to(ROOT)}")
    print(f"Wrote {MAP_OUT.relative_to(ROOT)}")
    print(f"Wrote {TRACE_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
