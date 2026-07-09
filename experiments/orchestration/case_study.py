#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import textwrap
from pathlib import Path
from typing import Any

from experiments.orchestration.prediction_io import safe_float, safe_int


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "case_studies" / "treatagent_arg"
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "case_studies" / "treatagent_arg_case_studies.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError(f"Unsupported JSON format: {path}")


def by_sample_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id") or row.get("id") or index): row for index, row in enumerate(rows)}


def factor(row: dict[str, Any], name: str) -> float:
    return safe_float((row.get("argument_factors") or {}).get(name), 0.0)


def select_cases(arg_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    def label(row: dict[str, Any]) -> int:
        return safe_int(row.get("label"), 0)

    def pred(row: dict[str, Any]) -> int:
        return safe_int(row.get("argument_prediction"), 0)

    def prob(row: dict[str, Any]) -> float:
        return safe_float(row.get("argument_probability"), 0.0)

    true_positive_pool = [
        row for row in arg_rows if label(row) == 1 and pred(row) == 1 and factor(row, "direct_support") > 0
    ] or [row for row in arg_rows if label(row) == 1 and pred(row) == 1]
    true_negative_pool = [row for row in arg_rows if label(row) == 0 and pred(row) == 0]
    conflict_pool = [
        row
        for row in arg_rows
        if row.get("top_conflict_arguments")
        or factor(row, "safety_conflict") > 0
        or factor(row, "conflict_strength") > 0.03
    ]

    if not true_positive_pool or not true_negative_pool or not conflict_pool:
        raise ValueError("Could not find enough ARG cases for TP/TN/conflict case study.")

    true_positive = max(
        true_positive_pool,
        key=lambda row: (factor(row, "direct_support"), prob(row), factor(row, "clinical_feasibility")),
    )
    true_negative = min(true_negative_pool, key=prob)
    conflict = max(
        conflict_pool,
        key=lambda row: (
            factor(row, "conflict_strength"),
            factor(row, "safety_conflict"),
            safe_float(row.get("argument_probability"), 0.0),
        ),
    )
    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "conflict_hard_case": conflict,
    }


def planner_trajectory(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    trajectory = row.get("trajectory") or []
    output = []
    for item in trajectory:
        if isinstance(item, dict):
            planner = item.get("planner_output") or {}
            action = planner.get("next_action") or planner.get("selected_action") or item.get("action")
            if action:
                output.append(str(action))
        elif item:
            output.append(str(item))
    return output


def wrap_svg_text(text: str, width: int = 36, max_lines: int = 3) -> list[str]:
    lines = textwrap.wrap(str(text), width=width)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][: max(0, width - 3)] + "..."]
    return lines or [""]


def svg_text(
    x: int,
    y: int,
    lines: list[str],
    *,
    size: int = 12,
    color: str = "#1f2933",
    weight: int = 400,
) -> str:
    parts = []
    for index, line in enumerate(lines):
        parts.append(
            f'<text x="{x}" y="{y + index * (size + 4)}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">'
            f"{html.escape(line)}</text>"
        )
    return "\n".join(parts)


def svg_node(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    subtitle: str,
    fill: str,
    stroke: str,
    *,
    title_color: str = "#111827",
) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>',
            svg_text(x + 14, y + 26, wrap_svg_text(title, 30, 1), size=13, color=title_color, weight=700),
            svg_text(x + 12, y + 46, wrap_svg_text(subtitle, 34, 2), size=11, color="#374151"),
        ]
    )


def svg_edge(x1: int, y1: int, x2: int, y2: int, color: str, label: str) -> str:
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2
    return "\n".join(
        [
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#arrow)"/>',
            f'<text x="{mx}" y="{my - 5}" font-size="10" fill="{color}">{html.escape(label)}</text>',
        ]
    )


def shorten(text: Any, length: int = 90) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= length:
        return value
    return value[: max(0, length - 3)].rstrip() + "..."


def svg_pill(x: int, y: int, text: str, fill: str, color: str) -> str:
    width = max(74, min(210, 11 * len(text) + 22))
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="26" rx="13" fill="{fill}"/>',
            svg_text(x + 11, y + 18, [text], size=11, color=color, weight=700),
        ]
    )


def score_bar(x: int, y: int, score: float, *, threshold: float = 0.36) -> str:
    width = 260
    filled = max(0, min(width, int(width * score)))
    threshold_x = x + int(width * threshold)
    color = "#16a34a" if score >= threshold else "#64748b"
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="14" rx="7" fill="#e5e7eb"/>',
            f'<rect x="{x}" y="{y}" width="{filled}" height="14" rx="7" fill="{color}"/>',
            f'<line x1="{threshold_x}" y1="{y - 5}" x2="{threshold_x}" y2="{y + 21}" stroke="#111827" stroke-width="1.4" stroke-dasharray="3 3"/>',
            svg_text(x, y + 38, ["0.0"], size=10, color="#64748b"),
            svg_text(x + width - 20, y + 38, ["1.0"], size=10, color="#64748b"),
            svg_text(threshold_x - 18, y - 10, ["threshold"], size=10, color="#111827"),
        ]
    )


def evidence_card(
    x: int,
    y: int,
    w: int,
    item: dict[str, Any],
    *,
    kind: str,
    index: int,
) -> str:
    if kind == "support":
        fill, stroke, accent, label_fill, label_color = "#f0fdf4", "#bbf7d0", "#16a34a", "#dcfce7", "#166534"
    else:
        fill, stroke, accent, label_fill, label_color = "#fff7ed", "#fed7aa", "#f97316", "#ffedd5", "#9a3412"
    expert = str(item.get("expert") or "Evidence")
    category = str(item.get("category") or "argument")
    strength = safe_float(item.get("strength"))
    claim = shorten(item.get("claim") or item.get("object"), 118)
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="108" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>',
            f'<rect x="{x}" y="{y}" width="6" height="108" rx="3" fill="{accent}"/>',
            svg_pill(x + 16, y + 14, f"{kind.title()} {index}", label_fill, label_color),
            svg_text(x + 16, y + 54, [f"{expert} / {category}"], size=12, color="#111827", weight=700),
            svg_text(x + w - 70, y + 54, [f"{strength:.2f}"], size=15, color=accent, weight=700),
            svg_text(x + 16, y + 78, wrap_svg_text(claim, 44, 2), size=10, color="#374151"),
        ]
    )


def write_case_svg(case_name: str, row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    supports = (row.get("top_support_arguments") or [])[:4]
    conflicts = (row.get("top_conflict_arguments") or [])[:3]
    factors = row.get("argument_factors") or {}
    prob = safe_float(row.get("argument_probability"), 0.0)
    label = safe_int(row.get("label"), 0)
    pred = safe_int(row.get("argument_prediction"), 0)
    disease = str(row.get("disease") or "")
    sample_id = str(row.get("sample_id") or "")
    decision_color = "#16a34a" if pred == 1 else "#64748b"
    decision_fill = "#dcfce7" if pred == 1 else "#f1f5f9"
    case_title = {
        "true_positive": "True Positive: strong convergent support",
        "true_negative": "True Negative: insufficient treatment evidence",
        "conflict_hard_case": "Conflict / Failure Case: support overwhelms weak conflict",
    }.get(case_name, case_name.replace("_", " ").title())

    width = 1240
    height = 720
    left_x = 42
    center_x = 430
    right_x = 842

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">',
        '<feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.08"/>',
        "</filter>",
        "</defs>",
        '<rect width="1240" height="720" fill="#f8fafc"/>',
        '<rect x="24" y="24" width="1192" height="672" rx="24" fill="#ffffff" filter="url(#shadow)"/>',
        svg_text(54, 64, [case_title], size=23, color="#0f172a", weight=800),
        svg_text(54, 94, [f"{sample_id} | disease: {disease}"], size=13, color="#475569"),
        svg_pill(54, 116, f"label {label}", "#eff6ff", "#1d4ed8"),
        svg_pill(142, 116, f"prediction {pred}", decision_fill, decision_color),
        svg_pill(272, 116, f"score {prob:.3f}", "#f8fafc", "#0f172a"),
        svg_text(left_x, 186, ["Support arguments"], size=16, color="#166534", weight=800),
        svg_text(right_x, 186, ["Conflict / caution"], size=16, color="#9a3412", weight=800),
        f'<rect x="{center_x}" y="166" width="360" height="424" rx="18" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.2"/>',
        svg_text(center_x + 28, 204, ["Argument EvidenceGraph"], size=18, color="#0f172a", weight=800),
        svg_text(center_x + 28, 232, ["Reliability-aware support/conflict aggregation"], size=12, color="#475569"),
        svg_text(center_x + 28, 282, ["ARG treatment score"], size=13, color="#334155", weight=700),
        score_bar(center_x + 28, 300, prob),
        svg_text(center_x + 28, 380, ["Key factors"], size=13, color="#334155", weight=700),
        svg_node(
            center_x + 28,
            402,
            138,
            64,
            "Direct",
            f"{safe_float(factors.get('direct_support')):.3f}",
            "#eef2ff",
            "#c7d2fe",
            title_color="#3730a3",
        ),
        svg_node(
            center_x + 194,
            402,
            138,
            64,
            "Clinical",
            f"{safe_float(factors.get('clinical_feasibility')):.3f}",
            "#ecfdf5",
            "#bbf7d0",
            title_color="#166534",
        ),
        svg_node(
            center_x + 28,
            486,
            138,
            64,
            "Consistency",
            f"{safe_float(factors.get('cross_source_consistency')):.3f}",
            "#f0f9ff",
            "#bae6fd",
            title_color="#0369a1",
        ),
        svg_node(
            center_x + 194,
            486,
            138,
            64,
            "Conflict",
            f"{safe_float(factors.get('conflict_strength')):.3f}",
            "#fff7ed",
            "#fed7aa",
            title_color="#9a3412",
        ),
    ]

    for index, item in enumerate(supports):
        y = 208 + index * 116
        parts.append(evidence_card(left_x, y, 340, item, kind="support", index=index + 1))
        line_y = y + 54
        parts.append(
            f'<path d="M {left_x + 340} {line_y} C {left_x + 374} {line_y}, {center_x - 28} {330 + index * 24}, {center_x} {330 + index * 24}" '
            'fill="none" stroke="#86efac" stroke-width="1.5" stroke-dasharray="4 5"/>'
        )

    if conflicts:
        for index, item in enumerate(conflicts):
            y = 220 + index * 126
            parts.append(evidence_card(right_x, y, 340, item, kind="conflict", index=index + 1))
            line_y = y + 54
            parts.append(
                f'<path d="M {right_x} {line_y} C {right_x - 36} {line_y}, {center_x + 388} {356 + index * 28}, {center_x + 360} {356 + index * 28}" '
                'fill="none" stroke="#fdba74" stroke-width="1.5" stroke-dasharray="4 5"/>'
            )
    else:
        parts.append(
            svg_node(
                right_x,
                308,
                340,
                94,
                "No dominant conflict",
                "No dominant structured conflict argument was selected.",
                "#f9fafb",
                "#9ca3af",
                title_color="#475569",
            )
        )
        parts.append(
            f'<path d="M {right_x} 356 C {right_x - 36} 356, {center_x + 388} 356, {center_x + 360} 356" '
            'fill="none" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4 5"/>'
        )

    footer = (
        "Decision trace: support evidence and conflict evidence are converted to typed tuples, "
        "then aggregated by ARG factors rather than free-form LLM scoring."
    )
    parts.append(svg_text(54, 654, wrap_svg_text(footer, 145, 1), size=11, color="#64748b"))

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| Field | Value |", "|---|---|"]
    for key, value in rows:
        lines.append(f"| {key} | {str(value).replace('|', '/')} |")
    return "\n".join(lines)


def case_markdown(
    case_name: str,
    row: dict[str, Any],
    source_row: dict[str, Any] | None,
    planner_row: dict[str, Any] | None,
    svg_rel_path: str,
) -> str:
    factors = row.get("argument_factors") or {}
    trajectory = planner_trajectory(planner_row) or planner_trajectory(source_row)
    support_rows = row.get("top_support_arguments") or []
    conflict_rows = row.get("top_conflict_arguments") or []
    direct = safe_float(factors.get("direct_support"))
    clinical = safe_float(factors.get("clinical_feasibility"))
    consistency = safe_float(factors.get("cross_source_consistency"))
    conflict = safe_float(factors.get("conflict_strength"))
    if safe_int(row.get("label"), 0) == 1 and safe_int(row.get("argument_prediction"), 0) == 1:
        explanation = (
            "This true positive is supported by strong direct or disease-context evidence, high clinical feasibility, "
            "and cross-source consistency. The graph view makes the positive decision traceable to explicit support arguments."
        )
    elif safe_int(row.get("label"), 0) == 0 and safe_int(row.get("argument_prediction"), 0) == 0:
        explanation = (
            "This true negative has weak direct support and very low clinical feasibility. "
            "The ARG score stays low despite some generic mechanism evidence, showing how the graph penalizes insufficient treatment-specific support."
        )
    else:
        explanation = (
            "This hard case is useful as a failure/conflict example: support arguments and clinical feasibility push the score upward, "
            "while conflict signals are not strong enough to reverse the decision. It highlights where stronger conflict modeling is still needed."
        )
    explanation += (
        f" Factor snapshot: direct={direct:.3f}, clinical={clinical:.3f}, "
        f"consistency={consistency:.3f}, conflict={conflict:.3f}."
    )
    return "\n\n".join(
        [
            f"## {case_name.replace('_', ' ').title()}",
            f"![{case_name}]({svg_rel_path})",
            markdown_table(
                [
                    ("sample_id", row.get("sample_id")),
                    ("disease", row.get("disease")),
                    ("smiles", str(row.get("smiles", ""))[:120]),
                    ("label", row.get("label")),
                    ("prediction", row.get("argument_prediction")),
                    ("ARG score", f"{safe_float(row.get('argument_probability')):.4f}"),
                    ("direct_support", f"{safe_float(factors.get('direct_support')):.4f}"),
                    ("clinical_feasibility", f"{safe_float(factors.get('clinical_feasibility')):.4f}"),
                    ("cross_source_consistency", f"{safe_float(factors.get('cross_source_consistency')):.4f}"),
                    ("conflict_strength", f"{safe_float(factors.get('conflict_strength')):.4f}"),
                    ("planner trajectory", " -> ".join(trajectory) if trajectory else "not available"),
                ]
            ),
            "### Top Support Arguments\n\n"
            + "\n".join(
                f"- **{item.get('expert')} / {item.get('category')}** "
                f"(strength={safe_float(item.get('strength')):.3f}): {item.get('claim') or item.get('object')}"
                for item in support_rows[:5]
            ),
            "### Top Conflict Arguments\n\n"
            + (
                "\n".join(
                    f"- **{item.get('expert')} / {item.get('category')}** "
                    f"(strength={safe_float(item.get('strength')):.3f}): {item.get('claim') or item.get('object')}"
                    for item in conflict_rows[:5]
                )
                if conflict_rows
                else "- No dominant structured conflict argument was selected."
            ),
            f"### Short Interpretation\n\n{explanation}",
        ]
    )


def case_takeaway(case_name: str) -> str:
    if case_name == "true_positive":
        return "Strong direct and disease-context support makes the positive decision traceable."
    if case_name == "true_negative":
        return "Low direct support and low clinical feasibility suppress generic mechanism evidence."
    if case_name == "conflict_hard_case":
        return "A failure case: weak conflict evidence does not fully offset disease-context support."
    return "Evidence is summarized through ARG support and conflict factors."


def mini_evidence_lines(items: list[dict[str, Any]], limit: int) -> list[str]:
    lines = []
    for item in items[:limit]:
        expert = str(item.get("expert") or "Evidence")
        category = str(item.get("category") or "argument")
        strength = safe_float(item.get("strength"))
        claim = shorten(item.get("claim") or item.get("object"), 72)
        lines.append(f"{expert}/{category} ({strength:.2f}): {claim}")
    return lines


def figure4_panel(x: int, y: int, w: int, h: int, case_name: str, row: dict[str, Any]) -> str:
    factors = row.get("argument_factors") or {}
    supports = mini_evidence_lines(row.get("top_support_arguments") or [], 2)
    conflicts = mini_evidence_lines(row.get("top_conflict_arguments") or [], 1)
    if not conflicts:
        conflicts = ["No dominant structured conflict."]
    prob = safe_float(row.get("argument_probability"))
    label = safe_int(row.get("label"), 0)
    pred = safe_int(row.get("argument_prediction"), 0)
    title = {
        "true_positive": "A. True positive",
        "true_negative": "B. True negative",
        "conflict_hard_case": "C. Conflict / failure",
    }.get(case_name, case_name.replace("_", " ").title())
    border = {
        "true_positive": "#16a34a",
        "true_negative": "#64748b",
        "conflict_hard_case": "#f97316",
    }.get(case_name, "#64748b")
    score_color = "#16a34a" if pred == 1 else "#64748b"
    score_width = max(0, min(220, int(220 * prob)))
    threshold_x = x + 30 + int(220 * 0.36)
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#ffffff" stroke="{border}" stroke-width="1.6"/>',
        svg_text(x + 24, y + 36, [title], size=18, color="#0f172a", weight=800),
        svg_text(
            x + 24,
            y + 66,
            [f"{row.get('sample_id')} | {shorten(row.get('disease'), 38)}"],
            size=11,
            color="#475569",
        ),
        svg_pill(x + 24, y + 82, f"label {label}", "#eff6ff", "#1d4ed8"),
        svg_pill(x + 112, y + 82, f"pred {pred}", "#f1f5f9" if pred == 0 else "#dcfce7", score_color),
        svg_text(x + 24, y + 140, ["ARG score"], size=12, color="#334155", weight=700),
        f'<rect x="{x + 30}" y="{y + 152}" width="220" height="12" rx="6" fill="#e5e7eb"/>',
        f'<rect x="{x + 30}" y="{y + 152}" width="{score_width}" height="12" rx="6" fill="{score_color}"/>',
        f'<line x1="{threshold_x}" y1="{y + 146}" x2="{threshold_x}" y2="{y + 170}" stroke="#111827" stroke-width="1.2" stroke-dasharray="3 3"/>',
        svg_text(x + 264, y + 164, [f"{prob:.3f}"], size=18, color=score_color, weight=800),
        svg_text(x + 24, y + 205, ["Key factors"], size=12, color="#334155", weight=700),
        svg_text(
            x + 24,
            y + 230,
            [
                f"direct {safe_float(factors.get('direct_support')):.2f}   clinical {safe_float(factors.get('clinical_feasibility')):.2f}",
                f"consistency {safe_float(factors.get('cross_source_consistency')):.2f}   conflict {safe_float(factors.get('conflict_strength')):.2f}",
            ],
            size=11,
            color="#475569",
        ),
        svg_text(x + 24, y + 288, ["Top support"], size=12, color="#166534", weight=800),
        svg_text(x + 34, y + 315, wrap_svg_text(supports[0] if supports else "No support evidence.", 48, 2), size=10, color="#166534"),
        svg_text(x + 34, y + 363, wrap_svg_text(supports[1] if len(supports) > 1 else "No second support evidence.", 48, 2), size=10, color="#166534"),
        svg_text(x + 24, y + 430, ["Top conflict / caution"], size=12, color="#9a3412", weight=800),
        svg_text(x + 34, y + 457, wrap_svg_text(conflicts[0], 48, 2), size=10, color="#9a3412"),
        f'<rect x="{x + 24}" y="{y + 512}" width="{w - 48}" height="84" rx="12" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>',
        svg_text(x + 40, y + 540, ["Take-away"], size=11, color="#334155", weight=800),
        svg_text(x + 40, y + 565, wrap_svg_text(case_takeaway(case_name), 48, 2), size=10, color="#475569"),
    ]
    return "\n".join(parts)


def write_figure4_svg(cases: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1500
    height = 760
    ordered = ["true_positive", "true_negative", "conflict_hard_case"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<filter id="figureShadow" x="-8%" y="-8%" width="116%" height="124%">',
        '<feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#0f172a" flood-opacity="0.08"/>',
        "</filter>",
        "</defs>",
        '<rect width="1500" height="760" fill="#f8fafc"/>',
        '<rect x="28" y="24" width="1444" height="704" rx="26" fill="#ffffff" filter="url(#figureShadow)"/>',
        svg_text(58, 70, ["Figure 4. EvidenceGraph case studies"], size=25, color="#0f172a", weight=800),
        svg_text(
            58,
            102,
            ["Each panel shows the top support/conflict evidence and the ARG factors used for the final treatment score."],
            size=13,
            color="#475569",
        ),
    ]
    for index, case_name in enumerate(ordered):
        parts.append(figure4_panel(58 + index * 476, 132, 430, 596, case_name, cases[case_name]))
    parts.append(svg_text(58, 710, ["Threshold shown as dashed line. Final scores are produced by ARG reasoning, not free-form LLM synthesis."], size=11, color="#64748b"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_case_studies(
    arg_predictions: Path,
    source_results: Path | None,
    planner_results: Path | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    doc_path: Path = DEFAULT_DOC_PATH,
) -> dict[str, Any]:
    arg_rows = load_rows(arg_predictions)
    source_by_id = by_sample_id(load_rows(source_results)) if source_results else {}
    planner_by_id = by_sample_id(load_rows(planner_results)) if planner_results else {}
    cases = select_cases(arg_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_parts = [
        "# TreatAgent-ARG Case Studies\n\n"
        "These cases are selected from the drug-disjoint test set using TreatAgent-ARG predictions."
    ]
    summary = {}
    for case_name, row in cases.items():
        svg_path = (output_dir / f"{case_name}_{row.get('sample_id')}.svg").resolve()
        write_case_svg(case_name, row, svg_path)
        rel_path = Path(os.path.relpath(svg_path, doc_path.parent.resolve()))
        sample_id = str(row.get("sample_id"))
        doc_parts.append(
            case_markdown(
                case_name,
                row,
                source_by_id.get(sample_id),
                planner_by_id.get(sample_id),
                rel_path.as_posix(),
            )
        )
        summary[case_name] = {
            "sample_id": sample_id,
            "label": row.get("label"),
            "prediction": row.get("argument_prediction"),
            "argument_probability": row.get("argument_probability"),
            "disease": row.get("disease"),
            "svg": str(svg_path),
        }

    figure4_path = (output_dir / "figure4_case_studies.svg").resolve()
    write_figure4_svg(cases, figure4_path)
    figure4_rel_path = Path(os.path.relpath(figure4_path, doc_path.parent.resolve()))
    doc_parts.insert(
        1,
        "## Figure 4 Combined Panel\n\n"
        f"![Figure 4]({figure4_rel_path.as_posix()})",
    )
    summary["figure4"] = {"svg": str(figure4_path)}

    doc_path.write_text("\n\n---\n\n".join(doc_parts) + "\n", encoding="utf-8")
    summary_path = output_dir / "case_study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"summary": summary, "doc_path": str(doc_path), "summary_path": str(summary_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TreatAgent-ARG case study markdown and SVG figures.")
    parser.add_argument("--arg_predictions", type=Path, required=True)
    parser.add_argument("--source_results", type=Path)
    parser.add_argument("--planner_results", type=Path)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc_path", type=Path, default=DEFAULT_DOC_PATH)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = build_case_studies(
        arg_predictions=args.arg_predictions,
        source_results=args.source_results,
        planner_results=args.planner_results,
        output_dir=args.output_dir,
        doc_path=args.doc_path,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

