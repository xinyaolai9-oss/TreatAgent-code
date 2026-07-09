from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "final_results" / "case_study_candidates.csv"
OUT_DIR = ROOT / "figure" / "final_results" / "case_study_panels"


MAIN_CASES = {
    "PAIR-000828": {
        "panel": "a",
        "short_title": "Direct indication support",
        "clinical_message": "A recognizable positive treatment relationship is recovered from direct drug-side evidence.",
        "evidence_state": "Direct support",
        "acquisition_summary": "4 expert calls; stopped before ADMET after direct indication and concordant context.",
        "graph_nodes": [
            ("support", "Direct indication match", "DrugKB", "r=0.52"),
            ("support", "Mechanistic context", "DTI + DiseaseKB", "r=0.42"),
            ("context", "Clinical feasibility prior", "Clinical", "r=0.59"),
            ("missing", "No dominant conflict", "Evidence state", ""),
        ],
        "biomedical_interpretation": [
            "Direct indication evidence links sumatriptan to migraine.",
            "Disease and mechanism context are concordant rather than contradictory.",
            "No dominant safety or missing-evidence boundary drives the decision.",
        ],
        "triage_meaning": "TreatAgent behaves as expected for a clear positive relationship: direct evidence dominates, and the candidate is prioritized.",
    },
    "PAIR-001194": {
        "panel": "b",
        "short_title": "Target-grounded support",
        "clinical_message": "Distributed drug, disease and mechanism signals are organized into a disease-relevant argument.",
        "evidence_state": "Mechanistic grounding",
        "acquisition_summary": "4 expert calls; DrugKB, DiseaseKB and DTI are connected through CASR.",
        "graph_nodes": [
            ("support", "Cross-source target grounding", "DTI + DiseaseKB", "r=0.61"),
            ("support", "Drug-side indication context", "DrugKB", "r=0.52"),
            ("context", "Moderate clinical feasibility", "Clinical", "r=0.33"),
            ("missing", "Direct evidence remains weak", "Evidence gap", ""),
        ],
        "biomedical_interpretation": [
            "Drug-side and disease-side evidence converge around CASR.",
            "DTI and DiseaseKB provide disease-relevant mechanistic grounding.",
            "Support is distributed across sources rather than coming from one direct match.",
        ],
        "triage_meaning": "The case illustrates why a claim-centered graph is useful: weak individual signals become interpretable when grounded across sources.",
    },
    "PAIR-001938": {
        "panel": "c",
        "short_title": "Negative boundary",
        "clinical_message": "Generic oncology plausibility is rejected when disease-specific support is missing and risk signals accumulate.",
        "evidence_state": "Conflict-limited",
        "acquisition_summary": "5 expert calls; ADMET is queried after support remains insufficient and risk is unresolved.",
        "graph_nodes": [
            ("missing", "No direct drug-disease support", "DrugKB", ""),
            ("missing", "Missing disease-specific grounding", "DrugKB / DiseaseKB / DTI", ""),
            ("conflict", "Safety conflict / toxicity risk", "ADMET", "r=0.40"),
            ("conflict", "Low clinical feasibility", "Clinical", "r=0.00"),
        ],
        "biomedical_interpretation": [
            "Direct drug-disease support and disease-specific grounding are absent.",
            "ADMET signals raise DILI, hERG and AMES-related safety concerns.",
            "Weak generic plausibility is not converted into a positive triage decision.",
        ],
        "triage_meaning": "TreatAgent exposes a negative evidence boundary: the candidate is not prioritized despite superficial disease-domain plausibility.",
    },
}


CSS = """
:root {
  --blue: #15549a;
  --blue-light: #eaf3ff;
  --green: #4f8b3b;
  --green-light: #eef8eb;
  --red: #d94d45;
  --red-light: #fff0ef;
  --gold: #dfaa00;
  --gold-light: #fff8df;
  --purple: #6c4c9a;
  --purple-light: #f4effb;
  --gray: #747474;
  --gray-light: #f4f4f4;
  --ink: #202833;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 28px;
  background: #ffffff;
  color: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
}

.figure-wrap {
  width: 1800px;
  margin: 0 auto;
}

.figure-title {
  display: flex;
  align-items: baseline;
  gap: 18px;
  margin: 0 0 20px 0;
}
.figure-title .label {
  font-size: 44px;
  font-weight: 700;
  color: #111;
}
.figure-title .text {
  font-size: 31px;
  font-weight: 500;
  color: #2c2c2c;
}

.panel-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 22px;
}

.case-panel {
  min-height: 790px;
  border: 3px dashed var(--blue);
  border-radius: 22px;
  background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
  padding: 24px 22px 22px;
  position: relative;
}

.case-panel.support { border-color: var(--green); }
.case-panel.boundary { border-color: var(--red); }

.case-header {
  display: grid;
  grid-template-columns: 48px 1fr;
  gap: 14px;
  align-items: start;
  margin-bottom: 18px;
}
.case-letter {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: var(--blue);
  color: white;
  font-size: 27px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  text-transform: lowercase;
}
.case-panel.support .case-letter { background: var(--green); }
.case-panel.boundary .case-letter { background: var(--red); }

.case-title {
  font-size: 25px;
  line-height: 1.1;
  color: var(--blue);
  font-weight: 700;
}
.case-panel.support .case-title { color: var(--green); }
.case-panel.boundary .case-title { color: var(--red); }
.case-subtitle {
  margin-top: 5px;
  font-size: 16px;
  color: #4d5866;
  line-height: 1.25;
}

.query-card {
  border: 2px solid #bdd3ee;
  background: var(--blue-light);
  border-radius: 16px;
  padding: 16px;
  margin-bottom: 16px;
}
.query-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.field-label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: #64758a;
  font-weight: 700;
  margin-bottom: 4px;
}
.field-value {
  font-size: 19px;
  line-height: 1.15;
  font-weight: 700;
  color: #182b43;
}
.field-value.small {
  font-size: 15px;
  font-weight: 600;
}

.score-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin: 14px 0 0;
}

.state-ribbon {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin: 13px 0 0;
  padding: 10px 12px;
  border-radius: 13px;
  border: 1.8px solid #d4e2f4;
  background: #fff;
}
.state-ribbon .state {
  font-size: 15px;
  font-weight: 800;
  color: var(--blue);
}
.case-panel.support .state-ribbon .state { color: var(--green); }
.case-panel.boundary .state-ribbon .state { color: var(--red); }
.state-ribbon .acq {
  font-size: 12.5px;
  line-height: 1.25;
  color: #586575;
  text-align: right;
}
.metric {
  background: #fff;
  border: 1.8px solid #d4e2f4;
  border-radius: 12px;
  padding: 9px 8px;
  text-align: center;
}
.metric .k {
  font-size: 11px;
  font-weight: 700;
  color: #667;
  text-transform: uppercase;
}
.metric .v {
  margin-top: 3px;
  font-size: 19px;
  font-weight: 800;
  color: var(--blue);
}
.case-panel.boundary .metric .v { color: var(--red); }

.section-title {
  font-size: 17px;
  font-weight: 800;
  color: var(--blue);
  margin: 16px 0 8px;
}
.case-panel.support .section-title { color: var(--green); }
.case-panel.boundary .section-title { color: var(--red); }

.graph-card {
  position: relative;
  height: 300px;
  border-radius: 18px;
  border: 2px dashed #9ab8dd;
  background: linear-gradient(180deg, #ffffff 0%, #f9fcff 100%);
  overflow: hidden;
}
.graph-card svg {
  width: 100%;
  height: 100%;
  display: block;
}

.evidence-grid {
  display: grid;
  grid-template-columns: 1.1fr .9fr;
  gap: 12px;
  margin-top: 12px;
}
.evidence-box {
  border-radius: 15px;
  padding: 13px 13px 12px;
  min-height: 145px;
}
.evidence-box.support-box {
  border: 2px solid #b4d7a8;
  background: var(--green-light);
}
.evidence-box.conflict-box {
  border: 2px solid #f1aaa6;
  background: var(--red-light);
}
.evidence-box.context-box {
  border: 2px solid #b9d1f0;
  background: var(--blue-light);
}
.evidence-box.gold-box {
  border: 2px solid #f0d892;
  background: var(--gold-light);
}
.evidence-box .box-title {
  font-size: 15px;
  font-weight: 800;
  margin-bottom: 7px;
}
.support-box .box-title { color: var(--green); }
.conflict-box .box-title { color: var(--red); }
.context-box .box-title { color: var(--blue); }
.gold-box .box-title { color: #a87d00; }
ul {
  margin: 0;
  padding-left: 18px;
}
li {
  font-size: 13.4px;
  line-height: 1.35;
  margin: 5px 0;
}

.triage-card {
  margin-top: 13px;
  padding: 14px 15px;
  border-radius: 15px;
  border: 2px solid #f0d892;
  background: var(--gold-light);
}
.triage-title {
  color: #a87d00;
  font-weight: 800;
  font-size: 15px;
  margin-bottom: 6px;
}
.triage-text {
  font-size: 14px;
  line-height: 1.38;
  color: #3d3d3d;
}

.decision {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 16px;
  background: #fff;
  border: 2px solid #e5edf6;
}
.decision .score {
  font-size: 34px;
  line-height: 1;
  font-weight: 800;
  color: var(--blue);
}
.case-panel.support .decision .score { color: var(--green); }
.case-panel.boundary .decision .score { color: var(--red); }
.decision .verdict {
  font-size: 18px;
  font-weight: 800;
  padding: 8px 12px;
  border-radius: 11px;
}
.verdict.prioritize {
  color: var(--green);
  background: var(--green-light);
  border: 1.8px solid #9dcc8f;
}
.verdict.no {
  color: var(--red);
  background: var(--red-light);
  border: 1.8px solid #e99b95;
}

@media print {
  body { padding: 0; }
  .figure-wrap { width: 1800px; }
}
"""


def esc(value: object) -> str:
    return html.escape(str(value or ""))


def fmt(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def read_cases() -> dict[str, dict[str, str]]:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["sample_id"]: row for row in rows if row.get("sample_id") in MAIN_CASES}


def evidence_list(items: list[str], fallback: str) -> str:
    values = items or [fallback]
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in values[:4]) + "</ul>"


def graph_svg(config: dict[str, object]) -> str:
    nodes = config["graph_nodes"]
    positions = [(118, 68), (382, 70), (404, 214), (116, 222)]
    center = (250, 150)

    def colors(kind: str) -> tuple[str, str, str]:
        if kind == "support":
            return "#4f8b3b", "#eef8eb", "#4f8b3b"
        if kind == "conflict":
            return "#d94d45", "#fff0ef", "#d94d45"
        if kind == "context":
            return "#15549a", "#eaf3ff", "#15549a"
        return "#777777", "#f4f4f4", "#777777"

    lines = [
        '<svg viewBox="0 0 500 300" xmlns="http://www.w3.org/2000/svg">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#6d7f92"/></marker></defs>',
    ]
    for idx, (kind, title, source, reliability) in enumerate(nodes):
        x, y = positions[idx]
        lines.append(
            f'<line x1="{center[0]}" y1="{center[1]}" x2="{x}" y2="{y}" '
            'stroke="#6d7f92" stroke-width="2.2" stroke-dasharray="5 5" marker-end="url(#arrow)" opacity="0.75"/>'
        )

    lines.append(
        f'<circle cx="{center[0]}" cy="{center[1]}" r="57" fill="#eaf3ff" stroke="#15549a" stroke-width="3"/>'
    )
    lines.append(
        f'<text x="{center[0]}" y="{center[1]-16}" text-anchor="middle" font-family="Arial" font-size="15" font-style="italic" fill="#202833">claim:</text>'
    )
    lines.append(
        f'<text x="{center[0]}" y="{center[1]+6}" text-anchor="middle" font-family="Arial" font-size="19" font-weight="700" fill="#15549a">DRUG treats</text>'
    )
    lines.append(
        f'<text x="{center[0]}" y="{center[1]+29}" text-anchor="middle" font-family="Arial" font-size="19" font-weight="700" fill="#15549a">DISEASE?</text>'
    )

    for idx, node in enumerate(nodes):
        kind, title, source, reliability = node
        stroke, fill, text_color = colors(kind)
        x, y = positions[idx]
        width, height = 172, 68
        rx, ry = x - width / 2, y - height / 2
        icon = "✓" if kind == "support" else ("×" if kind == "conflict" else ("i" if kind == "context" else "?"))
        lines.append(
            f'<rect x="{rx}" y="{ry}" width="{width}" height="{height}" rx="13" fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>'
        )
        lines.append(f'<circle cx="{rx+12}" cy="{ry+12}" r="16" fill="{stroke}"/>')
        lines.append(
            f'<text x="{rx+12}" y="{ry+19}" text-anchor="middle" font-family="Arial" font-size="22" font-weight="800" fill="#fff">{icon}</text>'
        )
        title_words = str(title).split()
        if len(title_words) > 3:
            line1 = " ".join(title_words[:3])
            line2 = " ".join(title_words[3:])
            lines.append(f'<text x="{x}" y="{ry+25}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="800" fill="{text_color}">{esc(line1)}</text>')
            lines.append(f'<text x="{x}" y="{ry+43}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="800" fill="{text_color}">{esc(line2)}</text>')
            source_y = ry + 60
        else:
            lines.append(f'<text x="{x}" y="{ry+30}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="800" fill="{text_color}">{esc(title)}</text>')
            source_y = ry + 50
        subtitle = f"{source} {reliability}".strip()
        lines.append(f'<text x="{x}" y="{source_y}" text-anchor="middle" font-family="Arial" font-size="12.5" font-weight="700" fill="#333">{esc(subtitle)}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def case_panel(row: dict[str, str], config: dict[str, object]) -> str:
    pred = int(float(row.get("prediction") or 0))
    label = int(float(row.get("label") or 0))
    score = float(row.get("judge_score") or 0.0)
    decision = "Prioritize" if pred == 1 else "Do not prioritize"
    decision_class = "prioritize" if pred == 1 else "no"
    kind_class = "support" if pred == 1 else "boundary"

    return f"""
    <section class="case-panel {kind_class}">
      <div class="case-header">
        <div class="case-letter">{esc(config['panel'])}</div>
        <div>
          <div class="case-title">{esc(config['short_title'])}</div>
          <div class="case-subtitle">{esc(config['clinical_message'])}</div>
        </div>
      </div>

      <div class="query-card">
        <div class="query-grid">
          <div>
            <div class="field-label">Candidate drug</div>
            <div class="field-value">{esc(row['drug'])}</div>
          </div>
          <div>
            <div class="field-label">Target disease</div>
            <div class="field-value">{esc(row['disease'])}</div>
          </div>
        </div>
        <div class="score-strip">
          <div class="metric"><div class="k">Label / Pred</div><div class="v">{label}/{pred}</div></div>
          <div class="metric"><div class="k">Score</div><div class="v">{score:.2f}</div></div>
          <div class="metric"><div class="k">Grade</div><div class="v">{esc(row['evidence_grade'])}</div></div>
          <div class="metric"><div class="k">Calls</div><div class="v">{esc(row['expert_calls'])}</div></div>
        </div>
        <div class="state-ribbon">
          <div>
            <div class="field-label">Evidence state</div>
            <div class="state">{esc(config['evidence_state'])}</div>
          </div>
          <div class="acq">{esc(config['acquisition_summary'])}</div>
        </div>
      </div>

      <div class="section-title">Claim-centered EvidenceGraph</div>
      <div class="graph-card">{graph_svg(config)}</div>

      <div class="evidence-grid">
        <div class="evidence-box support-box">
          <div class="box-title">Biomedical interpretation</div>
          {evidence_list(list(config['biomedical_interpretation']), "No dominant interpretation.")}
        </div>
        <div class="evidence-box gold-box">
          <div class="box-title">Triage implication</div>
          <div class="triage-text">{esc(config['triage_meaning'])}</div>
        </div>
      </div>

      <div class="triage-card">
        <div class="triage-title">Constrained evidence review</div>
        <div class="triage-text">{esc(row.get('judge_reason', ''))}</div>
      </div>

      <div class="decision">
        <div>
          <div class="field-label">Treatment-priority score</div>
          <div class="score">{score:.2f}</div>
        </div>
        <div class="verdict {decision_class}">{decision}</div>
      </div>
    </section>
    """


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_cases()
    panels: list[str] = []

    for sample_id, config in MAIN_CASES.items():
        row = rows[sample_id]
        panel = case_panel(row, config)
        panels.append(panel)
        slug = f"{config['panel']}_{row['drug']}_{row['disease']}".lower()
        slug = "".join(ch if ch.isalnum() else "_" for ch in slug)
        slug = "_".join(part for part in slug.split("_") if part)
        body = f'<div class="figure-wrap" style="width: 620px;">{panel}</div>'
        (OUT_DIR / f"{slug}.html").write_text(page(f"Case panel {sample_id}", body), encoding="utf-8")

    combined_body = f"""
    <div class="figure-wrap">
      <div class="figure-title">
        <div class="label">Case studies</div>
        <div class="text">Auditable treatment-priority decisions from claim-centered EvidenceGraphs</div>
      </div>
      <div class="panel-row">
        {''.join(panels)}
      </div>
    </div>
    """
    (OUT_DIR / "figure5_case_study_panels.html").write_text(
        page("Figure 5 case study panels", combined_body),
        encoding="utf-8",
    )
    print(f"Wrote HTML case panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
