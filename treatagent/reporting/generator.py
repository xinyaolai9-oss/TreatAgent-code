#!/usr/bin/env python3
import json
import os
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from treatagent.utils import build_sample_id

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError:  # pragma: no cover - optional dependency in some envs
    Chem = None
    rdMolDraw2D = None


class InteractiveReportGenerator:
    def __init__(self, report_dir: str = "reports", template_path: str = "assets/templates/report_template.html"):
        base_dir = Path(__file__).resolve().parents[2]
        self.report_dir = Path(report_dir)
        if not self.report_dir.is_absolute():
            self.report_dir = base_dir / self.report_dir
        self.template_path = Path(template_path)
        if not self.template_path.is_absolute():
            self.template_path = base_dir / self.template_path
        os.makedirs(self.report_dir, exist_ok=True)

    def generate(
        self,
        evidence_graph: Dict[str, Any],
        expert_outputs: Dict[str, Dict[str, Any]],
        smiles: str,
        disease: str,
        raw_score: float,
        calibrated_prob: float,
        synthesis_explanation: str,
        sample_id: Optional[str] = None,
        trajectory: Optional[List[Dict[str, Any]]] = None,
        synthesis_source: Optional[str] = None,
        group_scores: Optional[Dict[str, Any]] = None,
        memory_similar_cases: int = 0,
        knowledge_cutoff_date: Optional[str] = None,
        label: Optional[int] = None,
        final_prediction: Optional[int] = None,
        final_score: Optional[float] = None,
        final_threshold: Optional[float] = None,
        final_score_source: Optional[str] = None,
        argument_factors: Optional[Dict[str, Any]] = None,
        case_role: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        sample_id = sample_id or build_sample_id(smiles, disease)
        safe_name = self._slugify(f"{sample_id}_{disease}")
        report_path = str((self.report_dir / f"{safe_name}.html").resolve())

        admet_metrics = expert_outputs.get("ADMET", {}).get("metrics", {})
        dti_score = expert_outputs.get("DTI", {}).get("raw_data", {}).get("dti_score", 0.0)
        clinical_success_rate = expert_outputs.get("Clinical", {}).get("raw_data", {}).get("disease_success_rate", 0.0)
        evidence_rows = evidence_graph.get("evidence", [])
        typed_evidence_rows = evidence_graph.get("typed_evidence", []) or evidence_rows
        group_scores = group_scores or {}
        trajectory = trajectory or []
        argument_factors = argument_factors or {}
        final_score = calibrated_prob if final_score is None else final_score
        final_threshold = 0.5 if final_threshold is None else final_threshold
        if final_prediction is None:
            final_prediction = 1 if final_score >= final_threshold else 0
        matched_drug_name = self._drugkb_match_label(expert_outputs.get("DrugKB"))
        molecule_svg = self._build_molecule_svg(smiles)

        template = Template(self._load_template())
        html = template.safe_substitute(
            sample_id=self._escape_html(sample_id),
            disease=self._escape_html(disease),
            smiles=self._escape_html(smiles),
            matched_drug_name=self._escape_html(matched_drug_name),
            molecule_svg=molecule_svg,
            case_role=self._escape_html(case_role or "TreatAgent-ARG case report"),
            prediction_label="Positive" if final_prediction == 1 else "Negative",
            ground_truth=self._escape_html("Unknown" if label is None else str(label)),
            final_prediction=self._escape_html(str(final_prediction)),
            final_decision=self._escape_html("Treat" if final_prediction == 1 else "Do not treat"),
            final_score=f"{float(final_score):.4f}",
            final_threshold=f"{float(final_threshold):.4f}",
            final_score_source=self._escape_html(final_score_source or "unknown"),
            raw_score=f"{raw_score:.4f}",
            calibrated_prob=f"{calibrated_prob:.4f}",
            supportive_count=str(evidence_graph.get("supportive_evidence", 0)),
            risk_count=str(evidence_graph.get("risk_evidence", 0)),
            mean_confidence=f"{float(evidence_graph.get('mean_confidence', 0.0)):.4f}",
            conflict_level=f"{float(evidence_graph.get('conflict_level', 0.0)):.4f}",
            synthesis_source=self._escape_html(synthesis_source or "unknown"),
            memory_similar_cases=str(memory_similar_cases),
            knowledge_cutoff_date=self._escape_html(knowledge_cutoff_date or "None"),
            synthesis_explanation=self._escape_html(synthesis_explanation),
            evidence_rows=self._build_table_rows(evidence_rows),
            trajectory_rows=self._build_trajectory_rows(trajectory),
            expert_rows=self._build_expert_rows(expert_outputs, evidence_graph.get("coverage", {})),
            top_evidence_items=self._build_top_evidence_items(evidence_graph.get("top_evidence", [])),
            group_score_cards=self._build_group_score_cards(group_scores),
            hard_subset_badges=self._build_hard_subset_badges(argument_factors),
            argument_factor_cards=self._build_argument_factor_cards(argument_factors),
            evidence_graph_panel=self._build_evidence_graph_panel(typed_evidence_rows),
            support_argument_rows=self._build_argument_rows(typed_evidence_rows, "support"),
            conflict_argument_rows=self._build_argument_rows(typed_evidence_rows, "conflict"),
            chart_data=json.dumps(
                {
                    "admet": self._build_admet_chart_data(admet_metrics),
                    "dti_score": dti_score or 0.0,
                    "clinical_success_rate": clinical_success_rate or 0.0,
                    "group_scores": group_scores,
                },
                ensure_ascii=False,
            ),
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        report_summary = {
            "report_path": report_path,
            "sample_id": sample_id,
            "prediction_label": "Positive" if final_prediction == 1 else "Negative",
            "raw_score": round(raw_score, 4),
            "calibrated_probability": round(calibrated_prob, 4),
            "final_score": round(float(final_score), 4),
            "final_prediction": int(final_prediction),
            "evidence_count": len(evidence_rows),
        }
        return report_path, report_summary

    def generate_from_result(
        self,
        result: Dict[str, Any],
        case_role: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Render a report directly from a saved TreatAgent result row."""
        return self.generate(
            evidence_graph=result.get("evidence_graph") or {},
            expert_outputs=result.get("expert_outputs") or {},
            smiles=str(result.get("smiles") or result.get("drug") or ""),
            disease=str(result.get("disease") or ""),
            raw_score=float(result.get("raw_score") or 0.0),
            calibrated_prob=float(result.get("calibrated_probability") or result.get("prediction_score") or 0.0),
            synthesis_explanation=str(result.get("synthesis_explanation") or ""),
            sample_id=str(result.get("sample_id") or ""),
            trajectory=result.get("trajectory") or [],
            synthesis_source=str(result.get("synthesis_source") or "unknown"),
            group_scores=result.get("group_scores") or {},
            memory_similar_cases=int(result.get("memory_similar_cases") or 0),
            knowledge_cutoff_date=result.get("knowledge_cutoff_date"),
            label=result.get("label"),
            final_prediction=result.get("argument_prediction", result.get("prediction_binary")),
            final_score=result.get("argument_probability", result.get("prediction_score")),
            final_threshold=result.get("final_threshold"),
            final_score_source=result.get("final_score_source"),
            argument_factors=result.get("argument_factors") or {},
            case_role=case_role,
        )

    def _load_template(self) -> str:
        with open(self.template_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_admet_chart_data(self, metrics: Dict[str, Any]) -> Dict[str, List[float]]:
        return {
            "labels": ["Bioavailability", "HIA", "BBB", "Safety", "Clinical Tox"],
            "values": [
                self._normalize(metrics.get("bioavailability")),
                self._normalize(metrics.get("hia")),
                self._normalize(metrics.get("bbb")),
                1 - self._normalize(metrics.get("herg")),
                1 - self._normalize(metrics.get("clinical_toxicity")),
            ],
        }

    def _drugkb_match_label(self, drugkb_output: Optional[Dict[str, Any]]) -> str:
        if not drugkb_output:
            return "DrugKB not queried by planner"
        raw_data = drugkb_output.get("raw_data") or {}
        record = drugkb_output.get("record") or {}
        drug_name = raw_data.get("drug_name") or record.get("drug_name")
        if drug_name:
            return str(drug_name)
        status = str(drugkb_output.get("status") or "").strip()
        if status == "no_data":
            return "Candidate not matched in DrugKB"
        if status:
            return f"DrugKB status: {status}"
        return "Candidate not matched in DrugKB"

    def _build_table_rows(self, evidence_rows: List[Dict[str, Any]]) -> str:
        rows = []
        for item in evidence_rows:
            rows.append(
                "<tr>"
                f"<td>{self._escape_html(str(item.get('expert', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('category', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('claim', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('impact', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('confidence', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('source', '')))}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    def _build_trajectory_rows(self, trajectory: List[Dict[str, Any]]) -> str:
        rows = []
        for step in trajectory:
            planner_output = step.get("planner_output", {})
            response = step.get("expert_response_summary") or {}
            action = str(planner_output.get("next_action", ""))
            response_status = str(response.get("status") or "").strip()
            response_count = ""
            answer = str(response.get("answer_to_question") or "")
            if "structured evidence item" in answer:
                response_count = answer.split("structured evidence item")[0].split()[-1]
            details = []
            question = planner_output.get("planner_question")
            if question:
                details.append(f"Q: {self._clip(question, 150)}")
            if answer:
                details.append(f"A: {self._clip(answer, 120)}")
            if response.get("gap_resolved") is not None:
                resolved = response.get("gap_resolved") or {}
                if isinstance(resolved, dict):
                    resolved_names = [name for name, value in resolved.items() if value]
                    if resolved_names:
                        details.append(f"Resolved: {', '.join(resolved_names)}")
            detail_text = " | ".join(str(item) for item in details if item)
            if planner_output.get("reason"):
                reason = self._clip(planner_output.get("reason"), 110)
                detail_text = f"{reason} | {detail_text}" if detail_text else reason
            action_label = action
            if response_status:
                action_label += f" ({response_status})"
            if response_count:
                action_label += f", n={response_count}"
            rows.append(
                "<tr>"
                f"<td>{self._escape_html(str(step.get('round', '')))}</td>"
                f"<td>{self._escape_html(action_label)}</td>"
                f"<td>{self._escape_html(detail_text)}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    def _build_expert_rows(self, expert_outputs: Dict[str, Dict[str, Any]], coverage: Dict[str, Any]) -> str:
        rows = []
        for expert in ["DiseaseKB", "DrugKB", "DTI", "ADMET", "Clinical"]:
            payload = expert_outputs.get(expert, {})
            rows.append(
                "<tr>"
                f"<td>{self._escape_html(expert)}</td>"
                f"<td>{self._escape_html(str(payload.get('status', 'not_called')))}</td>"
                f"<td>{self._escape_html(str(coverage.get(expert, 0)))}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    def _build_top_evidence_items(self, claims: List[str]) -> str:
        if not claims:
            return "<li>No top evidence was extracted.</li>"
        return "\n".join(f"<li>{self._escape_html(claim)}</li>" for claim in claims)

    def _build_group_score_cards(self, group_scores: Dict[str, Any]) -> str:
        if not group_scores:
            return "<div class=\"mini-card\"><span class=\"mini-label\">No grouped scores</span></div>"
        cards = []
        ordered = ["disease_context", "drug_context", "mechanism", "safety", "clinical_prior", "knowledge"]
        labels = {
            "disease_context": "Disease Context",
            "drug_context": "Drug Context",
            "mechanism": "Mechanism",
            "safety": "Safety",
            "clinical_prior": "Clinical Prior",
            "knowledge": "Knowledge",
        }
        for key in ordered:
            if key not in group_scores:
                continue
            value = self._normalize(group_scores.get(key))
            cards.append(
                "<div class=\"mini-card\">"
                f"<div class=\"mini-label\">{self._escape_html(labels.get(key, key))}</div>"
                f"<div class=\"mini-value\">{value:.4f}</div>"
                f"<div class=\"bar\"><div class=\"bar-fill\" style=\"width:{value * 100:.2f}%\"></div></div>"
                "</div>"
            )
        return "\n".join(cards)

    def _build_hard_subset_badges(self, factors: Dict[str, Any]) -> str:
        tags = []
        direct = self._normalize(factors.get("direct_support"))
        clinical = self._normalize(factors.get("clinical_feasibility"))
        mechanism = self._normalize(factors.get("mechanism_support"))
        admet_risk = max(
            self._normalize(factors.get("safety_conflict")),
            self._normalize(factors.get("admet_bbb_non_cns_noise")),
        )
        if direct <= 1e-9:
            tags.append("No direct indication")
        if clinical <= 0.4:
            tags.append("Low clinical prior")
        if admet_risk >= 0.05:
            tags.append("ADMET-risk")
        if direct <= 1e-9 and mechanism >= 0.5:
            tags.append("Mechanism-only support")
        if not tags:
            tags.append("Standard evidence state")
        return "\n".join(f"<span class=\"tag\">{self._escape_html(tag)}</span>" for tag in tags)

    def _build_argument_factor_cards(self, factors: Dict[str, Any]) -> str:
        if not factors:
            return "<div class=\"signal-row\"><span>No ARG factors</span></div>"
        ordered = [
            ("direct_support", "Direct support", 1),
            ("mechanism_support", "Mechanism support", 1),
            ("clinical_feasibility", "Clinical feasibility", 1),
            ("cross_source_consistency", "Cross-source consistency", 1),
            ("conflict_strength", "Conflict strength", -1),
            ("admet_bbb_non_cns_noise", "ADMET risk", -1),
            ("missing_penalty", "Missing penalty", -1),
        ]
        rows = []
        for key, label, sign in ordered:
            value = self._normalize(factors.get(key))
            width = value * 50.0
            signed_value = value * sign
            if sign > 0:
                bar = f"<div class=\"signal-pos\" style=\"width:{width:.2f}%\"></div>"
            else:
                bar = f"<div class=\"signal-neg\" style=\"width:{width:.2f}%\"></div>"
            rows.append(
                "<div class=\"signal-row\">"
                f"<div class=\"signal-label\">{self._escape_html(label)}</div>"
                "<div class=\"signal-track\"><div class=\"axis\"></div>"
                f"{bar}</div>"
                f"<div class=\"signal-value\">{signed_value:+.3f}</div>"
                "</div>"
            )
        return "\n".join(rows)

    def _build_evidence_graph_panel(self, evidence_rows: List[Dict[str, Any]]) -> str:
        support_nodes = self._evidence_nodes(evidence_rows, "support", limit=4)
        conflict_nodes = self._evidence_nodes(evidence_rows, "conflict", limit=3)
        return (
            "<div class=\"evidence-graph\">"
            f"<div class=\"node-column support-col\">{support_nodes}</div>"
            "<div class=\"claim-core\">"
            "<div class=\"claim-title\">Drug treats disease?</div>"
            "<div class=\"claim-subtitle\">ARG claim</div>"
            "</div>"
            f"<div class=\"node-column conflict-col\">{conflict_nodes}</div>"
            "</div>"
        )

    def _evidence_nodes(self, evidence_rows: List[Dict[str, Any]], direction: str, limit: int) -> str:
        nodes = []
        for item in evidence_rows:
            item_direction = str(item.get("direction") or item.get("impact") or "").lower()
            if direction == "support" and "support" not in item_direction and item.get("impact") != "positive":
                continue
            if direction == "conflict" and not any(token in item_direction for token in ["conflict", "risk", "negative"]):
                continue
            reliability = item.get("reliability", item.get("confidence", item.get("score", item.get("strength", ""))))
            nodes.append(
                "<div class=\"evidence-node\">"
                f"<div class=\"node-head\">{self._escape_html(str(item.get('expert', '')))} · {self._escape_html(str(item.get('category', '')))}</div>"
                f"<div class=\"node-claim\">{self._escape_html(self._clip(item.get('claim', ''), 105))}</div>"
                f"<div class=\"node-r\">r={self._escape_html(str(reliability))}</div>"
                "</div>"
            )
            if len(nodes) >= limit:
                break
        if not nodes:
            return "<div class=\"evidence-node muted-node\">No dominant signal</div>"
        return "\n".join(nodes)

    def _build_argument_rows(self, evidence_rows: List[Dict[str, Any]], direction: str) -> str:
        rows = []
        for item in evidence_rows:
            item_direction = str(item.get("direction") or item.get("impact") or "").lower()
            if direction == "support" and "support" not in item_direction and item.get("impact") != "positive":
                continue
            if direction == "conflict" and not any(token in item_direction for token in ["conflict", "risk", "negative"]):
                continue
            reliability = item.get("reliability", item.get("confidence", ""))
            rows.append(
                "<tr>"
                f"<td>{self._escape_html(str(item.get('expert', '')))}</td>"
                f"<td>{self._escape_html(str(item.get('category', '')))}</td>"
                f"<td>{self._escape_html(self._clip(item.get('claim', ''), 135))}</td>"
                f"<td>{self._escape_html(str(reliability or item.get('score', item.get('strength', ''))))}</td>"
                "</tr>"
            )
        if not rows:
            return "<tr><td colspan=\"4\">No dominant evidence in this group.</td></tr>"
        return "\n".join(rows[:5])

    def _clip(self, text: Any, limit: int) -> str:
        value = str(text or "").replace("\n", " ").strip()
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)].rstrip() + "..."

    def _normalize(self, value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, numeric))

    def _build_molecule_svg(self, smiles: str) -> str:
        if not smiles or Chem is None or rdMolDraw2D is None:
            return "<div class=\"structure-fallback\">Structure preview unavailable.</div>"
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "<div class=\"structure-fallback\">Invalid SMILES for structure rendering.</div>"
            Chem.rdDepictor.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DSVG(420, 280)
            options = drawer.drawOptions()
            options.clearBackground = False
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            return drawer.GetDrawingText()
        except Exception:
            return "<div class=\"structure-fallback\">Structure preview unavailable.</div>"

    def _slugify(self, text: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:120]

    def _escape_html(self, text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
