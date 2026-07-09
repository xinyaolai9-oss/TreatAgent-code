#!/usr/bin/env python3
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests

from treatagent.config.runtime import AGENT_CONFIGS, API_CONFIG, get_api_headers, get_model_name, llm_synthesis_enabled
from treatagent.memory.manager import LongTermMemoryManager
from treatagent.skills import (
    SkillRegistry,
    SkillSpec,
    admet_data,
    get_disease_success_prior,
    get_dti_score_ensemble,
    resolve_target_sequence,
    DrugKBExpert,
    DiseaseKBExpert,
)
from treatagent.reporting.generator import InteractiveReportGenerator
from treatagent.utils import build_sample_id
from treatagent.orchestration.argument_graph_scorer import argument_factors_from_result
from treatagent.orchestration.evidence import TypedEvidenceTuple, legacy_evidence_to_tuple


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


RUBRIC_GRADE_INTERVALS = {
    "A": (0.75, 0.95),
    "B": (0.55, 0.74),
    "C": (0.35, 0.54),
    "D": (0.15, 0.34),
    "E": (0.00, 0.14),
}
RUBRIC_GRADE_RANK = {"E": 0, "D": 1, "C": 2, "B": 3, "A": 4}

DIRECT_EVIDENCE_GRADES = {"none", "weak", "moderate", "strong"}
MECHANISM_GRADES = {"none", "indirect", "disease_relevant", "cross_source_consistent"}
CLINICAL_FEASIBILITY_GRADES = {"low", "moderate", "high"}
SAFETY_CONFLICT_GRADES = {"none", "manageable", "significant", "severe"}


@dataclass
class EvidenceItem:
    expert: str
    category: str
    claim: str
    value: Any
    impact: str
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert": self.expert,
            "category": self.category,
            "claim": self.claim,
            "value": self.value,
            "impact": self.impact,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "metadata": self.metadata,
        }


class EvidenceGraph:
    def __init__(self, formula: str, disease: str, use_derived_claims: bool = True):
        self.formula = formula
        self.disease = disease
        self.use_derived_claims = use_derived_claims
        self.nodes = [
            {"id": "drug", "type": "Drug", "name": formula},
            {"id": "disease", "type": "Disease", "name": disease},
        ]
        self.edges: List[Dict[str, Any]] = []
        self.evidence_items: List[EvidenceItem] = []
        self.typed_evidence: List[TypedEvidenceTuple] = []

    def add_evidence(self, item: EvidenceItem) -> None:
        self.evidence_items.append(item)
        typed_tuple = legacy_evidence_to_tuple(item, self.formula, self.disease)
        self.typed_evidence.append(typed_tuple)
        evidence_id = f"evidence_{len(self.evidence_items)}"
        self.nodes.append({"id": evidence_id, "type": "Evidence", "expert": item.expert, "category": item.category, "claim": item.claim})
        relation = "supports" if typed_tuple.direction == "support" else "contraindicates" if typed_tuple.direction == "conflict" else "describes"
        self.edges.append({"from": evidence_id, "to": "drug", "relation": "describes", "confidence": round(item.confidence, 4), "typed_relation": typed_tuple.relation})
        self.edges.append({"from": evidence_id, "to": "disease", "relation": relation, "confidence": round(item.confidence, 4), "typed_direction": typed_tuple.direction, "reliability": round(typed_tuple.reliability, 4)})

    def expert_coverage(self) -> Dict[str, int]:
        coverage: Dict[str, int] = {}
        for item in self.evidence_items:
            coverage[item.expert] = coverage.get(item.expert, 0) + 1
        return coverage

    def supportive_count(self) -> int:
        return sum(1 for item in self.typed_evidence if item.direction == "support")

    def risk_count(self) -> int:
        return sum(1 for item in self.typed_evidence if item.direction == "conflict")

    def mean_confidence(self) -> float:
        if not self.evidence_items:
            return 0.0
        return sum(item.confidence for item in self.evidence_items) / len(self.evidence_items)

    def conflict_level(self) -> float:
        total = len(self.typed_evidence)
        if total == 0:
            return 0.0
        supportive = self.supportive_count()
        risk = self.risk_count()
        if supportive == 0 or risk == 0:
            return 0.0
        return min(supportive, risk) / total

    def top_evidence(self, limit: int = 4) -> List[str]:
        ranked = sorted(self.evidence_items, key=lambda item: item.confidence, reverse=True)
        return [item.claim for item in ranked[:limit]]

    def _target_key(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        return re.sub(r"[^a-z0-9]+", "", text)

    def _typed_target(self, item: TypedEvidenceTuple) -> str:
        metadata = item.metadata or {}
        return str(
            metadata.get("target_gene")
            or metadata.get("target_symbol")
            or metadata.get("target_name")
            or metadata.get("target_id")
            or item.object
            or ""
        ).strip()

    def _is_broad_disease(self) -> bool:
        disease = str(self.disease or "").strip().lower()
        if not disease:
            return False
        broad_terms = {
            "cancer",
            "tumor",
            "tumors",
            "neoplasm",
            "neoplasms",
            "carcinoma",
            "malignancy",
            "malignancies",
        }
        return disease in broad_terms

    def _claim_row(
        self,
        *,
        claim_id: str,
        claim: str,
        direction: str,
        semantic_role: str,
        confidence: float,
        reliability: float,
        derived_from: List[str],
        reason: str,
    ) -> Dict[str, Any]:
        strength = max(0.0, min(1.0, float(confidence) * float(reliability)))
        return {
            "claim_id": claim_id,
            "claim": claim,
            "direction": direction,
            "semantic_role": semantic_role,
            "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
            "reliability": round(max(0.0, min(1.0, float(reliability))), 4),
            "strength": round(strength, 4),
            "derived_from": derived_from,
            "reason": reason,
        }

    def derived_argument_claims(self) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        typed = self.typed_evidence

        direct_items = [
            item for item in typed
            if item.expert == "DrugKB" and item.semantic_role == "direct_indication" and item.direction == "support"
        ]
        for idx, item in enumerate(direct_items, 1):
            claims.append(self._claim_row(
                claim_id=f"direct_indication_{idx}",
                claim=f"Drug-side indication evidence directly overlaps the target disease: {item.claim}",
                direction="support",
                semantic_role="direct_indication_match",
                confidence=item.confidence,
                reliability=item.reliability,
                derived_from=[item.claim],
                reason="Direct disease indication overlap is treatment-relevant support.",
            ))

        drug_targets: Dict[str, List[TypedEvidenceTuple]] = {}
        disease_targets: Dict[str, List[TypedEvidenceTuple]] = {}
        for item in typed:
            if item.expert == "DrugKB" and item.semantic_role == "drug_target":
                key = self._target_key(self._typed_target(item))
                if key:
                    drug_targets.setdefault(key, []).append(item)
            if item.expert == "DiseaseKB" and item.semantic_role in {"disease_target", "clinical_target_context"}:
                key = self._target_key(self._typed_target(item))
                if key:
                    disease_targets.setdefault(key, []).append(item)

        for idx, target_key in enumerate(sorted(set(drug_targets) & set(disease_targets)), 1):
            drug_item = drug_targets[target_key][0]
            disease_item = disease_targets[target_key][0]
            target_label = self._typed_target(drug_item) or self._typed_target(disease_item) or target_key
            confidence = max(drug_item.confidence, disease_item.confidence)
            reliability = min(drug_item.reliability, disease_item.reliability)
            claims.append(self._claim_row(
                claim_id=f"target_overlap_{idx}",
                claim=f"The candidate drug and disease context converge on target {target_label}.",
                direction="support",
                semantic_role="target_overlap",
                confidence=confidence,
                reliability=reliability,
                derived_from=[drug_item.claim, disease_item.claim],
                reason="Drug-side target evidence is grounded by disease-side target evidence.",
            ))

        for idx, item in enumerate([item for item in typed if item.expert == "ADMET" and item.direction == "conflict"], 1):
            claims.append(self._claim_row(
                claim_id=f"safety_conflict_{idx}",
                claim=item.claim,
                direction="conflict",
                semantic_role="safety_conflict",
                confidence=item.confidence,
                reliability=item.reliability,
                derived_from=[item.claim],
                reason="Safety or developability risk can weaken treatment suitability.",
            ))

        for idx, item in enumerate([item for item in typed if item.expert == "Clinical"], 1):
            score = float(item.score or 0.0)
            if score >= 0.7:
                direction = "support"
                reason = "High disease-level clinical feasibility can strengthen translational plausibility."
            elif score <= 0.35:
                direction = "conflict"
                reason = "Low disease-level clinical feasibility weakens translational plausibility."
            else:
                direction = "neutral"
                reason = "Moderate disease-level clinical feasibility is contextual and should not by itself support treatment."
            claims.append(self._claim_row(
                claim_id=f"clinical_prior_{idx}",
                claim=item.claim,
                direction=direction,
                semantic_role="clinical_prior_modifier",
                confidence=item.confidence,
                reliability=item.reliability,
                derived_from=[item.claim],
                reason=reason,
            ))

        has_drugkb = any(item.expert == "DrugKB" for item in typed)
        has_direct = bool(direct_items)
        if has_drugkb and not has_direct:
            claims.append(self._claim_row(
                claim_id="missing_direct_indication",
                claim="No direct drug-side indication support was derived for the target disease.",
                direction="missing",
                semantic_role="missing_direct_evidence",
                confidence=0.8,
                reliability=0.8,
                derived_from=["DrugKB"],
                reason="A DrugKB match without direct disease indication should not be treated as support.",
            ))
        has_target_overlap = any(
            item.get("semantic_role") == "target_overlap" and item.get("direction") == "support"
            for item in claims
        )
        if self._is_broad_disease() and not has_direct and not has_target_overlap:
            claims.append(self._claim_row(
                claim_id="broad_disease_specificity_gap",
                claim=(
                    "The target disease is broad; generic disease context is insufficient "
                    "without direct indication or disease-specific target/pathway grounding."
                ),
                direction="missing",
                semantic_role="disease_specificity_gap",
                confidence=0.85,
                reliability=0.8,
                derived_from=["disease"],
                reason="Broad disease labels require stronger disease-specific grounding.",
            ))
        return claims

    def argument_graph_summary(self) -> Dict[str, Any]:
        claims = self.derived_argument_claims() if self.use_derived_claims else []
        support = [item for item in claims if item["direction"] == "support"]
        conflict = [item for item in claims if item["direction"] == "conflict"]
        missing = [item for item in claims if item["direction"] == "missing"]
        neutral_context = [
            item.to_dict() for item in self.typed_evidence
            if item.direction == "neutral"
        ][:20]
        return {
            "central_claim": f"The candidate molecule treats {self.disease}.",
            "support_claims": support,
            "conflict_claims": conflict,
            "missing_evidence": missing,
            "neutral_context": neutral_context,
            "derived_claims": claims,
            "derived_links": [
                {
                    "claim_id": item["claim_id"],
                    "semantic_role": item["semantic_role"],
                    "direction": item["direction"],
                    "derived_from": item["derived_from"],
                }
                for item in claims
            ],
            "source_coverage": self.expert_coverage(),
            "derived_claims_enabled": self.use_derived_claims,
        }

    def evidence_by_expert(self, expert: str) -> List[EvidenceItem]:
        return [item for item in self.evidence_items if item.expert == expert]

    def has_expert_evidence(self, expert: str) -> bool:
        return bool(self.evidence_by_expert(expert))

    def category_values(self, expert: Optional[str] = None, categories: Optional[List[str]] = None) -> List[float]:
        values: List[float] = []
        for item in self.evidence_items:
            if expert is not None and item.expert != expert:
                continue
            if categories is not None and item.category not in categories:
                continue
            if isinstance(item.value, (int, float)):
                values.append(float(item.value))
        return values

    def summary(self) -> Dict[str, Any]:
        return {
            "drug": self.formula,
            "disease": self.disease,
            "coverage": self.expert_coverage(),
            "supportive_evidence": self.supportive_count(),
            "risk_evidence": self.risk_count(),
            "mean_confidence": round(self.mean_confidence(), 4),
            "conflict_level": round(self.conflict_level(), 4),
            "top_evidence": self.top_evidence(),
            "evidence": [item.to_dict() for item in self.evidence_items],
            "typed_evidence": [item.to_dict() for item in self.typed_evidence],
            "argument_graph": self.argument_graph_summary(),
            "derived_argument_claims": self.derived_argument_claims(),
            "nodes": self.nodes,
            "edges": self.edges,
        }


class ScoreCalibrator:
    def __init__(self, slope: float = 1.25, intercept: float = 0.0):
        self.slope = slope
        self.intercept = intercept

    def calibrate(self, raw_score: float, graph: EvidenceGraph) -> Dict[str, float]:
        base_probability = max(0.0, min(1.0, raw_score / 10.0))
        # Re-center around 0.55 and amplify differences instead of shifting the
        # entire distribution upward.
        centered_probability = 0.5 + ((base_probability - 0.55) * self.slope)
        confidence_bonus = min(0.03, max(0.0, graph.mean_confidence() - 0.7) * 0.15)
        conflict_penalty = min(0.18, graph.conflict_level() * 0.28)
        calibrated = centered_probability + self.intercept + confidence_bonus - conflict_penalty
        calibrated = max(0.0, min(1.0, calibrated))
        return {
            "raw_probability": round(base_probability, 4),
            "centered_probability": round(centered_probability, 4),
            "calibrated_probability": round(calibrated, 4),
            "confidence_bonus": round(confidence_bonus, 4),
            "conflict_penalty": round(conflict_penalty, 4),
        }


class TreatAgentOrchestrator:
    def __init__(
        self,
        model: str = "gpt-4o",
        agent_version: str = "eg",
        max_rounds: int = 5,
        planner_budget: Optional[int] = None,
        generate_report: bool = False,
        use_memory: bool = False,
        knowledge_cutoff_date: Optional[str] = None,
    ):
        self.model = model
        self.agent_version = self._normalize_agent_version(agent_version)
        self.max_rounds = max_rounds
        self.planner_budget = self._init_planner_budget(planner_budget)
        self.arg_threshold = self._init_arg_threshold()
        self.generate_report = generate_report
        self.use_memory = use_memory
        self.knowledge_cutoff_date = knowledge_cutoff_date
        self.calibrator = ScoreCalibrator()
        self.report_generator = InteractiveReportGenerator()
        self.api_headers = get_api_headers()
        llm_available = llm_synthesis_enabled() and str(model).lower() != "local" and bool(API_CONFIG["url"])
        self.use_llm_planner = llm_available and self.agent_version == "full" and self._env_flag("TREATAGENT_USE_LLM_PLANNER", True)
        self.use_llm_explanation = llm_available and self.agent_version == "full"
        self.use_llm_judge = llm_available and self.agent_version == "full" and self._env_flag("TREATAGENT_USE_LLM_JUDGE", True)
        self.use_llm_synthesis = llm_available and self.agent_version == "ls"
        self.use_llm_experts = llm_available and self.agent_version == "full" and self._env_flag("TREATAGENT_USE_LLM_EXPERTS", False)
        self.force_all_experts = self.agent_version == "full" and self._env_flag("TREATAGENT_FORCE_ALL_EXPERTS", False)
        self.use_derived_argument_claims = self._env_flag("TREATAGENT_USE_DERIVED_ARGUMENT_CLAIMS", True)
        self.disabled_experts = self._init_disabled_experts()
        self.llm_expert_names = self._init_llm_expert_names()
        self.api_url = API_CONFIG["url"] if (self.use_llm_planner or self.use_llm_explanation or self.use_llm_judge or self.use_llm_synthesis) else ""
        if self.use_llm_experts:
            self.api_url = API_CONFIG["url"]
        self.drugkb_expert_client = DrugKBExpert(knowledge_cutoff_date=knowledge_cutoff_date)
        self.diseasekb_expert_client = DiseaseKBExpert(knowledge_cutoff_date=knowledge_cutoff_date)
        self.current_drug_names: List[str] = []
        self.current_drug_identifiers: List[Any] = []
        self.current_drug_inchikey: Optional[str] = None
        self.skill_registry = self._build_skill_registry()
        self.memory_manager = None
        self.memory_init_error = None
        if use_memory:
            try:
                self.memory_manager = LongTermMemoryManager(knowledge_cutoff_date=knowledge_cutoff_date)
            except Exception as exc:
                self.memory_init_error = str(exc)
                self.use_memory = False
                print(f"Memory manager disabled: {exc}")

    def _env_flag(self, name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    def _init_llm_expert_names(self) -> set[str]:
        raw = os.getenv("TREATAGENT_LLM_EXPERTS", "DrugKB,DiseaseKB,DTI,ADMET,Clinical")
        names = {item.strip() for item in raw.split(",") if item.strip()}
        allowed = {"DrugKB", "DiseaseKB", "DTI", "ADMET", "Clinical"}
        return names & allowed

    def _init_disabled_experts(self) -> set[str]:
        raw = os.getenv("TREATAGENT_DISABLED_EXPERTS", "")
        names = {item.strip() for item in raw.split(",") if item.strip()}
        allowed = {"DrugKB", "DiseaseKB", "DTI", "ADMET", "Clinical"}
        return names & allowed

    def _normalize_agent_version(self, agent_version: str) -> str:
        normalized = str(agent_version or "eg").strip().lower().replace("_", "-")
        aliases = {
            "eg": "eg",
            "base": "eg",
            "static": "eg",
            "treatagent-eg": "eg",
            "full": "full",
            "treatagent-full": "full",
            "ls": "ls",
            "llm-synthesis": "ls",
            "treatagent-ls": "ls",
            "legacy": "ls",
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported TreatAgent version: {agent_version}. Use eg, full, or ls.")
        return aliases[normalized]

    def _init_planner_budget(self, planner_budget: Optional[int]) -> int:
        if planner_budget is None:
            raw = os.getenv("TREATAGENT_FULL_PLANNER_BUDGET")
            if raw:
                try:
                    planner_budget = int(raw)
                except ValueError:
                    planner_budget = None
        if planner_budget is None:
            planner_budget = 4 if self.agent_version == "full" else self.max_rounds
        return max(2, min(int(planner_budget), self.max_rounds))

    def _init_arg_threshold(self) -> float:
        raw = os.getenv("TREATAGENT_ARG_THRESHOLD", "0.36")
        try:
            threshold = float(raw)
        except ValueError:
            threshold = 0.36
        return max(0.0, min(1.0, threshold))

    def planner_decide(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        available_skills: List[str],
        memory_context: str,
        similar_cases_count: int,
    ) -> Dict[str, Any]:
        static_output = self._static_planner_decide(
            graph,
            expert_outputs,
            available_skills,
            memory_context,
            similar_cases_count,
        )
        static_output = self._enrich_planner_output(
            static_output,
            graph=graph,
            expert_outputs=expert_outputs,
            available_skills=available_skills,
            similar_cases_count=similar_cases_count,
        )
        if not self.use_llm_planner or not self.api_url or static_output.get("next_action") == "STOP":
            static_output.setdefault("planner_type", "static")
            return static_output
        planner_state = static_output.get("planner_state") or {}
        if (
            static_output.get("planner_type") == "value_of_information_static"
            and int(planner_state.get("evidence_agents_covered") or 0) == 0
            and static_output.get("next_action") == "Clinical"
        ):
            return static_output
        llm_output = self._llm_planner_decide(
            graph,
            expert_outputs,
            available_skills,
            memory_context,
            similar_cases_count,
            static_output,
        )
        if llm_output is None:
            fallback_output = self._budget_aware_fallback(
                static_output=static_output,
                available_skills=available_skills,
                memory_context=memory_context,
                similar_cases_count=similar_cases_count,
            )
            return self._enrich_planner_output(
                fallback_output,
                graph=graph,
                expert_outputs=expert_outputs,
                available_skills=available_skills,
                similar_cases_count=similar_cases_count,
            )
        return self._enrich_planner_output(
            llm_output,
            graph=graph,
            expert_outputs=expert_outputs,
            available_skills=available_skills,
            similar_cases_count=similar_cases_count,
        )

    def _budget_aware_fallback(
        self,
        static_output: Dict[str, Any],
        available_skills: List[str],
        memory_context: str,
        similar_cases_count: int,
    ) -> Dict[str, Any]:
        planner_state = static_output.get("planner_state") or {}
        if not self.use_llm_planner:
            static_output["planner_type"] = "static_fallback"
            return static_output

        allowed_actions = list(available_skills) + ["STOP"]
        stop_policy = self._llm_stop_policy(planner_state)
        utilities = self._estimate_action_utilities(planner_state, allowed_actions, stop_policy)
        covered = int(planner_state.get("evidence_agents_covered") or 0)
        conflict_high = bool(planner_state.get("conflict_high") or planner_state.get("mechanism_conflict_high"))

        if covered >= self.planner_budget and not conflict_high:
            candidate_actions = [
                (action, payload.get("utility", 0.0))
                for action, payload in utilities.items()
                if action != "STOP" and action in available_skills
            ]
            candidate_actions.sort(key=lambda item: item[1], reverse=True)
            if candidate_actions and candidate_actions[0][1] >= 0.72:
                action = candidate_actions[0][0]
                output = {
                    "next_action": action,
                    "next_skill": action,
                    "reason": f"LLM output was rejected; budget-aware fallback selected {action} because it had the highest remaining utility.",
                    "memory_context": similar_cases_count,
                    "memory_hint": memory_context,
                    "planner_state": planner_state,
                    "planner_type": "budget_aware_static_fallback",
                    "action_utility_estimates": utilities,
                    "fallback_reference": {
                        "next_action": static_output.get("next_action"),
                        "reason": static_output.get("reason"),
                    },
                }
                if action in self.skill_registry.names():
                    output["selected_skill"] = self.skill_registry.get(action).to_metadata()
                return output

            return {
                "next_action": "STOP",
                "next_skill": "STOP",
                "reason": "LLM output was rejected and planner budget is exhausted; remaining actions had low utility, so the trajectory stops.",
                "memory_context": similar_cases_count,
                "memory_hint": memory_context,
                "planner_state": planner_state,
                "planner_type": "budget_aware_stop_fallback",
                "action_utility_estimates": utilities,
                "fallback_reference": {
                    "next_action": static_output.get("next_action"),
                    "reason": static_output.get("reason"),
                },
            }

        static_output["planner_type"] = "static_fallback"
        return static_output

    def _enrich_planner_output(
        self,
        output: Dict[str, Any],
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        available_skills: Optional[List[str]],
        similar_cases_count: int,
    ) -> Dict[str, Any]:
        enriched = dict(output)
        planner_state = enriched.get("planner_state") or self._build_planner_state(graph, expert_outputs, similar_cases_count)
        evidence_state = self.build_evidence_state(graph, expert_outputs, planner_state)
        sufficiency = self.assess_evidence_sufficiency(evidence_state)
        conflict = self.assess_evidence_conflict(evidence_state)
        action = str(enriched.get("next_action") or enriched.get("next_skill") or "STOP")
        stop_validation = self.validate_stop_reason(
            evidence_state=evidence_state,
            planner_state=planner_state,
            available_skills=available_skills or [],
        )
        if action == "STOP" and not stop_validation.get("stop_allowed"):
            forced_action = str(stop_validation.get("forced_action") or "")
            if forced_action and forced_action in (available_skills or []):
                action = forced_action
                enriched["next_action"] = forced_action
                enriched["next_skill"] = forced_action
                enriched["reason"] = stop_validation.get("reason")
                enriched["planner_type"] = f"{enriched.get('planner_type', 'planner')}_stop_validator"
                if forced_action in self.skill_registry.names():
                    enriched["selected_skill"] = self.skill_registry.get(forced_action).to_metadata()
        question_payload = self.build_planner_question(action, evidence_state)

        enriched["planner_state"] = planner_state
        enriched["evidence_state"] = evidence_state
        enriched["sufficiency_judgment"] = sufficiency
        enriched["conflict_judgment"] = conflict
        enriched["stop_validation"] = stop_validation
        if not str(enriched.get("planner_question") or "").strip():
            enriched["planner_question"] = question_payload.get("planner_question", "")
        if not enriched.get("expected_evidence"):
            enriched["expected_evidence"] = question_payload.get("expected_evidence", [])
        if not str(enriched.get("stop_condition") or "").strip():
            enriched["stop_condition"] = question_payload.get("stop_condition", "")
        return enriched

    def _static_planner_decide(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        available_skills: List[str],
        memory_context: str,
        similar_cases_count: int,
    ) -> Dict[str, Any]:
        planner_state = self._build_planner_state(graph, expert_outputs, similar_cases_count)
        coverage = graph.expert_coverage()

        value_decision = self._value_of_information_planner_decide(
            planner_state=planner_state,
            available_skills=available_skills,
            memory_context=memory_context,
            similar_cases_count=similar_cases_count,
        )
        if value_decision is not None:
            return value_decision

        if self._should_stop_early(graph, planner_state):
            return {
                "next_action": "STOP",
                "next_skill": "STOP",
                "reason": "All core evidence dimensions are covered and remain mutually consistent, so the trajectory can stop.",
                "memory_context": similar_cases_count,
                "memory_hint": memory_context,
                "planner_state": planner_state,
            }

        if planner_state["mechanism_conflict_high"]:
            for expert in ["DiseaseKB", "DrugKB", "DTI", "Clinical", "ADMET"]:
                if expert in available_skills:
                    return {
                        "next_action": expert,
                        "next_skill": expert,
                        "selected_skill": self.skill_registry.get(expert).to_metadata(),
                        "reason": "Mechanism-level conflict is high, so gather additional knowledge or mechanistic evidence before synthesizing.",
                        "memory_context": similar_cases_count,
                        "memory_hint": memory_context,
                        "planner_state": planner_state,
                    }

        if planner_state["conflict_high"]:
            for expert in ["DiseaseKB", "DrugKB", "DTI", "Clinical", "ADMET"]:
                if expert in available_skills:
                    return {
                        "next_action": expert,
                        "next_skill": expert,
                        "selected_skill": self.skill_registry.get(expert).to_metadata(),
                        "reason": "Evidence conflict is elevated, so prioritize a knowledge-rich or mechanistic expert to resolve disagreement.",
                        "memory_context": similar_cases_count,
                        "memory_hint": memory_context,
                        "planner_state": planner_state,
                    }

        if planner_state["needs_disease_context"] and "DiseaseKB" in available_skills:
            return {
                "next_action": "DiseaseKB",
                "next_skill": "DiseaseKB",
                "selected_skill": self.skill_registry.get("DiseaseKB").to_metadata(),
                "reason": "Disease-side treatment context is still weak or unfamiliar, so fetch disease knowledge first.",
                "memory_context": similar_cases_count,
                "memory_hint": memory_context,
                "planner_state": planner_state,
            }

        if planner_state["needs_drug_context"] and "DrugKB" in available_skills:
            return {
                "next_action": "DrugKB",
                "next_skill": "DrugKB",
                "selected_skill": self.skill_registry.get("DrugKB").to_metadata(),
                "reason": "Drug-side historical mechanism or repurposing context is missing, so fetch drug knowledge next.",
                "memory_context": similar_cases_count,
                "memory_hint": memory_context,
                "planner_state": planner_state,
            }

        if planner_state["evidence_gaps"]["DTI"] and "DTI" in available_skills:
            reason = "Mechanism evidence is still missing or uncertain, so query DTI."
            if self.use_memory and similar_cases_count > 0:
                reason = "Mechanism evidence is missing and similar cases suggest it is especially informative, so query DTI."
            return {
                "next_action": "DTI",
                "next_skill": "DTI",
                "selected_skill": self.skill_registry.get("DTI").to_metadata(),
                "reason": reason,
                "memory_context": similar_cases_count,
                "memory_hint": memory_context,
                "planner_state": planner_state,
            }

        if planner_state["evidence_gaps"]["ADMET"] and "ADMET" in available_skills:
            return {
                "next_action": "ADMET",
                "next_skill": "ADMET",
                "selected_skill": self.skill_registry.get("ADMET").to_metadata(),
                "reason": "Developability and safety evidence are missing, so query ADMET.",
                "memory_context": similar_cases_count,
                "memory_hint": memory_context,
                "planner_state": planner_state,
            }

        if planner_state["evidence_gaps"]["Clinical"] and "Clinical" in available_skills:
            return {
                "next_action": "Clinical",
                "next_skill": "Clinical",
                "selected_skill": self.skill_registry.get("Clinical").to_metadata(),
                "reason": "Clinical prior evidence is missing, so query the clinical expert.",
                "memory_context": similar_cases_count,
                "memory_hint": memory_context,
                "planner_state": planner_state,
            }

        for expert in ["DiseaseKB", "DrugKB", "DTI", "ADMET", "Clinical"]:
            if expert in available_skills and coverage.get(expert, 0) == 0:
                return {
                    "next_action": expert,
                    "next_skill": expert,
                    "selected_skill": self.skill_registry.get(expert).to_metadata(),
                    "reason": f"{expert} has not yet contributed evidence and remains a coverage gap.",
                    "memory_context": similar_cases_count,
                    "memory_hint": memory_context,
                    "planner_state": planner_state,
                }

        return {
            "next_action": "STOP",
            "next_skill": "STOP",
            "reason": "All core experts have contributed or the remaining evidence is unlikely to materially change the decision.",
            "memory_context": similar_cases_count,
            "memory_hint": memory_context,
            "planner_state": planner_state,
        }

    def _value_of_information_planner_decide(
        self,
        planner_state: Dict[str, Any],
        available_skills: List[str],
        memory_context: str,
        similar_cases_count: int,
    ) -> Optional[Dict[str, Any]]:
        coverage = planner_state.get("coverage") or {}
        covered = int(planner_state.get("evidence_agents_covered") or 0)

        if covered == 0 and "Clinical" in available_skills:
            return self._planner_action(
                action="Clinical",
                planner_state=planner_state,
                memory_context=memory_context,
                similar_cases_count=similar_cases_count,
                reason="Clinical prior is the cheapest high-yield evidence source, so query it first for budgeted triage.",
                planner_type="value_of_information_static",
            )

        if self._should_stop_by_decision_margin(planner_state):
            return {
                "next_action": "STOP",
                "next_skill": "STOP",
                "reason": "The ARG score is far from the decision threshold and conflict is low, so additional expert calls have low marginal value.",
                "memory_context": similar_cases_count,
                "memory_hint": memory_context,
                "planner_state": planner_state,
                "planner_type": "value_of_information_static",
            }

        allowed_actions = list(available_skills) + ["STOP"]
        stop_policy = self._llm_stop_policy(planner_state)
        utilities = self._estimate_action_utilities(planner_state, allowed_actions, stop_policy)
        candidates = [
            (action, (utilities.get(action) or {}).get("utility", 0.0))
            for action in available_skills
            if action in self.skill_registry.names() and coverage.get(action, 0) == 0
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        if candidates:
            action, utility = candidates[0]
            if utility >= 0.18:
                return self._planner_action(
                    action=action,
                    planner_state=planner_state,
                    memory_context=memory_context,
                    similar_cases_count=similar_cases_count,
                    reason=f"{action} has the highest estimated marginal information value under the current ARG state.",
                    planner_type="value_of_information_static",
                    action_utility_estimates=utilities,
                )

        return None

    def _planner_action(
        self,
        action: str,
        planner_state: Dict[str, Any],
        memory_context: str,
        similar_cases_count: int,
        reason: str,
        planner_type: str,
        action_utility_estimates: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        output = {
            "next_action": action,
            "next_skill": action,
            "reason": reason,
            "memory_context": similar_cases_count,
            "memory_hint": memory_context,
            "planner_state": planner_state,
            "planner_type": planner_type,
        }
        if action_utility_estimates is not None:
            output["action_utility_estimates"] = action_utility_estimates
        if action in self.skill_registry.names():
            output["selected_skill"] = self.skill_registry.get(action).to_metadata()
        return output

    def _llm_planner_decide(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        available_skills: List[str],
        memory_context: str,
        similar_cases_count: int,
        static_output: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        allowed_actions = list(available_skills) + ["STOP"]
        planner_state = self._build_planner_state(graph, expert_outputs, similar_cases_count)
        payload = self._build_llm_planner_payload(
            graph=graph,
            expert_outputs=expert_outputs,
            allowed_actions=allowed_actions,
            planner_state=planner_state,
            memory_context=memory_context,
            static_output=static_output,
        )
        prompt = self._format_llm_planner_prompt(payload)
        response_text = self._call_api(prompt, "agent4")
        parsed = self._parse_llm_json(response_text) if response_text else None
        if not parsed:
            return None
        output = self._validate_llm_planner_decision(
            parsed=parsed,
            allowed_actions=allowed_actions,
            static_output=static_output,
            planner_state=planner_state,
            memory_context=memory_context,
            similar_cases_count=similar_cases_count,
        )
        if output is None:
            return None
        output["llm_planner_raw_response"] = response_text
        output["llm_planner_payload"] = payload
        next_skill = output.get("next_skill")
        if next_skill != "STOP" and next_skill in self.skill_registry.names():
            output["selected_skill"] = self.skill_registry.get(next_skill).to_metadata()
        return output

    def _build_llm_planner_payload(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        allowed_actions: Sequence[str],
        planner_state: Dict[str, Any],
        memory_context: str,
        static_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        available_skill_metadata = []
        for name in allowed_actions:
            if name == "STOP" or name not in self.skill_registry.names():
                continue
            available_skill_metadata.append(self.skill_registry.get(name).to_metadata())

        stop_policy = self._llm_stop_policy(planner_state)
        action_utilities = self._estimate_action_utilities(planner_state, allowed_actions, stop_policy)
        covered = int(planner_state.get("evidence_agents_covered") or 0)
        evidence_state = self.build_evidence_state(graph, expert_outputs, planner_state)
        stop_validation = self.validate_stop_reason(
            evidence_state,
            planner_state,
            [action for action in allowed_actions if action != "STOP"],
        )
        return {
            "task": "Choose the next biomedical evidence source for drug-disease treatment assessment.",
            "routing_objective": [
                "Maximize expected evidence value per expert call.",
                "Do not call every expert by default.",
                "Skip experts whose marginal utility is low given the current graph state.",
                "Use STOP only when stop_validation provides a valid positive, negative/conflict, or insufficient-evidence reason.",
                "Treat the planner budget as a real budget: only exceed it when conflict is high or a remaining action has high utility.",
            ],
            "decision_scope": [
                "The planner routes evidence acquisition only.",
                "The planner must not predict treatment efficacy.",
                "The final decision is produced later by a constrained evidence judge.",
            ],
            "hard_constraints": [
                "Return strict JSON only.",
                "Choose exactly one next_action from allowed_actions.",
                "Do not invent tools, expert names, evidence, probabilities, or labels.",
                "Use STOP only when stop_validation.stop_allowed is true.",
                "Ground the reason in planner_state, evidence gaps, conflicts, stop_policy, or fallback_reference.",
            ],
            "routing_policy": {
                "Clinical": "Highest initial utility for cheap disease-level triage and translational prior.",
                "DiseaseKB": "High utility when disease context is unknown, target/pathway context is weak, or mechanism conflict needs disease-side resolution.",
                "DrugKB": "High utility when drug indication, target, class, or repurposing history is missing.",
                "DTI": "High utility when the current score is near the decision threshold or clinical prior needs mechanistic confirmation.",
                "ADMET": "High utility only after plausible support exists and safety can change the decision; it is optional when support is weak or no safety-sensitive conflict is present.",
                "STOP": "Use when evidence is sufficient, conflict is low, and remaining actions have low marginal utility.",
            },
            "planner_budget": {
                "max_evidence_agent_calls": self.planner_budget,
                "current_evidence_agent_calls": covered,
                "remaining_budget": max(0, self.planner_budget - covered),
                "budget_exhausted": covered >= self.planner_budget,
                "budget_rule": "If budget is exhausted and conflict is low, choose STOP unless a remaining action has high utility.",
            },
            "action_utility_estimates": action_utilities,
            "evidence_state_memory": evidence_state,
            "stop_validation": stop_validation,
            "allowed_actions": list(allowed_actions),
            "available_skill_metadata": available_skill_metadata,
            "fallback_reference": {
                "next_action": static_output.get("next_action"),
                "reason": static_output.get("reason"),
                "note": "This is a deterministic fallback reference, not a command. Prefer a different allowed action if it has higher expected information gain.",
            },
            "stop_policy": stop_policy,
            "planner_state": planner_state,
            "graph_snapshot": {
                "coverage": graph.expert_coverage(),
                "supportive_evidence": graph.supportive_count(),
                "risk_evidence": graph.risk_count(),
                "mean_confidence": round(graph.mean_confidence(), 4),
                "conflict_level": round(graph.conflict_level(), 4),
                "top_evidence": graph.top_evidence(limit=6),
            },
            "expert_status": {
                name: {
                    "status": output.get("status"),
                    "evidence_count": len(output.get("evidence") or []),
                }
                for name, output in expert_outputs.items()
            },
            "memory_context": memory_context[:1200] if memory_context else "",
            "required_output_schema": {
                "next_action": "string; one of allowed_actions",
                "reason": "one short sentence grounded in provided state",
                "planner_question": "query-specific question for the selected expert, or empty for STOP",
                "expected_evidence": "list of evidence types expected from the selected expert",
                "stop_condition": "condition under which the planner should stop after this action",
                "expected_information_gain": "optional short phrase",
                "risk_if_skipped": "optional short phrase",
            },
        }

    def _format_llm_planner_prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "You are an evidence acquisition planner for drug-disease treatment assessment.\n"
            "You do not score drug-disease treatment potential.\n"
            "You only decide the next evidence source to query.\n"
            "If you select an expert, write a query-specific planner_question for that expert.\n"
            "Use evidence_state_memory to judge missing evidence, sufficiency, and unresolved conflict.\n"
            "Be cost-aware: do not call all experts unless each remaining expert has clear marginal value.\n"
            "The fallback_reference is only a fallback, not an instruction to copy.\n"
            "Follow all hard constraints and return strict JSON with next_action, reason, planner_question, expected_evidence, and stop_condition.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def _llm_stop_policy(self, planner_state: Dict[str, Any]) -> Dict[str, Any]:
        stop_allowed = self._llm_stop_allowed(planner_state)
        coverage_count = int(planner_state.get("evidence_agents_covered") or 0)
        if stop_allowed:
            reason = "Enough non-conflicting evidence has been collected for a final evidence-grounded judgment."
        elif coverage_count >= self.planner_budget and not planner_state.get("conflict_high") and not planner_state.get("mechanism_conflict_high"):
            reason = "Planner budget is exhausted; STOP is allowed if remaining action utility is low."
        elif coverage_count < 2:
            reason = "Too few evidence agents have contributed."
        elif planner_state.get("conflict_high") or planner_state.get("mechanism_conflict_high"):
            reason = "Evidence conflict is still high and needs additional resolution."
        else:
            reason = "Important evidence gaps remain."
        return {
            "stop_allowed": stop_allowed,
            "reason": reason,
            "minimum_evidence_agents": 2,
            "current_evidence_agents": coverage_count,
            "planner_budget": self.planner_budget,
        }

    def _llm_stop_allowed(self, planner_state: Dict[str, Any]) -> bool:
        coverage_count = int(planner_state.get("evidence_agents_covered") or 0)
        if coverage_count < 2:
            return self._should_stop_by_decision_margin(planner_state)

        if planner_state.get("all_core_covered"):
            return True

        if planner_state.get("conflict_high") or planner_state.get("mechanism_conflict_high"):
            return False

        rescue_state = planner_state.get("rescue_state") or {}
        if rescue_state.get("low_clinical_prior") and not rescue_state.get("mechanism_rescue_attempted"):
            return False

        if self._should_stop_by_decision_margin(planner_state):
            return True

        mean_confidence = float(planner_state.get("mean_confidence") or 0.0)
        mechanism_score = float(planner_state.get("mechanism_score") or 0.0)
        safety_score = float(planner_state.get("admet_score") or 0.0)
        clinical_score = float(planner_state.get("clinical_score") or 0.0)
        disease_context_score = float(planner_state.get("disease_context_score") or 0.0)
        drug_context_score = float(planner_state.get("drug_context_score") or 0.0)

        knowledge_support = max(disease_context_score, drug_context_score)
        strong_support = (
            coverage_count >= 3
            and mean_confidence >= 0.62
            and mechanism_score >= 0.68
            and safety_score >= 0.45
            and knowledge_support >= 0.58
        )
        strong_negative_or_weak_mechanism = (
            coverage_count >= 3
            and mean_confidence >= 0.62
            and (mechanism_score <= 0.25 or safety_score <= 0.25)
            and knowledge_support <= 0.5
        )
        clinical_not_required = clinical_score >= 0.35 or coverage_count >= 4
        budget_exhausted_with_reasonable_evidence = (
            coverage_count >= self.planner_budget
            and mean_confidence >= 0.55
            and max(mechanism_score, knowledge_support, safety_score) >= 0.5
        )
        return bool(((strong_support or strong_negative_or_weak_mechanism) and clinical_not_required) or budget_exhausted_with_reasonable_evidence)

    def _should_stop_by_decision_margin(self, planner_state: Dict[str, Any]) -> bool:
        coverage_count = int(planner_state.get("evidence_agents_covered") or 0)
        if coverage_count < 1:
            return False
        if planner_state.get("conflict_high") or planner_state.get("mechanism_conflict_high"):
            return False
        clinical_score = float(planner_state.get("clinical_score") or 0.0)
        if clinical_score <= 0.0:
            return False
        rescue_state = planner_state.get("rescue_state") or {}
        if rescue_state.get("low_clinical_prior") and not rescue_state.get("mechanism_rescue_attempted"):
            return False
        current_score = float(planner_state.get("arg_current_score") or 0.0)
        decision_margin = float(planner_state.get("arg_decision_margin") or 0.0)
        conflict_strength = float(planner_state.get("arg_conflict_strength") or 0.0)
        return bool(conflict_strength < 0.08 and (decision_margin >= 0.18 or current_score <= 0.14))

    def _estimate_action_utilities(
        self,
        planner_state: Dict[str, Any],
        allowed_actions: Sequence[str],
        stop_policy: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        utilities: Dict[str, Dict[str, Any]] = {}
        gaps = planner_state.get("evidence_gaps") or {}
        covered = int(planner_state.get("evidence_agents_covered") or 0)
        budget_exhausted = covered >= self.planner_budget
        conflict_high = bool(planner_state.get("conflict_high") or planner_state.get("mechanism_conflict_high"))
        mechanism_score = float(planner_state.get("mechanism_score") or 0.0)
        safety_score = float(planner_state.get("admet_score") or 0.0)
        clinical_score = float(planner_state.get("clinical_score") or 0.0)
        disease_context_score = float(planner_state.get("disease_context_score") or 0.0)
        drug_context_score = float(planner_state.get("drug_context_score") or 0.0)
        current_score = float(planner_state.get("arg_current_score") or 0.0)
        decision_margin = float(planner_state.get("arg_decision_margin") or 0.0)
        support_strength = float(planner_state.get("arg_support_strength") or 0.0)
        conflict_strength = float(planner_state.get("arg_conflict_strength") or 0.0)
        rescue_state = planner_state.get("rescue_state") or {}
        low_clinical_needs_rescue = bool(
            rescue_state.get("low_clinical_prior")
            and not rescue_state.get("mechanism_rescue_attempted")
        )
        near_threshold = decision_margin < 0.18

        for action in allowed_actions:
            score = 0.0
            reason = "Not applicable."
            if action == "STOP":
                score = 0.9 if stop_policy.get("stop_allowed") else 0.05
                if self._should_stop_by_decision_margin(planner_state):
                    score = max(score, 0.88)
                if low_clinical_needs_rescue:
                    score = min(score, 0.02)
                reason = str(stop_policy.get("reason") or "Stop policy controls STOP utility.")
            elif action == "DiseaseKB":
                score = 0.68 if gaps.get("DiseaseKB") else 0.18
                if disease_context_score < 0.45:
                    score += 0.15
                if conflict_high:
                    score += 0.1
                if near_threshold and clinical_score > 0.0:
                    score += 0.08
                if low_clinical_needs_rescue:
                    score += 0.18
                reason = "Disease-side context resolves target/pathway gaps and mechanism conflicts."
            elif action == "DrugKB":
                score = 0.70 if gaps.get("DrugKB") else 0.18
                if drug_context_score < 0.45:
                    score += 0.15
                if conflict_high:
                    score += 0.1
                if clinical_score >= 0.65 or clinical_score <= 0.25:
                    score += 0.08
                if low_clinical_needs_rescue:
                    score += 0.14
                reason = "Drug-side indications, targets, and class priors are useful when drug context is weak."
            elif action == "DTI":
                score = 0.76 if gaps.get("DTI") else 0.18
                if mechanism_score < 0.45:
                    score += 0.15
                if near_threshold:
                    score += 0.1
                if clinical_score <= 0.25:
                    score += 0.12
                if low_clinical_needs_rescue:
                    score += 0.24
                reason = "Mechanism evidence is high value when DTI is missing or weak."
            elif action == "ADMET":
                score = 0.24 if gaps.get("ADMET") else 0.10
                if (mechanism_score >= 0.6 or support_strength >= 0.42 or current_score >= self.arg_threshold) and safety_score < 0.45:
                    score += 0.12
                if conflict_strength >= 0.12:
                    score += 0.15
                reason = "Safety/developability is high value after plausible mechanism support."
            elif action == "Clinical":
                score = 0.92 if gaps.get("Clinical") else 0.12
                if covered == 0:
                    score += 0.05
                knowledge_support = max(disease_context_score, drug_context_score)
                if clinical_score <= 0.0:
                    score += 0.08
                if mechanism_score >= 0.6 and knowledge_support >= 0.55 and clinical_score < 0.4:
                    score += 0.15
                if mechanism_score >= 0.75 and knowledge_support >= 0.6:
                    score += 0.04
                reason = "Clinical prior is a low-cost high-yield triage signal before expensive mechanism or safety checks."

            if budget_exhausted and action != "STOP" and not conflict_high:
                if low_clinical_needs_rescue and action in {"DTI", "DiseaseKB", "DrugKB"}:
                    score -= 0.05
                    reason += " Budget is exhausted, but low clinical prior still requires mechanism rescue."
                else:
                    score -= 0.1 if action == "Clinical" and score >= 0.8 else 0.25
                    reason += " Budget is exhausted, so marginal utility is penalized unless conflict is high."
            utilities[action] = {
                "utility": round(max(0.0, min(1.0, score)), 4),
                "reason": reason,
            }
        return utilities

    def _validate_llm_planner_decision(
        self,
        parsed: Dict[str, Any],
        allowed_actions: Sequence[str],
        static_output: Dict[str, Any],
        planner_state: Dict[str, Any],
        memory_context: str,
        similar_cases_count: int,
    ) -> Optional[Dict[str, Any]]:
        action_lookup = {str(action).lower(): str(action) for action in allowed_actions}
        raw_action = str(parsed.get("next_action") or "").strip()
        next_action = action_lookup.get(raw_action.lower())
        if next_action is None:
            return None

        static_action = static_output.get("next_action")
        if next_action == "STOP" and not self._llm_stop_allowed(planner_state):
            return None
        if next_action != "STOP":
            covered = int(planner_state.get("evidence_agents_covered") or 0)
            budget_exhausted = covered >= self.planner_budget
            conflict_high = bool(planner_state.get("conflict_high") or planner_state.get("mechanism_conflict_high"))
            if budget_exhausted and not conflict_high:
                utilities = self._estimate_action_utilities(
                    planner_state,
                    allowed_actions,
                    self._llm_stop_policy(planner_state),
                )
                action_utility = (utilities.get(next_action) or {}).get("utility", 0.0)
                if action_utility < 0.72:
                    return None

        reason = str(parsed.get("reason") or "").strip()
        if not reason:
            reason = str(static_output.get("reason") or "LLM planner selected the next evidence acquisition action.")

        next_skill = "STOP" if next_action == "STOP" else next_action
        return {
            "next_action": next_action,
            "next_skill": next_skill,
            "reason": reason,
            "planner_question": str(parsed.get("planner_question") or "").strip(),
            "expected_evidence": parsed.get("expected_evidence") if isinstance(parsed.get("expected_evidence"), list) else [],
            "stop_condition": str(parsed.get("stop_condition") or "").strip(),
            "expected_information_gain": str(parsed.get("expected_information_gain") or "").strip(),
            "risk_if_skipped": str(parsed.get("risk_if_skipped") or "").strip(),
            "memory_context": similar_cases_count,
            "memory_hint": memory_context,
            "planner_state": planner_state,
            "planner_type": "llm_constrained",
            "static_recommendation": {
                "next_action": static_action,
                "reason": static_output.get("reason"),
            },
        }

    def _build_skill_registry(self) -> SkillRegistry:
        registry = SkillRegistry()
        registry.register(SkillSpec(
            name="DiseaseKB",
            description="Retrieve disease-side targets, pathways, and therapy priors from the local disease knowledge base.",
            evidence_category="disease_context",
            triggers=["disease context gap", "mechanism conflict", "unknown disease"],
            input_schema=["disease"],
            output_schema=["structured disease evidence"],
            cost=0.8,
            local_only=True,
            supports_cutoff_filtering=True,
            executor=lambda smiles, disease: self.diseasekb_expert(disease),
        ))
        registry.register(SkillSpec(
            name="DrugKB",
            description="Retrieve drug-side indication and mechanism priors from the local DrugCentral snapshot.",
            evidence_category="drug_context",
            triggers=["repurposing hypothesis", "mechanism conflict", "missing drug history"],
            input_schema=["smiles", "disease"],
            output_schema=["structured drug evidence"],
            cost=0.8,
            local_only=True,
            supports_cutoff_filtering=True,
            executor=lambda smiles, disease: self.drugkb_expert(smiles, disease),
        ))
        registry.register(SkillSpec(
            name="DTI",
            description="Assess target-grounded mechanism plausibility using drug targets, disease targets, and sequence-level DTI when target sequences are available.",
            evidence_category="mechanism",
            triggers=["mechanism gap", "target validation", "conflict resolution"],
            input_schema=["smiles", "disease"],
            output_schema=["target-grounded mechanistic evidence"],
            cost=1.0,
            local_only=True,
            supports_cutoff_filtering=False,
            executor=lambda smiles, disease: self.dti_expert(smiles, disease),
        ))
        registry.register(SkillSpec(
            name="ADMET",
            description="Estimate disease-contextual developability, exposure, and safety properties for the molecule.",
            evidence_category="safety",
            triggers=["safety gap", "developability check"],
            input_schema=["smiles"],
            output_schema=["admet metrics and safety evidence"],
            cost=1.0,
            local_only=True,
            supports_cutoff_filtering=False,
            executor=lambda smiles, disease: self.admet_expert(smiles, disease),
        ))
        registry.register(SkillSpec(
            name="Clinical",
            description="Provide disease-level translational prior from historical outcome statistics.",
            evidence_category="clinical_prior",
            triggers=["clinical prior gap", "late-stage confidence check"],
            input_schema=["disease"],
            output_schema=["clinical prior evidence"],
            cost=0.4,
            local_only=True,
            supports_cutoff_filtering=False,
            executor=lambda smiles, disease: self.clinical_expert(disease),
        ))
        return registry

    def run_skill(
        self,
        skill_name: str,
        formula: str,
        disease: str,
        planner_question: str = "",
        evidence_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_output = self.skill_registry.execute(skill_name, formula, disease)
        return self.wrap_expert_response(
            expert_name=skill_name,
            raw_output=raw_output,
            planner_question=planner_question,
            evidence_state=evidence_state or {},
            response_mode="deterministic_tool",
        )

    def run_expert(self, expert_name: str, formula: str, disease: str) -> Dict[str, Any]:
        if expert_name not in self.skill_registry.names():
            return {"expert": expert_name, "status": "unknown", "evidence": []}
        return self.run_skill(expert_name, formula, disease)

    def _fixed_all_expert_planner_output(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        available_skills: List[str],
        similar_cases_count: int,
    ) -> Dict[str, Any]:
        if not available_skills:
            planner_state = self._build_planner_state(graph, expert_outputs, similar_cases_count)
            evidence_state = self.build_evidence_state(graph, expert_outputs, planner_state)
            return {
                "next_action": "STOP",
                "next_skill": "STOP",
                "reason": "Fixed all-expert ablation has queried every enabled expert.",
                "planner_type": "fixed_all_experts",
                "planner_state": planner_state,
                "evidence_state": evidence_state,
                "planner_question": "",
                "expected_evidence": [],
                "stop_condition": "all_enabled_experts_queried",
            }

        action = available_skills[0]
        planner_state = self._build_planner_state(graph, expert_outputs, similar_cases_count)
        evidence_state = self.build_evidence_state(graph, expert_outputs, planner_state)
        question_payload = self.build_planner_question(action, evidence_state)
        return {
            "next_action": action,
            "next_skill": action,
            "reason": f"Fixed all-expert ablation queries {action} according to the predefined expert order.",
            "planner_type": "fixed_all_experts",
            "planner_state": planner_state,
            "evidence_state": evidence_state,
            "planner_question": question_payload.get("planner_question", ""),
            "expected_evidence": question_payload.get("expected_evidence", []),
            "stop_condition": "",
        }

    def wrap_expert_response(
        self,
        expert_name: str,
        raw_output: Dict[str, Any],
        planner_question: str,
        evidence_state: Dict[str, Any],
        response_mode: str,
    ) -> Dict[str, Any]:
        output = dict(raw_output or {})
        output.setdefault("expert", expert_name)
        output.setdefault("status", "partial")
        output.setdefault("evidence", [])
        output["planner_question"] = planner_question

        llm_response = None
        effective_mode = response_mode
        if self.use_llm_experts and expert_name in self.llm_expert_names and self.api_url:
            llm_response = self._llm_interpret_expert_response(
                expert_name=expert_name,
                raw_output=output,
                planner_question=planner_question,
                evidence_state=evidence_state,
            )
            if llm_response:
                effective_mode = "llm_tool_grounded"
                output["llm_expert_response"] = llm_response
                output["tool_evidence"] = list(output.get("evidence") or [])
                updates = [
                    self._coerce_llm_evidence_update(expert_name, item)
                    for item in llm_response.get("evidence_updates", [])
                    if isinstance(item, dict)
                ]
                updates = [item for item in updates if item is not None]
                if updates:
                    output["evidence"] = updates
                    output["evidence_interpretation_source"] = "llm_tool_grounded"
                else:
                    output["evidence_interpretation_source"] = "llm_tool_grounded_no_updates_fallback"
            else:
                effective_mode = "llm_tool_grounded_failed_fallback"

        response_summary = self._build_expert_response_summary(
            expert_name=expert_name,
            output=output,
            planner_question=planner_question,
            evidence_state=evidence_state,
            response_mode=effective_mode,
            llm_response=llm_response,
        )
        output.update(response_summary)
        return output

    def _build_expert_response_summary(
        self,
        expert_name: str,
        output: Dict[str, Any],
        planner_question: str,
        evidence_state: Dict[str, Any],
        response_mode: str,
        llm_response: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        evidence = output.get("evidence") or []
        impacts = {str(item.get("impact") or "").lower() for item in evidence if isinstance(item, dict)}
        categories = {str(item.get("category") or "") for item in evidence if isinstance(item, dict)}
        confidence_values = [
            float(item.get("confidence"))
            for item in evidence
            if isinstance(item, dict) and _safe_float(item.get("confidence")) is not None
        ]
        mean_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        gap_resolved = self._infer_gap_resolved(expert_name, evidence_state, categories, bool(evidence))
        new_conflicts = [
            {
                "claim": str(item.get("claim") or ""),
                "category": str(item.get("category") or ""),
                "confidence": _safe_float(item.get("confidence"), 0.0),
            }
            for item in evidence
            if isinstance(item, dict) and str(item.get("impact") or "").lower() == "risk"
        ][:3]

        if llm_response:
            answer = str(llm_response.get("answer_to_question") or "").strip()
            recommended = str(llm_response.get("recommended_next_action") or "").strip()
            if isinstance(llm_response.get("gap_resolved"), dict):
                gap_resolved.update({str(k): bool(v) for k, v in llm_response["gap_resolved"].items()})
            llm_confidence = _safe_float(llm_response.get("confidence"))
            if llm_confidence is not None:
                mean_confidence = llm_confidence
            for conflict in llm_response.get("new_conflicts") or []:
                if isinstance(conflict, str) and conflict.strip():
                    new_conflicts.append({"claim": conflict.strip(), "category": "llm_reported_conflict", "confidence": round(mean_confidence, 4)})
                elif isinstance(conflict, dict) and conflict.get("claim"):
                    new_conflicts.append({
                        "claim": str(conflict.get("claim") or ""),
                        "category": str(conflict.get("category") or "llm_reported_conflict"),
                        "confidence": _safe_float(conflict.get("confidence"), mean_confidence),
                    })
            new_conflicts = new_conflicts[:3]
        else:
            answer = self._default_answer_to_planner_question(expert_name, evidence, impacts)
            recommended = self._default_recommended_next_action(expert_name, evidence_state, impacts)

        return {
            "answer_to_question": answer,
            "gap_resolved": gap_resolved,
            "new_conflicts": new_conflicts,
            "response_confidence": round(mean_confidence, 4),
            "recommended_next_action": recommended,
            "response_mode": response_mode,
        }

    def _infer_gap_resolved(
        self,
        expert_name: str,
        evidence_state: Dict[str, Any],
        categories: set[str],
        has_evidence: bool,
    ) -> Dict[str, bool]:
        missing = evidence_state.get("missing_evidence") or {}
        resolved = {key: False for key in missing}
        if not has_evidence:
            return resolved
        if expert_name == "Clinical":
            resolved["clinical_prior"] = True
        elif expert_name == "DTI":
            resolved["mechanism"] = True
        elif expert_name == "ADMET":
            resolved["safety"] = True
        elif expert_name == "DiseaseKB":
            resolved["disease_context"] = True
            if categories & {"disease_target_prior", "pathway_prior", "clinical_target_bridge"}:
                resolved["mechanism"] = True
        elif expert_name == "DrugKB":
            resolved["drug_context"] = True
            if categories & {"drug_history", "drug_identity"}:
                resolved["direct_indication"] = True
            if categories & {"mechanism_prior", "drug_class"}:
                resolved["mechanism"] = True
        return resolved

    def _default_answer_to_planner_question(self, expert_name: str, evidence: List[Dict[str, Any]], impacts: set[str]) -> str:
        if not evidence:
            return f"{expert_name} did not return usable evidence for the planner question."
        if "supportive" in impacts and "risk" in impacts:
            stance = "mixed support and conflict"
        elif "supportive" in impacts:
            stance = "supportive evidence"
        elif "risk" in impacts:
            stance = "conflicting or risk evidence"
        else:
            stance = "neutral evidence"
        return f"{expert_name} answered the planner question with {len(evidence)} structured evidence item(s), mainly providing {stance}."

    def _default_recommended_next_action(self, expert_name: str, evidence_state: Dict[str, Any], impacts: set[str]) -> str:
        missing = evidence_state.get("missing_evidence") or {}
        if "risk" in impacts:
            return "DiseaseKB" if missing.get("disease_context") else "STOP"
        for key, action in [
            ("clinical_prior", "Clinical"),
            ("mechanism", "DTI"),
            ("disease_context", "DiseaseKB"),
            ("drug_context", "DrugKB"),
            ("safety", "ADMET"),
        ]:
            if missing.get(key) and action != expert_name:
                return action
        return "STOP"

    def _llm_interpret_expert_response(
        self,
        expert_name: str,
        raw_output: Dict[str, Any],
        planner_question: str,
        evidence_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        payload = {
            "task": "Act as a tool-grounded biomedical evidence agent and answer the planner question.",
            "expert": expert_name,
            "constraints": [
                "Use only the provided tool/database result and evidence_state.",
                "Do not introduce external biomedical facts.",
                "Do not output the final treatment label or final treatment score.",
                "Do not treat background retrieval as treatment support unless it is grounded by the query-specific relation.",
                "Classify each evidence_update as supportive, risk, or neutral from the perspective of the drug treating the disease.",
                "Return strict JSON only.",
            ],
            "planner_question": planner_question,
            "evidence_state": self._compact_evidence_state_for_prompt(evidence_state),
            "tool_result": self._compact_tool_result_for_llm_expert(raw_output),
            "required_output_schema": {
                "answer_to_question": "short string",
                "evidence_updates": "optional list of evidence dicts using expert/category/claim/value/impact/confidence/source/metadata",
                "gap_resolved": "object mapping gap name to boolean",
                "new_conflicts": "list of short conflict strings",
                "confidence": "float in [0,1]",
                "recommended_next_action": "DrugKB|DiseaseKB|DTI|ADMET|Clinical|STOP",
            },
        }
        prompt = (
            "You are a tool-grounded biomedical evidence agent for drug-disease treatment assessment.\n"
            "Answer the planner question using only the provided tool_result and evidence_state.\n"
            "Do not treat retrieved background context as treatment support unless it is directly grounded in the query.\n"
            "If evidence_updates are included, they must be directly grounded in tool_result evidence.\n"
            "Prefer concise, treatment-relevant evidence updates over copying the raw tool output.\n"
            "Return strict JSON with the required schema.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        response_text = self._call_api(prompt, "agent4")
        parsed = self._parse_llm_json(response_text) if response_text else None
        if not parsed:
            return None
        return {
            "answer_to_question": str(parsed.get("answer_to_question") or "").strip(),
            "evidence_updates": parsed.get("evidence_updates") if isinstance(parsed.get("evidence_updates"), list) else [],
            "gap_resolved": parsed.get("gap_resolved") if isinstance(parsed.get("gap_resolved"), dict) else {},
            "new_conflicts": parsed.get("new_conflicts") if isinstance(parsed.get("new_conflicts"), list) else [],
            "confidence": _safe_float(parsed.get("confidence"), 0.0),
            "recommended_next_action": str(parsed.get("recommended_next_action") or "").strip(),
        }

    def _compact_tool_result_for_llm_expert(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        raw_data = raw_output.get("raw_data")
        if isinstance(raw_data, str) and len(raw_data) > 3000:
            raw_data = raw_data[:3000] + "\n...[truncated]"
        elif isinstance(raw_data, dict):
            raw_data = self._truncate_nested(raw_data, max_string=1200, max_list=8)
        elif isinstance(raw_data, list):
            raw_data = self._truncate_nested(raw_data, max_string=1200, max_list=8)
        return {
            "status": raw_output.get("status"),
            "raw_data": raw_data,
            "metrics": self._truncate_nested(raw_output.get("metrics"), max_string=800, max_list=8),
            "limitations": raw_output.get("limitations"),
            "evidence": self._truncate_nested((raw_output.get("evidence") or [])[:8], max_string=1000, max_list=8),
        }

    def _truncate_nested(self, value: Any, max_string: int = 1000, max_list: int = 10) -> Any:
        if isinstance(value, str):
            return value if len(value) <= max_string else value[:max_string] + "...[truncated]"
        if isinstance(value, list):
            return [self._truncate_nested(item, max_string=max_string, max_list=max_list) for item in value[:max_list]]
        if isinstance(value, dict):
            return {
                str(key): self._truncate_nested(item, max_string=max_string, max_list=max_list)
                for key, item in list(value.items())[:max_list]
            }
        return value

    def _compact_evidence_state_for_prompt(self, evidence_state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "coverage": evidence_state.get("coverage"),
            "acquisition_state": evidence_state.get("acquisition_state"),
            "current_support": (evidence_state.get("current_support") or [])[:4],
            "current_conflict": (evidence_state.get("current_conflict") or [])[:4],
            "missing_evidence": evidence_state.get("missing_evidence"),
            "arg_state": evidence_state.get("arg_state"),
            "sufficiency": evidence_state.get("sufficiency"),
            "conflict": evidence_state.get("conflict"),
            "unresolved_questions": evidence_state.get("unresolved_questions"),
        }

    def _coerce_llm_evidence_update(self, expert_name: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        claim = str(item.get("claim") or "").strip()
        if not claim:
            return None
        impact = str(item.get("impact") or item.get("direction") or "supportive").strip().lower()
        if impact in {"support", "supports", "supportive"}:
            impact = "supportive"
        elif impact in {"conflict", "risk", "contraindicates", "negative"}:
            impact = "risk"
        else:
            impact = "neutral"
        value = _safe_float(item.get("value"), 0.5)
        confidence = _safe_float(item.get("confidence"), 0.5)
        return {
            "expert": expert_name,
            "category": str(item.get("category") or "llm_interpreted_evidence"),
            "claim": claim,
            "value": round(max(0.0, min(1.0, value if value is not None else 0.5)), 4),
            "impact": impact,
            "confidence": round(max(0.0, min(1.0, confidence if confidence is not None else 0.5)), 4),
            "source": str(item.get("source") or f"{expert_name}.tool_grounded_llm"),
            "metadata": {
                **(item.get("metadata") if isinstance(item.get("metadata"), dict) else {}),
                "tool_grounded_llm_interpretation": True,
            },
        }

    def _is_cns_disease_name(self, disease: str) -> bool:
        text = str(disease or "").lower()
        keywords = [
            "brain",
            "central nervous",
            "cns",
            "neuro",
            "epilep",
            "seizure",
            "alzheimer",
            "parkinson",
            "multiple sclerosis",
            "bipolar",
            "schizophrenia",
            "depression",
            "anxiety",
            "migraine",
            "glioma",
            "meningitis",
            "spinal",
            "stroke",
            "dementia",
        ]
        return any(keyword in text for keyword in keywords)

    def _target_key(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        return re.sub(r"[^a-z0-9]+", "", text)

    def _split_target_symbols(self, value: Any) -> List[str]:
        text = str(value or "").strip()
        if not text:
            return []
        parts = re.split(r"[|,;/]+", text)
        return [part.strip() for part in parts if part.strip()]

    def _disease_target_panel(self, disease: str, limit: int = 8) -> Dict[str, Any]:
        record, matched_by, match_score = self.diseasekb_expert_client.lookup(disease)
        if record is None:
            return {
                "status": "unmatched",
                "matched_by": matched_by,
                "match_score": match_score,
                "targets": [],
                "pathways": [],
            }

        targets = []
        for item in (record.get("known_targets") or [])[:limit]:
            symbol = item.get("target_symbol") or item.get("target_id") or item.get("target_name")
            key = self._target_key(symbol)
            if not key:
                continue
            targets.append({
                "key": key,
                "symbol": symbol,
                "target_id": item.get("target_id"),
                "target_name": item.get("target_name"),
                "accession": item.get("accession") or item.get("uniprot_accession"),
                "support_score": _safe_float(item.get("support_score"), 0.0) or 0.0,
                "evidence_count": item.get("evidence_count"),
                "source": item.get("source"),
                "sequence": item.get("sequence") or item.get("protein_sequence") or item.get("target_sequence"),
            })

        pathways = []
        for item in (record.get("known_pathways") or [])[:limit]:
            pathways.append({
                "pathway_id": item.get("pathway_id"),
                "pathway_name": item.get("pathway_name"),
                "top_level_term": item.get("top_level_term"),
                "target_symbol": item.get("target_symbol"),
                "support_score": _safe_float(item.get("support_score"), 0.0) or 0.0,
                "source": item.get("source"),
            })

        return {
            "status": "ok",
            "matched_by": matched_by,
            "match_score": match_score,
            "disease_id": record.get("canonical_disease_id") or record.get("disease_id"),
            "disease_name": record.get("disease_name") or disease,
            "targets": targets,
            "pathways": pathways,
            "therapeutic_areas": record.get("therapeutic_areas") or [],
        }

    def _drug_target_panel(self, formula: str, limit: int = 8) -> Dict[str, Any]:
        record, matched_by, match_score = self.drugkb_expert_client.lookup(
            formula,
            drug_names=self.current_drug_names,
            identifiers=self.current_drug_identifiers,
            inchikey=self.current_drug_inchikey,
        )
        if record is None:
            return {
                "status": "unmatched",
                "matched_by": matched_by,
                "match_score": match_score,
                "targets": [],
            }

        targets = []
        for item in (record.get("known_targets") or [])[:limit]:
            symbols = self._split_target_symbols(item.get("target_gene") or item.get("target_name") or item.get("accession"))
            accessions = self._split_target_symbols(item.get("accession"))
            for idx, symbol in enumerate(symbols or accessions):
                key = self._target_key(symbol)
                if not key:
                    continue
                targets.append({
                    "key": key,
                    "symbol": symbol,
                    "target_name": item.get("target_name"),
                    "accession": accessions[idx] if idx < len(accessions) else item.get("accession"),
                    "target_class": item.get("target_class"),
                    "action_type": item.get("action_type"),
                    "moa": item.get("moa"),
                    "activity_type": item.get("activity_type"),
                    "activity_value": item.get("activity_value"),
                    "activity_source": item.get("activity_source"),
                    "organism": item.get("organism"),
                    "sequence": item.get("sequence") or item.get("protein_sequence") or item.get("target_sequence"),
                })

        return {
            "status": "ok",
            "matched_by": matched_by,
            "match_score": match_score,
            "drug_name": record.get("drug_name"),
            "drugcentral_id": record.get("drugcentral_id"),
            "targets": targets,
        }

    def _pathway_consistency_score(self, drug_targets: List[Dict[str, Any]], disease_pathways: List[Dict[str, Any]]) -> tuple[float, Optional[Dict[str, Any]]]:
        if not drug_targets or not disease_pathways:
            return 0.0, None
        best_score = 0.0
        best_match = None
        for drug_target in drug_targets:
            target_text = " ".join(
                str(value or "")
                for value in [drug_target.get("symbol"), drug_target.get("target_name"), drug_target.get("target_class")]
            ).lower()
            target_tokens = {token for token in re.findall(r"[a-z0-9]+", target_text) if len(token) >= 3}
            for pathway in disease_pathways:
                pathway_text = " ".join(
                    str(value or "")
                    for value in [pathway.get("pathway_name"), pathway.get("top_level_term"), pathway.get("target_symbol")]
                ).lower()
                pathway_tokens = {token for token in re.findall(r"[a-z0-9]+", pathway_text) if len(token) >= 3}
                if not target_tokens or not pathway_tokens:
                    continue
                linked_target_match = self._target_key(drug_target.get("symbol")) == self._target_key(pathway.get("target_symbol"))
                overlap = len(target_tokens & pathway_tokens) / len(target_tokens | pathway_tokens)
                if overlap <= 0.0 and not linked_target_match:
                    continue
                support = _safe_float(pathway.get("support_score"), 0.0) or 0.0
                score = min(1.0, overlap * 1.8 + support * (0.65 if linked_target_match else 0.25))
                if score > best_score:
                    best_score = score
                    best_match = {
                        "drug_target": drug_target,
                        "disease_pathway": pathway,
                        "token_overlap": sorted(target_tokens & pathway_tokens),
                        "linked_target_match": linked_target_match,
                    }
        return round(best_score, 4), best_match

    def _sequence_dti_targets(
        self,
        drug_targets: List[Dict[str, Any]],
        disease_targets: List[Dict[str, Any]],
        overlap: List[Dict[str, Any]],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        seen = set()

        def add_candidate(target: Dict[str, Any], drug_target: Optional[Dict[str, Any]] = None, priority: str = "disease_target") -> None:
            symbol = target.get("symbol") or target.get("target_symbol") or target.get("target_id") or target.get("target_name")
            key = self._target_key(symbol)
            if not key or key in seen:
                return
            seen.add(key)
            merged = dict(target)
            if drug_target:
                merged["drug_target"] = drug_target
                merged["accession"] = drug_target.get("accession") or target.get("accession")
                merged["symbol"] = drug_target.get("symbol") or target.get("symbol")
                merged["target_name"] = drug_target.get("target_name") or target.get("target_name")
            merged["selection_priority"] = priority
            selected.append(merged)

        for match in overlap:
            add_candidate(match.get("disease_target") or {}, match.get("drug_target"), "drug_disease_target_overlap")
            if len(selected) >= limit:
                return selected

        for target in sorted(disease_targets, key=lambda item: _safe_float(item.get("support_score"), 0.0) or 0.0, reverse=True):
            add_candidate(target, None, "top_disease_target")
            if len(selected) >= limit:
                return selected

        for target in drug_targets:
            add_candidate(target, target, "known_drug_target")
            if len(selected) >= limit:
                break
        return selected

    def admet_expert(self, formula: str, disease: str = "") -> Dict[str, Any]:
        raw = admet_data(formula)
        if raw is None:
            return {"expert": "ADMET", "status": "no_data", "evidence": []}
        raw_text = str(raw)
        evidence: List[EvidenceItem] = []
        is_cns = self._is_cns_disease_name(disease)
        metrics = {
            "bioavailability": self._extract_metric(raw_text, "Bioavailability"),
            "hia": self._extract_metric(raw_text, "HIA"),
            "bbb": self._extract_metric(raw_text, "BBB Penetration"),
            "herg": self._extract_metric(raw_text, "hERG Blocking"),
            "clinical_toxicity": self._extract_metric(raw_text, "Clinical Toxicity"),
        }
        if metrics["bioavailability"] is not None:
            evidence.append(self._build_admet_evidence(
                "absorption",
                "Predicted bioavailability informs systemic exposure but is not direct efficacy evidence.",
                metrics["bioavailability"],
                0.35,
                "tool1.admet_data",
                {"endpoint": "Bioavailability", "disease_context": disease, "decision_role": "exposure_feasibility"},
            ))
        if metrics["bbb"] is not None:
            if is_cns:
                bbb_claim = "Predicted BBB penetration is disease-context relevant for CNS accessibility."
                bbb_threshold = 0.35
            else:
                bbb_claim = "Predicted BBB penetration is recorded as distribution context; for non-CNS disease it should not be treated as efficacy support."
                bbb_threshold = 0.05
            evidence.append(self._build_admet_evidence(
                "distribution",
                bbb_claim,
                metrics["bbb"],
                bbb_threshold,
                "tool1.admet_data",
                {"endpoint": "BBB Penetration", "disease_context": disease, "is_cns_disease": is_cns, "decision_role": "distribution_context"},
            ))
        if metrics["herg"] is not None:
            evidence.append(self._build_admet_evidence("toxicity", "Predicted hERG liability affects safety risk and may conflict with treatment suitability.", 1 - metrics["herg"], 0.55, "tool1.admet_data", {"endpoint": "hERG", "raw_risk_score": round(metrics["herg"], 4), "decision_role": "safety_conflict"}))
        if metrics["clinical_toxicity"] is not None:
            evidence.append(self._build_admet_evidence("toxicity", "Predicted clinical toxicity influences developability and may require downstream validation.", 1 - metrics["clinical_toxicity"], 0.55, "tool1.admet_data", {"endpoint": "ClinTox", "raw_risk_score": round(metrics["clinical_toxicity"], 4), "decision_role": "developability_conflict"}))
        return {
            "expert": "ADMET",
            "status": "ok" if evidence else "partial",
            "raw_data": raw_text,
            "metrics": metrics,
            "disease_context": {"disease": disease, "is_cns_disease": is_cns},
            "limitations": [
                "ADMET endpoints describe exposure, distribution, and safety/developability context.",
                "Favorable ADMET is not direct evidence that the drug treats the disease.",
            ],
            "evidence": [item.to_dict() for item in evidence],
        }

    def dti_expert(self, formula: str, disease: str) -> Dict[str, Any]:
        drug_panel = self._drug_target_panel(formula)
        disease_panel = self._disease_target_panel(disease)
        drug_targets = drug_panel.get("targets") or []
        disease_targets = disease_panel.get("targets") or []
        disease_pathways = disease_panel.get("pathways") or []
        evidence: List[EvidenceItem] = []

        disease_by_key = {item["key"]: item for item in disease_targets if item.get("key")}
        overlap = []
        for drug_target in drug_targets:
            match = disease_by_key.get(drug_target.get("key"))
            if match:
                overlap.append({"drug_target": drug_target, "disease_target": match})

        for match in overlap[:3]:
            drug_target = match["drug_target"]
            disease_target = match["disease_target"]
            disease_support = _safe_float(disease_target.get("support_score"), 0.0) or 0.0
            activity_value = _safe_float(drug_target.get("activity_value"), 0.5)
            activity_component = 0.55 if activity_value is None else max(0.0, min(1.0, activity_value / 10.0))
            score = max(0.45, min(0.95, 0.55 * disease_support + 0.45 * activity_component))
            target_label = drug_target.get("symbol") or disease_target.get("symbol") or drug_target.get("target_name")
            evidence.append(EvidenceItem(
                expert="DTI",
                category="mechanism",
                claim=f"Drug and disease evidence converge on target {target_label}, supporting disease-grounded mechanistic plausibility.",
                value=round(score, 4),
                impact="supportive" if score >= 0.45 else "neutral",
                confidence=round(max(0.58, min(0.92, 0.62 + score * 0.25)), 4),
                source="DrugKB-DiseaseKB target grounding",
                metadata={
                    "target_symbol": target_label,
                    "target_name": drug_target.get("target_name") or disease_target.get("target_name"),
                    "target_id": disease_target.get("target_id"),
                    "drug_target_class": drug_target.get("target_class"),
                    "drug_action_type": drug_target.get("action_type"),
                    "drug_activity_type": drug_target.get("activity_type"),
                    "drug_activity_value": drug_target.get("activity_value"),
                    "disease_target_support": disease_support,
                    "mechanism_basis": "drug_target_disease_target_overlap",
                },
            ))

        pathway_score, pathway_match = self._pathway_consistency_score(drug_targets, disease_pathways)
        if pathway_match and pathway_score >= 0.18:
            drug_target = pathway_match["drug_target"]
            pathway = pathway_match["disease_pathway"]
            evidence.append(EvidenceItem(
                expert="DTI",
                category="mechanism",
                claim=(
                    f"Drug target context ({drug_target.get('symbol') or drug_target.get('target_name')}) "
                    f"is weakly consistent with disease pathway context ({pathway.get('pathway_name')})."
                ),
                value=round(min(0.65, 0.35 + pathway_score), 4),
                impact="neutral" if pathway_score < 0.3 else "supportive",
                confidence=round(max(0.5, min(0.78, 0.5 + pathway_score * 0.4)), 4),
                source="DrugKB-DiseaseKB pathway grounding",
                metadata={
                    "target_symbol": drug_target.get("symbol"),
                    "target_name": drug_target.get("target_name"),
                    "pathway_name": pathway.get("pathway_name"),
                    "pathway_id": pathway.get("pathway_id"),
                    "token_overlap": pathway_match.get("token_overlap"),
                    "linked_target_match": pathway_match.get("linked_target_match"),
                    "mechanism_basis": "pathway_context_consistency",
                },
            ))

        sequence_scores = []
        sequence_candidates = self._sequence_dti_targets(drug_targets, disease_targets, overlap, limit=5)
        for target in sequence_candidates:
            sequence_record = None
            sequence = target.get("sequence")
            if sequence:
                sequence_record = resolve_target_sequence(
                    accession=sequence,
                    gene_symbol=target.get("symbol"),
                    target_name=target.get("target_name"),
                    allow_online=False,
                )
            if sequence_record is None:
                sequence_record = resolve_target_sequence(
                    accession=target.get("accession"),
                    gene_symbol=target.get("symbol") or target.get("target_symbol"),
                    target_name=target.get("target_name"),
                )
            if sequence_record is None:
                continue
            score = _safe_float(get_dti_score_ensemble(formula, sequence_record.get("sequence")))
            if score is not None:
                sequence_scores.append({"target": target, "score": score, "sequence_record": sequence_record})
        for item in sequence_scores[:3]:
            target = item["target"]
            score = item["score"]
            sequence_record = item.get("sequence_record") or {}
            target_label = sequence_record.get("gene_symbol") or target.get("symbol") or target.get("target_id")
            evidence.append(EvidenceItem(
                expert="DTI",
                category="mechanism",
                claim=f"DeepPurpose predicts sequence-level binding plausibility for protein target {target_label}.",
                value=round(float(score), 4),
                impact="supportive" if score >= 0.55 else "neutral",
                confidence=round(max(0.52, min(0.9, 0.52 + abs(float(score) - 0.55) * 0.55)), 4),
                source="tool2.get_dti_score_ensemble",
                metadata={
                    "target_symbol": sequence_record.get("gene_symbol") or target.get("symbol"),
                    "target_id": target.get("target_id"),
                    "target_name": sequence_record.get("protein_name") or target.get("target_name"),
                    "uniprot_accession": sequence_record.get("accession") or target.get("accession"),
                    "sequence_source": sequence_record.get("source"),
                    "sequence_matched_by": sequence_record.get("matched_by"),
                    "sequence_cache_hit": sequence_record.get("cache_hit"),
                    "target_selection_priority": target.get("selection_priority"),
                    "mechanism_basis": "sequence_level_dti",
                },
            ))

        if not evidence:
            return {
                "expert": "DTI",
                "status": "no_data",
                "raw_data": {
                    "drug_target_count": len(drug_targets),
                    "disease_target_count": len(disease_targets),
                    "disease_pathway_count": len(disease_pathways),
                    "sequence_candidate_count": len(sequence_candidates),
                    "drug_match_status": drug_panel.get("status"),
                    "disease_match_status": disease_panel.get("status"),
                },
                "limitations": [
                    "No target overlap, pathway consistency, or sequence-level DTI input was available.",
                    "The system did not use a disease name as a DeepPurpose target.",
                ],
                "evidence": [],
            }
        max_score = max(float(item.value) for item in evidence if isinstance(item.value, (int, float)))
        return {
            "expert": "DTI",
            "status": "ok",
            "raw_data": {
                "dti_score": round(max_score, 4),
                "drug_target_count": len(drug_targets),
                "disease_target_count": len(disease_targets),
                "disease_pathway_count": len(disease_pathways),
                "target_overlap_count": len(overlap),
                "sequence_candidate_count": len(sequence_candidates),
                "sequence_dti_count": len(sequence_scores),
                "drug_match_status": drug_panel.get("status"),
                "disease_match_status": disease_panel.get("status"),
            },
            "limitations": [
                "DTI evidence is sequence-level mechanism plausibility and requires disease-side grounding.",
                "Protein sequences are resolved from cached or online UniProt records when accession or gene-symbol targets are available.",
            ],
            "evidence": [item.to_dict() for item in evidence],
        }

    def clinical_expert(self, disease: str) -> Dict[str, Any]:
        prior = get_disease_success_prior(disease)
        success_rate = _safe_float((prior or {}).get("success_rate"))
        if success_rate is None:
            return {"expert": "Clinical", "status": "no_data", "evidence": []}
        evidence = [self._build_numeric_evidence(
            "Clinical",
            "clinical_prior",
            "Historical disease-level success rate informs translational feasibility but is not drug-specific treatment evidence.",
            success_rate,
            0.45,
            "tool3.get_disease_risk",
            {
                "disease_name": disease,
                "matched_disease": prior.get("matched_disease"),
                "matched_by": prior.get("matched_by"),
                "match_score": prior.get("match_score"),
                "decision_role": "disease_level_translational_prior",
            },
        )]
        return {
            "expert": "Clinical",
            "status": "ok",
            "raw_data": {
                "disease_success_rate": round(success_rate, 4),
                "disease": disease,
                "matched_disease": prior.get("matched_disease"),
                "matched_by": prior.get("matched_by"),
                "match_score": prior.get("match_score"),
            },
            "limitations": [
                "Clinical prior is disease-level and not specific to the candidate drug.",
                "It should modify confidence rather than serve as direct indication support.",
            ],
            "evidence": [item.to_dict() for item in evidence],
        }

    def drugkb_expert(self, formula: str, disease: str) -> Dict[str, Any]:
        return self.drugkb_expert_client.analyze(
            formula,
            disease,
            drug_names=self.current_drug_names,
            identifiers=self.current_drug_identifiers,
            inchikey=self.current_drug_inchikey,
        )

    def diseasekb_expert(self, disease: str) -> Dict[str, Any]:
        return self.diseasekb_expert_client.analyze(disease)

    def synthesize(self, graph: EvidenceGraph, expert_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        heuristic = self._heuristic_synthesize(graph, expert_outputs)
        llm_result = self._llm_synthesize(graph, expert_outputs, heuristic)
        if llm_result is not None:
            return llm_result
        return heuristic

    def _heuristic_synthesize(self, graph: EvidenceGraph, expert_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        planner_state = self._build_planner_state(graph, expert_outputs, similar_cases_count=0)
        category_scores, skill_scores = self._category_group_scores(graph)

        category_weights = {
            "safety": 0.28,
            "mechanism": 0.24,
            "clinical_prior": 0.18,
            "drug_context": 0.15,
            "disease_context": 0.15,
        }
        weighted_probability = sum(
            category_weights[category] * category_scores.get(category, self._default_category_score(category))
            for category in category_weights
        )

        missing_expert_penalty = 0.0
        for skill_name in self.skill_registry.names():
            if not graph.evidence_by_expert(skill_name):
                missing_expert_penalty += 0.04
        conflict_penalty = min(0.18, graph.conflict_level() * 0.25)
        knowledge_conflict_penalty = min(0.12, planner_state["knowledge_conflict_score"] * 0.18)
        weighted_probability = max(
            0.0,
            min(1.0, weighted_probability - conflict_penalty - knowledge_conflict_penalty - missing_expert_penalty),
        )
        raw_score = weighted_probability * 10.0

        supportive_claims = [item.claim for item in graph.evidence_items if item.impact == "supportive"][:3]
        risk_claims = [item.claim for item in graph.evidence_items if item.impact == "risk"][:3]
        explanation = (
            f"The synthesized raw score is {raw_score:.2f}/10. "
            f"Category scores - Safety: {category_scores.get('safety', 0.0):.2f}, Mechanism: {category_scores.get('mechanism', 0.0):.2f}, "
            f"Clinical Prior: {category_scores.get('clinical_prior', 0.0):.2f}, Drug Context: {category_scores.get('drug_context', 0.0):.2f}, "
            f"Disease Context: {category_scores.get('disease_context', 0.0):.2f}. "
            f"Key supporting evidence: {'; '.join(supportive_claims) if supportive_claims else 'limited supportive evidence was collected'}. "
            f"Key risk evidence: {'; '.join(risk_claims) if risk_claims else 'no major structured risk signal dominated the evidence'}."
        )
        return {
            "raw_score": round(raw_score, 4),
            "explanation": explanation,
            "supportive_evidence_count": graph.supportive_count(),
            "risk_evidence_count": graph.risk_count(),
            "group_scores": {key: round(value, 4) for key, value in category_scores.items()},
            "skill_scores": {key: round(value, 4) for key, value in skill_scores.items()},
            "missing_expert_penalty": round(missing_expert_penalty, 4),
            "conflict_penalty": round(conflict_penalty, 4),
            "knowledge_conflict_penalty": round(knowledge_conflict_penalty, 4),
            "synthesis_source": "heuristic",
        }

    def _default_category_score(self, category: str) -> float:
        defaults = {
            "safety": 0.42,
            "mechanism": 0.32,
            "clinical_prior": 0.38,
            "drug_context": 0.45,
            "disease_context": 0.45,
        }
        return defaults.get(category, 0.4)

    def _category_group_scores(self, graph: EvidenceGraph):
        category_buckets: Dict[str, List[float]] = {}
        skill_scores: Dict[str, float] = {}
        for skill_name in self.skill_registry.names():
            skill = self.skill_registry.get(skill_name)
            score = self._expert_group_score(
                graph.evidence_by_expert(skill_name),
                default=self._default_category_score(skill.evidence_category),
            )
            skill_scores[skill_name] = score
            category_buckets.setdefault(skill.evidence_category, []).append(score)

        category_scores: Dict[str, float] = {}
        for category, scores in category_buckets.items():
            if scores:
                category_scores[category] = sum(scores) / len(scores)
            else:
                category_scores[category] = self._default_category_score(category)

        knowledge_components = [
            category_scores.get("drug_context"),
            category_scores.get("disease_context"),
        ]
        knowledge_values = [value for value in knowledge_components if value is not None]
        if knowledge_values:
            category_scores["knowledge"] = sum(knowledge_values) / len(knowledge_values)
        return category_scores, skill_scores

    def analyze(
        self,
        formula: str,
        disease: str,
        label: Optional[int] = None,
        sample_id: Optional[str] = None,
        drug_names: Optional[List[str]] = None,
        drug_identifiers: Optional[List[Any]] = None,
        drug_inchikey: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.current_drug_names = [str(item).strip() for item in (drug_names or []) if str(item).strip()]
        self.current_drug_identifiers = list(drug_identifiers or [])
        self.current_drug_inchikey = drug_inchikey
        graph = EvidenceGraph(formula, disease, use_derived_claims=self.use_derived_argument_claims)
        sample_id = sample_id or build_sample_id(formula, disease)
        trajectory = []
        expert_outputs: Dict[str, Dict[str, Any]] = {}
        available_skills = [name for name in self.skill_registry.names() if name not in self.disabled_experts]
        similar_cases = (
            self.memory_manager.retrieve_similar_cases(
                formula,
                disease,
                knowledge_cutoff_date=self.knowledge_cutoff_date,
            )
            if self.memory_manager
            else []
        )
        memory_context = self.memory_manager.format_for_planner(similar_cases) if self.memory_manager else ""

        for round_idx in range(1, self.max_rounds + 1):
            if self.force_all_experts:
                planner_output = self._fixed_all_expert_planner_output(
                    graph,
                    expert_outputs,
                    available_skills,
                    len(similar_cases),
                )
            else:
                planner_output = self.planner_decide(graph, expert_outputs, available_skills, memory_context, len(similar_cases))
            action = planner_output["next_action"]
            skill_name = planner_output.get("next_skill", action)
            step_record = {
                "round": round_idx,
                "planner_output": planner_output,
                "selected_skill": skill_name,
                "planner_question": planner_output.get("planner_question", ""),
                "evidence_state_before": planner_output.get("evidence_state"),
                "graph_snapshot": {
                    "coverage": graph.expert_coverage(),
                    "supportive_evidence": graph.supportive_count(),
                    "risk_evidence": graph.risk_count(),
                    "conflict_level": round(graph.conflict_level(), 4),
                },
            }
            trajectory.append(step_record)
            if action == "STOP":
                break
            expert_output = self.run_skill(
                skill_name,
                formula,
                disease,
                planner_question=planner_output.get("planner_question", ""),
                evidence_state=planner_output.get("evidence_state") or {},
            )
            expert_outputs[skill_name] = expert_output
            for evidence_dict in expert_output.get("evidence", []):
                graph.add_evidence(EvidenceItem(
                    expert=evidence_dict["expert"],
                    category=evidence_dict["category"],
                    claim=evidence_dict["claim"],
                    value=evidence_dict["value"],
                    impact=evidence_dict["impact"],
                    confidence=evidence_dict["confidence"],
                    source=evidence_dict["source"],
                    metadata=evidence_dict.get("metadata", {}),
                ))
            updated_planner_state = self._build_planner_state(graph, expert_outputs, len(similar_cases))
            step_record["expert_response_summary"] = {
                "expert": expert_output.get("expert", skill_name),
                "status": expert_output.get("status"),
                "answer_to_question": expert_output.get("answer_to_question"),
                "gap_resolved": expert_output.get("gap_resolved"),
                "new_conflicts": expert_output.get("new_conflicts"),
                "response_confidence": expert_output.get("response_confidence"),
                "recommended_next_action": expert_output.get("recommended_next_action"),
                "response_mode": expert_output.get("response_mode"),
            }
            step_record["evidence_state_after"] = self.build_evidence_state(graph, expert_outputs, updated_planner_state)
            if skill_name in available_skills:
                available_skills.remove(skill_name)

        synthesis = self.synthesize(graph, expert_outputs)
        calibration = self.calibrator.calibrate(synthesis["raw_score"], graph)
        llm_judge = self._llm_judge(graph, expert_outputs, synthesis, calibration)
        if llm_judge is not None:
            synthesis = dict(synthesis)
            if llm_judge.get("reasoning_summary"):
                synthesis["explanation"] = str(llm_judge["reasoning_summary"])
                synthesis["explanation_source"] = "llm_judge"
            synthesis["llm_judge_treatment_score"] = llm_judge.get("treatment_score")
            synthesis["llm_judge_decision"] = llm_judge.get("decision")
        if self.use_llm_explanation and llm_judge is None:
            llm_explanation = self._llm_explain(graph, expert_outputs, synthesis, calibration)
            if llm_explanation:
                synthesis = dict(synthesis)
                synthesis["explanation"] = llm_explanation
                synthesis["explanation_source"] = "llm"
            else:
                synthesis.setdefault("explanation_source", synthesis.get("synthesis_source", "heuristic"))
        else:
            synthesis.setdefault(
                "explanation_source",
                "llm_synthesis" if synthesis.get("synthesis_source") == "llm" else "heuristic",
            )
        predicted_binary = 1 if calibration["calibrated_probability"] >= 0.5 else 0
        stored_case_id = None
        if self.memory_manager:
            stored_case_id = self.memory_manager.store_case(
                smiles=formula,
                disease=disease,
                trajectory=trajectory,
                final_prediction=predicted_binary,
                calibrated_prob=calibration["calibrated_probability"],
                evidence_summary=graph.summary(),
                case_date=None,
            )

        report_path = None
        report_summary = None
        if self.generate_report:
            report_path, report_summary = self.report_generator.generate(
                evidence_graph=graph.summary(),
                expert_outputs=expert_outputs,
                smiles=formula,
                disease=disease,
                raw_score=synthesis["raw_score"],
                calibrated_prob=calibration["calibrated_probability"],
                synthesis_explanation=synthesis["explanation"],
                sample_id=sample_id,
                trajectory=trajectory,
                synthesis_source=synthesis.get("synthesis_source"),
                group_scores=synthesis.get("group_scores", {}),
                memory_similar_cases=len(similar_cases),
                knowledge_cutoff_date=self.knowledge_cutoff_date,
            )

        return {
            "sample_id": sample_id,
            "agent_version": self.agent_version,
            "llm_planner_enabled": self.use_llm_planner,
            "llm_explanation_enabled": self.use_llm_explanation,
            "llm_judge_enabled": self.use_llm_judge,
            "llm_synthesis_enabled": self.use_llm_synthesis,
            "llm_experts_enabled": self.use_llm_experts,
            "llm_expert_names": sorted(self.llm_expert_names),
            "force_all_experts": self.force_all_experts,
            "derived_argument_claims_enabled": self.use_derived_argument_claims,
            "disabled_experts": sorted(self.disabled_experts),
            "drug": formula,
            "disease": disease,
            "trajectory": trajectory,
            "expert_outputs": expert_outputs,
            "evidence_graph": graph.summary(),
            "synthesis": synthesis,
            "calibration": calibration,
            "llm_judge": llm_judge,
            "report_path": report_path,
            "report_summary": report_summary,
            "memory_similar_cases": len(similar_cases),
            "memory_context": memory_context if self.use_memory else None,
            "stored_case_id": stored_case_id,
            "memory_enabled": self.memory_manager is not None,
            "memory_init_error": self.memory_init_error,
            "knowledge_cutoff_date": self.knowledge_cutoff_date,
            "skill_registry": self.skill_registry.metadata(),
        }

    def _build_planner_state(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        similar_cases_count: int,
    ) -> Dict[str, Any]:
        coverage = graph.expert_coverage()
        arg_factors = self._current_argument_factors(graph)
        disease_context_score = self._expert_group_score(graph.evidence_by_expert("DiseaseKB"), default=0.0)
        drug_context_score = self._expert_group_score(graph.evidence_by_expert("DrugKB"), default=0.0)
        mechanism_score = self._expert_group_score(graph.evidence_by_expert("DTI"), default=0.0)
        admet_score = self._expert_group_score(graph.evidence_by_expert("ADMET"), default=0.0)
        clinical_score = self._expert_group_score(graph.evidence_by_expert("Clinical"), default=0.0)
        disease_unknown = not graph.has_expert_evidence("DiseaseKB") or disease_context_score < 0.48
        repurposing_candidate = (
            not graph.has_expert_evidence("DrugKB")
            and (similar_cases_count > 0 or disease_context_score >= 0.52 or not graph.has_expert_evidence("DTI"))
        )

        dti_raw_score = _safe_float(expert_outputs.get("DTI", {}).get("raw_data", {}).get("dti_score"), 0.0) or 0.0
        disease_target_score = self._mean_category_value(
            graph,
            "DiseaseKB",
            ["disease_target_prior", "pathway_prior", "clinical_target_bridge"],
        )
        drug_mechanism_score = self._mean_category_value(
            graph,
            "DrugKB",
            ["mechanism_prior", "drug_history", "drug_class"],
        )
        target_overlap = self._target_overlap_score(graph)

        mechanism_conflict_score = 0.0
        if dti_raw_score >= 0.7 and disease_target_score <= 0.42:
            mechanism_conflict_score += 0.45
        if dti_raw_score >= 0.7 and graph.has_expert_evidence("DiseaseKB") and disease_target_score <= 0.35:
            mechanism_conflict_score += 0.25
        if graph.has_expert_evidence("DrugKB") and graph.has_expert_evidence("DiseaseKB") and target_overlap == 0.0:
            mechanism_conflict_score += 0.18
        if graph.has_expert_evidence("DrugKB") and drug_mechanism_score <= 0.4 and dti_raw_score >= 0.72:
            mechanism_conflict_score += 0.12
        mechanism_conflict_score = max(mechanism_conflict_score, graph.conflict_level())

        evidence_gaps = {
            "ADMET": not graph.has_expert_evidence("ADMET") or admet_score < 0.35,
            "DTI": not graph.has_expert_evidence("DTI") or mechanism_score < 0.45,
            "Clinical": not graph.has_expert_evidence("Clinical") or clinical_score < 0.35,
            "DiseaseKB": not graph.has_expert_evidence("DiseaseKB") or disease_context_score < 0.45,
            "DrugKB": not graph.has_expert_evidence("DrugKB") or drug_context_score < 0.45,
        }

        has_dti = graph.has_expert_evidence("DTI")
        has_disease_context = graph.has_expert_evidence("DiseaseKB")
        has_drug_context = graph.has_expert_evidence("DrugKB")
        clinical_called = graph.has_expert_evidence("Clinical")
        low_clinical_prior = clinical_called and 0.0 < clinical_score < 0.35
        mechanism_rescue_attempted = has_dti and (has_disease_context or has_drug_context)
        mechanism_rescue_success = (
            mechanism_score >= 0.65
            or disease_context_score >= 0.60
            or drug_context_score >= 0.60
            or arg_factors.get("cross_source_consistency", 0.0) >= 0.20
        )
        rescue_state = {
            "low_clinical_prior": bool(low_clinical_prior),
            "clinical_called": bool(clinical_called),
            "has_dti": bool(has_dti),
            "has_disease_context": bool(has_disease_context),
            "has_drug_context": bool(has_drug_context),
            "mechanism_rescue_attempted": bool(mechanism_rescue_attempted),
            "mechanism_rescue_success": bool(mechanism_rescue_success),
            "rescue_required": bool(low_clinical_prior and not mechanism_rescue_attempted),
        }

        all_core_covered = all(not evidence_gaps[key] for key in ["ADMET", "DTI", "Clinical", "DiseaseKB", "DrugKB"])
        return {
            "coverage": coverage,
            "evidence_agents_covered": len([name for name in ["ADMET", "DTI", "Clinical", "DiseaseKB", "DrugKB"] if coverage.get(name, 0) > 0]),
            "supportive_evidence": graph.supportive_count(),
            "risk_evidence": graph.risk_count(),
            "mean_confidence": round(graph.mean_confidence(), 4),
            "conflict_level": round(graph.conflict_level(), 4),
            "disease_context_score": round(disease_context_score, 4),
            "drug_context_score": round(drug_context_score, 4),
            "mechanism_score": round(mechanism_score, 4),
            "admet_score": round(admet_score, 4),
            "clinical_score": round(clinical_score, 4),
            "disease_unknown": disease_unknown,
            "repurposing_candidate": repurposing_candidate,
            "disease_target_score": round(disease_target_score, 4),
            "drug_mechanism_score": round(drug_mechanism_score, 4),
            "target_overlap_score": round(target_overlap, 4),
            "knowledge_conflict_score": round(mechanism_conflict_score, 4),
            "mechanism_conflict_high": mechanism_conflict_score >= 0.4,
            "conflict_high": max(graph.conflict_level(), mechanism_conflict_score) >= 0.3,
            "needs_disease_context": evidence_gaps["DiseaseKB"] or disease_unknown,
            "needs_drug_context": evidence_gaps["DrugKB"] or repurposing_candidate,
            "evidence_gaps": evidence_gaps,
            "all_core_covered": all_core_covered,
            "arg_threshold": round(self.arg_threshold, 4),
            "arg_current_score": round(arg_factors.get("raw_argument_score", 0.0), 4),
            "arg_decision_margin": round(abs(arg_factors.get("raw_argument_score", 0.0) - self.arg_threshold), 4),
            "arg_support_strength": round(arg_factors.get("support_strength", 0.0), 4),
            "arg_conflict_strength": round(arg_factors.get("conflict_strength", 0.0), 4),
            "arg_direct_support": round(arg_factors.get("direct_support", 0.0), 4),
            "arg_cross_source_consistency": round(arg_factors.get("cross_source_consistency", 0.0), 4),
            "rescue_state": rescue_state,
        }

    def build_evidence_state(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        planner_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        support_items = self._compact_typed_evidence(graph, "support", limit=5)
        conflict_items = self._compact_typed_evidence(graph, "conflict", limit=5)
        gaps = planner_state.get("evidence_gaps") or {}
        acquisition_state = self._build_acquisition_state(graph, expert_outputs)
        missing_evidence = {
            "direct_indication": bool(gaps.get("DrugKB")),
            "mechanism": bool(gaps.get("DTI") or gaps.get("DiseaseKB")),
            "safety": bool(gaps.get("ADMET")),
            "clinical_prior": bool(gaps.get("Clinical")),
            "disease_context": bool(gaps.get("DiseaseKB")),
            "drug_context": bool(gaps.get("DrugKB")),
        }
        evidence_state = {
            "query": {
                "smiles": graph.formula,
                "disease": graph.disease,
            },
            "coverage": planner_state.get("coverage") or graph.expert_coverage(),
            "acquisition_state": acquisition_state,
            "current_support": support_items,
            "current_conflict": conflict_items,
            "missing_evidence": missing_evidence,
            "arg_state": {
                "score": planner_state.get("arg_current_score", 0.0),
                "threshold": planner_state.get("arg_threshold", self.arg_threshold),
                "decision_margin": planner_state.get("arg_decision_margin", 0.0),
                "support_strength": planner_state.get("arg_support_strength", 0.0),
                "conflict_strength": planner_state.get("arg_conflict_strength", 0.0),
            },
            "rescue_state": planner_state.get("rescue_state", {}),
            "confidence": {
                "mean_confidence": planner_state.get("mean_confidence", 0.0),
                "supportive_evidence": planner_state.get("supportive_evidence", 0),
                "risk_evidence": planner_state.get("risk_evidence", 0),
            },
            "expert_status": {
                name: {
                    "status": output.get("status"),
                    "evidence_count": len(output.get("evidence") or []),
                    "response_mode": output.get("response_mode"),
                }
                for name, output in expert_outputs.items()
            },
            "unresolved_questions": self._unresolved_questions_from_state(planner_state, missing_evidence),
        }
        evidence_state["conflict"] = self.assess_evidence_conflict(evidence_state)
        evidence_state["sufficiency"] = self.assess_evidence_sufficiency(evidence_state)
        return evidence_state

    def _build_acquisition_state(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        expected = ["Clinical", "DTI", "DrugKB", "DiseaseKB", "ADMET"]
        queried = sorted([name for name in expected if name in expert_outputs])
        not_queried = sorted([name for name in expected if name not in expert_outputs])
        retrieval_failures: List[str] = []
        no_match: List[str] = []
        matched_no_direct: List[str] = []
        partial: List[str] = []
        for name in queried:
            output = expert_outputs.get(name) or {}
            status = str(output.get("status") or "").lower()
            evidence_count = len(output.get("evidence") or [])
            if status in {"error", "failed", "timeout"}:
                retrieval_failures.append(name)
            elif status in {"no_data", "not_found"}:
                no_match.append(name)
            elif evidence_count == 0:
                partial.append(name)

        argument_graph = graph.argument_graph_summary()
        missing_roles = {
            str(item.get("semantic_role") or "")
            for item in (argument_graph.get("missing_evidence") or [])
            if isinstance(item, dict)
        }
        support_roles = {
            str(item.get("semantic_role") or "")
            for item in (argument_graph.get("support_claims") or [])
            if isinstance(item, dict)
        }
        if "DrugKB" in queried and "missing_direct_evidence" in missing_roles:
            matched_no_direct.append("DrugKB")

        has_clinical = "Clinical" in queried
        has_mechanism = "DTI" in queried or "target_overlap" in support_roles
        has_knowledge = "DrugKB" in queried or "DiseaseKB" in queried
        critical_missing = []
        if not has_clinical:
            critical_missing.append("Clinical")
        if not has_mechanism:
            critical_missing.append("mechanism")
        if not has_knowledge:
            critical_missing.append("DrugKB_or_DiseaseKB")

        completeness = "complete"
        if critical_missing:
            completeness = "partial" if len(critical_missing) == 1 else "insufficient"
        if retrieval_failures:
            completeness = "partial" if completeness == "complete" else completeness
        if len(queried) <= 1 and critical_missing:
            completeness = "insufficient"

        return {
            "queried_experts": queried,
            "not_queried_experts": not_queried,
            "retrieval_failures": retrieval_failures,
            "no_match_experts": sorted(set(no_match)),
            "matched_no_direct_evidence": sorted(set(matched_no_direct)),
            "partial_experts": sorted(set(partial)),
            "critical_missing": critical_missing,
            "acquisition_completeness": completeness,
            "missingness_semantics": {
                "not_queried": not_queried,
                "retrieval_failure": retrieval_failures,
                "no_match": sorted(set(no_match)),
                "matched_no_direct_indication": sorted(set(matched_no_direct)),
            },
        }

    def assess_evidence_sufficiency(self, evidence_state: Dict[str, Any]) -> Dict[str, Any]:
        arg_state = evidence_state.get("arg_state") or {}
        confidence = evidence_state.get("confidence") or {}
        missing = evidence_state.get("missing_evidence") or {}
        conflict = evidence_state.get("conflict") or {}
        rescue_state = evidence_state.get("rescue_state") or {}
        score = float(arg_state.get("score") or 0.0)
        margin = float(arg_state.get("decision_margin") or 0.0)
        mean_confidence = float(confidence.get("mean_confidence") or 0.0)
        covered_count = sum(1 for count in (evidence_state.get("coverage") or {}).values() if count)
        missing_core = [key for key, value in missing.items() if value and key in {"mechanism", "safety", "clinical_prior", "disease_context", "drug_context"}]
        unresolved_conflict = bool(conflict.get("has_unresolved_conflict"))

        sufficient = (
            covered_count >= 2
            and margin >= 0.18
            and mean_confidence >= 0.45
            and not unresolved_conflict
        )
        if score <= 0.14 and covered_count >= 1 and not unresolved_conflict:
            sufficient = True
        if missing.get("clinical_prior") and covered_count < 3:
            sufficient = False
        if rescue_state.get("rescue_required"):
            sufficient = False
        reason = "Evidence is sufficient because the ARG score is separated from the threshold and unresolved conflict is low."
        if not sufficient:
            if rescue_state.get("rescue_required"):
                reason = "Evidence is insufficient because clinical prior is low and mechanism rescue has not yet been attempted."
            elif unresolved_conflict:
                reason = "Evidence is insufficient because support/conflict disagreement remains unresolved."
            elif missing_core:
                reason = f"Evidence is insufficient because core evidence is missing: {', '.join(missing_core[:3])}."
            else:
                reason = "Evidence is insufficient because the ARG score remains near the decision threshold."
        return {
            "is_sufficient": bool(sufficient),
            "reason": reason,
            "covered_expert_count": covered_count,
            "missing_core_evidence": missing_core,
        }

    def assess_evidence_conflict(self, evidence_state: Dict[str, Any]) -> Dict[str, Any]:
        arg_state = evidence_state.get("arg_state") or {}
        support_strength = float(arg_state.get("support_strength") or 0.0)
        conflict_strength = float(arg_state.get("conflict_strength") or 0.0)
        conflict_items = evidence_state.get("current_conflict") or []
        support_items = evidence_state.get("current_support") or []
        unresolved = bool(
            conflict_strength >= 0.14
            and support_strength >= 0.20
            and len(conflict_items) > 0
            and len(support_items) > 0
        )
        reason = "No strong unresolved support-conflict disagreement is present."
        if unresolved:
            reason = "Support and conflict evidence are both present with non-trivial strength."
        return {
            "has_unresolved_conflict": unresolved,
            "reason": reason,
            "support_strength": round(support_strength, 4),
            "conflict_strength": round(conflict_strength, 4),
            "conflict_count": len(conflict_items),
        }

    def validate_stop_reason(
        self,
        evidence_state: Dict[str, Any],
        planner_state: Dict[str, Any],
        available_skills: List[str],
    ) -> Dict[str, Any]:
        coverage_count = int(planner_state.get("evidence_agents_covered") or 0)
        support_strength = float(planner_state.get("arg_support_strength") or 0.0)
        conflict_strength = float(planner_state.get("arg_conflict_strength") or 0.0)
        direct_support = float(planner_state.get("arg_direct_support") or 0.0)
        cross_source = float(planner_state.get("arg_cross_source_consistency") or 0.0)
        mechanism_score = float(planner_state.get("mechanism_score") or 0.0)
        clinical_score = float(planner_state.get("clinical_score") or 0.0)
        safety_score = float(planner_state.get("admet_score") or 0.0)
        disease_context_score = float(planner_state.get("disease_context_score") or 0.0)
        drug_context_score = float(planner_state.get("drug_context_score") or 0.0)
        conflict_high = bool(planner_state.get("conflict_high") or planner_state.get("mechanism_conflict_high"))
        rescue_state = planner_state.get("rescue_state") or {}
        acquisition_state = evidence_state.get("acquisition_state") or {}
        queried = set(acquisition_state.get("queried_experts") or [])
        has_clinical = "Clinical" in queried
        has_dti = "DTI" in queried
        has_knowledge = bool({"DrugKB", "DiseaseKB"} & queried)
        has_safety = "ADMET" in queried
        safety_ok = (not has_safety) or safety_score >= 0.25
        triangular_evidence = has_clinical and has_dti and has_knowledge

        has_positive_stop = bool(
            coverage_count >= 2
            and not conflict_high
            and (
                direct_support >= 0.12
                or (cross_source >= 0.16 and clinical_score >= 0.35 and safety_score >= 0.30)
                or (
                    triangular_evidence
                    and
                    mechanism_score >= 0.68
                    and max(disease_context_score, drug_context_score) >= 0.55
                    and clinical_score >= 0.35
                    and safety_ok
                )
                or (
                    triangular_evidence
                    and mechanism_score >= 0.58
                    and clinical_score >= 0.42
                    and support_strength >= 0.10
                    and safety_ok
                )
            )
        )
        if has_positive_stop:
            return {
                "stop_allowed": True,
                "stop_reason_category": "positive_support_sufficient",
                "reason": "STOP is valid because disease-specific support is sufficient and unresolved conflict is low.",
                "forced_action": None,
            }

        has_negative_stop = bool(
            coverage_count >= 2
            and (
                conflict_high
                or conflict_strength >= 0.18
                or (safety_score > 0.0 and safety_score <= 0.25)
                or (
                    rescue_state.get("low_clinical_prior")
                    and rescue_state.get("mechanism_rescue_attempted")
                    and not rescue_state.get("mechanism_rescue_success")
                )
                or (
                    "DrugKB" in acquisition_state.get("matched_no_direct_evidence", [])
                    and mechanism_score <= 0.35
                    and clinical_score <= 0.40
                )
            )
        )
        if has_negative_stop:
            return {
                "stop_allowed": True,
                "stop_reason_category": "negative_or_conflict_evidence_sufficient",
                "reason": "STOP is valid because available evidence is sufficiently weak or conflicting after targeted acquisition.",
                "forced_action": None,
            }

        budget_exhausted = coverage_count >= self.planner_budget
        no_available = not available_skills
        low_remaining_utility = False
        best_action = self._best_remaining_action(planner_state, available_skills)
        low_utility_threshold = 0.30 if coverage_count >= 3 and triangular_evidence and not conflict_high else 0.18
        if best_action is None or float(best_action.get("utility") or 0.0) < low_utility_threshold:
            low_remaining_utility = True
        if budget_exhausted or no_available or low_remaining_utility:
            return {
                "stop_allowed": True,
                "stop_reason_category": "insufficient_evidence_budget_or_low_utility",
                "reason": "STOP is valid as an insufficient-evidence stop because budget is exhausted or remaining evidence utility is low.",
                "forced_action": None,
                "best_remaining_action": best_action,
            }

        return {
            "stop_allowed": False,
            "stop_reason_category": "stop_not_valid",
            "reason": "STOP is not valid yet; evidence is neither sufficient nor exhausted, so the planner should query the highest-value remaining expert.",
            "forced_action": best_action.get("action") if best_action else None,
            "best_remaining_action": best_action,
        }

    def _best_remaining_action(
        self,
        planner_state: Dict[str, Any],
        available_skills: List[str],
    ) -> Optional[Dict[str, Any]]:
        candidates = [action for action in available_skills if action in self.skill_registry.names()]
        if not candidates:
            return None
        utilities = self._estimate_action_utilities(
            planner_state,
            candidates + ["STOP"],
            {"stop_allowed": False, "reason": "STOP is being validated by evidence sufficiency rules."},
        )
        ranked = [
            {
                "action": action,
                "utility": float((utilities.get(action) or {}).get("utility") or 0.0),
                "reason": (utilities.get(action) or {}).get("reason"),
            }
            for action in candidates
        ]
        ranked.sort(key=lambda item: item["utility"], reverse=True)
        return ranked[0] if ranked else None

    def build_planner_question(self, action: str, evidence_state: Dict[str, Any]) -> Dict[str, Any]:
        if action == "STOP":
            sufficiency = evidence_state.get("sufficiency") or {}
            return {
                "planner_question": "No further expert question is needed; decide whether the current evidence is sufficient for STOP.",
                "expected_evidence": [],
                "stop_condition": str(sufficiency.get("reason") or "STOP selected by planner."),
            }

        missing = evidence_state.get("missing_evidence") or {}
        arg_state = evidence_state.get("arg_state") or {}
        disease = (evidence_state.get("query") or {}).get("disease", "the target disease")
        score = float(arg_state.get("score") or 0.0)
        threshold = float(arg_state.get("threshold") or self.arg_threshold)
        near_threshold = abs(score - threshold) < 0.18

        templates = {
            "Clinical": {
                "planner_question": f"Estimate whether {disease} has favorable disease-level translational feasibility and whether clinical prior should support or weaken the treatment hypothesis.",
                "expected_evidence": ["clinical feasibility", "disease-level success prior"],
                "stop_condition": "Stop only if clinical prior makes the ARG score clearly separated from the threshold and no major mechanism or safety gap remains.",
            },
            "DTI": {
                "planner_question": f"Check whether the molecule has mechanistic support for {disease} through plausible drug-target interaction evidence, especially because the current ARG score is near the threshold." if near_threshold else f"Check whether the molecule has disease-relevant mechanistic support for {disease}.",
                "expected_evidence": ["drug-target interaction support", "mechanism conflict", "target relevance"],
                "stop_condition": "Stop if DTI is weak and no direct or disease-context support exists.",
            },
            "DiseaseKB": {
                "planner_question": f"Retrieve disease-side targets, pathways, and known therapy context for {disease} to determine whether current support is disease-specific.",
                "expected_evidence": ["disease targets", "pathway context", "therapy prior"],
                "stop_condition": "Stop if disease context does not support the mechanism and direct drug support is absent.",
            },
            "DrugKB": {
                "planner_question": f"Retrieve drug-side indications, targets, classes, and repurposing history relevant to {disease}, and identify direct support or contradictions.",
                "expected_evidence": ["direct indication", "drug target", "drug class", "repurposing history"],
                "stop_condition": "Stop if direct support is absent and no mechanism or disease context can rescue the hypothesis.",
            },
            "ADMET": {
                "planner_question": "Check whether ADMET or toxicity evidence introduces a safety/developability conflict that could invalidate otherwise plausible treatment support.",
                "expected_evidence": ["toxicity risk", "exposure support", "developability conflict"],
                "stop_condition": "Stop if safety risk is low or if safety conflict clearly outweighs weak support.",
            },
        }
        payload = templates.get(action, {
            "planner_question": f"Collect the most decision-relevant evidence from {action}.",
            "expected_evidence": ["decision-relevant evidence"],
            "stop_condition": "Stop when the added evidence cannot change the ARG decision.",
        })
        if missing:
            payload["missing_evidence_context"] = [key for key, value in missing.items() if value]
        return payload

    def _compact_typed_evidence(self, graph: EvidenceGraph, direction: str, limit: int = 5) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in graph.typed_evidence:
            data = item.to_dict()
            if data.get("direction") != direction:
                continue
            strength = float(data.get("score") or 0.0) * float(data.get("reliability") or 0.0)
            rows.append({
                "expert": data.get("expert"),
                "category": data.get("category"),
                "claim": data.get("claim"),
                "strength": round(strength, 4),
                "reliability": data.get("reliability"),
                "source": data.get("source"),
            })
        rows.sort(key=lambda item: float(item.get("strength") or 0.0), reverse=True)
        return rows[:limit]

    def _unresolved_questions_from_state(self, planner_state: Dict[str, Any], missing_evidence: Dict[str, bool]) -> List[str]:
        questions: List[str] = []
        if missing_evidence.get("direct_indication"):
            questions.append("Is there direct or drug-history support for this disease?")
        if missing_evidence.get("mechanism"):
            questions.append("Is there disease-relevant mechanism or target support?")
        if missing_evidence.get("safety"):
            questions.append("Could ADMET or toxicity evidence overturn current support?")
        if missing_evidence.get("clinical_prior"):
            questions.append("Is the disease-level clinical prior favorable enough for translation?")
        rescue_state = planner_state.get("rescue_state") or {}
        if rescue_state.get("rescue_required"):
            questions.append("Clinical prior is low; can mechanism or disease-context evidence rescue the hypothesis?")
        if planner_state.get("conflict_high") or planner_state.get("mechanism_conflict_high"):
            questions.append("Which source explains the current support-conflict disagreement?")
        return questions[:5]

    def _current_argument_factors(self, graph: EvidenceGraph) -> Dict[str, float]:
        try:
            result = {
                "disease": graph.disease,
                "evidence_graph": graph.summary(),
            }
            return argument_factors_from_result(result).get("factors", {})
        except Exception:
            return {
                "raw_argument_score": 0.0,
                "support_strength": 0.0,
                "conflict_strength": 0.0,
                "direct_support": 0.0,
                "cross_source_consistency": 0.0,
            }

    def _llm_judge(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        synthesis: Dict[str, Any],
        calibration: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        if not self.use_llm_judge or not self.api_url:
            return None

        argument_graph = graph.argument_graph_summary()
        acquisition_state = self._build_acquisition_state(graph, expert_outputs)
        payload = {
            "task": (
                "Assess whether the candidate molecule should be prioritized as a plausible "
                "drug-disease treatment candidate for further expert review or validation."
            ),
            "task_boundary": {
                "positive_meaning": "evidence supports candidate-level treatment potential or repurposing plausibility",
                "negative_meaning": "available evidence does not support prioritizing this candidate",
                "not_required": [
                    "regulatory approval for this exact indication",
                    "complete clinical proof",
                ],
                "safety_boundary": "The output is evidence triage, not autonomous prescribing or clinical recommendation.",
            },
            "query": {
                "smiles": graph.formula,
                "disease": graph.disease,
            },
            "central_claim": argument_graph.get("central_claim"),
            "argument_graph": {
                "support_claims": argument_graph.get("support_claims", [])[:10],
                "conflict_claims": argument_graph.get("conflict_claims", [])[:10],
                "missing_evidence": argument_graph.get("missing_evidence", [])[:8],
                "neutral_context": argument_graph.get("neutral_context", [])[:12],
                "derived_links": argument_graph.get("derived_links", [])[:12],
                "source_coverage": argument_graph.get("source_coverage", {}),
            },
            "acquisition_state": acquisition_state,
            "sanity_checks": {
                "argument_rule_score": calibration.get("raw_probability"),
                "heuristic_raw_score_0_to_10": synthesis.get("raw_score"),
                "supportive_evidence_count": graph.supportive_count(),
                "conflict_evidence_count": graph.risk_count(),
            },
            "evidence_sufficiency_constraints": [
                "A positive score should be grounded by direct indication evidence, target/pathway overlap, disease-relevant mechanism, or convergent disease-specific support.",
                "DTI evidence alone is mechanistic plausibility, but DTI plus disease-side target/pathway context and no safety conflict can support repurposing prioritization.",
                "Disease-level clinical prior is a modifier; it must not be the main reason for a positive decision.",
                "Favorable ADMET means no obvious developability conflict; it is not efficacy support.",
                "For broad disease labels such as cancer, tumor, neoplasm, or malignancy, generic disease context is insufficient without disease-specific grounding.",
                "Do not require direct drug-side indication evidence when disease-relevant mechanism, clinical feasibility, and no major safety conflict jointly support repurposing triage.",
                "If both direct drug-side evidence and disease-specific mechanistic grounding are missing, keep treatment_score below 0.50.",
                "Distinguish unqueried evidence from true absence: not_queried lowers acquisition completeness and confidence, but is not negative biomedical evidence.",
                "A no_match or matched_no_direct_indication result is stronger missing evidence than an unqueried expert.",
            ],
            "rubric_protocol": {
                "step_1_subgrades": {
                    "direct_evidence_grade": "none | weak | moderate | strong",
                    "mechanistic_grounding_grade": "none | indirect | disease_relevant | cross_source_consistent",
                    "clinical_feasibility_grade": "low | moderate | high",
                    "safety_conflict_grade": "none | manageable | significant | severe",
                },
                "step_2_evidence_grade": {
                    "A": {
                        "score_interval": [0.75, 0.95],
                        "meaning": "direct indication or strong disease-specific cross-source support with no major conflict",
                    },
                    "B": {
                        "score_interval": [0.55, 0.74],
                        "meaning": "prioritizable repurposing support, including disease-relevant mechanism plus clinical feasibility and no major safety conflict",
                    },
                    "C": {
                        "score_interval": [0.35, 0.54],
                        "meaning": "borderline but plausible triage candidate; indirect mechanism or feasibility exists but evidence remains incomplete",
                    },
                    "D": {
                        "score_interval": [0.15, 0.34],
                        "meaning": "mostly indirect/background evidence, missing direct or disease-specific grounding",
                    },
                    "E": {
                        "score_interval": [0.00, 0.14],
                        "meaning": "little support, severe missing evidence, or strong conflict/safety concern",
                    },
                },
                "step_3_score": (
                    "Choose a continuous treatment_score inside the selected grade interval. "
                    "Do not use coarse anchor values such as 0.30 or 0.50 unless the evidence exactly lies at that boundary."
                ),
            },
            "required_output": {
                "acquisition_completeness": "complete, partial, or insufficient",
                "missing_is_due_to": "not_queried, retrieval_failure, no_match, matched_no_direct_indication, mixed, or not_applicable",
                "decision_basis": "positive_support, negative_evidence, insufficient_evidence, or conflict_limited",
                "direct_evidence_grade": "none, weak, moderate, or strong",
                "mechanistic_grounding_grade": "none, indirect, disease_relevant, or cross_source_consistent",
                "clinical_feasibility_grade": "low, moderate, or high",
                "safety_conflict_grade": "none, manageable, significant, or severe",
                "evidence_grade": "A, B, C, D, or E",
                "score_interval": "two-number list matching evidence_grade interval",
                "treatment_score": "float in [0, 1]",
                "decision": "treat or not_treat",
                "confidence": "float in [0, 1]",
                "uncertainty": "low, medium, or high",
                "grade_reason": "short explanation for the selected evidence grade",
                "key_support": "list of short strings",
                "key_conflict": "list of short strings",
                "missing_evidence": "list of short strings",
                "reasoning_summary": "short paragraph based only on provided evidence",
                "safety_note": "short safety caveat",
                "recommended_next_validation": "short next validation step",
            },
        }
        prompt = (
            "You are a constrained biomedical evidence judge for drug-disease treatment assessment.\n"
            "Use only the provided Argument EvidenceGraph. Do not introduce external biomedical facts.\n"
            "The task is evidence triage for drug repurposing and treatment-potential screening, not approval or prescribing.\n"
            "Do not require definitive clinical proof before assigning a positive score.\n"
            "Score for repurposing triage priority: mechanism support, disease-related targets/pathways, clinical feasibility, and absence of major safety conflict can justify a prioritization score even without direct indication evidence.\n"
            "Do not treat retrieved background context as treatment support unless it appears as a derived support claim.\n"
            "DrugKB and DiseaseKB background facts are neutral unless cross-source grounding converts them into support or conflict.\n"
            "Favorable ADMET is only absence of a safety/developability objection, not treatment efficacy support.\n"
            "Clinical prior is a translational modifier, not disease-specific treatment evidence by itself.\n"
            "DTI support can justify a positive triage decision when it is paired with disease-side target/pathway context, favorable or moderate clinical feasibility, and no significant safety conflict.\n"
            "For broad disease labels, require direct or disease-specific grounding; do not generalize from generic disease context.\n"
            "Separate acquisition missingness from biomedical negative evidence: unqueried sources reduce completeness/confidence, while queried no-match or matched-no-direct evidence may count as missing evidence.\n"
            "First assign the four subgrades, then assign evidence_grade A-E, then choose treatment_score inside that grade's score interval.\n"
            "Scores above 0.5 mean the candidate is worth prioritizing for follow-up validation, not that it is clinically proven.\n"
            "Treat direct indication overlap, target overlap, or consistent disease-specific multi-source support as potentially actionable.\n"
            "Penalize missing direct evidence, missing disease-specific grounding, and safety/conflict evidence.\n"
            "Use continuous scores inside the interval. Avoid repeated coarse anchors such as 0.30, 0.40, 0.50, or 0.60 unless exactly justified.\n"
            "Use argument_rule_score as a structured prior sanity check; deviate only when the evidence graph justifies it.\n"
            "The decision must be treat when treatment_score >= 0.5, otherwise not_treat.\n"
            "Return strict JSON only with the required keys.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        response_text = self._call_api(prompt, "agent4")
        parsed = self._parse_llm_json(response_text) if response_text else None
        if not parsed:
            return None

        treatment_score = _safe_float(parsed.get("treatment_score"))
        if treatment_score is None:
            return None
        treatment_score = max(0.0, min(1.0, treatment_score))
        rubric = self._parse_llm_rubric(parsed, graph)
        acquisition_judgment = self._parse_acquisition_judgment(parsed, acquisition_state, rubric["evidence_grade"])
        treatment_score, rubric_guardrails = self._apply_rubric_interval(
            treatment_score,
            rubric["evidence_grade"],
            calibration.get("raw_probability"),
        )
        treatment_score, acquisition_guardrails = self._apply_acquisition_guardrails(
            treatment_score,
            acquisition_judgment,
        )
        treatment_score, guardrails = self._apply_llm_judge_guardrails(graph, treatment_score, acquisition_state)
        guardrails = rubric_guardrails + acquisition_guardrails + guardrails
        decision = "treat" if treatment_score >= 0.5 else "not_treat"
        confidence = _safe_float(parsed.get("confidence"), 0.0) or 0.0
        confidence = max(0.0, min(1.0, confidence))
        if acquisition_judgment["acquisition_completeness"] == "insufficient":
            confidence = min(confidence, 0.45)
        elif acquisition_judgment["acquisition_completeness"] == "partial":
            confidence = min(confidence, 0.70)
        uncertainty = str(parsed.get("uncertainty") or "medium").strip().lower()
        if uncertainty not in {"low", "medium", "high"}:
            uncertainty = "medium"
        return {
            "treatment_score": round(treatment_score, 6),
            "decision": decision,
            "prediction_binary": 1 if decision == "treat" else 0,
            "confidence": round(confidence, 4),
            "uncertainty": uncertainty,
            "acquisition_completeness": acquisition_judgment["acquisition_completeness"],
            "missing_is_due_to": acquisition_judgment["missing_is_due_to"],
            "decision_basis": acquisition_judgment["decision_basis"],
            "direct_evidence_grade": rubric["direct_evidence_grade"],
            "mechanistic_grounding_grade": rubric["mechanistic_grounding_grade"],
            "clinical_feasibility_grade": rubric["clinical_feasibility_grade"],
            "safety_conflict_grade": rubric["safety_conflict_grade"],
            "evidence_grade": rubric["evidence_grade"],
            "score_interval": rubric["score_interval"],
            "grade_reason": str(parsed.get("grade_reason") or "").strip(),
            "key_support": parsed.get("key_support") if isinstance(parsed.get("key_support"), list) else [],
            "key_conflict": parsed.get("key_conflict") if isinstance(parsed.get("key_conflict"), list) else [],
            "missing_evidence": parsed.get("missing_evidence") if isinstance(parsed.get("missing_evidence"), list) else [],
            "reasoning_summary": str(parsed.get("reasoning_summary") or "").strip(),
            "safety_note": str(parsed.get("safety_note") or "").strip(),
            "recommended_next_validation": str(parsed.get("recommended_next_validation") or "").strip(),
            "judge_source": "llm",
            "guardrails": guardrails,
        }

    def _parse_acquisition_judgment(
        self,
        parsed: Dict[str, Any],
        acquisition_state: Dict[str, Any],
        evidence_grade: str,
    ) -> Dict[str, str]:
        completeness_allowed = {"complete", "partial", "insufficient"}
        missing_allowed = {
            "not_queried",
            "retrieval_failure",
            "no_match",
            "matched_no_direct_indication",
            "mixed",
            "not_applicable",
        }
        basis_allowed = {
            "positive_support",
            "negative_evidence",
            "insufficient_evidence",
            "conflict_limited",
        }

        completeness = self._normalize_choice(
            parsed.get("acquisition_completeness"),
            completeness_allowed,
            str(acquisition_state.get("acquisition_completeness") or "partial"),
        )
        missing_is_due_to = self._normalize_choice(
            parsed.get("missing_is_due_to"),
            missing_allowed,
            self._infer_missingness_cause(acquisition_state),
        )
        decision_basis = self._normalize_choice(
            parsed.get("decision_basis"),
            basis_allowed,
            self._infer_decision_basis(acquisition_state, evidence_grade),
        )
        inferred_basis = self._infer_decision_basis(acquisition_state, evidence_grade)
        if completeness == "complete" and decision_basis == "insufficient_evidence":
            decision_basis = inferred_basis
        if completeness == "partial" and evidence_grade in {"A", "B", "C"} and decision_basis == "insufficient_evidence":
            decision_basis = inferred_basis
        if completeness == "insufficient":
            decision_basis = "insufficient_evidence"
        return {
            "acquisition_completeness": completeness,
            "missing_is_due_to": missing_is_due_to,
            "decision_basis": decision_basis,
        }

    def _infer_missingness_cause(self, acquisition_state: Dict[str, Any]) -> str:
        semantics = acquisition_state.get("missingness_semantics") or {}
        active = [
            key
            for key, values in semantics.items()
            if isinstance(values, list) and values
        ]
        if not active:
            return "not_applicable"
        if len(active) > 1:
            return "mixed"
        mapping = {
            "not_queried": "not_queried",
            "retrieval_failure": "retrieval_failure",
            "no_match": "no_match",
            "matched_no_direct_indication": "matched_no_direct_indication",
        }
        return mapping.get(active[0], "mixed")

    def _infer_decision_basis(self, acquisition_state: Dict[str, Any], evidence_grade: str) -> str:
        if acquisition_state.get("acquisition_completeness") == "insufficient":
            return "insufficient_evidence"
        if evidence_grade in {"A", "B"}:
            return "positive_support"
        if evidence_grade == "E":
            return "conflict_limited"
        return "negative_evidence"

    def _apply_acquisition_guardrails(
        self,
        score: float,
        acquisition_judgment: Dict[str, str],
    ) -> tuple[float, List[str]]:
        adjusted = float(score)
        guardrails: List[str] = []
        completeness = acquisition_judgment.get("acquisition_completeness")
        basis = acquisition_judgment.get("decision_basis")
        if completeness == "insufficient" and adjusted > 0.49:
            adjusted = 0.49
            guardrails.append("Capped score because acquisition completeness is insufficient; decision is evidence-insufficient rather than positive.")
        if completeness == "partial" and basis == "insufficient_evidence" and adjusted > 0.59:
            adjusted = 0.59
            guardrails.append("Capped score because acquisition is partial and the decision basis is evidence-insufficient.")
        if completeness == "insufficient" and basis == "insufficient_evidence" and adjusted > 0.49:
            adjusted = 0.49
            guardrails.append("Capped score because the decision basis is insufficient evidence.")
        return adjusted, guardrails

    def _parse_llm_rubric(self, parsed: Dict[str, Any], graph: EvidenceGraph) -> Dict[str, Any]:
        evidence_grade = self._normalize_evidence_grade(parsed.get("evidence_grade"))
        inferred_grade = self._infer_evidence_grade_from_graph(graph)
        if evidence_grade is None:
            evidence_grade = inferred_grade
        elif RUBRIC_GRADE_RANK.get(evidence_grade, 0) < RUBRIC_GRADE_RANK.get(inferred_grade, 0):
            evidence_grade = inferred_grade

        interval = RUBRIC_GRADE_INTERVALS[evidence_grade]
        return {
            "direct_evidence_grade": self._normalize_choice(
                parsed.get("direct_evidence_grade"),
                DIRECT_EVIDENCE_GRADES,
                self._infer_direct_evidence_grade(graph),
            ),
            "mechanistic_grounding_grade": self._normalize_choice(
                parsed.get("mechanistic_grounding_grade"),
                MECHANISM_GRADES,
                self._infer_mechanistic_grounding_grade(graph),
            ),
            "clinical_feasibility_grade": self._normalize_choice(
                parsed.get("clinical_feasibility_grade"),
                CLINICAL_FEASIBILITY_GRADES,
                self._infer_clinical_feasibility_grade(graph),
            ),
            "safety_conflict_grade": self._normalize_choice(
                parsed.get("safety_conflict_grade"),
                SAFETY_CONFLICT_GRADES,
                self._infer_safety_conflict_grade(graph),
            ),
            "evidence_grade": evidence_grade,
            "score_interval": [round(interval[0], 4), round(interval[1], 4)],
        }

    def _normalize_choice(self, value: Any, allowed: set[str], default: str) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return normalized if normalized in allowed else default

    def _normalize_evidence_grade(self, value: Any) -> Optional[str]:
        text = str(value or "").strip().upper()
        if text.startswith("GRADE "):
            text = text.split()[-1]
        return text if text in RUBRIC_GRADE_INTERVALS else None

    def _apply_rubric_interval(
        self,
        score: float,
        evidence_grade: str,
        structured_prior: Optional[float],
    ) -> tuple[float, List[str]]:
        low, high = RUBRIC_GRADE_INTERVALS.get(evidence_grade, RUBRIC_GRADE_INTERVALS["D"])
        guardrails: List[str] = []
        adjusted = max(low, min(high, float(score)))
        if adjusted != score:
            guardrails.append(
                f"Adjusted score into Grade {evidence_grade} interval [{low:.2f}, {high:.2f}]."
            )

        # If the LLM used a coarse anchor (e.g. 0.30), use the structured prior
        # only to place the sample more continuously inside the same grade band.
        prior = _safe_float(structured_prior)
        one_decimal_anchor = abs((adjusted * 10.0) - round(adjusted * 10.0)) < 1e-8
        if prior is not None and one_decimal_anchor and high > low:
            prior_position = max(0.0, min(1.0, prior))
            interval_score = low + ((high - low) * prior_position)
            adjusted = (0.65 * adjusted) + (0.35 * interval_score)
            adjusted = max(low, min(high, adjusted))
            guardrails.append(
                f"Refined coarse Grade {evidence_grade} anchor using structured prior within the same interval."
            )
        return adjusted, guardrails

    def _argument_role_sets(self, graph: EvidenceGraph) -> tuple[set[str], set[str], set[str]]:
        argument_graph = graph.argument_graph_summary()
        support_roles = {
            str(item.get("semantic_role") or "")
            for item in (argument_graph.get("support_claims") or [])
            if isinstance(item, dict)
        }
        conflict_roles = {
            str(item.get("semantic_role") or "")
            for item in (argument_graph.get("conflict_claims") or [])
            if isinstance(item, dict)
        }
        missing_roles = {
            str(item.get("semantic_role") or "")
            for item in (argument_graph.get("missing_evidence") or [])
            if isinstance(item, dict)
        }
        return support_roles, conflict_roles, missing_roles

    def _infer_direct_evidence_grade(self, graph: EvidenceGraph) -> str:
        support_roles, _, missing_roles = self._argument_role_sets(graph)
        if "direct_indication_match" in support_roles:
            return "strong"
        if "missing_direct_evidence" in missing_roles:
            return "none"
        return "weak" if any(item.expert == "DrugKB" for item in graph.typed_evidence) else "none"

    def _infer_mechanistic_grounding_grade(self, graph: EvidenceGraph) -> str:
        support_roles, _, missing_roles = self._argument_role_sets(graph)
        if "target_overlap" in support_roles:
            return "cross_source_consistent"
        if "disease_specificity_gap" in missing_roles:
            return "none"
        if any(item.expert == "DTI" for item in graph.typed_evidence) and any(
            item.expert == "DiseaseKB" for item in graph.typed_evidence
        ):
            return "disease_relevant"
        if any(item.expert == "DTI" for item in graph.typed_evidence):
            return "indirect"
        return "none"

    def _infer_clinical_feasibility_grade(self, graph: EvidenceGraph) -> str:
        clinical_scores = [float(item.score) for item in graph.typed_evidence if item.expert == "Clinical"]
        if not clinical_scores:
            return "moderate"
        score = max(clinical_scores)
        if score >= 0.7:
            return "high"
        if score <= 0.35:
            return "low"
        return "moderate"

    def _infer_safety_conflict_grade(self, graph: EvidenceGraph) -> str:
        _, conflict_roles, _ = self._argument_role_sets(graph)
        if "safety_conflict" not in conflict_roles:
            return "none"
        risk_scores = [float(item.score) for item in graph.typed_evidence if item.expert == "ADMET" and item.direction == "conflict"]
        if risk_scores and min(risk_scores) <= 0.25:
            return "severe"
        return "significant"

    def _infer_evidence_grade_from_graph(self, graph: EvidenceGraph) -> str:
        support_roles, conflict_roles, missing_roles = self._argument_role_sets(graph)
        has_direct = "direct_indication_match" in support_roles
        has_target_overlap = "target_overlap" in support_roles
        has_specificity_gap = "disease_specificity_gap" in missing_roles
        has_missing_direct = "missing_direct_evidence" in missing_roles
        has_safety_conflict = "safety_conflict" in conflict_roles
        dti_scores = [float(item.score or 0.0) for item in graph.typed_evidence if item.expert == "DTI"]
        clinical_scores = [float(item.score or 0.0) for item in graph.typed_evidence if item.expert == "Clinical"]
        disease_context_scores = [float(item.score or 0.0) for item in graph.typed_evidence if item.expert == "DiseaseKB"]
        drug_context_scores = [float(item.score or 0.0) for item in graph.typed_evidence if item.expert == "DrugKB"]
        dti_score = max(dti_scores) if dti_scores else 0.0
        clinical_score = max(clinical_scores) if clinical_scores else 0.0
        knowledge_score = max(disease_context_scores + drug_context_scores) if (disease_context_scores or drug_context_scores) else 0.0
        has_indirect_mechanism = bool(dti_scores)
        has_clinical_prior = bool(clinical_scores)
        has_knowledge_context = bool(disease_context_scores or drug_context_scores)

        if has_safety_conflict and not (has_direct or has_target_overlap):
            return "E"
        if has_direct and has_target_overlap and not has_safety_conflict:
            return "A"
        if has_direct or has_target_overlap:
            return "B" if not has_safety_conflict else "C"
        has_safety_conflict = "safety_conflict" in conflict_roles

        if (
            not has_specificity_gap
            and not has_safety_conflict
            and has_indirect_mechanism
            and has_clinical_prior
            and has_knowledge_context
            and dti_score >= 0.62
            and clinical_score >= 0.45
            and knowledge_score >= 0.45
        ):
            return "B"
        if (
            has_indirect_mechanism
            and has_clinical_prior
            and has_knowledge_context
            and not has_safety_conflict
            and dti_score >= 0.50
            and clinical_score >= 0.35
        ):
            return "C"
        if has_specificity_gap or has_missing_direct:
            return "D"
        if support_roles:
            return "C"
        if has_indirect_mechanism or has_clinical_prior:
            return "D"
        return "E"

    def _apply_llm_judge_guardrails(
        self,
        graph: EvidenceGraph,
        score: float,
        acquisition_state: Optional[Dict[str, Any]] = None,
    ) -> tuple[float, List[str]]:
        argument_graph = graph.argument_graph_summary()
        acquisition_state = acquisition_state or {}
        support_claims = argument_graph.get("support_claims") or []
        missing_claims = argument_graph.get("missing_evidence") or []
        conflict_claims = argument_graph.get("conflict_claims") or []
        support_roles = {
            str(item.get("semantic_role") or "")
            for item in support_claims
            if isinstance(item, dict)
        }
        missing_roles = {
            str(item.get("semantic_role") or "")
            for item in missing_claims
            if isinstance(item, dict)
        }
        has_direct = "direct_indication_match" in support_roles
        has_target_overlap = "target_overlap" in support_roles
        has_disease_specific_support = has_direct or has_target_overlap
        has_mechanism_evidence = any(item.expert == "DTI" for item in graph.typed_evidence)
        has_disease_context = any(item.expert == "DiseaseKB" for item in graph.typed_evidence)
        has_clinical_context = any(item.expert == "Clinical" for item in graph.typed_evidence)
        has_repurposing_triage_support = (
            (has_disease_specific_support or (has_mechanism_evidence and has_disease_context))
            and has_clinical_context
            and not any(
                str(item.get("semantic_role") or "") == "safety_conflict"
                for item in conflict_claims
                if isinstance(item, dict)
            )
        )
        clinical_only_support = bool(support_roles) and support_roles <= {"clinical_prior_modifier"}
        not_queried = set(acquisition_state.get("not_queried_experts") or [])
        no_match = set(acquisition_state.get("no_match_experts") or [])
        matched_no_direct = set(acquisition_state.get("matched_no_direct_evidence") or [])
        drugkb_not_queried = "DrugKB" in not_queried
        drugkb_no_match = "DrugKB" in no_match
        drugkb_matched_no_direct = "DrugKB" in matched_no_direct
        has_safety_conflict = any(
            str(item.get("semantic_role") or "") == "safety_conflict"
            for item in conflict_claims
            if isinstance(item, dict)
        )

        guardrails: List[str] = []
        capped_score = float(score)

        def cap(limit: float, reason: str) -> None:
            nonlocal capped_score
            if capped_score > limit:
                capped_score = limit
                guardrails.append(reason)

        if "disease_specificity_gap" in missing_roles and not has_disease_specific_support:
            cap(0.35, "Capped score because broad disease evidence lacks direct or disease-specific grounding.")
        if "missing_direct_evidence" in missing_roles and not has_disease_specific_support and not has_repurposing_triage_support:
            cap(0.49, "Capped score because direct drug-side support and disease-specific grounding are both missing.")
        if (drugkb_no_match or drugkb_matched_no_direct) and not has_disease_specific_support and not has_repurposing_triage_support:
            cap(0.49, "Capped score because queried DrugKB did not provide direct drug-side support and no grounded mechanism link is present.")
        if drugkb_not_queried and not has_disease_specific_support and not has_repurposing_triage_support and acquisition_state.get("acquisition_completeness") == "insufficient":
            cap(0.49, "Capped score because DrugKB was not queried and acquisition completeness is insufficient.")
        if clinical_only_support:
            cap(0.45, "Capped score because clinical prior is the only derived support signal.")
        if has_safety_conflict and not has_disease_specific_support:
            cap(0.45, "Capped score because safety conflict is present without strong grounded support.")

        return max(0.0, min(1.0, capped_score)), guardrails

    def _should_stop_early(self, graph: EvidenceGraph, planner_state: Dict[str, Any]) -> bool:
        return (
            graph.evidence_items
            and planner_state["all_core_covered"]
            and graph.mean_confidence() >= 0.74
            and graph.conflict_level() <= 0.12
            and planner_state["knowledge_conflict_score"] <= 0.22
            and not planner_state["needs_disease_context"]
            and not planner_state["needs_drug_context"]
            and not any(planner_state["evidence_gaps"].values())
        )

    def _mean_category_value(self, graph: EvidenceGraph, expert: str, categories: List[str]) -> float:
        values = graph.category_values(expert=expert, categories=categories)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _target_overlap_score(self, graph: EvidenceGraph) -> float:
        drug_targets = set()
        disease_targets = set()
        for item in graph.evidence_by_expert("DrugKB"):
            metadata = item.metadata or {}
            target = metadata.get("target_gene") or metadata.get("target_name")
            if target:
                drug_targets.add(str(target).lower())
        for item in graph.evidence_by_expert("DiseaseKB"):
            metadata = item.metadata or {}
            target = metadata.get("target_symbol") or metadata.get("target_id")
            if target:
                disease_targets.add(str(target).lower())
        if not drug_targets or not disease_targets:
            return 0.0
        overlap = drug_targets & disease_targets
        union = drug_targets | disease_targets
        return len(overlap) / len(union)

    def _llm_synthesize(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        heuristic: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.use_llm_synthesis:
            return None
        if not self.api_url:
            return None

        payload = {
            "drug": graph.formula,
            "disease": graph.disease,
            "evidence_graph_summary": graph.summary(),
            "expert_outputs": expert_outputs,
            "heuristic_summary": {
                "raw_score": heuristic["raw_score"],
                "group_scores": heuristic.get("group_scores", {}),
                "missing_expert_penalty": heuristic.get("missing_expert_penalty", 0.0),
                "conflict_penalty": heuristic.get("conflict_penalty", 0.0),
            },
            "required_output": {
                "raw_score": "float in [0, 10]",
                "explanation": "short paragraph",
            },
        }
        prompt = (
            "You are a constrained biomedical evidence synthesis expert. "
            "Use the structured evidence graph and expert outputs to judge therapeutic potential. "
            "Be conservative with safety risks and weak mechanism evidence. "
            "Return strict JSON with keys raw_score and explanation only.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        response_text = self._call_api(prompt, "agent4")
        if not response_text:
            return None

        parsed = self._parse_llm_json(response_text)
        if not parsed:
            return None

        raw_score = _safe_float(parsed.get("raw_score"))
        explanation = parsed.get("explanation")
        if raw_score is None or explanation is None:
            return None

        raw_score = max(0.0, min(10.0, raw_score))
        return {
            "raw_score": round(raw_score, 4),
            "explanation": str(explanation),
            "supportive_evidence_count": graph.supportive_count(),
            "risk_evidence_count": graph.risk_count(),
            "group_scores": heuristic.get("group_scores", {}),
            "skill_scores": heuristic.get("skill_scores", {}),
            "missing_expert_penalty": heuristic.get("missing_expert_penalty", 0.0),
            "conflict_penalty": heuristic.get("conflict_penalty", 0.0),
            "knowledge_conflict_penalty": heuristic.get("knowledge_conflict_penalty", 0.0),
            "synthesis_source": "llm",
        }

    def _llm_explain(
        self,
        graph: EvidenceGraph,
        expert_outputs: Dict[str, Dict[str, Any]],
        synthesis: Dict[str, Any],
        calibration: Dict[str, float],
    ) -> Optional[str]:
        if not self.use_llm_explanation:
            return None
        if not self.api_url:
            return None

        payload = {
            "task": "Explain a drug-disease treatment assessment from structured evidence.",
            "constraints": [
                "Do not change raw_score or calibrated_probability.",
                "Do not introduce evidence that is absent from the graph.",
                "Mention the main supporting evidence and main risk or missing evidence.",
                "Keep the explanation concise and evidence-grounded.",
            ],
            "drug": graph.formula,
            "disease": graph.disease,
            "evidence_graph_summary": {
                "coverage": graph.expert_coverage(),
                "supportive_evidence": graph.supportive_count(),
                "risk_evidence": graph.risk_count(),
                "conflict_level": round(graph.conflict_level(), 4),
                "top_evidence": graph.top_evidence(limit=8),
                "typed_evidence": [item.to_dict() for item in graph.typed_evidence[:20]],
            },
            "expert_status": {
                name: output.get("status")
                for name, output in expert_outputs.items()
            },
            "prediction": {
                "raw_score": synthesis.get("raw_score"),
                "calibrated_probability": calibration.get("calibrated_probability"),
                "group_scores": synthesis.get("group_scores", {}),
                "synthesis_source": synthesis.get("synthesis_source"),
            },
            "required_output": {
                "explanation": "short evidence-grounded paragraph",
            },
        }
        prompt = (
            "You are an evidence-grounded explanation generator for drug-disease treatment assessment. "
            "The decision score has already been produced; do not change it. "
            "Return strict JSON with one key: explanation.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        response_text = self._call_api(prompt, "agent4")
        parsed = self._parse_llm_json(response_text) if response_text else None
        if not parsed:
            return None
        explanation = parsed.get("explanation")
        if explanation is None:
            return None
        return str(explanation).strip()

    def _extract_metric(self, raw_text: str, key: str) -> Optional[float]:
        prefix = f"{key}:"
        for line in raw_text.splitlines():
            if line.strip().startswith(prefix):
                return _safe_float(line.split(":", 1)[1].strip())
        return None

    def _build_numeric_evidence(self, expert: str, category: str, claim: str, value: float, supportive_threshold: float, source: str, metadata: Optional[Dict[str, Any]] = None) -> EvidenceItem:
        normalized_value = max(0.0, min(1.0, float(value)))
        impact = "supportive" if normalized_value >= supportive_threshold else "risk"
        confidence = 0.55 + abs(normalized_value - supportive_threshold) * 0.6
        confidence = max(0.5, min(0.95, confidence))
        return EvidenceItem(expert=expert, category=category, claim=claim, value=round(normalized_value, 4), impact=impact, confidence=round(confidence, 4), source=source, metadata=metadata or {})

    def _build_admet_evidence(self, category: str, claim: str, value: float, risk_threshold: float, source: str, metadata: Optional[Dict[str, Any]] = None) -> EvidenceItem:
        normalized_value = max(0.0, min(1.0, float(value)))
        impact = "risk" if normalized_value < risk_threshold else "neutral"
        confidence = 0.55 + abs(normalized_value - risk_threshold) * 0.45
        confidence = max(0.5, min(0.9, confidence))
        return EvidenceItem(
            expert="ADMET",
            category=category,
            claim=claim,
            value=round(normalized_value, 4),
            impact=impact,
            confidence=round(confidence, 4),
            source=source,
            metadata=metadata or {},
        )

    def _expert_group_score(self, items: List[EvidenceItem], default: float) -> float:
        if not items:
            return default

        weighted_sum = 0.0
        weight_total = 0.0
        for item in items:
            if not isinstance(item.value, (int, float)):
                continue
            value = float(item.value)
            # Center each signal around 0.5 so mediocre evidence contributes
            # near-zero while very good / very poor evidence separates samples.
            signed_value = (value * 2.0) - 1.0
            weight = max(0.05, item.confidence)
            weighted_sum += signed_value * weight
            weight_total += weight

        if weight_total == 0:
            return default

        score = 0.5 + ((weighted_sum / weight_total) * 0.5)
        return max(0.0, min(1.0, score))

    def _call_api(self, prompt: str, agent_type: str) -> str:
        config = AGENT_CONFIGS.get(agent_type, AGENT_CONFIGS["agent4"])
        data = {
            "model": get_model_name(self.model),
            "messages": [
                {"role": "system", "content": config["system_role"]},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            response = requests.post(self.api_url, headers=self.api_headers, json=data, timeout=240)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            print(f"LLM synthesis disabled for this sample: {exc}")
            return ""

    def _parse_llm_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None


def analyze_drug_disease(
    formula: str,
    disease: str,
    model: str = "gpt-4o",
    agent_version: str = "eg",
    label: Optional[int] = None,
    sample_id: Optional[str] = None,
    generate_report: bool = False,
    use_memory: bool = False,
    knowledge_cutoff_date: Optional[str] = None,
) -> Dict[str, Any]:
    return TreatAgentOrchestrator(
        model=model,
        agent_version=agent_version,
        generate_report=generate_report,
        use_memory=use_memory,
        knowledge_cutoff_date=knowledge_cutoff_date,
    ).analyze(formula, disease, label=label, sample_id=sample_id)
