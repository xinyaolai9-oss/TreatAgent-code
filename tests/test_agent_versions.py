from treatagent.orchestration.orchestrator import TreatAgentOrchestrator


def test_agent_version_normalization():
    orchestrator = TreatAgentOrchestrator.__new__(TreatAgentOrchestrator)

    assert orchestrator._normalize_agent_version("eg") == "eg"
    assert orchestrator._normalize_agent_version("TreatAgent-EG") == "eg"
    assert orchestrator._normalize_agent_version("full") == "full"
    assert orchestrator._normalize_agent_version("llm-synthesis") == "ls"

    try:
        orchestrator._normalize_agent_version("unknown")
    except ValueError as exc:
        assert "Unsupported TreatAgent version" in str(exc)
    else:
        raise AssertionError("unknown agent version should fail")


def test_local_full_disables_llm_calls():
    orchestrator = TreatAgentOrchestrator(model="local", agent_version="full", max_rounds=1)

    assert orchestrator.agent_version == "full"
    assert not orchestrator.use_llm_planner
    assert not orchestrator.use_llm_explanation
    assert not orchestrator.use_llm_synthesis
    assert orchestrator.api_url == ""


def test_llm_planner_rejects_invalid_action():
    orchestrator = TreatAgentOrchestrator.__new__(TreatAgentOrchestrator)
    parsed = {"next_action": "UnknownTool", "reason": "Need another tool."}

    decision = orchestrator._validate_llm_planner_decision(
        parsed=parsed,
        allowed_actions=["DrugKB", "STOP"],
        static_output={"next_action": "DrugKB", "reason": "Need drug context."},
        planner_state={"evidence_gaps": {"DrugKB": True}},
        memory_context="",
        similar_cases_count=0,
    )

    assert decision is None


def test_llm_planner_rejects_early_stop_when_static_planner_has_gap():
    orchestrator = TreatAgentOrchestrator.__new__(TreatAgentOrchestrator)
    orchestrator.planner_budget = 4
    parsed = {"next_action": "STOP", "reason": "Enough evidence."}

    decision = orchestrator._validate_llm_planner_decision(
        parsed=parsed,
        allowed_actions=["DrugKB", "STOP"],
        static_output={"next_action": "DrugKB", "reason": "Need drug context."},
        planner_state={
            "evidence_gaps": {"DrugKB": True},
            "evidence_agents_covered": 1,
            "conflict_high": False,
            "mechanism_conflict_high": False,
        },
        memory_context="",
        similar_cases_count=0,
    )

    assert decision is None


def test_llm_planner_accepts_stop_when_stop_policy_is_satisfied():
    orchestrator = TreatAgentOrchestrator.__new__(TreatAgentOrchestrator)
    orchestrator.planner_budget = 4
    parsed = {"next_action": "STOP", "reason": "Evidence is sufficient and conflict is low."}

    decision = orchestrator._validate_llm_planner_decision(
        parsed=parsed,
        allowed_actions=["ADMET", "Clinical", "STOP"],
        static_output={"next_action": "ADMET", "reason": "Safety evidence is still useful."},
        planner_state={
            "evidence_gaps": {"ADMET": True, "Clinical": True},
            "evidence_agents_covered": 3,
            "conflict_high": False,
            "mechanism_conflict_high": False,
            "all_core_covered": False,
            "mean_confidence": 0.75,
            "mechanism_score": 0.8,
            "admet_score": 0.55,
            "clinical_score": 0.4,
            "disease_context_score": 0.7,
            "drug_context_score": 0.6,
        },
        memory_context="",
        similar_cases_count=0,
    )

    assert decision is not None
    assert decision["next_action"] == "STOP"
    assert decision["next_skill"] == "STOP"


def test_llm_planner_accepts_valid_action_case_insensitive():
    orchestrator = TreatAgentOrchestrator.__new__(TreatAgentOrchestrator)
    orchestrator.planner_budget = 4
    parsed = {
        "next_action": "drugkb",
        "reason": "Drug-side history is missing.",
        "expected_information_gain": "indication and target context",
        "risk_if_skipped": "drug prior remains unknown",
    }

    decision = orchestrator._validate_llm_planner_decision(
        parsed=parsed,
        allowed_actions=["DrugKB", "STOP"],
        static_output={"next_action": "DrugKB", "reason": "Need drug context."},
        planner_state={"evidence_gaps": {"DrugKB": True}},
        memory_context="",
        similar_cases_count=0,
    )

    assert decision is not None
    assert decision["next_action"] == "DrugKB"
    assert decision["next_skill"] == "DrugKB"
    assert decision["planner_type"] == "llm_constrained"


def test_llm_planner_rejects_low_utility_admet_action_after_budget():
    orchestrator = TreatAgentOrchestrator.__new__(TreatAgentOrchestrator)
    orchestrator.planner_budget = 3
    orchestrator.arg_threshold = 0.36
    parsed = {"next_action": "ADMET", "reason": "Safety might help."}

    decision = orchestrator._validate_llm_planner_decision(
        parsed=parsed,
        allowed_actions=["ADMET", "STOP"],
        static_output={"next_action": "ADMET", "reason": "Safety evidence is missing."},
        planner_state={
            "evidence_gaps": {"ADMET": True},
            "evidence_agents_covered": 3,
            "conflict_high": False,
            "mechanism_conflict_high": False,
            "all_core_covered": False,
            "mean_confidence": 0.52,
            "mechanism_score": 0.42,
            "admet_score": 0.55,
            "clinical_score": 0.35,
            "disease_context_score": 0.45,
            "drug_context_score": 0.42,
        },
        memory_context="",
        similar_cases_count=0,
    )

    assert decision is None


def test_budget_aware_fallback_prefers_high_utility_clinical_after_budget():
    orchestrator = TreatAgentOrchestrator.__new__(TreatAgentOrchestrator)
    orchestrator.planner_budget = 3
    orchestrator.use_llm_planner = True

    class _Registry:
        def names(self):
            return ["Clinical", "ADMET"]

        def get(self, name):
            class _Skill:
                def to_metadata(self):
                    return {"name": name}

            return _Skill()

    orchestrator.skill_registry = _Registry()
    planner_state = {
        "evidence_gaps": {"Clinical": True, "ADMET": True},
        "evidence_agents_covered": 3,
        "conflict_high": False,
        "mechanism_conflict_high": False,
        "all_core_covered": False,
        "mean_confidence": 0.72,
        "mechanism_score": 0.82,
        "admet_score": 0.35,
        "clinical_score": 0.2,
        "disease_context_score": 0.7,
        "drug_context_score": 0.62,
    }

    decision = orchestrator._budget_aware_fallback(
        static_output={
            "next_action": "ADMET",
            "reason": "Safety is missing.",
            "planner_state": planner_state,
        },
        available_skills=["Clinical", "ADMET"],
        memory_context="",
        similar_cases_count=0,
    )

    assert decision["next_action"] == "Clinical"
    assert decision["planner_type"] == "budget_aware_static_fallback"
