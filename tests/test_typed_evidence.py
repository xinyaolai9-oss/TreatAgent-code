from treatagent.orchestration.evidence import legacy_evidence_to_tuple
from treatagent.orchestration.orchestrator import EvidenceGraph, EvidenceItem


def test_legacy_evidence_to_typed_tuple_for_drugkb_target():
    evidence = {
        "expert": "DrugKB",
        "category": "mechanism_prior",
        "claim": "DrugCentral annotates the molecule with target TNF.",
        "value": 0.72,
        "impact": "supportive",
        "confidence": 0.8,
        "source": "DrugCentral snapshot",
        "metadata": {
            "target_gene": "TNF",
            "matched_by": "canonical_smiles",
            "snapshot_date": "2024-01-01",
        },
    }

    typed = legacy_evidence_to_tuple(evidence, "CCO", "rheumatoid arthritis")

    assert typed.subject == "CCO"
    assert typed.relation == "targets"
    assert typed.object == "TNF"
    assert typed.direction == "neutral"
    assert typed.semantic_role == "drug_target"
    assert "grounding" in typed.grounding_requirement.lower()
    assert typed.match_type == "canonical_smiles"
    assert typed.timestamp == "2024-01-01"
    assert 0 < typed.reliability <= 1
    assert 0 < typed.confidence <= 1


def test_evidence_graph_exports_typed_evidence():
    graph = EvidenceGraph("CCO", "asthma")
    graph.add_evidence(
        EvidenceItem(
            expert="DTI",
            category="mechanism",
            claim="Predicted drug-target interaction supports mechanistic plausibility.",
            value=0.7,
            impact="supportive",
            confidence=0.65,
            source="tool2.get_dti_score_ensemble",
            metadata={},
        )
    )

    summary = graph.summary()

    assert len(summary["evidence"]) == 1
    assert len(summary["typed_evidence"]) == 1
    assert summary["typed_evidence"][0]["relation"] == "predicted_mechanistic_support_for"
    assert summary["typed_evidence"][0]["direction"] == "support"
