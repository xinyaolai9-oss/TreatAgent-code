from treatagent.orchestration.argument_graph_scorer import argument_factors_from_result


def test_argument_factors_are_bounded_and_explainable():
    result = {
        "sample_id": "s1",
        "label": 1,
        "smiles": "CCO",
        "disease": "epilepsy",
        "evidence_graph": {
            "typed_evidence": [
                {
                    "expert": "DrugKB",
                    "category": "drug_history",
                    "direction": "support",
                    "score": 1.0,
                    "confidence": 0.9,
                    "reliability": 0.9,
                    "relation": "has_indication",
                    "object": "epilepsy",
                    "match_type": "canonical_smiles",
                    "source": "DrugKB",
                    "claim": "known indication",
                    "metadata": {},
                },
                {
                    "expert": "DTI",
                    "category": "mechanism",
                    "direction": "support",
                    "score": 0.8,
                    "confidence": 0.5,
                    "reliability": 0.6,
                    "relation": "predicted_mechanistic_support_for",
                    "object": "epilepsy",
                    "match_type": "model_prediction",
                    "source": "DTI",
                    "claim": "mechanistic support",
                    "metadata": {},
                },
            ]
        },
    }
    row = argument_factors_from_result(result)
    factors = row["factors"]
    assert 0.0 <= factors["raw_argument_score"] <= 1.0
    assert factors["direct_support"] > 0.0
    assert factors["mechanism_support"] > 0.0
    assert row["top_support_arguments"]
