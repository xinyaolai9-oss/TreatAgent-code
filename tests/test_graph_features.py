from treatagent.orchestration.features import extract_graph_features, feature_row_from_result


def test_extract_graph_features_from_typed_evidence():
    graph = {
        "typed_evidence": [
            {
                "subject": "drug",
                "relation": "has_indication",
                "object": "asthma",
                "score": 0.9,
                "direction": "support",
                "confidence": 0.7,
                "source": "DrugCentral snapshot",
                "match_type": "canonical_smiles",
                "reliability": 0.9,
                "expert": "DrugKB",
                "category": "drug_history",
                "metadata": {"match_score": 0.8},
            },
            {
                "subject": "drug",
                "relation": "has_toxicity_risk",
                "object": "toxicity",
                "score": 0.6,
                "direction": "conflict",
                "confidence": 0.3,
                "source": "tool1.admet_data",
                "match_type": "model_prediction",
                "reliability": 0.45,
                "expert": "ADMET",
                "category": "toxicity",
                "metadata": {},
            },
        ]
    }

    features = extract_graph_features(graph, {"DrugKB": {"status": "ok"}, "ADMET": {"status": "ok"}})

    assert features["direct_indication_match"] == 1.0
    assert features["indication_similarity"] == 0.8
    assert features["support_score"] >= 0.7
    assert features["conflict_score"] >= 0.3
    assert 0.0 <= features["support_conflict_ratio"] <= 1.0
    assert features["drugkb_present"] == 1.0
    assert features["admet_present"] == 1.0
    assert features["missing_evidence_count"] == 3.0
    assert features["drugkb_support_x_dti_support"] == 0.0
    assert features["admet_conflict_x_low_clinical_prior"] > 0.0


def test_feature_row_from_legacy_result():
    result = {
        "sample_id": "PAIR-1",
        "label": 1,
        "prediction_binary": 1,
        "smiles": "CCO",
        "disease": "asthma",
        "expert_outputs": {"DTI": {"status": "ok"}},
        "evidence_graph": {
            "drug": "CCO",
            "disease": "asthma",
            "evidence": [
                {
                    "expert": "DTI",
                    "category": "mechanism",
                    "claim": "Predicted support.",
                    "value": 0.75,
                    "impact": "supportive",
                    "confidence": 0.65,
                    "source": "tool2.get_dti_score_ensemble",
                    "metadata": {},
                }
            ],
        },
    }

    row = feature_row_from_result(result)

    assert row["sample_id"] == "PAIR-1"
    assert row["label"] == 1
    assert row["dti_strength"] == 0.75
    assert row["dti_present"] == 1.0
