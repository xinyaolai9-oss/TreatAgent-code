from experiments.orchestration.raeg_scorer import (
    RAEG_FEATURE_NAMES,
    raeg_attention_summary,
    raeg_feature_row_from_result,
)


def _sample_result():
    return {
        "sample_id": "s1",
        "label": 1,
        "smiles": "CCO",
        "disease": "demo disease",
        "expert_outputs": {"DrugKB": {"status": "success"}},
        "evidence_graph": {
            "drug": "CCO",
            "disease": "demo disease",
            "typed_evidence": [
                {
                    "subject": "CCO",
                    "relation": "has_indication",
                    "object": "demo disease",
                    "score": 1.0,
                    "direction": "support",
                    "confidence": 0.9,
                    "source": "DrugKB",
                    "match_type": "canonical_smiles",
                    "reliability": 0.9,
                    "expert": "DrugKB",
                    "category": "drug_history",
                    "claim": "known indication",
                    "metadata": {},
                },
                {
                    "subject": "CCO",
                    "relation": "has_toxicity_risk",
                    "object": "hERG",
                    "score": 0.8,
                    "direction": "conflict",
                    "confidence": 0.4,
                    "source": "ADMET",
                    "match_type": "model_prediction",
                    "reliability": 0.5,
                    "expert": "ADMET",
                    "category": "toxicity",
                    "claim": "toxicity risk",
                    "metadata": {},
                },
            ],
        },
    }


def test_raeg_feature_row_has_fixed_features():
    row = raeg_feature_row_from_result(_sample_result())
    assert set(RAEG_FEATURE_NAMES).issubset(row)
    assert row["support_att_score"] > row["conflict_att_score"]
    assert row["support_drugkb_weight"] == 1.0
    assert row["conflict_admet_weight"] == 1.0


def test_raeg_attention_summary_groups_by_direction():
    summary = raeg_attention_summary(_sample_result())
    assert summary["support"][0]["expert"] == "DrugKB"
    assert summary["conflict"][0]["expert"] == "ADMET"

