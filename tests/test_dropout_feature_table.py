from experiments.orchestration.build_dropout_feature_table import mask_result_experts
from treatagent.orchestration.features import feature_row_from_result


def test_mask_result_experts_removes_typed_evidence_and_marks_missing():
    result = {
        "sample_id": "x",
        "label": 1,
        "prediction_binary": 1,
        "prediction_score": 1,
        "calibrated_probability": 0.7,
        "raw_score": 7.0,
        "smiles": "CC",
        "disease": "test disease",
        "evidence_graph": {
            "drug": "CC",
            "disease": "test disease",
            "typed_evidence": [
                {
                    "expert": "DTI",
                    "relation": "binds_target",
                    "category": "mechanism",
                    "direction": "support",
                    "score": 0.9,
                    "confidence": 0.8,
                    "reliability": 0.7,
                    "source": "dti",
                },
                {
                    "expert": "Clinical",
                    "relation": "clinical_prior",
                    "category": "clinical_prior",
                    "direction": "support",
                    "score": 0.6,
                    "confidence": 0.7,
                    "reliability": 0.8,
                    "source": "clinical",
                },
            ],
        },
        "expert_outputs": {
            "DTI": {"status": "ok", "evidence": [{}]},
            "Clinical": {"status": "ok", "evidence": [{}]},
        },
    }

    masked = mask_result_experts(result, ["Clinical"])
    row = feature_row_from_result(masked)

    assert row["dti_present"] == 1.0
    assert row["clinical_present"] == 0.0
    assert row["clinical_prior"] == 0.0
    assert row["missing_evidence_count"] >= 1.0

