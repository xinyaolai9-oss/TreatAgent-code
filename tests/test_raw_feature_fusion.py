from experiments.orchestration.raw_feature_fusion import (
    RAW_FEATURE_FUSION_FEATURES,
    train_raw_feature_fusion_scorer,
)


def test_raw_feature_fusion_feature_set_excludes_graph_features():
    excluded = {
        "support_count",
        "conflict_count",
        "support_score",
        "conflict_score",
        "support_conflict_ratio",
        "reliability_mean",
        "reliability_max",
        "confidence_mean",
        "confidence_max",
        "evidence_count",
        "evidence_coverage",
        "source_diversity",
        "missing_evidence_count",
        "agent_failure_count",
    }

    assert "dti_strength" in RAW_FEATURE_FUSION_FEATURES
    assert "admet_safety_score" in RAW_FEATURE_FUSION_FEATURES
    assert "clinical_prior" in RAW_FEATURE_FUSION_FEATURES
    assert not excluded.intersection(RAW_FEATURE_FUSION_FEATURES)


def test_train_raw_feature_fusion_predicts_probabilities():
    pad = [0.0] * (len(RAW_FEATURE_FUSION_FEATURES) - 2)
    train_x = [[0.0, 0.1] + pad, [0.1, 0.2] + pad, [0.9, 0.8] + pad, [1.0, 0.9] + pad]
    train_y = [0, 0, 1, 1]
    val_x = [[0.0, 0.2] + pad, [1.0, 0.8] + pad]
    val_y = [0, 1]

    scorer, info = train_raw_feature_fusion_scorer(train_x, train_y, val_x, val_y)
    probabilities = scorer.predict_proba(val_x)

    assert len(probabilities) == 2
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert info["validation"]["f1"] >= 0.0

