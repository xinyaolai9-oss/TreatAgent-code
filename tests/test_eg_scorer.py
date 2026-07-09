from experiments.orchestration.eg_scorer import (
    choose_threshold,
    classification_metrics,
    expected_calibration_error,
    train_eg_scorer,
)
from treatagent.orchestration.features import FEATURE_NAMES


def _pad(values):
    return values + [0.0] * (len(FEATURE_NAMES) - len(values))


def test_calibration_metrics_are_bounded():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    metrics = classification_metrics(y_true, y_prob, 0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= expected_calibration_error(y_true, y_prob) <= 1.0


def test_train_eg_scorer_predicts_probabilities():
    train_x = [_pad([0.1, 0.0]), _pad([0.2, 0.1]), _pad([0.8, 0.9]), _pad([0.9, 0.8])]
    train_y = [0, 0, 1, 1]
    val_x = [_pad([0.15, 0.1]), _pad([0.85, 0.8])]
    val_y = [0, 1]

    scorer, info = train_eg_scorer(train_x, train_y, val_x, val_y)
    probabilities = scorer.predict_proba(val_x)
    threshold, metrics = choose_threshold(val_y, probabilities)

    assert len(probabilities) == 2
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert 0.0 < scorer.threshold < 1.0
    assert info["validation"]["f1"] >= 0.0
    assert metrics["threshold"] == round(threshold, 4)

