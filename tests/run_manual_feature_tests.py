import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(path, names):
    namespace = runpy.run_path(path)
    for name in names:
        namespace[name]()


run(
    "tests/test_typed_evidence.py",
    [
        "test_legacy_evidence_to_typed_tuple_for_drugkb_target",
        "test_evidence_graph_exports_typed_evidence",
    ],
)
run(
    "tests/test_graph_features.py",
    [
        "test_extract_graph_features_from_typed_evidence",
        "test_feature_row_from_legacy_result",
    ],
)
run(
    "tests/test_eg_scorer.py",
    [
        "test_calibration_metrics_are_bounded",
        "test_train_eg_scorer_predicts_probabilities",
    ],
)
run(
    "tests/test_agent_versions.py",
    [
        "test_agent_version_normalization",
        "test_local_full_disables_llm_calls",
    ],
)
print("manual feature tests passed")
