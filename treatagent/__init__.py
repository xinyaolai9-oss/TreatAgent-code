__all__ = [
    "EvidenceGraph",
    "EvidenceItem",
    "ScoreCalibrator",
    "TreatAgentOrchestrator",
    "analyze_drug_disease",
]


def __getattr__(name):
    if name in __all__:
        from .orchestration.orchestrator import (
            EvidenceGraph,
            EvidenceItem,
            ScoreCalibrator,
            TreatAgentOrchestrator,
            analyze_drug_disease,
        )

        exports = {
            "EvidenceGraph": EvidenceGraph,
            "EvidenceItem": EvidenceItem,
            "ScoreCalibrator": ScoreCalibrator,
            "TreatAgentOrchestrator": TreatAgentOrchestrator,
            "analyze_drug_disease": analyze_drug_disease,
        }
        return exports[name]
    raise AttributeError(f"module 'treatagent' has no attribute {name!r}")
