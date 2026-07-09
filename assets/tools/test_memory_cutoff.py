import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from treatagent.memory.manager import LongTermMemoryManager


class DummyEmbeddingModel:
    def encode(self, texts):
        class _Vector(list):
            def tolist(self):
                return list(self)

        return [_Vector([float(len(text) % 13), 0.5, 1.0]) for text in texts]


class FakeCollection:
    def __init__(self):
        self.items = []

    def count(self):
        return len(self.items)

    def add(self, ids, embeddings, documents, metadatas):
        for idx in range(len(ids)):
            self.items.append(
                {
                    "id": ids[idx],
                    "embedding": embeddings[idx],
                    "document": documents[idx],
                    "metadata": metadatas[idx],
                }
            )

    def query(self, query_embeddings, n_results):
        limited = self.items[:n_results]
        return {
            "documents": [[item["document"] for item in limited]],
            "metadatas": [[item["metadata"] for item in limited]],
            "ids": [[item["id"] for item in limited]],
        }


manager = LongTermMemoryManager.__new__(LongTermMemoryManager)
manager.collection = FakeCollection()
manager.embedding_model = DummyEmbeddingModel()
manager.min_store_probability = 0.0
manager.knowledge_cutoff_date = LongTermMemoryManager.__dict__["__init__"].__globals__["_parse_date"]("2024-12-31")

no_date_case = manager.store_case(
    smiles="CCO",
    disease="test disease",
    trajectory=[],
    final_prediction=1,
    calibrated_prob=0.9,
    evidence_summary={"top_evidence": ["no date case"]},
)

old_case = manager.store_case(
    smiles="CCN",
    disease="test disease",
    trajectory=[],
    final_prediction=1,
    calibrated_prob=0.8,
    evidence_summary={"top_evidence": ["old case"]},
    case_date="2024-01-01",
)

future_case = manager.store_case(
    smiles="CCC",
    disease="test disease",
    trajectory=[],
    final_prediction=0,
    calibrated_prob=0.7,
    evidence_summary={"top_evidence": ["future case"]},
    case_date="2025-01-01",
)

cases = manager.retrieve_similar_cases("CCN", "test disease", top_k=5)

print("no_date_case", no_date_case)
print("old_case", old_case)
print("future_case", future_case)
print("retrieved_count", len(cases))
print("retrieved_case_dates", [case.get("case_date") for case in cases])
print("retrieved_ids", [case.get("case_id") for case in cases])
