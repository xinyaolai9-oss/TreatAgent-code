#!/usr/bin/env python3
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import certifi


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None

    for parser in (
        lambda item: datetime.fromisoformat(item.replace("Z", "+00:00")).date(),
        lambda item: datetime.strptime(item, "%Y-%m-%d").date(),
        lambda item: datetime.strptime(item, "%Y/%m/%d").date(),
        lambda item: datetime.strptime(item, "%Y%m%d").date(),
    ):
        try:
            return parser(text)
        except ValueError:
            continue
    return None


class LongTermMemoryManager:
    _shared_embedding_model = None
    _shared_embedding_model_name = None

    os.environ["SSL_CERT_FILE"] = certifi.where()

    def __init__(
        self,
        persist_directory: str = "memory_db",
        collection_name: str = "treatagent_cases",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_store_probability: float = 0.0,
        knowledge_cutoff_date: Optional[str] = None,
    ):
        base_dir = Path(__file__).resolve().parents[2]
        persist_path = Path(persist_directory)
        if not persist_path.is_absolute():
            persist_path = base_dir / persist_path
        self.persist_directory = str(persist_path)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.min_store_probability = min_store_probability
        self.knowledge_cutoff_date = _parse_date(knowledge_cutoff_date)
        os.makedirs(self.persist_directory, exist_ok=True)

        import chromadb

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.embedding_model = self._load_embedding_model()

    def _load_embedding_model(self):
        ssl_cert_file = os.environ.get("SSL_CERT_FILE")
        if ssl_cert_file and not os.path.exists(ssl_cert_file):
            raise RuntimeError(
                f"SSL_CERT_FILE points to a missing file: {ssl_cert_file}. "
                "Unset SSL_CERT_FILE or point it to a valid certificate bundle before enabling --use_memory."
            )

        if (
            LongTermMemoryManager._shared_embedding_model is not None
            and LongTermMemoryManager._shared_embedding_model_name == self.embedding_model_name
        ):
            return LongTermMemoryManager._shared_embedding_model

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.embedding_model_name)
        LongTermMemoryManager._shared_embedding_model = model
        LongTermMemoryManager._shared_embedding_model_name = self.embedding_model_name
        return model

    def retrieve_similar_cases(
        self,
        smiles: str,
        disease: str,
        top_k: int = 3,
        knowledge_cutoff_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []

        effective_cutoff = _parse_date(knowledge_cutoff_date) or self.knowledge_cutoff_date
        query_text = self._build_case_text(smiles, disease, {}, [], None, None)
        embedding = self.embedding_model.encode([query_text])[0].tolist()
        query_size = max(top_k * 5, top_k)
        result = self.collection.query(query_embeddings=[embedding], n_results=query_size)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]

        similar_cases = []
        for idx, metadata in enumerate(metadatas):
            item = dict(metadata)
            if effective_cutoff is not None and not self._case_allowed(item, effective_cutoff):
                continue
            item["case_id"] = ids[idx]
            item["document"] = documents[idx]
            similar_cases.append(item)
            if len(similar_cases) >= top_k:
                break
        return similar_cases

    def store_case(
        self,
        smiles: str,
        disease: str,
        trajectory: List[Dict[str, Any]],
        final_prediction: int,
        calibrated_prob: float,
        evidence_summary: Dict[str, Any],
        case_date: Optional[str] = None,
    ) -> Optional[str]:
        if calibrated_prob < self.min_store_probability:
            return None

        parsed_case_date = _parse_date(case_date)
        if self.knowledge_cutoff_date is not None:
            # Avoid leaking future / same-run evaluation cases into memory when a
            # temporal cutoff is active unless an explicit pre-cutoff case date
            # is provided.
            if parsed_case_date is None or parsed_case_date > self.knowledge_cutoff_date:
                return None

        case_id = f"{disease}_{self.collection.count() + 1}"
        document = self._build_case_text(smiles, disease, evidence_summary, trajectory, final_prediction, calibrated_prob)
        embedding = self.embedding_model.encode([document])[0].tolist()
        stored_at = datetime.now().isoformat()
        metadata = {
            "smiles": smiles,
            "disease": disease,
            "final_prediction": int(final_prediction),
            "calibrated_prob": round(float(calibrated_prob), 4),
            "evidence_summary": json.dumps(evidence_summary, ensure_ascii=False),
            "trajectory_summary": json.dumps(trajectory[-2:], ensure_ascii=False),
            "stored_at": stored_at,
            "stored_at_date": stored_at[:10],
            "case_date": parsed_case_date.isoformat() if parsed_case_date else None,
        }
        metadata = self._sanitize_metadata(metadata)
        self.collection.add(
            ids=[case_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )
        return case_id

    def format_for_planner(self, similar_cases: List[Dict[str, Any]]) -> str:
        if not similar_cases:
            return "No similar historical cases were retrieved."

        lines = ["You have previously handled similar cases:"]
        for index, case in enumerate(similar_cases, start=1):
            decision = "Positive" if int(case.get("final_prediction", 0)) == 1 else "Negative"
            evidence_summary = case.get("evidence_summary", "{}")
            try:
                summary_obj = json.loads(evidence_summary)
            except json.JSONDecodeError:
                summary_obj = {}
            key_evidence = summary_obj.get("top_evidence", [])
            key_evidence_text = ", ".join(key_evidence) if key_evidence else "limited structured evidence summary"
            lines.append(
                f"- Case {index}: {case.get('disease', 'Unknown disease')}, final decision: {decision} "
                f"(calibrated probability {float(case.get('calibrated_prob', 0.0)):.2f}). "
                f"Key evidence: {key_evidence_text}."
            )
        lines.append("Use these as reference but not as definitive rules.")
        return "\n".join(lines)

    def _case_allowed(self, metadata: Dict[str, Any], cutoff: date) -> bool:
        case_date = _parse_date(metadata.get("case_date"))
        if case_date is not None:
            return case_date <= cutoff

        stored_at_date = _parse_date(metadata.get("stored_at_date"))
        if stored_at_date is not None:
            return stored_at_date <= cutoff

        # If legacy memory entries have no date metadata, keep them out when a
        # cutoff is requested to avoid silent temporal leakage.
        return False

    def _build_case_text(
        self,
        smiles: str,
        disease: str,
        evidence_summary: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        final_prediction: Optional[int],
        calibrated_prob: Optional[float],
    ) -> str:
        return (
            f"Drug SMILES: {smiles}\n"
            f"Disease: {disease}\n"
            f"Final prediction: {final_prediction}\n"
            f"Calibrated probability: {calibrated_prob}\n"
            f"Evidence summary: {json.dumps(evidence_summary, ensure_ascii=False)}\n"
            f"Trajectory: {json.dumps(trajectory, ensure_ascii=False)}"
        )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = json.dumps(value, ensure_ascii=False)
        return sanitized
