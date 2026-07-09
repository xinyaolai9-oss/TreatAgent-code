import json
import math
import os
import re
import shutil
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import requests

try:
    from DeepPurpose import DTI as models
except Exception:  # pragma: no cover - optional runtime dependency
    models = None

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAINED_MODEL = "MPNN_CNN_BindingDB"
DEFAULT_LOCAL_MODEL_DIR = str(PROJECT_ROOT / "assets" / "models" / "model_MPNN_CNN")
DEFAULT_DOWNLOAD_DIR = str(PROJECT_ROOT /"assets" / "models" / "model_MPNN_CNN")
DEFAULT_SEQUENCE_CACHE_PATH = PROJECT_ROOT / "data" / "dti" / "uniprot_sequence_cache.json"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def _clip_legacy_dti_score(raw_binding_score):
    # DeepPurpose BindingDB models often return affinity-like scores rather
    # than calibrated probabilities. Treat values in the common 0-10 range as
    # pAffinity-like scores and map them to [0, 1] instead of clipping all
    # values above 1.0 to a saturated positive signal.
    try:
        score = float(raw_binding_score)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    if 0.0 <= score <= 10.0:
        return max(0.0, min(1.0, score / 10.0))
    return max(0.0, min(1.0, score))


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_protein_sequence(value: Any) -> bool:
    if not value or not isinstance(value, str):
        return False
    raw = value.strip()
    sequence = re.sub(r"\s+", "", raw.upper())
    if len(sequence) < 30:
        return False
    # Short natural-language target names such as "BRCA2 DNA repair associated"
    # can otherwise look like amino-acid strings after whitespace removal.
    if re.search(r"\s", raw) and len(sequence) < 80:
        return False
    if re.search(r"[^A-Za-z\s]", raw):
        return False
    return sum(1 for char in sequence if char in VALID_AA) / len(sequence) >= 0.95


def _clean_sequence(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").upper())


def _normalize_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_:-]+", "", str(value or "").strip())


def _load_sequence_cache(path: Path = DEFAULT_SEQUENCE_CACHE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_sequence_cache(cache: dict[str, Any], path: Path = DEFAULT_SEQUENCE_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _protein_name(entry: dict[str, Any]) -> str:
    description = entry.get("proteinDescription") or {}
    recommended = description.get("recommendedName") or {}
    full_name = recommended.get("fullName") or {}
    return str(full_name.get("value") or "")


def _gene_symbol(entry: dict[str, Any]) -> str:
    genes = entry.get("genes") or []
    if not genes:
        return ""
    gene_name = genes[0].get("geneName") or {}
    return str(gene_name.get("value") or "")


def _sequence_record_from_uniprot_entry(entry: dict[str, Any], matched_by: str, query: str) -> Optional[dict[str, Any]]:
    sequence = ((entry.get("sequence") or {}).get("value") or "").strip()
    if not is_protein_sequence(sequence):
        return None
    return {
        "sequence": _clean_sequence(sequence),
        "accession": entry.get("primaryAccession"),
        "gene_symbol": _gene_symbol(entry),
        "protein_name": _protein_name(entry),
        "organism": ((entry.get("organism") or {}).get("scientificName") or ""),
        "matched_by": matched_by,
        "query": query,
        "source": "UniProt REST",
    }


def _fetch_uniprot_by_accession(accession: str, timeout: int = 20) -> Optional[dict[str, Any]]:
    accession = _normalize_token(accession)
    if not accession:
        return None
    try:
        response = requests.get(UNIPROT_ENTRY_URL.format(accession=accession), timeout=timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return _sequence_record_from_uniprot_entry(response.json(), "accession", accession)
    except Exception:
        return None


def _fetch_uniprot_by_gene(gene_symbol: str, timeout: int = 20) -> Optional[dict[str, Any]]:
    gene_symbol = _normalize_token(gene_symbol)
    if not gene_symbol:
        return None
    queries = [
        f"(gene_exact:{gene_symbol}) AND (organism_id:9606) AND (reviewed:true)",
        f"(gene:{gene_symbol}) AND (organism_id:9606) AND (reviewed:true)",
        f"(gene_exact:{gene_symbol}) AND (organism_id:9606)",
    ]
    for query in queries:
        try:
            response = requests.get(
                UNIPROT_SEARCH_URL,
                params={
                    "query": query,
                    "fields": "accession,gene_primary,protein_name,organism_name,sequence",
                    "format": "json",
                    "size": 1,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            results = response.json().get("results") or []
            if not results:
                continue
            record = _sequence_record_from_uniprot_entry(results[0], "gene_symbol", gene_symbol)
            if record:
                return record
        except Exception:
            continue
    return None


def resolve_target_sequence(
    *,
    accession: Any = None,
    gene_symbol: Any = None,
    target_name: Any = None,
    cache_path: Path = DEFAULT_SEQUENCE_CACHE_PATH,
    allow_online: Optional[bool] = None,
) -> Optional[dict[str, Any]]:
    """Resolve a human protein sequence for a biomedical target.

    Resolution order:
    1. direct sequence if any input already contains a protein sequence;
    2. local UniProt cache by accession or gene symbol;
    3. UniProt REST lookup by accession, then gene symbol.

    NCBI/other providers can be added behind this function without changing the
    orchestrator or DTI scoring code.
    """
    for raw_value in [accession, gene_symbol, target_name]:
        if is_protein_sequence(raw_value):
            return {
                "sequence": _clean_sequence(str(raw_value)),
                "accession": None,
                "gene_symbol": None,
                "protein_name": None,
                "organism": "",
                "matched_by": "provided_sequence",
                "query": "provided_sequence",
                "source": "input",
            }

    allow_online = _env_flag("TREATAGENT_ENABLE_ONLINE_SEQUENCE_FETCH", True) if allow_online is None else allow_online
    accession = _normalize_token(accession)
    gene_symbol = _normalize_token(gene_symbol)
    target_name = str(target_name or "").strip()
    cache = _load_sequence_cache(cache_path)
    cache_keys = []
    if accession:
        cache_keys.append(f"accession:{accession.upper()}")
    if gene_symbol:
        cache_keys.append(f"gene:{gene_symbol.upper()}")
    if target_name:
        cache_keys.append(f"name:{target_name.lower()}")

    for key in cache_keys:
        record = cache.get(key)
        if isinstance(record, dict) and is_protein_sequence(record.get("sequence")):
            cached = dict(record)
            cached.setdefault("source", "UniProt REST cache")
            cached["cache_hit"] = True
            return cached

    if not allow_online:
        return None

    record = None
    if accession:
        record = _fetch_uniprot_by_accession(accession)
    if record is None and gene_symbol:
        record = _fetch_uniprot_by_gene(gene_symbol)
    if record is None:
        return None

    for key in cache_keys:
        cache[key] = record
    if record.get("accession"):
        cache[f"accession:{str(record['accession']).upper()}"] = record
    if record.get("gene_symbol"):
        cache[f"gene:{str(record['gene_symbol']).upper()}"] = record
    _save_sequence_cache(cache, cache_path)
    record["cache_hit"] = False
    return record


@lru_cache(maxsize=4)
def load_dti_model(model_name=DEFAULT_PRETRAINED_MODEL, local_model_dir=DEFAULT_LOCAL_MODEL_DIR):
    if models is None:
        return None

    local_model_dir = str(Path(local_model_dir).resolve())
    if os.path.exists(local_model_dir):
        try:
            print(f"Falling back to local DeepPurpose model directory: {local_model_dir}")
            return models.model_pretrained(path_dir=local_model_dir)
        except Exception as exc:
            print(f"DeepPurpose local model load failed: {exc}")
            print("The local model directory exists but could not be loaded. Re-downloading the pretrained model.")

    download_error = None
    try:
        print(f"Loading DeepPurpose pretrained model by official name: {model_name}")
        model = models.model_pretrained(model=model_name)
        if os.path.exists(DEFAULT_DOWNLOAD_DIR):
            if os.path.exists(local_model_dir):
                shutil.rmtree(local_model_dir)
            shutil.copytree(DEFAULT_DOWNLOAD_DIR, local_model_dir)
            print(f"Copied pretrained DTI model to stable local path: {local_model_dir}")
        return model
    except Exception as exc:
        download_error = exc

    print(
        "DeepPurpose DTI model is unavailable. "
        f"Tried local directory '{local_model_dir}' and official model '{model_name}'."
    )
    if download_error is not None:
        print(f"Last DeepPurpose error: {download_error}")
    return None


def get_dti_score_deeppurpose(smiles, target_sequence):
    try:
        model = load_dti_model()
        if model is None:
            return None
        x_drug = [smiles]
        x_target = [target_sequence]
        y_pred = models.virtual_screening(x_drug, x_target, model)
        raw_binding_score = float(y_pred[0])
        dti_score = _clip_legacy_dti_score(raw_binding_score)
        if dti_score is None:
            return None
        print(f"Raw DTI binding score: {raw_binding_score:.4f}")
        return dti_score
    except Exception as exc:
        print(f"DTI inference error: {exc}")
        return None


def get_dti_score_ensemble(smiles, target_sequence, kg_path="knowledge_graph.pkl"):
    """Return a DeepPurpose DTI score for a drug-target sequence pair.

    The second argument must be a protein sequence. Disease names are not valid
    DeepPurpose DTI targets, so callers should first map a disease to one or
    more target proteins and pass sequence-level inputs here.
    """
    if not target_sequence or not isinstance(target_sequence, str):
        return None
    sequence = target_sequence.strip()
    # Protein sequences should be long amino-acid strings. This guard prevents
    # accidentally passing disease names or target symbols into DeepPurpose.
    if len(sequence) < 20 or " " in sequence:
        return None

    dp_score = get_dti_score_deeppurpose(smiles, sequence)
    if dp_score is not None:
        print(f"DTI score: {dp_score:.4f}")
        return dp_score
    return None


if __name__ == "__main__":
    result = get_dti_score_ensemble(
        smiles="CCN1C=C(C(O)=O)C(=O)C2=C1C=C(C=C2)C1=CC=NC=C1",
        target_sequence="MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP",
    )
    print(result)
