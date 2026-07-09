import hashlib


def build_sample_id(smiles: str, disease: str) -> str:
    payload = f"{(smiles or '').strip()}||{(disease or '').strip()}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=6).hexdigest().upper()
    return f"TA-{digest}"
