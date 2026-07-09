from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("results")
        if isinstance(rows, list):
            return rows
    raise ValueError(f"Unsupported prediction JSON format: {path}")


def parse_method_spec(spec: str) -> dict[str, str]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(
            "Prediction spec must be name:path[:probability_key[:prediction_key]], "
            f"got: {spec}"
        )
    return {
        "name": parts[0],
        "path": parts[1],
        "probability_key": parts[2] if len(parts) >= 3 and parts[2] else "",
        "prediction_key": parts[3] if len(parts) >= 4 and parts[3] else "",
    }


def prediction_table(
    path: Path,
    probability_key: str = "",
    prediction_key: str = "",
) -> dict[str, dict[str, Any]]:
    rows = load_prediction_rows(path)
    table: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or row.get("id") or index)
        label = safe_int(row.get("label"), 0)
        if probability_key:
            probability = safe_float(row.get(probability_key), 0.0)
        elif prediction_key:
            probability = float(safe_int(row.get(prediction_key), 0))
        else:
            probability = safe_float(row.get("prediction_score"), safe_float(row.get("prediction_binary"), 0.0))

        if prediction_key:
            prediction = safe_int(row.get(prediction_key), 1 if probability >= 0.5 else 0)
        else:
            prediction = 1 if probability >= 0.5 else 0

        table[sample_id] = {
            "sample_id": sample_id,
            "label": label,
            "probability": probability,
            "prediction": prediction,
            "row": row,
        }
    return table


def infer_threshold(table: dict[str, dict[str, Any]]) -> float:
    positives = [row["probability"] for row in table.values() if row["prediction"] == 1]
    negatives = [row["probability"] for row in table.values() if row["prediction"] == 0]
    if positives and negatives:
        return (min(positives) + max(negatives)) / 2.0
    return 0.5

