#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import json
import random
from pathlib import Path
from typing import Iterable

from experiments.orchestration.build_graph_feature_table import ID_COLUMNS, iter_result_files, read_results
from treatagent.orchestration.features import CORE_EXPERTS, FEATURE_NAMES, feature_row_from_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "feature_tables" / "graph_features_dropout.csv"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "results" / "feature_tables" / "graph_features_dropout.jsonl"
EXTRA_COLUMNS = ["dropout_variant", "dropout_mask"]


def _item_expert(item: dict) -> str:
    return str(item.get("expert") or "")


def mask_result_experts(result: dict, masked_experts: Iterable[str]) -> dict:
    masked = set(masked_experts)
    if not masked:
        return copy.deepcopy(result)

    output = copy.deepcopy(result)
    graph = output.get("evidence_graph") or {}
    for key in ["evidence", "typed_evidence"]:
        items = graph.get(key) or []
        graph[key] = [item for item in items if _item_expert(item) not in masked]

    expert_outputs = output.get("expert_outputs") or {}
    for expert in masked:
        if expert in expert_outputs:
            expert_outputs[expert] = {
                "status": "masked",
                "evidence": [],
                "raw_data": {},
                "message": "Masked during evidence-dropout feature augmentation.",
            }

    output["evidence_graph"] = graph
    output["expert_outputs"] = expert_outputs
    return output


def _present_experts(result: dict) -> list[str]:
    graph = result.get("evidence_graph") or {}
    typed = graph.get("typed_evidence") or graph.get("evidence") or []
    present = {_item_expert(item) for item in typed if _item_expert(item)}
    return [expert for expert in CORE_EXPERTS if expert in present]


def _feature_row(path: Path, payload: dict, result: dict, variant: str, masked: list[str]) -> dict:
    row = feature_row_from_result(result)
    row["source_file"] = str(path)
    row["method"] = payload.get("method") or result.get("method")
    row["backbone"] = path.parent.name
    row["dropout_variant"] = variant
    row["dropout_mask"] = ",".join(masked)
    return row


def build_dropout_rows(
    input_paths: list[Path],
    include_original: bool = True,
    include_single_expert_masks: bool = True,
    random_repeats: int = 2,
    max_drop: int = 2,
    min_keep: int = 2,
    seed: int = 13,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for path in iter_result_files(input_paths):
        payload, results = read_results(path)
        for result_index, result in enumerate(results):
            if not result.get("evidence_graph"):
                continue
            present = _present_experts(result)
            if include_original:
                rows.append(_feature_row(path, payload, result, "original", []))

            max_allowed_drop = max(0, min(max_drop, len(present) - min_keep))
            if max_allowed_drop <= 0:
                continue

            if include_single_expert_masks:
                for expert in present:
                    if len(present) - 1 < min_keep:
                        continue
                    masked_result = mask_result_experts(result, [expert])
                    rows.append(_feature_row(path, payload, masked_result, f"mask_{expert}", [expert]))

            for repeat_index in range(random_repeats):
                drop_count = rng.randint(1, max_allowed_drop)
                masked = sorted(rng.sample(present, drop_count))
                masked_result = mask_result_experts(result, masked)
                variant = f"random_{result_index}_{repeat_index}"
                rows.append(_feature_row(path, payload, masked_result, variant, masked))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ID_COLUMNS + EXTRA_COLUMNS + FEATURE_NAMES
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build evidence-dropout feature tables from TreatAgent result JSON files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Result JSON files or directories containing result JSON files.")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output_jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--random_repeats", type=int, default=2)
    parser.add_argument("--max_drop", type=int, default=2)
    parser.add_argument("--min_keep", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no_original", action="store_true")
    parser.add_argument("--no_single_expert_masks", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = build_dropout_rows(
        input_paths=args.inputs,
        include_original=not args.no_original,
        include_single_expert_masks=not args.no_single_expert_masks,
        random_repeats=args.random_repeats,
        max_drop=args.max_drop,
        min_keep=args.min_keep,
        seed=args.seed,
    )
    write_csv(args.output_csv, rows)
    write_jsonl(args.output_jsonl, rows)
    print(f"Wrote {len(rows)} dropout feature rows to {args.output_csv}")
    print(f"Wrote {len(rows)} dropout feature rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()

