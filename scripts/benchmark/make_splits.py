#!/usr/bin/env python3
"""Create random and drug-disjoint benchmark splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data" / "benchmark" / "processed" / "pair_level_dataset.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "benchmark" / "splits"
DEFAULT_STATS = ROOT / "data" / "benchmark" / "processed" / "split_stats.json"


def stratified_random_split(rows: list[dict], seed: int, ratios=(0.7, 0.1, 0.2)) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    splits = {"train": [], "val": [], "test": []}
    for label_rows in by_label.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        n = len(label_rows)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        splits["train"].extend(label_rows[:n_train])
        splits["val"].extend(label_rows[n_train : n_train + n_val])
        splits["test"].extend(label_rows[n_train + n_val :])

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def drug_disjoint_split(rows: list[dict], seed: int, ratios=(0.7, 0.1, 0.2)) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["canonical_smiles"]].append(row)

    group_list = list(groups.values())
    rng.shuffle(group_list)
    group_list.sort(key=lambda group: len(group), reverse=True)

    total = len(rows)
    targets = {"train": total * ratios[0], "val": total * ratios[1], "test": total * ratios[2]}
    splits = {"train": [], "val": [], "test": []}

    for group in group_list:
        # Greedy fill by relative under-target size.
        best = max(splits, key=lambda name: targets[name] - len(splits[name]))
        splits[best].extend(group)

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def summarize_split(splits: dict[str, list[dict]]) -> dict:
    stats = {}
    drug_sets = {}
    for name, rows in splits.items():
        drug_sets[name] = {row["canonical_smiles"] for row in rows}
        stats[name] = {
            "rows": len(rows),
            "label_counts": dict(Counter(str(row["label"]) for row in rows)),
            "unique_drugs": len(drug_sets[name]),
            "unique_diseases": len({row["normalized_disease"] for row in rows}),
        }

    stats["drug_overlap"] = {
        "train_val": len(drug_sets.get("train", set()) & drug_sets.get("val", set())),
        "train_test": len(drug_sets.get("train", set()) & drug_sets.get("test", set())),
        "val_test": len(drug_sets.get("val", set()) & drug_sets.get("test", set())),
    }
    return stats


def write_splits(prefix: str, splits: dict[str, list[dict]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = output_dir / f"{prefix}_{name}.json"
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def build(input_json: Path, output_dir: Path, stats_json: Path, seed: int) -> None:
    rows = json.loads(input_json.read_text(encoding="utf-8"))
    random_splits = stratified_random_split(rows, seed)
    drug_splits = drug_disjoint_split(rows, seed)

    write_splits("random", random_splits, output_dir)
    write_splits("drug_disjoint", drug_splits, output_dir)

    stats = {
        "seed": seed,
        "random": summarize_split(random_splits),
        "drug_disjoint": summarize_split(drug_splits),
    }
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote splits to {output_dir}")
    print(f"Wrote stats to {stats_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stats_json", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    build(args.input_json, args.output_dir, args.stats_json, args.seed)


if __name__ == "__main__":
    main()
