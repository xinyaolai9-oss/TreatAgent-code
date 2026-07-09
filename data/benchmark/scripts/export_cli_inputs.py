import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_DIR = ROOT / "data" / "benchmark" / "splits"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "benchmark" / "split_inputs"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def to_cli_sample(row: dict) -> dict:
    temporal_date = row.get("temporal_date") or row.get("pair_date")
    temporal_date_source = row.get("temporal_date_source") or row.get("pair_date_source")
    return {
        "sample_id": row.get("pair_id"),
        "label": int(row["label"]),
        "disease": row.get("normalized_disease") or row.get("example_disease") or row.get("disease"),
        "smiles": row.get("canonical_smiles") or row.get("example_smiles") or row.get("smiles"),
        "drugs": row.get("drugs", []),
        "pair_id": row.get("pair_id"),
        "nctids": row.get("nctids", []),
        "phases": row.get("phases", []),
        "statuses": row.get("statuses", []),
        "trial_count": row.get("trial_count"),
        "record_count": row.get("record_count"),
        "temporal_date": temporal_date,
        "temporal_date_source": temporal_date_source,
    }


def export_split_inputs(split_dir: Path, output_dir: Path) -> dict:
    stats = {}
    for split_path in sorted(split_dir.glob("*.json")):
        rows = load_json(split_path)
        samples = [to_cli_sample(row) for row in rows]
        output_path = output_dir / split_path.name
        write_json(output_path, samples)
        stats[split_path.stem] = {
            "rows": len(samples),
            "output": str(output_path.relative_to(ROOT)),
        }
    write_json(output_dir / "export_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark splits into TreatAgent CLI input format.")
    parser.add_argument("--split_dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    stats = export_split_inputs(args.split_dir, args.output_dir)
    print(f"Exported {len(stats)} split input files to {args.output_dir}")
    for name, row in stats.items():
        print(f"{name}: {row['rows']} rows -> {row['output']}")


if __name__ == "__main__":
    main()
