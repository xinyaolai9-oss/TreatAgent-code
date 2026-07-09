#!/usr/bin/env python3
"""Summarize frozen TreatAgent batch runs from background logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRIC_PATTERNS = {
    "accuracy": re.compile(r"Accuracy:\s+([0-9.]+)"),
    "f1": re.compile(r"F1 Score:\s+([0-9.]+)"),
    "precision": re.compile(r"Precision:\s+([0-9.]+)"),
    "recall": re.compile(r"Recall:\s+([0-9.]+)"),
}


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    progress = re.findall(r"Processing sample\s+(\d+)/(\d+)", text)
    saved = re.findall(r"Detailed results saved to:\s+(\S+)", text)
    errors = re.findall(r"(Traceback|API call failed|ERROR:|Error processing sample)", text)
    result = {
        "background_log": str(path),
        "status": "completed" if saved else "running_or_failed",
        "processed": "",
        "total": "",
        "result_json": saved[-1] if saved else "",
        "error_markers": len(errors),
    }
    if progress:
        result["processed"], result["total"] = progress[-1]
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.findall(text)
        result[key] = match[-1] if match else ""
    return result


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return []
    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        values = line.split("\t")
        row = {headers[i]: values[i] if i < len(values) else "" for i in range(len(headers))}
        rows.append(row)
    return rows


def summarize(run_dir: Path) -> tuple[list[dict], str]:
    manifests = sorted(run_dir.glob("*manifest.tsv"))
    rows: list[dict] = []
    for manifest in manifests:
        for row in load_manifest(manifest):
            log_path = Path(row.get("background_log", ""))
            parsed = parse_log(log_path)
            rows.append({**row, **parsed})

    if not rows:
        for log_path in sorted(run_dir.glob("*.background.log")):
            rows.append(parse_log(log_path))

    headers = [
        "kind",
        "split_prefix",
        "split_name",
        "status",
        "processed",
        "total",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "error_markers",
        "result_json",
    ]
    lines = ["# Frozen Run Summary", ""]
    lines.append(f"Run directory: `{run_dir}`")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    lines.append("")
    lines.append("Raw JSON:")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(rows, indent=2, ensure_ascii=False))
    lines.append("```")
    return rows, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=None, help="results/frozen_runs/<RUN_TAG>")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        candidates = sorted(Path("results/frozen_runs").glob("*"))
        if not candidates:
            raise SystemExit("No frozen run directory found.")
        run_dir = candidates[-1]

    _, markdown = summarize(run_dir)
    output = run_dir / "summary.md"
    output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

