#!/usr/bin/env python3
"""Fetch ClinicalTrials.gov date metadata for NCT IDs.

This script is intentionally separate from the main benchmark pipeline because it
requires network access. It writes a reusable cache keyed by NCT ID.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data" / "benchmark" / "processed" / "extracted_with_meta.json"
DEFAULT_OUTPUT = ROOT / "data" / "benchmark" / "processed" / "nctid_trial_dates.json"


def collect_nctids(input_json: Path) -> list[str]:
    records = json.loads(input_json.read_text(encoding="utf-8"))
    return sorted({item["nctid"] for item in records if item.get("nctid")})


def get_date_value(date_struct: dict | None) -> str | None:
    if not isinstance(date_struct, dict):
        return None
    return date_struct.get("date")


def fetch_one(nctid: str, timeout: int) -> dict:
    url = f"https://clinicaltrials.gov/api/v2/studies/{nctid}"
    request = urllib.request.Request(url, headers={"User-Agent": "TreatAgentBenchmark/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    status = payload.get("protocolSection", {}).get("statusModule", {})
    return {
        "nctid": nctid,
        "start_date": get_date_value(status.get("startDateStruct")),
        "primary_completion_date": get_date_value(status.get("primaryCompletionDateStruct")),
        "completion_date": get_date_value(status.get("completionDateStruct")),
        "study_first_submit_date": status.get("studyFirstSubmitDate"),
        "last_update_submit_date": status.get("lastUpdateSubmitDate"),
        "overall_status": status.get("overallStatus"),
        "fetch_status": "ok",
        "error": None,
    }


def build(input_json: Path, output_json: Path, sleep_seconds: float, timeout: int, limit: int | None) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    nctids = collect_nctids(input_json)
    if limit is not None:
        nctids = nctids[:limit]

    existing = {}
    if output_json.exists():
        existing = json.loads(output_json.read_text(encoding="utf-8"))

    for idx, nctid in enumerate(nctids, start=1):
        if nctid in existing and existing[nctid].get("fetch_status") == "ok":
            continue
        print(f"[{idx}/{len(nctids)}] fetching {nctid}")
        try:
            existing[nctid] = fetch_one(nctid, timeout)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            existing[nctid] = {
                "nctid": nctid,
                "start_date": None,
                "primary_completion_date": None,
                "completion_date": None,
                "study_first_submit_date": None,
                "last_update_submit_date": None,
                "overall_status": None,
                "fetch_status": "error",
                "error": str(exc),
            }
        output_json.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        time.sleep(sleep_seconds)

    print(f"Wrote {len(existing)} records to {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build(args.input_json, args.output_json, args.sleep, args.timeout, args.limit)


if __name__ == "__main__":
    main()
