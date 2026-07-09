#!/usr/bin/env python3
import json
import os
import sys

import requests

from treatagent.config.runtime import MODEL_MAPPING, get_api_headers


def main() -> int:
    url = os.getenv("URL_VALUE", "")
    if not url:
        print("URL_VALUE is missing", file=sys.stderr)
        return 2
    if not os.getenv("API_VALUE"):
        print("API_VALUE is missing", file=sys.stderr)
        return 2

    aliases = sys.argv[1:] or ["gpt-4o", "gpt5", "kimi-k2", "claude", "gemini"]
    for alias in aliases:
        model = MODEL_MAPPING.get(alias, alias)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: ANSWER: 1"}],
            "temperature": 0,
        }
        print(f"\n== {alias} -> {model} ==")
        try:
            response = requests.post(url, headers=get_api_headers(), json=payload, timeout=60)
            print(f"status={response.status_code}")
            if response.status_code != 200:
                print(response.text[:1000])
                continue
            data = response.json()
            print(json.dumps(data.get("usage", {}), ensure_ascii=False))
            print(data["choices"][0]["message"]["content"][:500])
        except Exception as exc:
            print(f"ERROR: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
