import os

import requests

from .runtime import get_api_headers, get_model_name


def get_response(prompt, model="gpt-4o"):
    url = os.getenv("URL_VALUE", "")
    data = {
        "model": get_model_name(model),
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(url, headers=get_api_headers(), json=data, timeout=240)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"{exc}")
        return "Error: API call failed"
