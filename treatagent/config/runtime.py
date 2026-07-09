"""
Runtime API configuration shared by the TreatAgent orchestrator.
"""

import os


API_value = os.getenv("API_VALUE", "sk-")
URL_value = os.getenv("URL_VALUE", "")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


LOCAL_ONLY_INFERENCE = _env_flag("TREATAGENT_LOCAL_ONLY", False)
USE_LLM_SYNTHESIS = _env_flag("TREATAGENT_USE_LLM_SYNTHESIS", True) and not LOCAL_ONLY_INFERENCE

API_CONFIG = {
    "url": URL_value if USE_LLM_SYNTHESIS else "",
    "authorization": API_value,
    "headers": {
        "Content-Type": "application/json",
    },
}

MODEL_MAPPING = {
    "local": "local",
    # Current Table 1 DMX model set.
    "gpt-4o": "gpt-4o-mini",
    "gpt5": "gpt-5-mini",
    "gemini": "gemini-2.5-flash",
    "deepseek-v3": "deepseek-chat",
    "qwen": "Qwen3-32B",
    "glm-4.5": "GLM-4.5",
    # The following names are DMX-published candidates, but availability
    # depends on the token group/channel configured in the DMX dashboard.
    "kimi-k2": "Kimi-K2-0905",
    "claude": "claude-sonnet-4-20250514",
    "llama-4": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "deepseek-r1": "deepseek-reasoner",
    "o3": "o3-mini",
    "grok": "grok-4",
}


AGENT_CONFIGS = {
    "agent1": {
        "system_role": "You are a Molecule Expert specializing in ADMET analysis.",
        "model": "gpt-4o",
    },
    "agent2": {
        "system_role": "You are a Mechanism Validation Expert specializing in drug-target interactions.",
        "model": "gpt-4o",
    },
    "agent3": {
        "system_role": "You are a Clinical Expert specializing in therapeutic development.",
        "model": "gpt-4o",
    },
    "agent4": {
        "system_role": "You are a Pharmaceutical Expert specializing in multi-criteria decision making.",
        "model": "gpt-4o",
    },
}


def get_api_headers():
    headers = API_CONFIG["headers"].copy()
    authorization = API_CONFIG["authorization"]
    auth_scheme = os.getenv("DMX_AUTH_SCHEME", "raw").strip().lower()
    if auth_scheme == "bearer" and authorization and not authorization.lower().startswith("bearer "):
        authorization = f"Bearer {authorization}"
    headers["Authorization"] = authorization
    return headers


def get_model_name(model_key):
    return MODEL_MAPPING.get(model_key, model_key)


def update_api_config(url=None, authorization=None):
    if url:
        API_CONFIG["url"] = url
    if authorization:
        API_CONFIG["authorization"] = authorization


def llm_synthesis_enabled():
    return USE_LLM_SYNTHESIS
