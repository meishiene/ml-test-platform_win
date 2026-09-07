from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import httpx

from .config import ApiConfig
from .protocols import headers


def connectivity_endpoint(config: ApiConfig) -> str:
    base = config.base_url.rstrip("/")
    if config.protocol in {"openai", "volcengine"}:
        return config.endpoint
    if config.protocol == "anthropic":
        return f"{base}/messages"
    if config.protocol == "gemini":
        return f"{base}/models/{quote(config.model, safe='')}:generateContent"
    raise ValueError(f"不支持的协议: {config.protocol}")


def connectivity_payload(config: ApiConfig) -> dict[str, Any]:
    prompt = "Reply with OK only."
    if config.protocol in {"openai", "volcengine"}:
        return {"model": config.model, "stream": False, "max_tokens": 32,
                "reasoning_effort": "low", "messages": [{"role": "user", "content": prompt}]}
    if config.protocol == "anthropic":
        return {"model": config.model, "stream": False, "max_tokens": 32,
                "messages": [{"role": "user", "content": prompt}]}
    if config.protocol == "gemini":
        return {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 32, "thinkingConfig": {"thinkingBudget": 0}}}
    raise ValueError(f"不支持的协议: {config.protocol}")


def check_connectivity(config: ApiConfig) -> dict[str, Any]:
    """Send one tiny request. HTTP 2xx + a valid JSON response means reachable.

    Visible content is deliberately not required: a reasoning model can consume the
    tiny output budget in its thinking phase while the endpoint itself is healthy.
    """
    started = time.perf_counter()
    result: dict[str, Any] = {"name": "connectivity", "passed": False, "protocol": config.protocol,
                              "endpoint": connectivity_endpoint(config), "http_status": None,
                              "duration_ms": None, "error": None}
    try:
        with httpx.Client(timeout=min(config.request_timeout_seconds, 60), verify=config.verify_ssl,
                          trust_env=config.trust_env) as client:
            response = client.post(result["endpoint"], headers=headers(config.protocol, config.api_key, "connectivity-check"),
                                   json=connectivity_payload(config))
        result["http_status"] = response.status_code
        if 200 <= response.status_code < 300:
            try:
                payload = response.json()
                result["passed"] = isinstance(payload, dict) and bool(payload)
                if not result["passed"]:
                    result["error"] = "接口返回了空 JSON 对象"
            except ValueError:
                result["error"] = "接口返回 HTTP 2xx，但响应不是有效 JSON"
        else:
            result["error"] = f"HTTP {response.status_code}: {response.text[:500]}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["duration_ms"] = (time.perf_counter() - started) * 1000
    return result
