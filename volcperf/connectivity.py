from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import httpx

from .config import ApiConfig
from .protocols import api_root
from .protocols import headers


_PROTOCOL_NAMES = {
    "openai": "OpenAI Chat Completions",
    "openai_responses": "OpenAI Responses",
    "openai_images": "OpenAI Images",
    "anthropic": "Claude Messages",
    "gemini": "Gemini generateContent",
    "volcengine": "火山云 Chat Completions",
}


def connectivity_endpoint(config: ApiConfig) -> str:
    base = api_root(config.protocol, config.base_url)
    if config.protocol == "openai_images":
        return f"{base}/images/generations"
    if config.protocol == "openai_responses":
        return f"{base}/responses"
    if config.protocol in {"openai", "volcengine"}:
        return config.endpoint
    if config.protocol == "anthropic":
        return f"{base}/messages"
    if config.protocol == "gemini":
        return f"{base}/models/{quote(config.model, safe='')}:generateContent"
    raise ValueError(f"不支持的协议: {config.protocol}")


def connectivity_payload(config: ApiConfig) -> dict[str, Any]:
    prompt = "Reply with OK only."
    if config.protocol == "openai_images":
        return {"model": config.model, "prompt": "A small green circle on a white background.",
                "n": 1, "size": "1024x1024", "quality": "low"}
    if config.protocol == "openai_responses":
        # Keep the probe to the portable Responses subset. Third-party gateways
        # often implement /responses before optional reasoning controls.
        return {"model": config.model, "max_output_tokens": 2048, "input": prompt}
    if config.protocol in {"openai", "volcengine"}:
        # Thinking 模型的兼容网关可能要求输出预算大于 1024；与随机压测
        # 的默认输出上限保持一致，避免健康探针在正式请求前误判失败。
        return {"model": config.model, "stream": False, "max_tokens": 2048,
                "reasoning_effort": "low", "messages": [{"role": "user", "content": prompt}]}
    if config.protocol == "anthropic":
        # K3/Thinking routes commonly reject output budgets <= 1024 even when
        # the client does not explicitly include a thinking block.
        return {"model": config.model, "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}]}
    if config.protocol == "gemini":
        return {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 32, "thinkingConfig": {"thinkingBudget": 0}}}
    raise ValueError(f"不支持的协议: {config.protocol}")


def connectivity_headers(config: ApiConfig) -> dict[str, str]:
    """Headers for a non-streaming probe, including common gateway auth."""
    result = headers(config.protocol, config.api_key, "connectivity-check")
    result["Accept"] = "application/json"
    if config.protocol == "anthropic":
        # Native Anthropic uses x-api-key. Many multi-protocol gateways accept
        # the same key through Bearer auth, so sending both is the safest
        # compatible shape and does not expose either value in diagnostics.
        result["Authorization"] = f"Bearer {config.api_key}"
    return result


def _request_shape(config: ApiConfig, request_headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "POST",
        "endpoint": connectivity_endpoint(config),
        "protocol": _PROTOCOL_NAMES.get(config.protocol, config.protocol),
        "model": config.model,
        "header_names": sorted(request_headers),
        "body_keys": sorted(body),
    }


def _upstream_diagnosis(config: ApiConfig, status_code: int, response_text: str) -> str | None:
    lowered = response_text.lower()
    protocol_name = _PROTOCOL_NAMES.get(config.protocol, config.protocol)
    if "invalid character '<' looking for beginning of value" in lowered or "<!doctype html" in lowered or "<html" in lowered:
        return (
            f"请求已到达网关，但网关收到的上游响应是 HTML 而不是 JSON。"
            f"通常是模型路由未启用 {protocol_name}，或该协议的上游地址配置异常；"
            "这不是本地 Base URL 拼装错误。"
        )
    if status_code in {404, 405}:
        return f"网关未开放 {protocol_name} 路径，或该路径不允许 POST。"
    return None


def check_connectivity(config: ApiConfig) -> dict[str, Any]:
    """Send one tiny request. HTTP 2xx + a valid JSON response means reachable.

    Visible content is deliberately not required: a reasoning model can consume the
    tiny output budget in its thinking phase while the endpoint itself is healthy.
    """
    started = time.perf_counter()
    body = connectivity_payload(config)
    request_headers = connectivity_headers(config)
    result: dict[str, Any] = {"name": "connectivity", "passed": False, "protocol": config.protocol,
                              "endpoint": connectivity_endpoint(config), "http_status": None,
                              "duration_ms": None, "error": None, "diagnosis": None,
                              "request": _request_shape(config, request_headers, body)}
    try:
        with httpx.Client(timeout=min(config.request_timeout_seconds, 60), verify=config.verify_ssl,
                          trust_env=config.trust_env) as client:
            response = client.post(result["endpoint"], headers=request_headers, json=body)
        result["http_status"] = response.status_code
        result["content_type"] = response.headers.get("content-type")
        result["provider_request_id"] = (response.headers.get("x-request-id")
                                         or response.headers.get("request-id")
                                         or response.headers.get("cf-ray"))
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
            result["diagnosis"] = _upstream_diagnosis(config, response.status_code, response.text)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["duration_ms"] = (time.perf_counter() - started) * 1000
    return result
