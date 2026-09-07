from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote


@dataclass(frozen=True)
class Protocol:
    id: str
    name: str
    default_base_url: str
    description: str
    model_hint: str


PROTOCOLS = {
    "openai": Protocol("openai", "OpenAI", "https://api.openai.com/v1", "OpenAI Chat Completions 兼容协议", "例如 gpt-5-mini"),
    "anthropic": Protocol("anthropic", "Anthropic", "https://api.anthropic.com/v1", "Anthropic Messages SSE 协议", "例如 claude-sonnet-4-5"),
    "gemini": Protocol("gemini", "Gemini", "https://generativelanguage.googleapis.com/v1beta", "Google Gemini streamGenerateContent 协议", "例如 gemini-2.5-flash"),
    "volcengine": Protocol("volcengine", "火山云", "https://ark.cn-beijing.volces.com/api/v3", "火山方舟 OpenAI 兼容协议", "填写推理接入点或模型 ID"),
}


def endpoint(protocol: str, base_url: str, model: str, chat_path: str = "/chat/completions") -> str:
    base = base_url.rstrip("/")
    if protocol in {"openai", "volcengine"}:
        return f"{base}/{chat_path.lstrip('/')}"
    if protocol == "anthropic":
        return f"{base}/messages"
    if protocol == "gemini":
        return f"{base}/models/{quote(model, safe='')}:streamGenerateContent?alt=sse"
    raise ValueError(f"不支持的协议: {protocol}")


def headers(protocol: str, api_key: str, request_id: str) -> dict[str, str]:
    common = {"Content-Type": "application/json", "Accept": "text/event-stream", "X-Request-ID": request_id}
    if protocol in {"openai", "volcengine"}:
        common["Authorization"] = f"Bearer {api_key}"
    elif protocol == "anthropic":
        common.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    elif protocol == "gemini":
        common["x-goog-api-key"] = api_key
    return common


def payload(protocol: str, model: str, messages: list[dict[str, str]] | str, max_tokens: int,
            include_usage: bool, reasoning_effort: str | None = None) -> dict[str, Any]:
    system = "You are a performance-test assistant. Follow the final user task exactly; do not call tools."
    normalized = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
    if protocol in {"openai", "volcengine"}:
        body: dict[str, Any] = {"model": model, "stream": True, "max_tokens": max_tokens,
                                "messages": [{"role": "system", "content": system}, *normalized]}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if include_usage:
            body["stream_options"] = {"include_usage": True}
        return body
    if protocol == "anthropic":
        return {"model": model, "max_tokens": max_tokens, "stream": True, "system": system,
                "messages": normalized}
    if protocol == "gemini":
        role_map = {"assistant": "model", "user": "user"}
        return {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": role_map.get(message["role"], "user"), "parts": [{"text": message["content"]}]} for message in normalized],
                "generationConfig": {"maxOutputTokens": max_tokens}}
    raise ValueError(f"不支持的协议: {protocol}")


def models_endpoint(protocol: str, base_url: str) -> str:
    base = base_url.rstrip("/")
    if protocol in {"openai", "volcengine", "anthropic"}:
        return f"{base}/models"
    if protocol == "gemini":
        return f"{base}/models?pageSize=1000"
    raise ValueError(f"不支持的协议: {protocol}")


def models_headers(protocol: str, api_key: str) -> dict[str, str]:
    result = {"Accept": "application/json"}
    if protocol in {"openai", "volcengine"}:
        result["Authorization"] = f"Bearer {api_key}"
    elif protocol == "anthropic":
        result.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    elif protocol == "gemini":
        result["x-goog-api-key"] = api_key
    return result


def parse_models(protocol: str, obj: dict[str, Any]) -> list[dict[str, str]]:
    raw_models = obj.get("models") if protocol == "gemini" else obj.get("data")
    raw_models = raw_models or []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("name") if protocol == "gemini" else item.get("id")
        if not raw_id:
            continue
        model_id = str(raw_id)
        if protocol == "gemini" and model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        if protocol == "gemini":
            methods = item.get("supportedGenerationMethods") or []
            if methods and not any(method in methods for method in ("generateContent", "streamGenerateContent")):
                continue
        if model_id in seen:
            continue
        seen.add(model_id)
        result.append({"id": model_id, "name": str(item.get("displayName") or model_id)})
    return sorted(result, key=lambda model: model["id"].lower())
