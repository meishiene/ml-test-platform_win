from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse


@dataclass(frozen=True)
class Protocol:
    id: str
    provider: str
    name: str
    default_base_url: str
    path_template: str
    description: str
    model_hint: str
    family: str
    test_kinds: tuple[str, ...] = ()


PROTOCOLS = {
    "openai": Protocol(
        "openai", "OpenAI", "Chat Completions", "https://api.openai.com", "/v1/chat/completions",
        "文字、视觉输入、工具调用与流式输出", "例如 gpt-5-mini", "openai",
        ("load", "kvv", "deepseek", "glm", "gpt", "authenticity", "moderation", "temporary",
         "arxivmath", "deepswe", "swe-atlas-tw", "swe-atlas-rf", "legal-agent-benchmark"),
    ),
    "openai_responses": Protocol(
        "openai_responses", "OpenAI", "Responses", "https://api.openai.com", "/v1/responses",
        "统一文本/多模态输入、内置工具与 typed SSE", "例如 gpt-5-mini", "openai",
        ("load",),
    ),
    "openai_images": Protocol(
        "openai_images", "OpenAI", "Images · Generations", "https://api.openai.com", "/v1/images/generations",
        "图片生成接口（GPT Image）", "例如 gpt-image-2", "openai",
        ("gpt-image",),
    ),
    "anthropic": Protocol(
        "anthropic", "Claude", "Messages", "https://api.anthropic.com", "/v1/messages",
        "文字、视觉输入、Thinking、工具调用与 SSE", "例如 claude-sonnet-4-5", "anthropic",
        ("load", "awsb"),
    ),
    "gemini": Protocol(
        "gemini", "Gemini", "streamGenerateContent", "https://generativelanguage.googleapis.com",
        "/v1beta/models/{model}:streamGenerateContent?alt=sse",
        "文字、视觉/文件输入及支持模型的图像输出", "例如 gemini-2.5-flash", "gemini",
        ("load",),
    ),
    "volcengine": Protocol(
        "volcengine", "火山云", "Chat Completions", "https://ark.cn-beijing.volces.com",
        "/api/v3/chat/completions", "火山方舟 OpenAI 兼容接口", "填写推理接入点或模型 ID", "volcengine",
    ),
}


def protocol_spec(protocol: str) -> Protocol:
    try:
        return PROTOCOLS[protocol]
    except KeyError as exc:
        raise ValueError(f"不支持的协议: {protocol}") from exc


def normalize_base_url(base_url: str) -> str:
    """Validate and normalize the user-facing, host-only base URL."""
    value = base_url.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url 必须是有效的 HTTP(S) 地址")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("base_url 只能填写域名（可含端口），不要带 / 或路径")
    if parsed.username or parsed.password:
        raise ValueError("base_url 不能包含用户名或密码")
    return f"{parsed.scheme}://{parsed.netloc}"


def api_root(protocol: str, base_url: str) -> str:
    """Return the SDK-style API root while accepting legacy pathful CLI values."""
    base = base_url.rstrip("/")
    if urlparse(base).path not in {"", "/"}:
        return base
    family = protocol_spec(protocol).family
    suffix = "/api/v3" if family == "volcengine" else "/v1beta" if family == "gemini" else "/v1"
    return base if base.endswith(suffix) else f"{base}{suffix}"


def display_endpoint(protocol: str, base_url: str, model: str = "{model}") -> str:
    spec = protocol_spec(protocol)
    return f"{base_url.rstrip('/')}{spec.path_template.format(model=quote(model, safe=''))}"


def supports_test_kind(protocol: str, test_kind: str, accepted_families: tuple[str, ...] | list[str]) -> bool:
    spec = protocol_spec(protocol)
    return spec.family in accepted_families and (not spec.test_kinds or test_kind in spec.test_kinds)


def endpoint(protocol: str, base_url: str, model: str, chat_path: str = "/chat/completions") -> str:
    root = api_root(protocol, base_url)
    if protocol == "openai_responses":
        return f"{root}/responses"
    if protocol in {"openai", "volcengine"}:
        return f"{root}/{chat_path.lstrip('/')}"
    if protocol == "anthropic":
        return f"{root}/messages"
    if protocol == "gemini":
        return f"{root}/models/{quote(model, safe='')}:streamGenerateContent?alt=sse"
    if protocol == "openai_images":
        return f"{root}/images/generations"
    raise ValueError(f"协议 {protocol} 不支持文字流式请求")


def headers(protocol: str, api_key: str, request_id: str) -> dict[str, str]:
    accept = "application/json" if protocol == "openai_images" else "text/event-stream"
    common = {"Content-Type": "application/json", "Accept": accept, "X-Request-ID": request_id}
    family = protocol_spec(protocol).family
    if family in {"openai", "volcengine"}:
        common["Authorization"] = f"Bearer {api_key}"
    elif family == "anthropic":
        common.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    elif family == "gemini":
        common["x-goog-api-key"] = api_key
    return common


def payload(protocol: str, model: str, messages: list[dict[str, str]] | str, max_tokens: int,
            include_usage: bool, reasoning_effort: str | None = None) -> dict[str, Any]:
    system = "You are a performance-test assistant. Follow the final user task exactly; do not call tools."
    normalized = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
    if protocol == "openai_responses":
        body = {"model": model, "stream": True, "max_output_tokens": max_tokens,
                "instructions": system, "input": normalized}
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        return body
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
    root = api_root(protocol, base_url)
    if protocol_spec(protocol).family in {"openai", "volcengine", "anthropic"}:
        return f"{root}/models"
    if protocol_spec(protocol).family == "gemini":
        return f"{root}/models?pageSize=1000"
    raise ValueError(f"不支持的协议: {protocol}")


def models_headers(protocol: str, api_key: str) -> dict[str, str]:
    result = {"Accept": "application/json"}
    family = protocol_spec(protocol).family
    if family in {"openai", "volcengine"}:
        result["Authorization"] = f"Bearer {api_key}"
    elif family == "anthropic":
        result.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    elif family == "gemini":
        result["x-goog-api-key"] = api_key
    return result


def parse_models(protocol: str, obj: dict[str, Any]) -> list[dict[str, str]]:
    family = protocol_spec(protocol).family
    raw_models = obj.get("models") if family == "gemini" else obj.get("data")
    raw_models = raw_models or []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("name") if family == "gemini" else item.get("id")
        if not raw_id:
            continue
        model_id = str(raw_id)
        if family == "gemini" and model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        if family == "gemini":
            methods = item.get("supportedGenerationMethods") or []
            if methods and not any(method in methods for method in ("generateContent", "streamGenerateContent")):
                continue
        if model_id in seen:
            continue
        seen.add(model_id)
        result.append({"id": model_id, "name": str(item.get("displayName") or model_id)})
    return sorted(result, key=lambda model: model["id"].lower())
