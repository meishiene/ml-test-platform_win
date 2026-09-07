"""跨模型可复用的契约探针。

只复用测试方法；每个模型通过 Profile 提供自己的字段、数据和验收标准。
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any
from pathlib import Path

@dataclass(frozen=True)
class ModelProfile:
    family: str
    model: str
    thinking_on: dict[str, Any] | None
    thinking_off: dict[str, Any] | None
    expected_model_tokens: tuple[str, ...]
    answer_marker: str
    tool_name: str
    supports_reasoning: bool

def load_profile_data(family: str) -> dict[str, Any]:
    path = Path(__file__).parent / "data" / f"{family}.json"
    return json.loads(path.read_text(encoding="utf-8"))

PROFILES = {
    "kvv": ModelProfile("kvv", "", {"thinking": {"type": "enabled"}}, {"thinking": {"type": "disabled"}}, ("kimi", "k3"), "KVV-CONTRACT-OK", "lookup_order", True),
    "deepseek": ModelProfile("deepseek", "", {"thinking": {"type": "enabled"}}, {"thinking": {"type": "disabled"}}, ("deepseek-v4",), "DS-CONTRACT-OK", "lookup_order", True),
    "glm": ModelProfile("glm", "", {"thinking": {"type": "enabled"}}, None, ("glm",), "GLM-CONTRACT-OK", "lookup_order", True),
}

@dataclass(frozen=True)
class ProbeSpec:
    id: str
    name: str
    layer: str
    description: str

PROBES = (
    ProbeSpec("CONTRACT-01", "身份与路由", "identity", "模型返回标识、alias 和未知模型路由"),
    ProbeSpec("CONTRACT-02", "参数错误契约", "boundary", "非法 temperature、messages 类型应得到 4xx"),
    ProbeSpec("CONTRACT-03", "流式协议完整性", "protocol", "SSE 内容、结束标记和 usage"),
    ProbeSpec("CONTRACT-04", "工具调用多轮闭环", "protocol", "tool_choice、tool call 参数和 tool result"),
    ProbeSpec("CONTRACT-05", "Thinking 与 reasoning_effort", "capability", "固定开启 thinking，逐档验证模型支持的 reasoning_effort"),
)

def text(body: dict[str, Any] | None) -> str:
    msg = (((body or {}).get("choices") or [{}])[0].get("message") or {})
    value = msg.get("content", "")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

def message(body: dict[str, Any] | None) -> dict[str, Any]:
    return (((body or {}).get("choices") or [{}])[0].get("message") or {})

def json_object(value: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(value)
        return obj if isinstance(obj, dict) else None
    except (TypeError, ValueError):
        return None

def validate_identity(status: int, body: dict[str, Any] | None, expected: tuple[str, ...]) -> tuple[bool, dict[str, Any]]:
    returned = str((body or {}).get("model", ""))
    return status == 200 and bool(returned) and any(token.lower() in returned.lower() for token in expected), {"response_model": returned}

def validate_usage(body: dict[str, Any] | None) -> bool:
    usage = (body or {}).get("usage") or {}
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return all(isinstance(usage.get(k), (int, float)) and usage[k] >= 0 for k in keys) and usage.get("total_tokens") == usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
