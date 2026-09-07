"""页面 KVV 选项到官方 Kimi Vendor Verifier pytest 套件的映射。"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class KvvCase:
    id: str
    title: str
    description: str
    cost: str
    request_count: int
    concurrency: int
    suites: tuple[str, ...]
    protocols: tuple[str, ...] = ("openai", "volcengine")


CASES = {
    "KVV_K3_FEATURES": KvvCase(
        "KVV_K3_FEATURES", "官方 K3 Features（核心）", "执行 response_format、tokenization、tool_choice；默认排除 K3 专属动态工具与 Thinking Effort", "高", 46, 1,
        ("k3_features_core",),
    ),
    "KVV_PARAMS": KvvCase(
        "KVV_PARAMS", "官方参数约束", "直接执行官方 tests/params 全部 18 个用例", "中", 18, 1,
        ("params",),
    ),
    "KVV_TOOL_SCHEMA": KvvCase(
        "KVV_TOOL_SCHEMA", "官方 Tool JSON Schema", "直接执行官方 tests/tool_call_json_schema 全部 408 个流式/非流式用例", "极高", 408, 4,
        ("tool_call_json_schema",),
    ),
}
