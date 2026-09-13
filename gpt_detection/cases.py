from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    case_id: str
    title: str
    description: str
    cost: str = "低"
    protocols: tuple[str, ...] = ("openai", "volcengine")
    request_count: int = 1


GPT_MODEL_MATCH = Case(
    case_id="GPT_MODEL_MATCH",
    title="GPT 模型身份一致性",
    description="比较请求中的 model 与响应顶层 model，检查网关是否实际使用了目标模型。",
)

CASES: dict[str, Case] = {GPT_MODEL_MATCH.case_id: GPT_MODEL_MATCH}
