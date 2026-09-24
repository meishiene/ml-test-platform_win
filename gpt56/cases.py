from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GPT56Case:
    id: str
    title: str
    description: str
    cost: str
    priorities: tuple[str, ...]
    request_count: int
    concurrency: int = 1
    protocols: tuple[str, ...] = ("openai",)


CASES = {
    "GPT56_P0": GPT56Case("GPT56_P0", "GPT-5.6 核心专项", "Responses API 推理全生命周期（默认档位/none/档位/摘要/context）、工具闭环、Structured Outputs、previous_response_id 与加密推理回放、隐式缓存、双协议流式与截断可检测", "中", ("P0",), 16),
    "GPT56_P1": GPT56Case("GPT56_P1", "GPT-5.6 增强专项", "pro 模式、reasoning.context=all_turns 持久化、Programmatic Tool Calling、显式缓存选项与 TTL、非法 json_schema 拒绝、未知模型拒绝", "中～高（pro 模式按聚合工作计费）", ("P1",), 7),
    "GPT56_P2": GPT56Case("GPT56_P2", "GPT-5.6 契约与健壮性", "双协议 Usage 契约、流式 usage chunk、非法 temperature/messages/input 拒绝、Tool Choice none/required、Stop 序列、max_completion_tokens 边界", "中", ("P2",), 10),
    "GPT56_FINGERPRINT": GPT56Case("GPT56_FINGERPRINT", "GPT-5.6 满血真实性指纹", "随机挑战动态工具、结构化输出绑定、多轮记忆、思考摘要、流式完整性与 effort 吞参对照，识别疑似降级或中转特征", "中～高", ("AUTH",), 6),
}
