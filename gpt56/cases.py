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
    "GPT56_P0": GPT56Case("GPT56_P0", "GPT-5.6 核心专项", "Responses API 推理全生命周期（默认档位/none/档位/摘要/context）、工具闭环、Structured Outputs、previous_response_id 与加密推理回放、隐式缓存、双协议流式与截断可检测；三档模型可调用、别名路由、图像识别与 detail 档位、后台模式、响应对象结构契约", "中～高（图像与后台轮询用例偏高）", ("P0",), 22),
    "GPT56_P1": GPT56Case("GPT56_P1", "GPT-5.6 增强专项", "pro 模式、reasoning.context=all_turns 持久化、Programmatic Tool Calling、显式缓存选项与 TTL、非法 json_schema 拒绝、未知模型拒绝；显式缓存断点两跳（增量判据）、multi-agent beta", "中～高（pro 模式按聚合工作计费、multi-agent 多 subagent 并发）", ("P1",), 9),
    "GPT56_P2": GPT56Case("GPT56_P2", "GPT-5.6 契约与健壮性", "双协议 Usage 契约、流式 usage chunk、非法 temperature/messages/input 拒绝、Tool Choice none/required、Stop 序列、max_completion_tokens 边界；无效 Key 鉴权、错误对象四字段、/models 与 token 计数端点、流式语义事件与无 [DONE]", "中", ("P2",), 15),
    "GPT56_FINGERPRINT": GPT56Case("GPT56_FINGERPRINT", "GPT-5.6 满血真实性指纹", "随机挑战动态工具、结构化输出绑定、多轮记忆、思考摘要、流式完整性与 effort 吞参对照，识别疑似降级或中转特征；中转注入识别（instructions/非标准字段）、档位能力梯度、response id 真实性、Chat 协议未被改写", "中～高（档位梯度双档对照）", ("AUTH",), 11),
}
