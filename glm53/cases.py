from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GLMCase:
    id: str
    title: str
    description: str
    cost: str
    priorities: tuple[str, ...]
    request_count: int
    concurrency: int = 1
    protocols: tuple[str, ...] = ("openai", "volcengine")

CASES = {
    "GLM_53_CORE": GLMCase("GLM_53_CORE", "GLM 5.3 核心能力", "基础对话、流式输出（必须开启深度思考）、结构化输出、长上下文与 Usage", "中", ("P0",), 10),
    "GLM_53_AGENT": GLMCase("GLM_53_AGENT", "GLM 5.3 Agent 能力", "工具调用、tool_choice、工具流式输出、多轮工具链与 Schema", "中", ("P1",), 8),
    "GLM_53_CACHE": GLMCase("GLM_53_CACHE", "GLM 5.3 缓存与 Session 专项", "数学、代码、中文业务三组固定前缀；每组种子请求后执行两次命中探测，并验证同身份多轮 Session 连续性", "中", ("PCACHE",), 9, concurrency=1),
    "GLM_53_FINGERPRINT": GLMCase("GLM_53_FINGERPRINT", "GLM 5.3 满血能力指纹", "通过官方能力字段、错误码、静默忽略和响应身份识别疑似降级或兼容层限制", "中", ("P2",), 8),
}
