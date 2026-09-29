"""GPT-5.6 专项兼容性测试。

测试依据 OpenAI GPT-5.6 官方 API 文档（《Model guidance》《Reasoning models》《Text
generation》），使用原始 HTTP 请求，避免 SDK 对扩展字段的过滤。

能力面依据（2026-09 文档检索）：
- 模型：gpt-5.6-sol / gpt-5.6-terra / gpt-5.6-luna；`gpt-5.6` 别名默认路由到 sol。
- 主推 Responses API（/v1/responses），Chat Completions（/v1/chat/completions）兼容保留。
- reasoning.effort 六档 none/low/medium/high/xhigh/max，缺省 medium；reasoning.mode=pro；
  reasoning.context（auto/all_turns/current_turn）；reasoning.summary。
- previous_response_id 状态续接；store=false 时以 encrypted_content 回放加密推理。
- 隐式缓存（cached_tokens / cache_write_tokens）；prompt_cache_options.mode/ttl。
- Programmatic Tool Calling（PTC）；Structured Outputs（text.format json_schema strict）。

与 dsv4 不同：GPT-5.6 无 beta / anthropic 特殊端点，所有请求都挂在 openai 家族根
（{base}/v1）下，不存在"剥版本后缀"的地址派生问题。

判定口径补充（2026-09-24）：上游 5xx / 连接错误属瞬时故障，先按 GPT56Config.max_retries
退避重试，重试耗尽仍失败则该用例记 `blocked`（平台级前提不满足），不记 `failed`。
4xx 与结构化 `model_not_found` 是确定性结论，既不重试也不降级。
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from common.png_fixture import png_data_url
from common.test_framework import HttpExecutor, ResultLog

MODELS = ("gpt-5.6",)

# 模型身份令牌白名单：别名 gpt-5.6 官方承诺路由到 gpt-5.6-sol，
# 因此响应 model 落在任一 GPT-5.6 令牌上都视为身份一致。
MODEL_TOKENS = ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")

REASONING_CONTEXT_MODES = {"auto", "all_turns", "current_turn"}


@dataclass(frozen=True)
class GPT56Config:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 300.0
    verify_ssl: bool = True
    trust_env: bool = False
    # 瞬时故障（上游 5xx / 连接错误）的重试次数与退避基数：网关抖动不该被记成模型失败。
    # 重试耗尽后仍失败 → 用例按 blocked 收口（见 _transient_reason）。
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0


@dataclass
class ApiResponse:
    status: int
    body: dict[str, Any] | None
    text: str
    elapsed: float
    headers: dict[str, str] = field(default_factory=dict)
    upstream_request_id: str | None = None
    stream: dict[str, Any] | None = None


@dataclass
class CaseResult:
    case_id: str
    name: str
    model: str
    passed: bool
    status: str = "passed"
    details: str = ""
    duration_seconds: float = 0.0
    observations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    name: str
    priority: str
    handler: str


CASE_SPECS = (
    # ---- P0 核心专项（16 条）----
    CaseSpec("GPT56-01", "模型别名可调用", "P0", "basic"),
    CaseSpec("GPT56-02", "默认思考档位", "P0", "default_effort"),
    CaseSpec("GPT56-03", "effort=none 关闭思考", "P0", "effort_none"),
    CaseSpec("GPT56-04", "思考强度档位接受", "P0", "effort_levels"),
    CaseSpec("GPT56-05", "reasoning summary 摘要", "P0", "reasoning_summary"),
    CaseSpec("GPT56-06", "reasoning.context 回显", "P0", "reasoning_context"),
    CaseSpec("GPT56-07", "工具调用闭环（Responses）", "P0", "tool_loop"),
    CaseSpec("GPT56-08", "Structured Outputs", "P0", "json_schema_output"),
    CaseSpec("GPT56-09", "Chat Completions 兼容", "P0", "chat_compat"),
    CaseSpec("GPT56-10", "previous_response_id 多轮状态", "P0", "previous_response_state"),
    CaseSpec("GPT56-11", "store=false 加密推理回放", "P0", "encrypted_replay"),
    CaseSpec("GPT56-12", "隐式缓存命中", "P0", "cache_hit"),
    CaseSpec("GPT56-13", "小输出截断可检测", "P0", "truncation"),
    CaseSpec("GPT56-14", "Responses 流式", "P0", "responses_stream"),
    CaseSpec("GPT56-15", "Chat 流式", "P0", "chat_stream"),
    CaseSpec("GPT56-16", "instructions 系统指令生效", "P0", "instructions"),
    CaseSpec("GPT56-34", "三档模型均可调用", "P0", "tier_models"),
    CaseSpec("GPT56-35", "别名路由到旗舰档", "P0", "alias_flagship"),
    CaseSpec("GPT56-36", "图像识别（内联 base64）", "P0", "image_recognition"),
    CaseSpec("GPT56-37", "图像 detail 档位", "P0", "image_detail_tiers"),
    CaseSpec("GPT56-40", "后台模式（background）", "P0", "background_mode"),
    CaseSpec("GPT56-41", "响应对象结构契约", "P0", "response_object_contract"),
    # ---- P1 增强专项（7 条）----
    CaseSpec("GPT56-17", "pro 模式可用", "P1", "pro_mode"),
    CaseSpec("GPT56-18", "reasoning.context=all_turns 持久化", "P1", "context_all_turns"),
    CaseSpec("GPT56-19", "Programmatic Tool Calling 注册", "P1", "ptc"),
    CaseSpec("GPT56-20", "显式缓存选项", "P1", "explicit_cache"),
    CaseSpec("GPT56-21", "缓存 TTL 选项", "P1", "cache_ttl"),
    CaseSpec("GPT56-22", "非法 json_schema 拒绝", "P1", "strict_invalid_schema"),
    CaseSpec("GPT56-23", "未知模型拒绝", "P1", "unknown_model"),
    CaseSpec("GPT56-49", "显式缓存断点（增量判据）", "P1", "explicit_cache_breakpoint"),
    CaseSpec("GPT56-53", "multi-agent beta", "P1", "multi_agent"),
    # ---- P2 契约与健壮性（10 条）----
    CaseSpec("GPT56-24", "Responses Usage 完整性", "P2", "responses_usage"),
    CaseSpec("GPT56-25", "Chat Usage 完整性", "P2", "chat_usage"),
    CaseSpec("GPT56-26", "Chat 流式 Usage chunk", "P2", "chat_stream_usage"),
    CaseSpec("GPT56-27", "非法 temperature 拒绝", "P2", "temperature_invalid"),
    CaseSpec("GPT56-28", "非法 Messages 拒绝", "P2", "messages_invalid"),
    CaseSpec("GPT56-29", "Tool Choice none", "P2", "tool_choice_none"),
    CaseSpec("GPT56-30", "Tool Choice required", "P2", "tool_choice_required"),
    CaseSpec("GPT56-31", "Stop 停止序列", "P2", "stop_sequence"),
    CaseSpec("GPT56-32", "max_completion_tokens 边界接受", "P2", "max_completion_tokens"),
    CaseSpec("GPT56-33", "非法 input 拒绝", "P2", "invalid_input"),
    CaseSpec("GPT56-54", "无效 API Key 鉴权契约", "P2", "invalid_key_auth"),
    CaseSpec("GPT56-55", "错误对象结构契约", "P2", "error_object_contract"),
    CaseSpec("GPT56-56", "模型列表端点与三档一致性", "P2", "models_list"),
    CaseSpec("GPT56-57", "输入 token 计数端点", "P2", "input_tokens_endpoint"),
    CaseSpec("GPT56-61", "流式语义事件与无 DONE", "P2", "stream_semantic_events"),
    # ---- FINGERPRINT 满血真实性指纹（6 条）----
    CaseSpec("AUTH-G01", "随机动态工具诱导调用", "AUTH", "auth_random_tool"),
    CaseSpec("AUTH-G02", "随机 Structured Output 绑定", "AUTH", "auth_random_json"),
    CaseSpec("AUTH-G03", "随机多轮记忆", "AUTH", "auth_random_memory"),
    CaseSpec("AUTH-G04", "思考真实性（summary）", "AUTH", "auth_reasoning_summary"),
    CaseSpec("AUTH-G05", "流式事件完整性", "AUTH", "auth_stream_events"),
    CaseSpec("AUTH-G06", "effort 静默吞参对照", "AUTH", "auth_effort_contrast"),
    CaseSpec("AUTH-G07", "中转注入识别（instructions 回显）", "AUTH", "auth_injection"),
    CaseSpec("AUTH-G08", "非标准字段检测", "AUTH", "auth_nonstandard_fields"),
    CaseSpec("AUTH-G10", "档位能力梯度", "AUTH", "auth_tier_gradient"),
    CaseSpec("AUTH-G11", "previous_response_id 真实性", "AUTH", "auth_response_id_authenticity"),
    CaseSpec("AUTH-G12", "Chat 协议未被改写", "AUTH", "auth_chat_not_translated"),
)


def _rng_token(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8].upper()}"


def _response(body: dict[str, Any] | None) -> dict[str, Any]:
    """Responses API 的 response 对象（非流式 body 本身就是 response）。"""
    return body or {}


def _output_items(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    return (body or {}).get("output") or []


def _output_text(body: dict[str, Any] | None) -> str:
    """拼接 Responses 输出里的 message 文本。"""
    parts: list[str] = []
    for item in _output_items(body):
        if item.get("type") == "message":
            for block in item.get("content") or []:
                if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                    parts.append(str(block.get("text") or ""))
    return "".join(parts)


def _reasoning_items(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [i for i in _output_items(body) if i.get("type") == "reasoning"]


def _reasoning_object(body: dict[str, Any] | None) -> dict[str, Any]:
    obj = (body or {}).get("reasoning")
    return obj if isinstance(obj, dict) else {}


def _usage(body: dict[str, Any] | None) -> dict[str, Any]:
    return (body or {}).get("usage") or {}


def _reasoning_tokens(body: dict[str, Any] | None) -> int | None:
    details = _usage(body).get("output_tokens_details") or {}
    value = details.get("reasoning_tokens")
    return int(value) if isinstance(value, (int, float)) else None


def _cached_input_tokens(body: dict[str, Any] | None) -> int:
    usage = _usage(body)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    value = details.get("cached_tokens")
    return int(value) if isinstance(value, (int, float)) else 0


def _cache_write_tokens(body: dict[str, Any] | None) -> int:
    """缓存写入 token（显式缓存计价项，作为缓存能力的旁证）。"""
    usage = _usage(body)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    value = details.get("cache_write_tokens")
    return int(value) if isinstance(value, (int, float)) else 0


def _chat_message(body: dict[str, Any] | None) -> dict[str, Any]:
    choices = (body or {}).get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def _is_rejection(status: int) -> bool:
    """4xx 且非 404 —— 404 表示端点不存在（链路/配置问题），不构成"参数被正确拒绝"。"""
    return 400 <= status < 500 and status != 404


def _endpoint_missing_reason(status: int, endpoint: str) -> str | None:
    if status != 404:
        return None
    return f"{endpoint} 端点不存在（HTTP 404），用例目标特性未被触达；属链路/配置问题（检查 base_url），不是被测对象的行为结论"


def _error_body(body: dict[str, Any] | None) -> dict[str, Any] | None:
    """取结构化 error 对象；响应体可能是非 dict（JSON 字符串、列表等），须防御。"""
    err = (body if isinstance(body, dict) else {}).get("error")
    return err if isinstance(err, dict) else None


def _error_text(body: dict[str, Any] | None) -> str:
    """错误文案（结构化 error 优先，兼容纯文本错误体），统一小写便于匹配。"""
    err = _error_body(body)
    raw = json.dumps(err, ensure_ascii=False) if err else json.dumps(body or "", ensure_ascii=False)
    return raw.lower()


# 账号层限制文案：中转/账号通道不支持某能力时的显式拒绝（真实批次证据：
# mux.la 中转对 previous_response_id 返回 400 "previous_response_id requires
# an OpenAI API-key account for HTTP requests"）。命中即记为"平台级前提不满足"。
_ACCOUNT_LIMITATION_MARKERS = (
    "requires an openai api-key account",
    "requires an api key account",
    "requires an api-key account",
    "api-key account for http requests",
)


def _account_limitation_reason(status: int, body: dict[str, Any] | None, feature: str) -> str | None:
    """上游账号层明确拒绝该能力时给出 blocked 理由（前提不满足，无法观测）。"""
    if status == 200 or status == 404:
        return None
    text = _error_text(body)
    if not any(marker in text for marker in _ACCOUNT_LIMITATION_MARKERS):
        return None
    return (f"{feature}被上游账号层拒绝（HTTP {status}）：当前账号/通道类型不支持该能力，"
            f"属平台级前提不满足，用例目标特性未被触达（原文：{(_error_body(body) or {}).get('message') or text[:160]}）")


def _model_not_found_reason(requested_model: str, status: int, body: dict[str, Any] | None) -> str | None:
    """结构化"模型不存在"错误——含网关用非 4xx 状态码承载该语义的情况。

    必须同时满足：① 响应体是结构化 error；② 错误码/文案明确指向"模型不存在"，
    且指向的是被请求的那个模型（避免把"渠道整体不可用"误读成"模型校验存在"）。
    """
    err = _error_body(body)
    if not err:
        return None
    code = str(err.get("code") or "").strip().lower()
    message = str(err.get("message") or "").strip().lower()
    if "model_not_found" not in code and "model not found" not in message and "no available channel for model" not in message:
        return None
    if requested_model.lower() not in message and "model_not_found" not in code:
        return None
    if 400 <= status < 500:
        return f"未知模型被拒绝（HTTP {status}，code={code or 'n/a'}）且返回结构化 error，模型校验契约成立"
    return f"未知模型被拒绝（HTTP {status}，code={code or 'n/a'}）：网关用非 4xx 状态码承载了「模型不存在」的语义，模型校验契约成立"


# ---------------------------------------------------------------------------
# 瞬时故障（上游 5xx / 连接错误）
#
# 真实批次证据（2026-09-24，results/GPT56_P0-20260924-153058/requests.jsonl）：
# mux.la 中转对 GPT56-12 / GPT56-16 返回
#   HTTP 503 {"code":"system_memory_overloaded",
#             "message":"system memory overloaded (current: 91.5%, threshold: 90%)"}
# 且 GPT56-12 第二跳叠加 ConnectError。这类故障不由被测模型决定，请求也未产生有效响应，
# 因此：① 先做有限重试（5xx / 连接错误）；② 重试耗尽后用例按 blocked 收口，不记 failed。
# 注意与"上游明确拒绝某参数"（4xx）区分：后者是确定性判定，既不重试也不降级。
# ---------------------------------------------------------------------------
_TRANSIENT_STATUS_MIN = 500


def _is_transient_status(status: int | None) -> bool:
    return isinstance(status, int) and _TRANSIENT_STATUS_MIN <= status < 600


def _is_definitive_error(body: dict[str, Any] | None) -> bool:
    """结构化且明确指向"模型不存在"的错误是确定性结论，不是网关抖动，不重试。

    真实批次证据：New API 用 HTTP 503 + code=model_not_found 承载"未知模型"（GPT56-23），
    该状态码虽在 5xx 区间，语义上却是确定性回答，重试只会浪费配额。
    """
    err = _error_body(body)
    if not err:
        return False
    return str(err.get("code") or "").strip().lower() == "model_not_found"


def _is_transient_record(record: Any) -> bool:
    """传输层失败（连接错误/超时）或 5xx 视为瞬时故障；确定性错误与 4xx 不算。"""
    if record.transport_status != "completed":
        return True
    if _is_definitive_error(record.response_body):
        return False
    return _is_transient_status(record.http_status)


def _transient_event(record: Any, path: str, attempts: int) -> dict[str, Any]:
    error = record.transport_error or {}
    message = (_error_body(record.response_body) or {}).get("message") or str(record.response_text or "")[:160]
    return {"path": path, "attempts": attempts, "http_status": record.http_status,
            "transport_error": error.get("type"), "message": str(message)[:200]}


def _transient_reason(event: dict[str, Any], total: int) -> str:
    """瞬时故障的 blocked 理由，需自解释（谁报的错、重试了几次、为何不算模型失败）。"""
    retried = f"已重试 {event['attempts'] - 1} 次仍失败" if event["attempts"] > 1 else "未启用重试"
    more = f"；本用例累计 {total} 次瞬时故障" if total > 1 else ""
    if event.get("transport_error"):
        cause = f"链路错误 {event['transport_error']}（请求未完成）"
    else:
        cause = f"HTTP {event['http_status']}（原文：{event.get('message')}）"
    return (f"上游瞬时故障：{cause}，{retried}{more}。该故障不由被测模型决定、请求未产生有效响应，"
            f"用例目标特性未被触达，属平台级前提不满足（基础设施抖动），按 blocked 收口，建议重跑本批次")


# ---------------------------------------------------------------------------
# AUTH-G07/G08（中转注入 / 非标准字段）与 GPT56-34/35（档位）的判据
# ---------------------------------------------------------------------------
# Responses API 响应对象的字段白名单（官方 schema 超集；phase / safety_identifier
# 属合法字段）。凡出现在白名单外即视为中转/多供应商网关特征（AUTH-G08）。
_RESPONSE_SCHEMA_FIELDS = frozenset({
    "id", "object", "created_at", "status", "error", "incomplete_details",
    "instructions", "metadata", "model", "output", "parallel_tool_calls",
    "temperature", "tool_choice", "tools", "top_p", "background",
    "reasoning", "service_tier", "store", "truncation", "usage", "text",
    "safety_identifier", "previous_response_id", "conversation", "prompt",
    "prompt_cache_key", "prompt_cache_options", "include", "max_output_tokens",
    "max_tool_calls", "metadata", "stream", "sequence_number", "phase",
})

# 真实批次（2026-09-24 mux.la 中转）实测出现的非标准字段，作 AUTH-G08 的校准依据。
_RELAY_FIELD_MARKERS = ("billing_usage", "access_programs", "claude_cache_creation", "new_api_error")

# 别名 gpt-5.6 官方承诺路由到旗舰档（GPT56-35 的判据）。
FLAGSHIP_TIER = "gpt-5.6-sol"
ALL_TIERS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")

# AUTH-G07：短问句（输入体量 < 100 token）的 input_tokens 膨胀阈值。
# 真实批次证据：注入 Codex 系统提示词后简单问句 input_tokens 达 4400+。
_INPUT_TOKEN_INFLATION_THRESHOLD = 2000

# AUTH-G11：官方 response id 形如 resp_ + 长 hex（真实批次 chatcmpl- 前缀即协议混用）。
_RESPONSE_ID_PATTERN = re.compile(r"^resp_[0-9a-f]{20,}$")

# Responses 流式语义事件（GPT56-61）：规范内无 [DONE]，靠 response.completed 收尾。
_RESPONSES_STREAM_CORE_EVENTS = ("response.created", "response.output_text.delta", "response.completed")

# AUTH-G12：Chat 响应出现任一 Responses 特征即判"协议被改写"（GPT56-31 的 stop 丢参即此因）。
_CHAT_TRANSLATION_MARKERS = ("oai_responses",)


def _injected_instructions_reason(body: dict[str, Any] | None) -> str | None:
    """AUTH-G07：未传 instructions 却被上游替换/填充即注入（信任级失败）。"""
    if "instructions" not in (body or {}):
        return None
    value = (body or {}).get("instructions")
    if value is None:
        return None
    text = str(value)
    return (f"响应 instructions 被上游填充（长度 {len(text)}，开头：{text[:80]!r}）——"
            "未传 instructions 的请求收到注入内容，本次响应内容不由请求单独决定，属信任级失败")


def _relay_fields(body: dict[str, Any] | None) -> list[str]:
    """AUTH-G08：扫描响应顶层非标准字段。"""
    extra = [k for k in (body or {}) if k not in _RESPONSE_SCHEMA_FIELDS]
    marked = [k for k in extra if any(m in k for m in _RELAY_FIELD_MARKERS)]
    return sorted(set(marked) | {k for k in extra if k.startswith("claude_cache_creation")}) or sorted(extra)


def _chat_translation_evidence(body: dict[str, Any] | None) -> str | None:
    """AUTH-G12：Chat 响应出现 Responses 特征时给出证据描述。"""
    b = body or {}
    usage = b.get("usage") or {}
    source = ((usage.get("billing_usage") or {}).get("source")) if isinstance(usage, dict) else None
    if isinstance(source, str) and any(m in source for m in _CHAT_TRANSLATION_MARKERS):
        return f"usage.billing_usage.source={source!r}（Chat 被翻译成 Responses）"
    if b.get("object") == "response":
        return "object=='response'（Responses 特征出现在 Chat 端点）"
    if str(b.get("id") or "").startswith("resp_"):
        return f"id 以 resp_ 开头（{b.get('id')!r}）"
    return None



class GPT56Runner:
    def __init__(self, config: GPT56Config, client: httpx.AsyncClient | None = None, audit_dir: Path | None = None):
        self.config = config
        self._external_client = client
        self.client = client
        self.executor: HttpExecutor | None = None
        # 执行证据：每个请求的完整 record 实时落到 <audit_dir>/requests.jsonl，
        # 供统一准入校验的 evidence_present 使用。
        self.audit_dir = audit_dir
        self.log: ResultLog | None = None
        self.current_case_id: str | None = None
        # 本用例内"重试耗尽仍失败"的瞬时故障记录（5xx / 连接错误），供 run() 改判 blocked。
        self.transient_events: list[dict[str, Any]] = []

    async def __aenter__(self) -> "GPT56Runner":
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                trust_env=self.config.trust_env,
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            )
        self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
        await self.executor.__aenter__()
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            self.log = ResultLog(self.audit_dir / "requests.jsonl")
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.log:
            self.log.close()
            self.log = None
        if self.executor:
            await self.executor.__aexit__(*args)
        if self._external_client is None and self.client:
            await self.client.aclose()

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    async def request(self, path: str, payload: dict[str, Any], *, stream: bool = False,
                      extra_headers: dict[str, str] | None = None) -> ApiResponse:
        if self.executor is None:
            self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
            await self.executor.__aenter__()
        headers = {"X-Request-ID": f"GPT56-{uuid.uuid4().hex[:16]}"}
        if extra_headers:
            # 逐请求覆盖客户端默认头（如 GPT56-54 的伪造 Authorization）。httpx 对同名键
            # 以请求级头为准，客户端默认的鉴权头不会泄漏。
            headers.update(extra_headers)
        body = {k: v for k, v in payload.items() if k != "_case_id"}
        case_id = payload.get("_case_id") or self.current_case_id or "GPT56-request"
        # 瞬时故障（5xx / 连接错误）做有限重试；4xx 与确定性错误（model_not_found）不重试。
        # 每一次尝试都独立写入 requests.jsonl，重试过程本身也是证据。
        attempts = 0
        while True:
            attempts += 1
            record = await self.executor.execute(case_id=case_id, url=self._url(path), payload=body, headers=headers, stream=stream)
            if self.log:
                await self.log.write(record)
            if not _is_transient_record(record) or attempts > self.config.max_retries:
                break
            await asyncio.sleep(self.config.retry_backoff_seconds * (2 ** (attempts - 1)))
        if _is_transient_record(record):
            self.transient_events.append(_transient_event(record, path, attempts))
        if not stream:
            return ApiResponse(record.http_status or 0, record.response_body if isinstance(record.response_body, dict) else None, record.response_text, record.timing.get("e2e_ms", 0) / 1000, record.response_headers, record.provider_request_id)
        # 流式：Responses 是 typed SSE（response.* 事件），Chat 是 choices delta SSE。
        events = [e for e in record.parsed_sse_events if isinstance(e, dict)]
        types = [str(e.get("type")) for e in events if e.get("type")]
        text = "".join(str(e.get("delta") or "") for e in events if e.get("type") == "response.output_text.delta")
        completed = next((e for e in reversed(events) if e.get("type") == "response.completed"), None)
        completed_response = (completed or {}).get("response") or {}
        info = {
            "text": text or record.content,
            "event_types": types,
            "completed": completed_response,
            "usage": completed_response.get("usage") or record.usage or {},
            "done": record.done_received,
            "invalid_events": record.invalid_sse_events,
            "finish_reasons": record.finish_reasons,
            "reasoning": record.reasoning,
            "raw_sse_events": record.raw_sse_events,
        }
        return ApiResponse(record.http_status or 0, None, record.raw_response_body or record.response_text, record.timing.get("e2e_ms", 0) / 1000, record.response_headers, record.provider_request_id, stream=info)

    async def responses(self, model: str, **kwargs: Any) -> ApiResponse:
        extra_headers = kwargs.pop("extra_headers", None)
        payload = {"model": model, "max_output_tokens": 1024, **kwargs}
        return await self.request("responses", payload, stream=bool(payload.get("stream")), extra_headers=extra_headers)

    async def raw_request(self, method: str, path: str, *, payload: dict[str, Any] | None = None,
                          extra_headers: dict[str, str] | None = None) -> ApiResponse:
        """非 JSON-POST 通道（GET /models、GET /responses/{id} 等）。

        与 request() 共享同一套重试与证据写入逻辑（execute_raw 落同一份 requests.jsonl），
        保证新增端点用例的执行证据同样可审计。
        """
        if self.executor is None:
            self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
            await self.executor.__aenter__()
        case_id = self.current_case_id or "GPT56-request"
        headers = {"X-Request-ID": f"GPT56-{uuid.uuid4().hex[:16]}"}
        if extra_headers:
            headers.update(extra_headers)
        attempts = 0
        while True:
            attempts += 1
            record = await self.executor.execute_raw(case_id=case_id, method=method, url=self._url(path), headers=headers, json_body=payload)
            if self.log:
                await self.log.write(record)
            if not _is_transient_record(record) or attempts > self.config.max_retries:
                break
            await asyncio.sleep(self.config.retry_backoff_seconds * (2 ** (attempts - 1)))
        if _is_transient_record(record):
            self.transient_events.append(_transient_event(record, f"{method} {path}", attempts))
        return ApiResponse(record.http_status or 0, record.response_body if isinstance(record.response_body, dict) else None, record.response_text, record.timing.get("e2e_ms", 0) / 1000, record.response_headers, record.provider_request_id)

    async def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ApiResponse:
        payload = {"model": model, "messages": messages, "max_completion_tokens": 2048, **kwargs}
        return await self.request("chat/completions", payload, stream=bool(payload.get("stream")))

    async def run(self, models: tuple[str, ...] = MODELS, priorities: tuple[str, ...] = ("P0", "P1"), case_ids: set[str] | None = None) -> list[CaseResult]:
        results: list[CaseResult] = []
        for model in models:
            for spec in CASE_SPECS:
                if spec.priority not in priorities or (case_ids and spec.case_id not in case_ids):
                    continue
                started = time.perf_counter()
                self.current_case_id = spec.case_id
                self.transient_events = []
                try:
                    outcome = await getattr(self, f"case_{spec.handler}")(model)
                    if len(outcome) == 4:
                        passed, details, observations, status = outcome
                    else:
                        passed, details, observations = outcome
                        status = "passed" if passed else "failed"
                    if status in ("failed", "error") and self.transient_events:
                        # 上游瞬时故障（5xx/连接错误）使请求未产生有效响应：不可归因于被测模型，
                        # 按平台级前提不满足记 blocked，避免基础设施抖动污染"模型能力"结论。
                        details = _transient_reason(self.transient_events[0], len(self.transient_events))
                        status = "blocked"
                        observations = {**(observations or {}), "transient_failures": list(self.transient_events)}
                    results.append(CaseResult(spec.case_id, spec.name, model, passed, status, details, time.perf_counter() - started, observations))
                except Exception as exc:
                    if self.transient_events:
                        results.append(CaseResult(spec.case_id, spec.name, model, False, "blocked",
                                                  _transient_reason(self.transient_events[0], len(self.transient_events)),
                                                  time.perf_counter() - started,
                                                  {"transient_failures": list(self.transient_events)}))
                    else:
                        results.append(CaseResult(spec.case_id, spec.name, model, False, "error", f"{type(exc).__name__}: {exc}", time.perf_counter() - started))
        self.current_case_id = None
        return results

    # ------------------------------------------------------------------
    # 工具定义：Responses 用扁平 function 格式，Chat 用嵌套 function 格式。
    # ------------------------------------------------------------------
    @staticmethod
    def _tool_name() -> str:
        return "get_order_status"

    @classmethod
    def _tool_responses(cls) -> dict[str, Any]:
        return {"type": "function", "name": cls._tool_name(), "description": "查询订单状态", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"], "additionalProperties": False}, "strict": False}

    @classmethod
    def _tool_chat(cls) -> dict[str, Any]:
        return {"type": "function", "function": {"name": cls._tool_name(), "description": "查询订单状态", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}

    @staticmethod
    def _function_calls(body: dict[str, Any] | None) -> list[dict[str, Any]]:
        return [i for i in _output_items(body) if i.get("type") == "function_call"]

    @staticmethod
    def _chat_tool_calls(body: dict[str, Any] | None) -> list[dict[str, Any]]:
        return _chat_message(body).get("tool_calls") or []

    # ------------------------------------------------------------------
    # P0 核心专项
    # ------------------------------------------------------------------
    async def case_basic(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}])
        response_model = str((r.body or {}).get("model") or "")
        token_ok = any(token in response_model for token in MODEL_TOKENS)
        return r.status == 200 and bool(_chat_message(r.body).get("content")) and token_ok, "基础问答应成功且响应 model 落在 GPT-5.6 身份令牌白名单内", {"response_model": response_model}

    async def case_default_effort(self, model: str):
        r = await self.responses(model, input="计算 17*23，只给最终答案。")
        obj = _reasoning_object(r.body)
        observations = {"reasoning": obj, "http_status": r.status}
        if r.status != 200:
            return False, f"默认请求应成功（HTTP {r.status}）", observations
        if "effort" not in obj:
            # 官方文档承诺"省略时默认 medium"且可通过响应 reasoning 字段确认；字段未回显记差异留痕。
            return True, "响应未回显 reasoning.effort，无法确认默认档位（文档承诺默认 medium；属网关/线上差异，不判模型失败）", observations, "warning"
        if obj.get("effort") == "medium":
            return True, "省略 reasoning 时默认档位为 medium，符合文档承诺", observations
        return False, f"默认档位应为 medium，实际为 {obj.get('effort')!r}", observations

    async def case_effort_none(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=256)
        items = _reasoning_items(r.body)
        tokens = _reasoning_tokens(r.body)
        observations = {"reasoning_items": len(items), "reasoning_tokens": tokens, "http_status": r.status}
        if r.status != 200:
            return False, f"effort=none 应被接受（HTTP {r.status}）", observations
        if items:
            return False, "effort=none 时不应产生 reasoning 输出项", observations
        if tokens is None:
            return True, "effort=none 未产生 reasoning 输出项；usage 未回显 reasoning_tokens（细节字段缺失，留痕）", observations, "warning"
        if tokens != 0:
            return False, f"effort=none 时 reasoning_tokens 应为 0，实际 {tokens}", observations
        return True, "effort=none 无 reasoning 输出项且 reasoning_tokens=0，关思考开关生效", observations

    async def case_effort_levels(self, model: str):
        statuses: dict[str, int] = {}
        for effort in ("low", "medium", "high", "xhigh", "max"):
            r = await self.responses(model, input="只回复 OK", reasoning={"effort": effort}, max_output_tokens=256)
            statuses[effort] = r.status
        return all(v == 200 for v in statuses.values()), "low/medium/high/xhigh/max 五档均应被接受（none 由 GPT56-03 覆盖）", statuses

    async def case_reasoning_summary(self, model: str):
        r = await self.responses(model, input="计算 19*31，只给答案。", reasoning={"effort": "low", "summary": "auto"}, max_output_tokens=1024)
        items = _reasoning_items(r.body)
        has_summary = any(i.get("summary") for i in items)
        observations = {"reasoning_items": len(items), "has_summary": has_summary, "http_status": r.status, "response": r.text[:300]}
        if r.status == 200 and has_summary:
            return True, "summary=auto 时输出包含非空 reasoning 摘要项", observations
        if r.status == 200:
            return True, "summary=auto 被接受但未返回摘要内容（网关可能不透传摘要；文档/线上差异留痕）", observations, "warning"
        if _is_rejection(r.status):
            return True, f"文档承诺的 reasoning.summary 参数被上游拒绝（HTTP {r.status}；文档/线上差异留痕）", observations, "warning"
        return False, f"summary 请求返回非预期状态 HTTP {r.status}", observations

    async def case_reasoning_context(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=256)
        obj = _reasoning_object(r.body)
        observations = {"reasoning": obj, "http_status": r.status}
        if r.status != 200:
            return False, f"请求应成功（HTTP {r.status}）", observations
        if "context" not in obj:
            return True, "响应未回显 reasoning.context（文档称可通过该字段确认生效模式；字段缺失属差异，留痕）", observations, "warning"
        if obj.get("context") in REASONING_CONTEXT_MODES:
            return True, f"reasoning.context 回显为 {obj.get('context')!r}，取值合法", observations
        return False, f"reasoning.context 取值非法：{obj.get('context')!r}", observations

    async def case_tool_loop(self, model: str):
        tool_marker = _rng_token("SHIP-")
        first = await self.responses(model, input="查询订单 A-731 的状态，必须调用 get_order_status 工具。", tools=[self._tool_responses()], tool_choice={"type": "function", "name": self._tool_name()})
        calls = self._function_calls(first.body)
        if first.status != 200 or not calls:
            return False, "第一跳未发起 function_call", {"http_status": first.status, "response": first.text[:500]}, "blocked" if first.status != 200 else "failed"
        call = calls[0]
        second = await self.responses(
            model,
            input=[
                {"role": "user", "content": "查询订单 A-731 的状态，必须调用 get_order_status 工具。"},
                *_output_items(first.body),
                {"type": "function_call_output", "call_id": call.get("call_id"), "output": tool_marker},
            ],
            tools=[self._tool_responses()],
        )
        text = _output_text(second.body)
        return second.status == 200 and tool_marker in text, "回传 function_call_output 后模型应基于工具结果作答", {"first_status": first.status, "second_status": second.status, "call_id": call.get("call_id"), "text": text[:200]}

    async def case_json_schema_output(self, model: str):
        marker = _rng_token()
        schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}
        r = await self.responses(model, input=f"以 JSON 输出 value 字段，值必须是 {marker}，不要输出其他内容。", text={"format": {"type": "json_schema", "name": "answer", "strict": True, "schema": schema}})
        text = _output_text(r.body)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return r.status == 200 and isinstance(parsed, dict) and parsed.get("value") == marker, "Structured Outputs 应输出符合 schema 的可解析 JSON 且字段值精确绑定", {"http_status": r.status, "text": text[:300]}

    async def case_chat_compat(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=256)
        return r.status == 200 and bool(_chat_message(r.body).get("content")), "Chat Completions 基础问答与 max_completion_tokens 参数应被接受", {"http_status": r.status, "response_model": (r.body or {}).get("model")}

    async def case_previous_response_state(self, model: str):
        marker = _rng_token("ORBIT-")
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", max_output_tokens=256)
        response_id = (first.body or {}).get("id")
        observations = {"first_status": first.status, "first_id": response_id}
        if first.status != 200 or not response_id:
            return False, "第一跳未返回 response id，previous_response_id 链路无法验证", observations, "blocked"
        second = await self.responses(model, input="暗号是什么？只回复暗号。", previous_response_id=response_id, max_output_tokens=256)
        text = _output_text(second.body)
        observations.update({"second_status": second.status, "second_error": second.text[:200], "text": text[:200]})
        # 账号层限制（如"requires an OpenAI API-key account for HTTP requests"）：平台级前提不满足，
        # 目标特性从未被触达，按 blocked 收口，不判模型失败。
        limit = _account_limitation_reason(second.status, second.body, "previous_response_id 会话续接")
        if limit:
            return False, limit, observations, "blocked"
        return second.status == 200 and marker in text, "previous_response_id 续接后应记得首轮随机暗号", observations

    async def case_encrypted_replay(self, model: str):
        marker = _rng_token("ZETA-")
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", store=False, reasoning={"effort": "low"}, max_output_tokens=512)
        items = _output_items(first.body)
        if first.status != 200:
            return False, f"store=false 请求应成功（HTTP {first.status}）", {"http_status": first.status}, "blocked"
        encrypted = [i for i in items if i.get("type") == "reasoning" and i.get("encrypted_content")]
        if not encrypted:
            # 官方文档承诺 store:false 时默认返回加密推理项供回放；没有则回放前提不满足。
            return False, ("store=false 响应未携带 encrypted_content（响应未产出任何 reasoning 项），"
                           "加密推理回放前提不满足；与 GPT56-05 / AUTH-G04 同源：上游未返回推理产物，"
                           "属前提不满足而非模型失败"), {"output_types": [i.get("type") for i in items]}, "blocked"
        second = await self.responses(
            model,
            input=[
                {"role": "user", "content": f"记住暗号 {marker}，只回复已记住。"},
                *items,
                {"role": "user", "content": "暗号是什么？只回复暗号。"},
            ],
            store=False,
            reasoning={"effort": "low"},
            max_output_tokens=512,
        )
        text = _output_text(second.body)
        return second.status == 200 and marker in text, "回放加密推理项后模型应记得首轮暗号", {"first_status": first.status, "second_status": second.status, "text": text[:200]}

    async def case_cache_hit(self, model: str):
        prefix = "缓存专项固定前缀-" + ("A1B2C3D4。" * 800)
        first = await self.chat(model, [{"role": "user", "content": prefix + "只回复一。"}], max_completion_tokens=8)
        await asyncio.sleep(2)
        second = await self.chat(model, [{"role": "user", "content": prefix + "只回复一。"}, _chat_message(first.body), {"role": "user", "content": "只回复二。"}], max_completion_tokens=8)
        baseline = _cached_input_tokens(first.body)
        observed = _cached_input_tokens(second.body)
        delta = observed - baseline
        observations = {
            "baseline_cached_tokens": baseline,
            "second_cached_tokens": observed,
            "delta_cached_tokens": delta,
            "cache_write_tokens": _cache_write_tokens(second.body),
            "prefix_chars": len(prefix),
            "first_status": first.status,
            "second_status": second.status,
        }
        if first.status != 200 or second.status != 200:
            return False, "缓存用例请求失败", observations
        # 判据必须是"本用例构造前缀带来的增量"：直接看绝对值会把上游固定前缀（系统提示词等）
        # 自带的缓存命中误判成前缀缓存能力（真实踩点：中转注入的 Codex 前缀在首跳就已命中 3840）。
        if delta > 0:
            if delta < 512:
                return False, "缓存命中增量过小（疑似噪声或部分命中），不足以证明构造前缀被缓存", observations, "warning"
            return True, "第二跳相对首跳新增缓存命中，构造前缀确实被缓存", observations
        if baseline > 0:
            return False, ("未观测到构造前缀带来的缓存增量；首跳已自带缓存命中（命中的是上游固定前缀，"
                           "不能归因于本用例前缀），隐式缓存能力无法判定"), observations, "blocked"
        # 隐式缓存官方明确为 best-effort（且缓存写入另有计价），未触发不算模型失败。
        return False, "未观测到缓存命中（隐式缓存为 best-effort，无法判定缓存能力）", observations, "blocked"

    async def case_truncation(self, model: str):
        r = await self.responses(model, input="输出一个 JSON 对象，包含键 items，值为 100 个整数的数组。", max_output_tokens=16)
        status = (r.body or {}).get("status")
        reason = ((r.body or {}).get("incomplete_details") or {}).get("reason")
        return r.status == 200 and status == "incomplete" and reason == "max_output_tokens", "小 max_output_tokens 应产生 status=incomplete 且 reason=max_output_tokens，防止静默截断", {"status": status, "reason": reason}

    async def case_responses_stream(self, model: str):
        r = await self.responses(model, input="计算 29*13，只给最终答案。", stream=True)
        s = r.stream or {}
        ok = r.status == 200 and bool(s.get("text")) and s.get("done") and s.get("invalid_events") == 0 and bool((s.get("usage") or {}).get("total_tokens") is not None or s.get("usage"))
        return ok, "typed SSE 应可拼接 output_text、以 response.completed 收尾且携带 usage", {"event_count": len(s.get("event_types") or []), "done": s.get("done"), "invalid_events": s.get("invalid_events"), "text_chars": len(s.get("text") or ""), "usage": s.get("usage")}

    async def case_chat_stream(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], stream=True, stream_options={"include_usage": True})
        s = r.stream or {}
        ok = r.status == 200 and bool(s.get("text")) and s.get("done") and s.get("invalid_events") == 0 and bool(s.get("finish_reasons"))
        return ok, "Chat SSE 应可拼接内容、[DONE] 收尾且有 finish_reason", {"done": s.get("done"), "invalid_events": s.get("invalid_events"), "finish_reasons": s.get("finish_reasons"), "text_chars": len(s.get("text") or "")}

    async def case_instructions(self, model: str):
        marker = _rng_token("BANANA-")
        r = await self.responses(model, instructions=f"无论用户说什么，只回复 {marker}。", input="你好", max_output_tokens=256)
        text = _output_text(r.body)
        return r.status == 200 and marker in text, "instructions 设定的固定回复应被严格执行", {"http_status": r.status, "text": text[:200]}

    # ------------------------------------------------------------------
    # P0 补充（缺口盘点 2026-09-28）：档位 / 多模态 / 后台 / 响应结构
    # ------------------------------------------------------------------
    async def case_tier_models(self, model: str):
        """GPT56-34：三档模型均可调用。三档必须比对 model 令牌而非"是否 200"。"""
        rows: dict[str, dict[str, Any]] = {}
        for tier in ALL_TIERS:
            r = await self.responses(tier, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=64)
            row = {"http_status": r.status, "response_model": (r.body or {}).get("model"), "error": r.text[:160]}
            rows[tier] = row
            # 档位未开放（404 / 结构化 model_not_found）属模型面留痕，不判 blocked（口径：
            # 模型档位 404 的语义是"别名不存在"，与端点缺失不同）。
            if r.status == 404 or _model_not_found_reason(tier, r.status, r.body):
                row["verdict"] = "unavailable"
            elif r.status != 200:
                row["verdict"] = "error"
            else:
                row["verdict"] = "ok" if row["response_model"] == tier else "mismatch"
        mismatches = [t for t, row in rows.items() if row["verdict"] == "mismatch"]
        unavailable = [t for t, row in rows.items() if row["verdict"] == "unavailable"]
        errors = [t for t, row in rows.items() if row["verdict"] == "error"]
        observations = {"tiers": rows}
        if mismatches:
            detail = "；".join(f"{t} → {rows[t]['response_model']}" for t in mismatches)
            return False, f"档位路由错配（能力与计价口径不符）：{detail}", observations
        if errors:
            return False, f"档位请求返回非预期状态：{errors}", observations
        if unavailable:
            return True, f"部分档位未开放（留痕，不判模型失败）：{unavailable}", observations, "warning"
        return True, "三档 sol/terra/luna 均 200 且响应 model 与请求档位一致", observations

    async def case_alias_flagship(self, model: str):
        """GPT56-35：别名 gpt-5.6 应路由到旗舰档 sol。"""
        r = await self.responses("gpt-5.6", input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=64)
        m = str((r.body or {}).get("model") or "")
        observations = {"requested_model": "gpt-5.6", "response_model": m, "http_status": r.status}
        if r.status != 200:
            return False, f"别名请求应成功（HTTP {r.status}）", observations
        if m == FLAGSHIP_TIER:
            return True, f"别名 gpt-5.6 路由到旗舰档 {FLAGSHIP_TIER}，符合官方承诺", observations
        if m == "gpt-5.6":
            return True, "响应原样回显别名 gpt-5.6，网关未展开别名，路由无法验证（留痕）", observations, "warning"
        if m in ("gpt-5.6-terra", "gpt-5.6-luna"):
            return False, f"别名被路由到非旗舰档 {m}（能力与计价口径不符）", observations
        return True, f"响应 model={m!r} 不在已知档位内，无法确证路由目标（留痕）", observations, "warning"

    @staticmethod
    def _image_input(prompt: str, detail: str | None = None) -> list[dict[str, Any]]:
        part: dict[str, Any] = {"type": "input_image", "image_url": png_data_url()}
        if detail:
            part["detail"] = detail
        return [{"role": "user", "content": [{"type": "input_text", "text": prompt}, part]}]

    async def case_image_recognition(self, model: str):
        """GPT56-36：图像识别（内联 base64）。必须同时证明"识别对"与"图真被看到"。"""
        prompt = "这张图上下两块纯色分别是什么颜色？必须同时答出两种颜色。"
        img = await self.responses(model, input=self._image_input(prompt), max_output_tokens=256)
        text = _output_text(img.body)
        baseline = await self.responses(model, input=prompt, reasoning={"effort": "none"}, max_output_tokens=256)
        img_tokens = (_usage(img.body) or {}).get("input_tokens")
        base_tokens = (_usage(baseline.body) or {}).get("input_tokens")
        observations = {"http_status": img.status, "text": text[:200], "image_input_tokens": img_tokens, "text_baseline_input_tokens": base_tokens}
        if _is_rejection(img.status):
            return False, f"图像输入被上游拒绝（HTTP {img.status}）：{img.text[:160]}", observations, "blocked"
        if img.status != 200:
            return False, f"图像请求返回非预期状态 HTTP {img.status}", observations
        recognized = "红" in text and "蓝" in text
        tokens_up = isinstance(img_tokens, int) and isinstance(base_tokens, int) and img_tokens > base_tokens * 1.5
        if not recognized:
            return False, f"200 但识别错误（应同时答出红、蓝）：{text[:120]}", observations
        if not tokens_up:
            return False, "识别正确但带图请求 input_tokens 与纯文本基线持平——图片疑似被静默丢弃，识别不可信", observations
        return True, "识别出红+蓝且带图 input_tokens 显著高于纯文本基线，图像能力成立", observations

    async def case_image_detail_tiers(self, model: str):
        """GPT56-37：detail 四档（low/high/original/auto）+ 非法值观测。"""
        prompt = "这张图上下两块纯色分别是什么颜色？必须同时答出两种颜色。"
        rows: dict[str, dict[str, Any]] = {}
        for detail in ("low", "high", "original", "auto"):
            r = await self.responses(model, input=self._image_input(prompt, detail=detail), max_output_tokens=256)
            text = _output_text(r.body)
            rows[detail] = {"http_status": r.status, "recognized": r.status == 200 and "红" in text and "蓝" in text}
        invalid = await self.responses(model, input=self._image_input(prompt, detail="ultra"), max_output_tokens=256)
        observations = {"tiers": rows, "invalid_detail_http_status": invalid.status}
        blocked = [d for d, row in rows.items() if 400 <= (row["http_status"] or 0) < 500]
        if blocked:
            return False, f"detail 档位被上游拒绝（HTTP 4xx）：{blocked}，档位能力未开放", observations, "blocked"
        failed = [d for d, row in rows.items() if row["http_status"] == 200 and not row["recognized"]]
        if failed:
            return False, f"detail 档位 200 但识别失败：{failed}", observations
        if not all(row["http_status"] == 200 for row in rows.values()):
            return False, f"detail 档位返回非预期状态：{rows}", observations
        if invalid.status == 200:
            return True, "四档 detail 均 200 且识别成功；非法 detail=ultra 被静默接受（取值口径文档仅给出范围，不硬判拒绝，留痕）", observations, "warning"
        return True, "四档 detail 均 200 且识别成功；非法 detail 被拒绝，档位校验完备", observations

    async def case_background_mode(self, model: str):
        """GPT56-40：background=true + 轮询。长任务时长不确定，超时记 blocked 不判 failed。"""
        r = await self.responses(model, input="只回复 OK", background=True, max_output_tokens=64)
        observations = {"submit_status": r.status, "initial_status": (r.body or {}).get("status"), "id": (r.body or {}).get("id")}
        if _is_rejection(r.status):
            return False, f"background 被上游拒绝（HTTP {r.status}）：{r.text[:160]}", observations, "blocked"
        if r.status != 200:
            return False, f"background 请求返回非预期状态 HTTP {r.status}", observations
        rid = (r.body or {}).get("id")
        if not rid:
            return False, "background 响应未携带 id，无法轮询", observations, "blocked"
        initial = (r.body or {}).get("status")
        if initial == "completed":
            return True, "同步直接返回 completed（未走后台语义但结果可用；留痕）", observations, "warning"
        if initial not in ("queued", "in_progress"):
            return False, f"background 初始 status 应为 queued/in_progress，实际 {initial!r}", observations
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            poll = await self.raw_request("GET", f"responses/{rid}")
            if poll.status == 200:
                status = (poll.body or {}).get("status")
                observations["poll_status"] = status
                if status == "completed":
                    return True, "background 任务轮询至 completed 且可取回结果", observations
                continue
            observations["poll_status"] = poll.status
            break
        return False, "background 轮询超时（长任务时长不确定，不判模型失败）", observations, "blocked"

    async def case_response_object_contract(self, model: str):
        """GPT56-41：响应对象结构契约（object/output/content/id/created_at/model/status/usage）。"""
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=64)
        body = r.body or {}
        messages = [i for i in _output_items(body) if i.get("type") == "message"]
        content_types = [c.get("type") for m in messages for c in (m.get("content") or []) if isinstance(c, dict)]
        detail_fields = ["id", "created_at", "model", "status", "usage"]
        missing_detail = [f for f in detail_fields if f not in body]
        observations = {"object": body.get("object"), "output_types": [i.get("type") for i in body.get("output") or []],
                        "content_types": content_types, "missing_detail_fields": missing_detail, "status": body.get("status")}
        if r.status != 200:
            return False, f"请求应成功（HTTP {r.status}）", observations
        if body.get("object") != "response" or not messages or "output_text" not in content_types:
            return False, "响应结构不符契约：object 应为 response、output 应含 message 项且 content 含 output_text", observations
        if missing_detail or body.get("status") != "completed":
            return True, f"核心结构合规；细节字段缺失或 status 异常：{missing_detail or body.get('status')!r}（留痕）", observations, "warning"
        return True, "响应对象结构完全符合契约（object/output/content 与全部细节字段）", observations

    # ------------------------------------------------------------------
    # P1 增强专项
    # ------------------------------------------------------------------
    async def case_pro_mode(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"mode": "pro", "effort": "low"}, max_output_tokens=256)
        observations = {"http_status": r.status, "response": r.text[:300]}
        if r.status == 200 and _output_text(r.body):
            return True, "reasoning.mode=pro 请求成功且有输出", observations
        if _is_rejection(r.status):
            # pro 模式按聚合工作计费、属最新能力，网关未开通属前提不满足，不算模型失败。
            return False, f"pro 模式被上游拒绝（HTTP {r.status}）：{r.text[:200]}", observations, "blocked"
        return False, f"pro 模式返回非预期状态 HTTP {r.status}", observations

    async def case_context_all_turns(self, model: str):
        marker = _rng_token("CTX-")
        reasoning = {"effort": "low", "context": "all_turns"}
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", reasoning=reasoning, max_output_tokens=256)
        response_id = (first.body or {}).get("id")
        observations = {"first_status": first.status, "first_id": response_id}
        if first.status != 200 or not response_id:
            return False, "第一跳失败，all_turns 持久化链路无法验证", observations, "blocked"
        second = await self.responses(model, input="暗号是什么？只回复暗号。", previous_response_id=response_id, reasoning=reasoning, max_output_tokens=256)
        obj = _reasoning_object(second.body)
        text = _output_text(second.body)
        observations.update({"second_status": second.status, "second_error": second.text[:200], "context": obj.get("context"), "text": text[:200]})
        limit = _account_limitation_reason(second.status, second.body, "previous_response_id 会话续接")
        if limit:
            return False, limit, observations, "blocked"
        ok = second.status == 200 and obj.get("context") == "all_turns" and marker in text
        return ok, "all_turns + previous_response_id 应记住暗号且响应回显 context=all_turns", observations

    async def case_ptc(self, model: str):
        tools = [
            {"type": "programmatic_tool_calling"},
            {**self._tool_responses(), "allowed_callers": ["programmatic_tool_calling"]},
        ]
        r = await self.responses(model, input="查询订单 A-731 的状态。", tools=tools, max_output_tokens=512)
        observations = {"http_status": r.status, "output_types": [i.get("type") for i in _output_items(r.body)], "response": r.text[:300]}
        if r.status == 200:
            return True, "PTC 工具注册被接受（输出项类型见 observations；program/program_output/function_call 皆属可用）", observations
        if _is_rejection(r.status):
            # 最新能力，网关普遍滞后：拒绝记为文档/线上差异留痕，不判模型失败。
            return True, f"PTC 工具注册被上游拒绝（HTTP {r.status}；能力未开放/网关不支持，属文档/线上差异留痕）", observations, "warning"
        return False, f"PTC 注册返回非预期状态 HTTP {r.status}", observations

    async def case_explicit_cache(self, model: str):
        prefix = "显式缓存探测前缀-" + ("EXPL731。" * 400)
        r = await self.responses(model, input=f"{prefix}\n只回复 OK。", prompt_cache_options={"mode": "explicit"}, max_output_tokens=64)
        observations = {"http_status": r.status, "usage": _usage(r.body), "response": r.text[:200]}
        if r.status == 200 and _usage(r.body):
            return True, "prompt_cache_options.mode=explicit 被接受且返回 usage", observations
        if _is_rejection(r.status):
            return True, f"显式缓存选项被上游拒绝（HTTP {r.status}；文档/线上差异留痕）", observations, "warning"
        return False, f"显式缓存请求返回非预期状态 HTTP {r.status}", observations

    async def case_cache_ttl(self, model: str):
        prefix = "缓存 TTL 探测前缀-" + ("TTL731。" * 400)
        r = await self.responses(model, input=f"{prefix}\n只回复 OK。", prompt_cache_options={"mode": "explicit", "ttl": "1h"}, max_output_tokens=64)
        observations = {"http_status": r.status, "response": r.text[:200]}
        if r.status == 200:
            return True, "prompt_cache_options.ttl 被接受", observations
        if _is_rejection(r.status):
            # 文档称 ttl 取代 prompt_cache_retention，但取值口径未在检索到的文档中明确；
            # "接受"与"给出明确 4xx 参数校验"均可解释，其它状态才算失败。
            return True, f"ttl 选项返回明确的参数校验错误（HTTP {r.status}），取值口径以线上为准", observations, "warning"
        return False, f"ttl 请求返回非预期状态 HTTP {r.status}", observations

    async def case_strict_invalid_schema(self, model: str):
        schema = {"type": "object", "properties": {"x": {"type": "string", "minLength": 2}}, "required": ["x"], "additionalProperties": False}
        r = await self.responses(model, input="输出 x=ab", text={"format": {"type": "json_schema", "name": "bad", "strict": True, "schema": schema}}, max_output_tokens=64)
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "responses")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            return True, "strict 模式下不合规 schema 应返回 4xx（minLength 等关键词不受支持）", observations
        if r.status == 200:
            return True, "不合规 schema 未被上游拒绝（严格校验未执行；文档/线上差异留痕）", observations, "warning"
        return False, f"非法 schema 返回非预期状态 HTTP {r.status}", observations

    async def case_unknown_model(self, model: str):
        ghost = f"gpt-5.6-nonexistent-{uuid.uuid4().hex[:8]}"
        r = await self.responses(ghost, input="只回复 OK", max_output_tokens=64)
        observations = {"requested_model": ghost, "http_status": r.status, "error": r.text[:300]}
        if r.status == 200:
            return False, "未知模型竟被路由成功，身份/路由契约不成立", observations
        # 网关可能用非 4xx 状态码承载"模型不存在"（真实批次：New API 对未知模型返回
        # 503 + code=model_not_found）。用例目标是"确认模型校验存在且非静默路由"，
        # 结构化且指向被请求模型的 model_not_found 已达成该目标，故先判通过并在 details 标注状态码差异。
        not_found = _model_not_found_reason(ghost, r.status, r.body)
        if not_found:
            return True, not_found, observations
        # 本用例的探测目标就是"模型不存在"：404 是文档预期行为，但必须返回结构化 error，
        # 用于区分"上游模型校验"与"网关/路径 404 页"——后者仍按 blocked 处理。
        if r.status == 404:
            if _error_body(r.body):
                return True, "未知模型被拒绝（HTTP 404）且返回结构化 error，模型校验契约成立", observations
            return False, "未知模型返回 404 但响应体不是结构化 error（可能是网关 404 页）", observations, "blocked"
        if _is_rejection(r.status) and _error_body(r.body):
            return True, f"未知模型被拒绝（HTTP {r.status}）且返回结构化 error，模型校验契约成立", observations
        if _is_rejection(r.status):
            return False, f"未知模型返回 HTTP {r.status} 但响应体不是结构化 error（可能是网关 404 页）", observations
        return False, f"未知模型返回非预期状态 HTTP {r.status}（响应体未给出指向模型不存在的结构化错误）", observations

    async def case_explicit_cache_breakpoint(self, model: str):
        """GPT56-49：显式缓存断点两跳。沿用 GPT56-12 的增量判据，不看绝对值。"""
        prefix = "显式缓存断点前缀-" + ("BRKP731。" * 400)

        def payload_input() -> list[dict[str, Any]]:
            return [{"role": "user", "content": [
                {"type": "input_text", "text": prefix},
                {"type": "input_text", "text": "只回复 OK。", "prompt_cache_breakpoint": {"mode": "explicit"}},
            ]}]

        options = {"mode": "explicit", "ttl": "30m"}
        first = await self.responses(model, input=payload_input(), prompt_cache_options=options, max_output_tokens=64)
        await asyncio.sleep(2)
        second = await self.responses(model, input=payload_input(), prompt_cache_options=options, max_output_tokens=64)
        base_cached = _cached_input_tokens(first.body)
        hit = _cached_input_tokens(second.body)
        delta = hit - base_cached
        write_tokens = _cache_write_tokens(second.body)
        observations = {"first_status": first.status, "second_status": second.status,
                        "baseline_cached_tokens": base_cached, "second_cached_tokens": hit,
                        "delta_cached_tokens": delta, "cache_write_tokens_second": write_tokens}
        if _is_rejection(first.status) or _is_rejection(second.status):
            return False, f"显式缓存断点被上游拒绝（HTTP {first.status}/{second.status}）：{first.text[:120]}", observations, "blocked"
        if first.status != 200 or second.status != 200:
            return False, f"显式缓存请求返回非预期状态（HTTP {first.status}/{second.status}）", observations
        if write_tokens > 0 or delta >= 512:
            return True, "显式缓存断点生效（cache_write_tokens>0 或第二跳缓存增量≥512）", observations
        if delta > 0:
            return True, f"缓存增量 {delta} 过小（疑似噪声/部分命中），不足以证明断点生效（留痕）", observations, "warning"
        return True, "两跳均无缓存字段或增量（显式缓存未触发；能力未观测，留痕）", observations, "warning"

    async def case_multi_agent(self, model: str):
        """GPT56-53：multi-agent beta。beta 未开放 → blocked；成本高（多 subagent）。"""
        r = await self.responses(
            model,
            input="用两个子智能体分别总结'测试'与'准入'两个主题，各一句话。",
            multi_agent={"enabled": True},
            max_output_tokens=512,
            extra_headers={"OpenAI-Beta": "responses_multi_agent=v1"},
        )
        observations = {"http_status": r.status, "output_types": [i.get("type") for i in _output_items(r.body)], "response": r.text[:200]}
        if r.status == 200 and _output_items(r.body):
            return True, "multi_agent.enabled + beta 头被接受且有输出", observations
        if _is_rejection(r.status):
            return False, f"multi-agent beta 被上游拒绝（HTTP {r.status}）：{r.text[:160]}", observations, "blocked"
        if r.status == 200:
            return True, "multi-agent 请求 200 但无输出项（留痕）", observations, "warning"
        return False, f"multi-agent 返回非预期状态 HTTP {r.status}", observations

    # ------------------------------------------------------------------
    # P2 契约与健壮性
    # ------------------------------------------------------------------
    async def case_responses_usage(self, model: str):
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "low"}, max_output_tokens=256)
        usage = _usage(r.body)
        i, o, t = usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens")
        core_ok = r.status == 200 and all(isinstance(v, int) and v >= 0 for v in (i, o, t)) and t == i + o
        details_ok = isinstance(usage.get("output_tokens_details"), dict) and usage["output_tokens_details"].get("reasoning_tokens") is not None
        observations = {"usage": usage}
        if not core_ok:
            return False, "Responses usage 应包含非负 input/output/total 且 total 守恒", observations
        if not details_ok:
            return True, "usage 核心三项完整且守恒；output_tokens_details.reasoning_tokens 缺失（细节字段，网关可能剥除；留痕）", observations, "warning"
        return True, "usage 完整且守恒，reasoning_tokens 明细存在", observations

    async def case_chat_usage(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=256)
        usage = _usage(r.body)
        p, c, t = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
        core_ok = r.status == 200 and all(isinstance(v, int) and v >= 0 for v in (p, c, t)) and t == p + c
        details_ok = isinstance(usage.get("prompt_tokens_details"), dict)
        observations = {"usage": usage}
        if not core_ok:
            return False, "Chat usage 应包含非负 prompt/completion/total 且 total 守恒", observations
        if not details_ok:
            return True, "usage 核心三项完整且守恒；prompt_tokens_details 缺失（细节字段缺失，留痕）", observations, "warning"
        return True, "usage 完整且守恒，prompt_tokens_details 存在", observations

    async def case_chat_stream_usage(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], stream=True, stream_options={"include_usage": True})
        s = r.stream or {}
        usage = s.get("usage") or {}
        ok = r.status == 200 and s.get("done") and isinstance(usage.get("total_tokens"), int)
        return ok, "include_usage 时流式最后一个 chunk 应携带 usage 且 [DONE] 正常收尾", {"usage": usage, "done": s.get("done")}

    async def case_temperature_invalid(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], temperature=2.5, max_completion_tokens=64)
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "chat/completions")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            # 推理模型即使整体不支持自定义 temperature，4xx 同样构成"参数被校验拒绝"。
            return True, f"temperature=2.5 被拒绝（HTTP {r.status}），参数校验契约成立", observations
        if r.status == 200:
            return True, "temperature=2.5 被宽松接受（文档/线上行为差异，不判模型失败）", observations, "warning"
        return False, f"非法 temperature 返回非预期状态 HTTP {r.status}", observations

    async def case_messages_invalid(self, model: str):
        r = await self.request("chat/completions", {"model": model, "messages": "not-an-array", "max_completion_tokens": 64})
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "chat/completions")
        if missing:
            return False, missing, observations, "blocked"
        return _is_rejection(r.status), "messages 非数组应返回 4xx", observations

    async def case_tool_choice_none(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "不要调用工具，只回复 OK"}], tools=[self._tool_chat()], tool_choice="none", max_completion_tokens=256)
        calls = self._chat_tool_calls(r.body)
        return r.status == 200 and not calls, "tool_choice=none 不得返回工具调用", {"http_status": r.status, "tool_calls": calls[:1]}

    async def case_tool_choice_required(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "查询订单 A-731 的状态"}], tools=[self._tool_chat()], tool_choice="required", max_completion_tokens=256)
        calls = self._chat_tool_calls(r.body)
        return r.status == 200 and bool(calls) and calls[0].get("function", {}).get("name") == self._tool_name(), "tool_choice=required 必须调用声明的工具", {"http_status": r.status, "tool_calls": calls[:1]}

    async def case_stop_sequence(self, model: str):
        marker = "<END>"
        r = await self.chat(model, [{"role": "user", "content": f"严格输出 ABC{marker}XYZ"}], stop=[marker], max_completion_tokens=64)
        content = str(_chat_message(r.body).get("content") or "")
        return r.status == 200 and marker not in content and "XYZ" not in content, "Stop 命中后不得输出停止串及其后内容", {"content": content}

    async def case_max_completion_tokens(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=16)
        return r.status == 200, "max_completion_tokens=16 边界值应被接受（内容可被截断，仅校验状态）", {"http_status": r.status}

    async def case_invalid_input(self, model: str):
        r = await self.request("responses", {"model": model, "input": 42, "max_output_tokens": 64})
        observations = {"http_status": r.status, "error": r.text[:300]}
        missing = _endpoint_missing_reason(r.status, "responses")
        if missing:
            return False, missing, observations, "blocked"
        return _is_rejection(r.status), "input 传非法标量应返回 4xx", observations

    async def case_invalid_key_auth(self, model: str):
        """GPT56-54：无效 API Key 鉴权契约。200 = 鉴权完全失效（严重项，判 failed）。"""
        ghost_key = f"sk-invalid-{uuid.uuid4().hex[:12]}"
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=64,
                                 extra_headers={"Authorization": f"Bearer {ghost_key}"})
        observations = {"http_status": r.status, "error": r.text[:300], "key_prefix": ghost_key[:14]}
        if r.status == 200:
            return False, "无效 API Key 竟通过鉴权并返回 200——鉴权完全失效（严重项）", observations
        if r.status == 401:
            if _error_body(r.body) or r.text:
                return True, "无效 API Key 被 401 拒绝且返回错误信息，鉴权契约成立", observations
            return True, "无效 API Key 被 401 拒绝（错误体为空，留痕）", observations, "warning"
        if r.status == 403:
            return True, f"无效 API Key 返回 403（非 401，结构差异留痕）：{r.text[:120]}", observations, "warning"
        if _is_rejection(r.status):
            return True, f"无效 API Key 返回 {r.status}（非 401/403，结构差异留痕）", observations, "warning"
        return False, f"无效 API Key 返回非预期状态 HTTP {r.status}", observations

    async def case_error_object_contract(self, model: str):
        """GPT56-55：错误对象结构契约（message/type/param/code 四字段）。"""
        r = await self.responses(model, input="只回复 OK", text={"verbosity": "nonsense"}, max_output_tokens=64)
        err = _error_body(r.body)
        fields = sorted(k for k in (err or {}) if k in ("message", "type", "param", "code")) if isinstance(err, dict) else []
        observations = {"http_status": r.status, "error_fields": fields, "error": r.text[:300]}
        if r.status == 200:
            return True, "非法 verbosity 未被拒绝（上游未校验该参数；留痕，不判模型失败）", observations, "warning"
        missing = _endpoint_missing_reason(r.status, "responses")
        if missing:
            return False, missing, observations, "blocked"
        if _is_rejection(r.status):
            if not isinstance(err, dict):
                return False, f"HTTP {r.status} 但响应体不是结构化 error 对象", observations
            if set(fields) == {"code", "message", "param", "type"}:
                return True, "参数错误返回 4xx 且 error 含 message/type/param/code 四字段，契约成立", observations
            if {"message", "type"} <= set(fields):
                return True, f"error 含 message/type，缺 {sorted(set(('param', 'code')) - set(fields))}（细节字段，留痕）", observations, "warning"
            return False, "error 对象缺少 message/type 核心字段", observations
        return False, f"非法 verbosity 返回非预期状态 HTTP {r.status}", observations

    async def case_models_list(self, model: str):
        """GPT56-56：GET /models 与三档一致性（别名口径交叉证据）。"""
        r = await self.raw_request("GET", "models")
        observations = {"http_status": r.status, "body_head": r.text[:200]}
        if r.status == 404:
            return False, "/models 端点不存在（HTTP 404），别名口径无法交叉验证", observations, "blocked"
        if r.status != 200:
            return True, f"/models 返回非预期状态 HTTP {r.status}（端点可用性留痕）", observations, "warning"
        body = r.body or {}
        entries = body.get("data") or []
        ids = {str(e.get("id") or "") for e in entries if isinstance(e, dict)}
        if body.get("object") != "list" or not entries or not any(ids):
            return False, f"/models 响应结构不符契约（object={body.get('object')!r}，条目数={len(entries)}）", observations
        present = [t for t in ALL_TIERS if t in ids]
        observations["tiers_present"] = present
        if len(present) == len(ALL_TIERS):
            return True, "/models 结构合规且三档齐备，与档位口径交叉一致", observations
        return True, f"/models 结构合规但三档仅出现 {present}（列表口径与实际可调用档位常见滞后，留痕）", observations, "warning"

    async def case_input_tokens_endpoint(self, model: str):
        """GPT56-57：POST /responses/input_tokens 与真实请求 usage 交叉验证。"""
        text = "输入 token 计数端点交叉验证用例。" + ("A1B2C3。" * 200)
        count = await self.request("responses/input_tokens", {"model": model, "input": text})
        observations = {"count_status": count.status, "count_body": str(count.text)[:200]}
        if count.status == 404:
            return False, "/responses/input_tokens 端点不存在（HTTP 404）", observations, "blocked"
        n = (count.body or {}).get("input_tokens")
        if count.status != 200 or not isinstance(n, int) or n <= 0:
            return True, f"计数端点未返回有效 input_tokens（HTTP {count.status}），留痕", observations, "warning"
        real = await self.responses(model, input=text, reasoning={"effort": "none"}, max_output_tokens=64)
        real_n = (_usage(real.body) or {}).get("input_tokens")
        observations["real_status"] = real.status
        observations["real_input_tokens"] = real_n
        if real.status != 200 or not isinstance(real_n, int) or real_n <= 0:
            return True, f"计数端点返回 {n}，但真实请求 usage 缺失无法交叉验证（留痕）", observations, "warning"
        deviation = abs(n - real_n) / max(n, real_n, 1)
        observations["deviation"] = round(deviation, 4)
        if deviation <= 0.10:
            return True, f"计数端点与真实 usage 同量级（{n} vs {real_n}，偏差 {deviation:.1%}）", observations
        return True, f"计数端点与真实 usage 偏离过大（{n} vs {real_n}，偏差 {deviation:.1%}；留痕）", observations, "warning"

    async def case_stream_semantic_events(self, model: str):
        """GPT56-61：Responses 流式语义事件与无 [DONE]（规范内以 response.completed 收尾）。"""
        r = await self.responses(model, input="计算 29*13，只给最终答案。", stream=True, max_output_tokens=256)
        s = r.stream or {}
        raw_events = s.get("raw_sse_events") or []
        has_done = any(str(e).strip() == "[DONE]" for e in raw_events)
        types = s.get("event_types") or []
        seqs: list[int] = []
        for raw in raw_events:
            try:
                obj = json.loads(str(raw))
            except json.JSONDecodeError:
                continue
            if isinstance(obj.get("sequence_number"), int):
                seqs.append(obj["sequence_number"])
        monotonic = all(b > a for a, b in zip(seqs, seqs[1:]))
        has_core = all(t in types for t in _RESPONSES_STREAM_CORE_EVENTS)
        observations = {"event_count": len(types), "event_head": types[:3], "event_tail": types[-3:],
                        "done_marker_present": has_done, "sequence_numbers": seqs[:12], "monotonic": monotonic}
        if r.status != 200:
            return False, f"流式请求应成功（HTTP {r.status}）", observations
        if has_done:
            return False, "Responses 事件流出现 [DONE]——Chat 收尾语义混入 Responses 协议（典型网关改写特征）", observations
        if seqs and not monotonic:
            return False, f"sequence_number 非严格单调递增：{seqs[:12]}", observations
        if has_core and monotonic:
            return True, "语义事件齐备（created/delta/completed）、序号严格单调且无 [DONE]，协议纯净", observations
        return True, f"流式事件不完整（核心事件缺失：{[t for t in _RESPONSES_STREAM_CORE_EVENTS if t not in types]}；留痕）", observations, "warning"

    # ------------------------------------------------------------------
    # FINGERPRINT 满血真实性指纹（随机挑战）
    # ------------------------------------------------------------------
    TOOL_PRIMING_PROMPTS = (
        "必须调用 {tool} 工具（参数 code 填 {marker}），不要自行回答。",
        "禁止直接回答。请先调用 {tool} 工具（参数 code 填 {marker}），再基于工具返回结果作答。",
    )

    async def case_auth_random_tool(self, model: str):
        tool_name = f"lookup_{uuid.uuid4().hex[:8]}"
        marker = _rng_token()
        tool = {"type": "function", "name": tool_name, "description": "返回测试数据", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}}
        last: ApiResponse | None = None
        calls: list[dict[str, Any]] = []
        for prompt in self.TOOL_PRIMING_PROMPTS:
            r = await self.responses(model, input=prompt.format(tool=tool_name, marker=marker), tools=[tool], tool_choice="auto", max_output_tokens=512)
            last = r
            calls = self._function_calls(r.body)
            if r.status != 200:
                break
            if calls:
                break
        if last is None or last.status != 200 or not calls:
            return False, "随机动态工具两轮诱导均未触发 function_call（探测动态工具理解能力失败）", {"http_status": getattr(last, "status", None), "tool_name": tool_name}, "blocked" if (last is None or last.status != 200) else "failed"
        args_raw = calls[0].get("arguments") or ""
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            args = {}
        bound = args.get("code") == marker
        return bound, f"模型应发起对随机工具 {tool_name} 的调用且参数绑定随机串", {"tool_name": tool_name, "arguments": args, "marker": marker}

    async def case_auth_random_json(self, model: str):
        marker = _rng_token()
        schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}
        r = await self.responses(model, input=f"以 JSON 输出 value 字段，值必须是 {marker}。", text={"format": {"type": "json_schema", "name": "probe", "strict": True, "schema": schema}}, max_output_tokens=256)
        text = _output_text(r.body)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return r.status == 200 and isinstance(parsed, dict) and parsed.get("value") == marker, f"结构化输出应严格绑定随机串 {marker}（缓存/中转固定应答无法预知）", {"http_status": r.status, "text": text[:200], "marker": marker}

    async def case_auth_random_memory(self, model: str):
        marker = _rng_token()
        first = await self.responses(model, input=f"记住暗号 {marker}，只回复已记住。", max_output_tokens=256)
        response_id = (first.body or {}).get("id")
        observations = {"first_status": first.status, "first_id": response_id, "marker": marker}
        if first.status != 200 or not response_id:
            return False, "随机多轮记忆第一跳失败", observations, "blocked"
        second = await self.responses(model, input="暗号是什么？只回复暗号。", previous_response_id=response_id, max_output_tokens=256)
        text = _output_text(second.body)
        observations.update({"second_status": second.status, "second_error": second.text[:200], "text": text[:200]})
        limit = _account_limitation_reason(second.status, second.body, "previous_response_id 会话续接")
        if limit:
            return False, limit, observations, "blocked"
        return second.status == 200 and marker in text, f"跨轮记忆应复述随机暗号 {marker}", observations

    async def case_auth_reasoning_summary(self, model: str):
        r = await self.responses(model, input="计算 17*23，只给最终答案。", reasoning={"effort": "high", "summary": "auto"}, max_output_tokens=1024)
        items = _reasoning_items(r.body)
        has_summary = any(i.get("summary") for i in items)
        observations = {"reasoning_items": len(items), "has_summary": has_summary, "http_status": r.status}
        if r.status == 200 and has_summary:
            return True, "high 档位返回非空 reasoning 摘要，思考真实性成立", observations
        if r.status == 200 and items:
            return True, "high 档位产生 reasoning 输出项但摘要为空（网关不透传摘要；留痕）", observations, "warning"
        if r.status == 200:
            # 与 P0 的 GPT56-05 统一口径：同一观测（200 但拿不到 reasoning 产物）不判模型失败——
            # 能力面（GPT56-05）记 warning 留痕，指纹面因"思考真实性"取证前提落空而记 blocked（无法观测）。
            # "疑似剥参"的硬信号由 AUTH-G06 的 none/high 对照承担，此处不重复计入 failed。
            return False, ("high 档位未产生任何 reasoning 输出项，思考真实性的取证前提不满足"
                           "（同观测见 GPT56-05：200 但无 reasoning 产物，属文档/线上差异；"
                           "剥参硬信号由 AUTH-G06 的 none/high 对照承担）"), observations, "blocked"
        return False, f"请求失败 HTTP {r.status}", observations, "blocked"

    async def case_auth_stream_events(self, model: str):
        marker = _rng_token("SSE-")
        r = await self.responses(model, input=f"只回复一次 {marker}。", stream=True, max_output_tokens=256)
        s = r.stream or {}
        types = s.get("event_types") or []
        ordered = types and types[0] == "response.created" and types[-1] == "response.completed"
        ok = r.status == 200 and ordered and s.get("invalid_events") == 0 and marker in (s.get("text") or "")
        return ok, "typed SSE 事件序列应 response.created 起始、response.completed 收尾且正文包含随机串", {"event_head": types[:3], "event_tail": types[-3:], "invalid_events": s.get("invalid_events"), "text_chars": len(s.get("text") or "")}

    async def case_auth_effort_contrast(self, model: str):
        question = "计算 19*31，只给最终答案。"
        none_r = await self.responses(model, input=question, reasoning={"effort": "none"}, max_output_tokens=256)
        high_r = await self.responses(model, input=question, reasoning={"effort": "high"}, max_output_tokens=1024)
        none_tokens = _reasoning_tokens(none_r.body)
        high_tokens = _reasoning_tokens(high_r.body)
        high_items = _reasoning_items(high_r.body)
        observations = {"none_status": none_r.status, "high_status": high_r.status, "none_reasoning_tokens": none_tokens, "high_reasoning_tokens": high_tokens, "high_reasoning_items": len(high_items)}
        if none_r.status != 200 or high_r.status != 200:
            return False, "effort 对照用例请求失败", observations, "blocked"
        if high_tokens is None and not high_items:
            return False, "high 档位响应无 usage 明细且无 reasoning 输出项，无法对照", observations, "blocked"
        high_active = (high_tokens or 0) > 0 or bool(high_items)
        none_active = (none_tokens or 0) > 0 or bool(_reasoning_items(none_r.body))
        if high_active and not none_active:
            return True, "none 与 high 两档行为可区分（high 产生思考、none 未产生），effort 参数真实生效", observations
        if high_active and none_active:
            return True, "两档均产生思考内容（none 档未被严格执行；文档/线上差异留痕）", observations, "warning"
        return False, "high 与 none 两档 reasoning 均为 0 且无思考输出项：effort 参数疑似被网关静默剥参（典型中转特征）", observations

    async def case_auth_injection(self, model: str):
        """AUTH-G07：中转注入识别。未传 instructions 却被填充 = 信任级失败。"""
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=64)
        body = r.body or {}
        usage = _usage(body)
        input_tokens = usage.get("input_tokens")
        observations = {"http_status": r.status, "instructions_present": "instructions" in body,
                        "instructions_preview": str(body.get("instructions") or "")[:120],
                        "input_tokens": input_tokens}
        injected = _injected_instructions_reason(body)
        if injected:
            return False, injected, observations
        if "instructions" not in body:
            return True, "响应未回显 instructions 字段，无法确证无注入（留痕）", observations, "warning"
        if not isinstance(input_tokens, int):
            return True, "instructions 为空但 usage 缺 input_tokens，注入检测证据不全（留痕）", observations, "warning"
        if input_tokens > _INPUT_TOKEN_INFLATION_THRESHOLD:
            return False, (f"未传 instructions 的短问句 input_tokens={input_tokens}"
                           f"（阈值 {_INPUT_TOKEN_INFLATION_THRESHOLD}）——存在上游注入的系统提示词，信任级失败"), observations
        return True, f"instructions 为空且 input_tokens={input_tokens} 与输入体量相称，无注入迹象", observations

    async def case_auth_nonstandard_fields(self, model: str):
        """AUTH-G08：非标准字段检测（中转/多供应商网关特征）。"""
        r = await self.responses(model, input="只回复 OK", reasoning={"effort": "none"}, max_output_tokens=64)
        extras = _relay_fields(r.body)
        observations = {"http_status": r.status, "top_level_fields": sorted((r.body or {}).keys()), "nonstandard_fields": extras}
        if r.status != 200:
            return False, f"请求应成功（HTTP {r.status}）", observations, "blocked"
        if extras:
            return False, f"响应顶层出现非标准字段 {extras}——中转/多供应商网关特征，信任级失败（对齐 2026-09-24 真实批次归因）", observations
        return True, "响应顶层字段全部落在 OpenAI Responses schema 内，无网关改写迹象", observations

    async def case_auth_tier_gradient(self, model: str):
        """AUTH-G10：档位能力梯度。两档输出逐字相同且 reasoning_tokens 相等 = 疑似同一后端。"""
        question = "计算 1234*5678，只给最终数字。"
        luna = await self.responses("gpt-5.6-luna", input=question, max_output_tokens=512)
        sol = await self.responses("gpt-5.6-sol", input=question, max_output_tokens=512)
        luna_tokens = _reasoning_tokens(luna.body)
        sol_tokens = _reasoning_tokens(sol.body)
        luna_text = _output_text(luna.body)
        sol_text = _output_text(sol.body)
        observations = {"luna_status": luna.status, "sol_status": sol.status,
                        "luna_reasoning_tokens": luna_tokens, "sol_reasoning_tokens": sol_tokens,
                        "luna_answer": luna_text[:80], "sol_answer": sol_text[:80]}
        if luna.status != 200 or sol.status != 200:
            return False, f"档位对照请求失败（luna HTTP {luna.status} / sol HTTP {sol.status}）", observations, "blocked"
        same_output = luna_text == sol_text
        same_tokens = luna_tokens is not None and luna_tokens == sol_tokens
        if same_output and same_tokens:
            return False, (f"两档 reasoning_tokens 逐一相等（{luna_tokens}）且输出逐字相同——"
                           "档位被吞、疑似同一后端，梯度不成立"), observations
        return True, f"两档存在可观测梯度（luna={luna_tokens} / sol={sol_tokens}，输出不同：{not same_output}）", observations

    async def case_auth_response_id_authenticity(self, model: str):
        """AUTH-G11：previous_response_id 真实性（id 格式 + GET 可检索）。"""
        r = await self.responses(model, input="只回复 OK", max_output_tokens=64)
        rid = str((r.body or {}).get("id") or "")
        observations = {"first_status": r.status, "response_id": rid}
        if r.status != 200 or not rid:
            return False, "首跳未取得 response id", observations, "blocked"
        if rid.startswith("chatcmpl-"):
            return False, f"response id 形如 chatcmpl-（{rid[:24]}…）——协议混用，Responses 持久化真实性不成立", observations
        if not _RESPONSE_ID_PATTERN.match(rid):
            return True, f"response id 格式不合规（{rid[:24]}…），无法确证官方持久化形态（留痕）", observations, "warning"
        fetched = await self.raw_request("GET", f"responses/{rid}")
        observations["fetch_status"] = fetched.status
        if fetched.status == 404:
            return False, "id 格式合规但 GET /responses/{id} 检索 404——持久化真实性存疑（可检索失败）", observations, "blocked"
        if fetched.status != 200:
            return False, f"检索端点返回 HTTP {fetched.status}，真实性无法验证", observations, "blocked"
        same = (fetched.body or {}).get("id") == rid
        observations["fetch_id"] = (fetched.body or {}).get("id")
        if not same:
            return False, "检索回的 id 与首跳不一致，持久化真实性不成立", observations
        return True, f"id 格式合规且可检索回同一对象（{rid[:24]}…），Responses 持久化真实性成立", observations

    async def case_auth_chat_not_translated(self, model: str):
        """AUTH-G12：Chat 协议未被改写（Responses 特征 = 网关翻译 = 信任级失败）。"""
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], max_completion_tokens=64)
        body = r.body or {}
        evidence = _chat_translation_evidence(body)
        observations = {"http_status": r.status, "object": body.get("object"), "id": str(body.get("id") or "")[:32],
                        "translation_evidence": evidence}
        if r.status != 200:
            return False, f"Chat 请求应成功（HTTP {r.status}）", observations, "blocked"
        if evidence:
            return False, f"Chat 响应被网关改写：{evidence}——参数语义将丢失（GPT56-31 stop 丢参即此因），信任级失败", observations
        if body.get("object") == "chat.completion" and str(body.get("id") or "").startswith("chatcmpl-"):
            return True, "Chat 响应严格符合 Chat 契约（object/id 形态正确），协议未被改写", observations
        return True, f"Chat 响应缺少契约标识字段（object={body.get('object')!r}），无法确证协议未被改写（留痕）", observations, "warning"


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    counts = {name: sum(r.status == name for r in results) for name in ("passed", "failed", "warning", "blocked", "error")}
    hard_failed = counts["failed"] + counts["error"]
    evaluated = counts["passed"] + hard_failed
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "total": len(results),
            "passed": counts["passed"], "failed": hard_failed, "warning": counts["warning"], "blocked": counts["blocked"],
            "error": counts["error"], "pass_rate": counts["passed"] / evaluated if evaluated else 0.0,
            "all_passed": hard_failed == 0, "results": [asdict(r) for r in results]}


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# GPT-5.6 专项测试报告", "", f"- 总计：{summary['total']}", f"- 通过：{summary['passed']}", f"- 失败：{summary['failed']}", f"- 警告：{summary['warning']}", f"- 阻塞/未观测：{summary['blocked']}", f"- 已评估用例通过率：{summary['pass_rate']:.1%}", "", "| 用例 | 模型 | 结果 | 耗时(s) | 说明 |", "|---|---|---:|---:|---|"]
    icons = {"passed": "✅ 通过", "failed": "❌ 失败", "error": "❌ 错误", "warning": "⚠️ 警告", "blocked": "⏸ 未验证"}
    for r in summary["results"]:
        detail = str(r["details"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r['case_id']} {r['name']} | {r['model']} | {icons.get(r['status'], r['status'])} | {r['duration_seconds']:.2f} | {detail} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
