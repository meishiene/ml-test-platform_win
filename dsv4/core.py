"""DeepSeek V4 Pro/Flash 专项兼容性测试。

测试依据 DeepSeek V4 官方 API 文档，使用原始 HTTP 请求，避免 SDK 对扩展字段的过滤。
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

MODELS = ("deepseek-v4-pro", "deepseek-v4-flash")


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    beta_base_url: str | None = None
    anthropic_base_url: str | None = None
    timeout: float = 300.0
    verify_ssl: bool = True
    trust_env: bool = False

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        key = os.getenv("DS_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("缺少 DS_API_KEY（或 DEEPSEEK_API_KEY）")
        base = (os.getenv("DS_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        return cls(
            api_key=key,
            base_url=base,
            beta_base_url=os.getenv("DS_BETA_BASE_URL"),
            anthropic_base_url=os.getenv("DS_ANTHROPIC_BASE_URL"),
            timeout=float(os.getenv("DS_REQUEST_TIMEOUT_SECONDS", "300")),
            verify_ssl=os.getenv("DS_VERIFY_SSL", "true").lower() not in {"0", "false", "no"},
            trust_env=os.getenv("DS_TRUST_ENV", "false").lower() in {"1", "true", "yes"},
        )

    def url(self, path: str, family: str = "openai") -> str:
        base = self.base_url.rstrip("/")
        if family == "beta":
            root = (self.beta_base_url or f"{base}/beta").rstrip("/")
        elif family == "anthropic":
            root = (self.anthropic_base_url or f"{base}/anthropic").rstrip("/")
        else:
            root = base
        return f"{root}/{path.lstrip('/')}"


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
    beta: bool = False
    destructive: bool = False


CASE_SPECS = (
    CaseSpec("DS-V4-01", "模型别名可调用", "P0", "basic"),
    CaseSpec("DS-V4-02", "默认思考模式", "P0", "default_thinking"),
    CaseSpec("DS-V4-03", "关闭思考模式", "P0", "disabled_thinking"),
    CaseSpec("DS-V4-04", "思考强度映射", "P0", "effort_mapping"),
    CaseSpec("DS-V4-05", "思考模式兼容参数静默接受", "P0", "ignored_sampling"),
    CaseSpec("DS-V4-06", "流式思考与正文分离", "P0", "stream_thinking"),
    CaseSpec("DS-V4-07", "无工具多轮回传思考", "P0", "replay_without_tools"),
    CaseSpec("DS-V4-08", "思考模式工具调用完整回传", "P0", "tool_replay"),
    CaseSpec("DS-V4-09", "工具调用漏传 reasoning_content 返回 400", "P0", "tool_missing_reasoning"),
    CaseSpec("DS-V4-10", "未进入工具链时 reasoning 回传行为", "P0", "tools_plain_turn_replay"),
    CaseSpec("DS-V4-11", "JSON Output", "P0", "json_output"),
    CaseSpec("DS-V4-12", "JSON Prompt 缺少关键字时可控失败", "P0", "json_missing_keyword"),
    CaseSpec("DS-V4-13", "JSON 小输出截断可检测", "P0", "json_truncation"),
    CaseSpec("DS-V4-14", "自动 Context Cache 命中", "P0", "cache_hit"),
    CaseSpec("DS-V4-15", "Context Cache 完整前缀规则", "P0", "cache_prefix_rule"),
    CaseSpec("DS-V4-16", "user_id 缓存隔离", "P0", "user_isolation"),
    CaseSpec("DS-V4-17", "非法 user_id 线上行为观察", "P0", "invalid_user_id"),
    CaseSpec("DS-V4-18", "同身份多轮 Session 连续性与缓存命中", "P0", "session_continuity_cache"),
    CaseSpec("DS-V4-19", "Strict Tool Call", "P1", "strict_tool", beta=True),
    CaseSpec("DS-V4-20", "Strict 非法 Schema 拒绝", "P1", "strict_invalid", beta=True),
    CaseSpec("DS-V4-21", "Chat Prefix Completion", "P1", "prefix", beta=True),
    CaseSpec("DS-V4-22", "Chat Prefix 参数约束", "P1", "prefix_invalid", beta=True),
    CaseSpec("DS-V4-23", "FIM Completion", "P1", "fim", beta=True),
    CaseSpec("DS-V4-24", "Anthropic 模型映射", "P1", "anthropic_mapping"),
    CaseSpec("DS-V4-25", "Responses API 无状态字段", "P1", "responses_stateless"),
    CaseSpec("DS-V4-26", "非流式 Usage 完整性", "P2", "usage_nonstream"),
    CaseSpec("DS-V4-27", "流式 Usage 与结束事件", "P2", "usage_stream"),
    CaseSpec("DS-V4-28", "Stop 停止序列", "P2", "stop_sequence"),
    CaseSpec("DS-V4-29", "Temperature 合法边界", "P2", "temperature_bounds"),
    CaseSpec("DS-V4-30", "Temperature 越界拒绝", "P2", "temperature_invalid"),
    CaseSpec("DS-V4-31", "非法 Messages 拒绝", "P2", "messages_invalid"),
    CaseSpec("DS-V4-32", "Tool Choice none", "P2", "tool_choice_none"),
    CaseSpec("DS-V4-33", "Tool Choice required", "P2", "tool_choice_required"),
    CaseSpec("DS-V4-34", "Responses API 基础响应", "P2", "responses_basic"),
    CaseSpec("DS-V4-35", "Anthropic API 基础响应", "P2", "anthropic_basic"),
)


def _message(body: dict[str, Any] | None) -> dict[str, Any]:
    choices = (body or {}).get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def _usage(body: dict[str, Any] | None) -> dict[str, Any]:
    return (body or {}).get("usage") or {}


def _cache_tokens(body: dict[str, Any] | None) -> int:
    usage = _usage(body)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    return int(usage.get("prompt_cache_hit_tokens") or details.get("cached_tokens") or 0)


class DeepSeekV4Runner:
    def __init__(self, config: DeepSeekConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._external_client = client
        self.client = client

    async def __aenter__(self) -> "DeepSeekV4Runner":
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                trust_env=self.config.trust_env,
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._external_client is None and self.client:
            await self.client.aclose()

    async def request(self, path: str, payload: dict[str, Any], *, family: str = "openai", stream: bool = False, headers: dict[str, str] | None = None) -> ApiResponse:
        assert self.client is not None
        started = time.perf_counter()
        request_headers = {"X-Request-ID": f"DSV4-{uuid.uuid4().hex[:16]}"}
        if family == "anthropic":
            request_headers.update({"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"})
        if headers:
            request_headers.update(headers)
        if not stream:
            response = await self.client.post(self.config.url(path, family), json=payload, headers=request_headers)
            text = response.text
            try:
                body = response.json()
            except (ValueError, json.JSONDecodeError):
                body = None
            return ApiResponse(response.status_code, body, text, time.perf_counter() - started, dict(response.headers), response.headers.get("x-request-id") or response.headers.get("request-id") or response.headers.get("x-trace-id") or response.headers.get("x-tt-logid"))

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        invalid_events = 0
        done = False
        usage: dict[str, Any] = {}
        finish_reasons: list[str] = []
        status = 0
        raw_error = ""
        async with self.client.stream("POST", self.config.url(path, family), json=payload, headers=request_headers) as response:
            status = response.status_code
            if status >= 300:
                raw_error = (await response.aread()).decode("utf-8", "replace")
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        invalid_events += 1
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices") or []:
                        if choice.get("finish_reason") is not None:
                            finish_reasons.append(str(choice["finish_reason"]))
                        delta = choice.get("delta") or {}
                        if isinstance(delta.get("reasoning_content"), str):
                            reasoning_parts.append(delta["reasoning_content"])
                        if isinstance(delta.get("content"), str):
                            content_parts.append(delta["content"])
        info = {"reasoning_content": "".join(reasoning_parts), "content": "".join(content_parts), "usage": usage, "done": done, "invalid_events": invalid_events, "finish_reasons": finish_reasons}
        return ApiResponse(status, None, raw_error, time.perf_counter() - started, dict(response.headers) if "response" in locals() else {}, (response.headers.get("x-request-id") or response.headers.get("request-id") or response.headers.get("x-trace-id") or response.headers.get("x-tt-logid")) if "response" in locals() else None, stream=info)

    async def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ApiResponse:
        payload = {"model": model, "messages": messages, "max_tokens": 2048, **kwargs}
        family = payload.pop("_family", "openai")
        return await self.request("chat/completions", payload, family=family, stream=bool(payload.get("stream")))

    async def run(self, models: tuple[str, ...] = MODELS, priorities: tuple[str, ...] = ("P0", "P1"), case_ids: set[str] | None = None) -> list[CaseResult]:
        results: list[CaseResult] = []
        for model in models:
            for spec in CASE_SPECS:
                if spec.priority not in priorities or (case_ids and spec.case_id not in case_ids):
                    continue
                started = time.perf_counter()
                try:
                    outcome = await getattr(self, f"case_{spec.handler}")(model)
                    if len(outcome) == 4:
                        passed, details, observations, status = outcome
                    else:
                        passed, details, observations = outcome
                        status = "passed" if passed else "failed"
                    results.append(CaseResult(spec.case_id, spec.name, model, passed, status, details, time.perf_counter() - started, observations))
                except Exception as exc:
                    results.append(CaseResult(spec.case_id, spec.name, model, False, "error", f"{type(exc).__name__}: {exc}", time.perf_counter() - started))
        return results

    async def case_basic(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"})
        return r.status == 200 and bool(_message(r.body).get("content")), f"HTTP {r.status}", {"response_model": (r.body or {}).get("model")}

    async def case_default_thinking(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "计算 17*23，只给最终答案。"}])
        rc = _message(r.body).get("reasoning_content")
        return r.status == 200 and isinstance(rc, str) and bool(rc), "默认请求应返回 reasoning_content", {"reasoning_chars": len(rc or "")}

    async def case_disabled_thinking(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"})
        rc = _message(r.body).get("reasoning_content")
        return r.status == 200 and not rc, "关闭思考后 reasoning_content 应为空", {}

    async def case_effort_mapping(self, model: str):
        statuses = {}
        for effort in ("low", "medium", "high", "xhigh", "max"):
            r = await self.chat(model, [{"role": "user", "content": "计算 19*31，只给答案。"}], thinking={"type": "enabled"}, reasoning_effort=effort, max_tokens=2048)
            statuses[effort] = r.status
        return all(v == 200 for v in statuses.values()), "五种 effort 均应被接受", statuses

    async def case_ignored_sampling(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "enabled"}, temperature=1.7, top_p=0.2, presence_penalty=1.0, frequency_penalty=1.0)
        return r.status == 200, "思考模式下兼容参数应静默接受", {"http_status": r.status}

    async def case_stream_thinking(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "计算 29*13，只给最终答案。"}], thinking={"type": "enabled"}, stream=True, stream_options={"include_usage": True})
        s = r.stream or {}
        ok = r.status == 200 and bool(s.get("reasoning_content")) and bool(s.get("content")) and s.get("done") and s.get("invalid_events") == 0
        return ok, "SSE 中 reasoning/content 应可独立拼接且收到 [DONE]", {k: len(v) if isinstance(v, str) else v for k, v in s.items() if k != "usage"}

    async def case_replay_without_tools(self, model: str):
        first = await self.chat(model, [{"role": "user", "content": "记住数字 731，并回复收到。"}])
        assistant = _message(first.body)
        second = await self.chat(model, [{"role": "user", "content": "记住数字 731，并回复收到。"}, assistant, {"role": "user", "content": "刚才的数字是什么？只回复数字。"}])
        return first.status == second.status == 200 and "731" in str(_message(second.body).get("content")), "无 tools 时回传 assistant 消息应成功", {}

    @staticmethod
    def tools() -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "get_number", "description": "返回测试数字", "parameters": {"type": "object", "properties": {}, "required": []}}}]

    async def _tool_first(self, model: str) -> tuple[ApiResponse, dict[str, Any]]:
        r = await self.chat(model, [{"role": "user", "content": "必须调用 get_number 获取测试数字，不要自行回答。"}], tools=self.tools(), tool_choice="required", thinking={"type": "enabled"})
        return r, _message(r.body)

    async def case_tool_replay(self, model: str):
        first, msg = await self._tool_first(model)
        calls = msg.get("tool_calls") or []
        if first.status != 200 or not calls or not msg.get("reasoning_content"):
            observations = {"http_status": first.status, "has_tool_calls": bool(calls), "has_reasoning": bool(msg.get("reasoning_content")), "response": first.text[:500]}
            return False, "前置工具调用未同时产生 reasoning_content 与 tool_calls，后续回传链路无法验证", observations, "blocked"
        messages = [{"role": "user", "content": "必须调用 get_number 获取测试数字，不要自行回答。"}, msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "731"}]
        second = await self.chat(model, messages, tools=self.tools(), thinking={"type": "enabled"})
        return second.status == 200 and "731" in str(_message(second.body).get("content")), "完整回传后应生成工具结果答案", {"second_status": second.status}

    async def case_tool_missing_reasoning(self, model: str):
        first, msg = await self._tool_first(model)
        calls = msg.get("tool_calls") or []
        if first.status != 200 or not calls or not msg.get("reasoning_content"):
            observations = {"http_status": first.status, "has_tool_calls": bool(calls), "has_reasoning": bool(msg.get("reasoning_content")), "response": first.text[:500]}
            return False, "前置工具调用未同时产生 reasoning_content 与 tool_calls，本负向用例不计失败", observations, "blocked"
        msg = dict(msg); msg.pop("reasoning_content", None)
        second = await self.chat(model, [{"role": "user", "content": "必须调用 get_number。"}, msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "731"}], tools=self.tools(), thinking={"type": "enabled"})
        return second.status == 400, "漏传 reasoning_content 应返回 400", {"http_status": second.status, "error": second.text[:300]}

    async def case_tools_plain_turn_replay(self, model: str):
        first = await self.chat(model, [{"role": "user", "content": "不要调用工具，只回复 OK。"}], tools=self.tools(), tool_choice="none", thinking={"type": "enabled"})
        msg = _message(first.body)
        if first.status != 200 or not msg.get("reasoning_content"):
            return False, "无法建立携带 tools 但未调用工具的前置轮次", {"http_status": first.status, "response": first.text[:500]}, "blocked"
        without_reasoning = dict(msg); without_reasoning.pop("reasoning_content", None)
        second = await self.chat(model, [{"role": "user", "content": "不要调用工具，只回复 OK。"}, without_reasoning, {"role": "user", "content": "再回复一次 OK。"}], tools=self.tools(), tool_choice="none", thinking={"type": "enabled"})
        observations = {"http_status": second.status, "response": second.text[:300]}
        if second.status == 400:
            return True, "未进入工具调用链时仍执行 reasoning_content 严格校验", observations
        if second.status == 200:
            return True, "未进入工具调用链时允许省略 reasoning_content（官方线上可接受行为）", observations, "warning"
        return False, "未进入工具调用链时返回非预期状态", observations

    async def case_json_output(self, model: str):
        r = await self.chat(model, [{"role": "system", "content": '请仅输出 json，例如 {"value":731}'}, {"role": "user", "content": "以 json 输出 value=731"}], response_format={"type": "json_object"}, thinking={"type": "disabled"})
        text = _message(r.body).get("content") or ""
        try: parsed = json.loads(text)
        except json.JSONDecodeError: parsed = None
        return r.status == 200 and isinstance(parsed, dict) and parsed.get("value") == 731, "响应必须是可解析 JSON", {"content": text[:300]}

    async def case_json_missing_keyword(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "返回一个包含 value=731 的对象。"}], response_format={"type": "json_object"}, thinking={"type": "disabled"}, max_tokens=32)
        # 官方要求 Prompt 包含 json；实现可以明确拒绝，也可能返回空或有效结构，关键是请求可终止且可分类。
        controlled = r.status == 400 or (r.status == 200 and isinstance(_message(r.body).get("content"), str))
        return controlled, "缺少 json 关键字时必须得到可分类响应，不能协议异常", {"http_status": r.status, "content": str(_message(r.body).get("content"))[:200]}

    async def case_json_truncation(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": '输出 json，包含键 items，值为 100 个整数。'}], response_format={"type": "json_object"}, thinking={"type": "disabled"}, max_tokens=2)
        text = _message(r.body).get("content") or ""
        finish = (((r.body or {}).get("choices") or [{}])[0]).get("finish_reason")
        detectable = r.status == 200 and (finish == "length" or not text or _invalid_json(text))
        return detectable, "小 max_tokens 导致的截断/空内容必须可检测", {"finish_reason": finish, "content": text}

    async def case_cache_hit(self, model: str):
        prefix = "缓存专项固定前缀-" + ("A1B2C3D4。" * 800)
        first = await self.chat(model, [{"role": "user", "content": prefix + "只回复一。"}], thinking={"type": "disabled"}, max_tokens=8)
        await asyncio.sleep(2)
        second = await self.chat(model, [{"role": "user", "content": prefix + "只回复一。"}, _message(first.body), {"role": "user", "content": "只回复二。"}], thinking={"type": "disabled"}, max_tokens=8)
        hit = _cache_tokens(second.body)
        return first.status == second.status == 200 and hit > 0, "重复完整前缀后应报告缓存命中", {"cache_hit_tokens": hit}

    async def case_cache_prefix_rule(self, model: str):
        common = "公共缓存前缀-" + ("PREFIX731。" * 700)
        a = await self.chat(model, [{"role": "user", "content": common + "分支A"}], thinking={"type": "disabled"}, max_tokens=8)
        await asyncio.sleep(2)
        b = await self.chat(model, [{"role": "user", "content": common + "分支B"}], thinking={"type": "disabled"}, max_tokens=8)
        await asyncio.sleep(2)
        c = await self.chat(model, [{"role": "user", "content": common + "分支C"}], thinking={"type": "disabled"}, max_tokens=8)
        hit_b, hit_c = _cache_tokens(b.body), _cache_tokens(c.body)
        return a.status == b.status == c.status == 200 and hit_c > 0, "公共前缀持久化后第三次请求应可命中（best-effort）", {"second_hit": hit_b, "third_hit": hit_c}

    async def case_user_isolation(self, model: str):
        prefix = "隔离测试固定前缀-" + ("XYZ731。" * 600)
        uid1, uid2 = f"ds-test-{uuid.uuid4().hex[:8]}", f"ds-test-{uuid.uuid4().hex[:8]}"
        await self.chat(model, [{"role": "user", "content": prefix}], user_id=uid1, thinking={"type": "disabled"}, max_tokens=8)
        await asyncio.sleep(2)
        same = await self.chat(model, [{"role": "user", "content": prefix}], user_id=uid1, thinking={"type": "disabled"}, max_tokens=8)
        other = await self.chat(model, [{"role": "user", "content": prefix}], user_id=uid2, thinking={"type": "disabled"}, max_tokens=8)
        same_hit, other_hit = _cache_tokens(same.body), _cache_tokens(other.body)
        observations = {"same_user_hit": same_hit, "other_user_hit": other_hit}
        if same.status != 200 or other.status != 200:
            return False, "user_id 缓存隔离请求失败", {**observations, "same_status": same.status, "other_status": other.status}
        if same_hit == 0:
            return True, "本轮未观测到同 user_id 缓存命中，无法判定隔离（缓存为 best-effort）", observations, "blocked"
        if other_hit > 0:
            return False, "已观测同 user_id 命中，但不同 user_id 也发生命中", observations
        return True, "相同 user_id 命中且不同 user_id 未命中", observations

    async def case_session_continuity_cache(self, model: str):
        """同一身份按会话追加消息，验证上下文连续性与前缀缓存是否同时成立。"""
        session_id = f"ds-session-{uuid.uuid4().hex[:10]}"
        prefix = "SESSION-CONTINUITY-DSV4-" + ("固定会话背景。" * 700)
        turns = [
            [{"role": "system", "content": prefix}, {"role": "user", "content": "记住会话暗号：ORBIT-731。只回复已记住。"}],
            [{"role": "system", "content": prefix}, {"role": "user", "content": "记住会话暗号：ORBIT-731。只回复已记住。"}, {"role": "assistant", "content": "已记住。"}, {"role": "user", "content": "会话暗号是什么？只回复暗号。"}],
            [{"role": "system", "content": prefix}, {"role": "user", "content": "记住会话暗号：ORBIT-731。只回复已记住。"}, {"role": "assistant", "content": "已记住。"}, {"role": "user", "content": "会话暗号是什么？只回复暗号。"}, {"role": "assistant", "content": "ORBIT-731"}, {"role": "user", "content": "再次确认暗号，只回复暗号。"}],
        ]
        rows = []
        for i, messages in enumerate(turns):
            r = await self.chat(model, messages, user_id=session_id, thinking={"type": "disabled"}, max_tokens=16)
            text = _message(r.body).get("content") or ""
            rows.append({"turn": i + 1, "status": r.status, "content": text[:100], "cache_hit_tokens": _cache_tokens(r.body), "prompt_tokens": _usage(r.body).get("prompt_tokens")})
            if i < len(turns) - 1:
                await asyncio.sleep(2)
        continuity = rows[1]["status"] == rows[2]["status"] == 200 and all("ORBIT-731" in rows[i]["content"] for i in (1, 2))
        cache_hits = [r["cache_hit_tokens"] for r in rows[1:]]
        cache_observed = any(x > 0 for x in cache_hits)
        ok = all(r["status"] == 200 for r in rows) and continuity and cache_observed
        status = "passed" if ok else ("blocked" if continuity and not cache_observed else "failed")
        return ok, f"Session 连续性={'通过' if continuity else '失败'}，后续轮缓存命中={cache_hits}", {"session_id": session_id, "turns": rows, "continuity": continuity, "cache_observed": cache_observed}, status

    async def case_invalid_user_id(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], user_id="非法 user id!", thinking={"type": "disabled"})
        observations = {"http_status": r.status, "response": r.text[:300]}
        if 400 <= r.status < 500:
            return True, "接口拒绝非法 user_id", observations
        if r.status == 200:
            return True, "接口接受非法 user_id；记录为文档/线上行为差异，不判模型失败", observations, "warning"
        return False, "非法 user_id 返回非预期状态", observations

    async def case_strict_tool(self, model: str):
        tool = {"type": "function", "function": {"name": "create_user", "strict": True, "description": "创建测试用户", "parameters": {"type": "object", "properties": {"email": {"type": "string", "format": "email"}, "score": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["email", "score"], "additionalProperties": False}}}
        r = await self.chat(model, [{"role": "user", "content": "调用 create_user，邮箱 a@example.com，评分 5。"}], _family="beta", tools=[tool], tool_choice={"type": "function", "function": {"name": "create_user"}}, thinking={"type": "disabled"})
        calls = _message(r.body).get("tool_calls") or []
        try: args = json.loads(calls[0]["function"]["arguments"])
        except (IndexError, KeyError, json.JSONDecodeError): args = {}
        return r.status == 200 and args == {"email": "a@example.com", "score": 5}, "Strict 参数必须严格符合 Schema", {"arguments": args}

    async def case_strict_invalid(self, model: str):
        tool = {"type": "function", "function": {"name": "bad", "strict": True, "description": "非法测试", "parameters": {"type": "object", "properties": {"x": {"type": "string", "minLength": 2}}, "required": [], "additionalProperties": True}}}
        r = await self.chat(model, [{"role": "user", "content": "调用 bad"}], _family="beta", tools=[tool], thinking={"type": "disabled"})
        return 400 <= r.status < 500, "不支持的 Strict Schema 应返回 4xx", {"http_status": r.status, "error": r.text[:300]}

    async def case_prefix(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "写一个 Python 加法函数，只输出代码。"}, {"role": "assistant", "content": "def add(a, b):\n", "prefix": True}], _family="beta", thinking={"type": "disabled"}, stop=["```"], max_tokens=64)
        text = _message(r.body).get("content") or ""
        return r.status == 200 and "return" in text and not text.startswith("def add"), "应从 assistant 前缀后继续生成", {"content": text[:300]}

    async def case_prefix_invalid(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "写代码", "prefix": True}], _family="beta", thinking={"type": "disabled"}, max_tokens=32)
        return 400 <= r.status < 500, "prefix=true 不在末尾 assistant 消息上时应返回 4xx", {"http_status": r.status, "error": r.text[:300]}

    async def case_fim(self, model: str):
        payload = {"model": model, "prompt": "def fib(n):\n", "suffix": "\n    return fib(n-1) + fib(n-2)", "max_tokens": 128}
        r = await self.request("completions", payload, family="beta")
        text = (((r.body or {}).get("choices") or [{}])[0]).get("text") or ""
        return r.status == 200 and bool(text), "Beta FIM 应返回中间补全文本", {"content": text[:300]}

    async def case_anthropic_mapping(self, model: str):
        alias = "claude-opus-4-test" if model.endswith("pro") else "claude-sonnet-4-test"
        payload = {"model": alias, "max_tokens": 32, "messages": [{"role": "user", "content": "只回复 OK"}]}
        r = await self.request("v1/messages", payload, family="anthropic")
        response_model = (r.body or {}).get("model")
        expected = "pro" if model.endswith("pro") else "flash"
        return r.status == 200 and expected in str(response_model).lower(), "Claude 别名应映射到目标 DeepSeek 型号", {"requested": alias, "response_model": response_model}

    async def case_responses_stateless(self, model: str):
        payload = {"model": model, "input": "只回复 OK", "store": True, "previous_response_id": "resp_nonexistent", "max_output_tokens": 32}
        r = await self.request("responses", payload)
        body = r.body or {}
        return r.status == 200 and body.get("store") is False and body.get("previous_response_id") is None, "无状态字段应返回固定值", {"store": body.get("store"), "previous_response_id": body.get("previous_response_id")}



    async def case_usage_nonstream(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"})
        usage = _usage(r.body)
        prompt, completion, total = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
        ok = r.status == 200 and all(isinstance(v, int) and v >= 0 for v in (prompt, completion, total)) and total == prompt + completion
        return ok, "Usage 应包含非负 prompt/completion/total，且 total 等于两者之和", usage

    async def case_usage_stream(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, stream=True, stream_options={"include_usage": True})
        info, usage = r.stream or {}, (r.stream or {}).get("usage") or {}
        ok = r.status == 200 and info.get("done") and info.get("invalid_events") == 0 and isinstance(usage.get("total_tokens"), int) and bool(info.get("finish_reasons"))
        return ok, "流式响应应有合法 SSE、[DONE]、finish_reason 与 Usage", {"usage": usage, "finish_reasons": info.get("finish_reasons"), "done": info.get("done")}

    async def case_stop_sequence(self, model: str):
        marker = "<END>"
        r = await self.chat(model, [{"role": "user", "content": f"严格输出 ABC{marker}XYZ"}], thinking={"type": "disabled"}, stop=[marker], max_tokens=32)
        content = str(_message(r.body).get("content") or "")
        return r.status == 200 and marker not in content and "XYZ" not in content, "Stop 命中后不得输出停止串及其后内容", {"content": content}

    async def case_temperature_bounds(self, model: str):
        statuses = []
        for value in (0.0, 2.0):
            r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, temperature=value, max_tokens=16)
            statuses.append(r.status)
        return all(status == 200 for status in statuses), "temperature=0 与 2 均应接受", {"statuses": statuses}

    async def case_temperature_invalid(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "只回复 OK"}], thinking={"type": "disabled"}, temperature=2.1, max_tokens=16)
        return 400 <= r.status < 500, "temperature 超出 [0,2] 应返回 4xx", {"http_status": r.status, "error": r.text[:300]}

    async def case_messages_invalid(self, model: str):
        r = await self.request("chat/completions", {"model": model, "messages": "not-an-array", "max_tokens": 16})
        return 400 <= r.status < 500, "messages 非数组应返回 4xx", {"http_status": r.status, "error": r.text[:300]}

    async def case_tool_choice_none(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "不要调用工具，只回复 OK"}], tools=self.tools(), tool_choice="none", thinking={"type": "disabled"})
        calls = _message(r.body).get("tool_calls") or []
        return r.status == 200 and not calls, "tool_choice=none 不得返回工具调用", {"tool_calls": calls}

    async def case_tool_choice_required(self, model: str):
        r = await self.chat(model, [{"role": "user", "content": "调用 get_number"}], tools=self.tools(), tool_choice="required", thinking={"type": "disabled"})
        calls = _message(r.body).get("tool_calls") or []
        return r.status == 200 and bool(calls) and calls[0].get("function", {}).get("name") == "get_number", "tool_choice=required 必须调用声明的工具", {"tool_calls": calls[:1]}

    async def case_responses_basic(self, model: str):
        r = await self.request("responses", {"model": model, "input": "只回复 OK", "max_output_tokens": 32})
        body = r.body or {}
        output = body.get("output")
        return r.status == 200 and isinstance(body.get("id"), str) and isinstance(output, list), "Responses API 应返回 id 与 output 数组", {"id": body.get("id"), "output_items": len(output or [])}

    async def case_anthropic_basic(self, model: str):
        payload = {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": "只回复 OK"}]}
        r = await self.request("v1/messages", payload, family="anthropic", headers={"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"})
        body = r.body or {}
        return r.status == 200 and body.get("role") == "assistant" and isinstance(body.get("content"), list), "Anthropic Messages 应返回 assistant 与 content 数组", {"role": body.get("role"), "model": body.get("model")}


def _invalid_json(text: str) -> bool:
    try: json.loads(text); return False
    except json.JSONDecodeError: return True


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
    lines = ["# DeepSeek V4 专项测试报告", "", f"- 总计：{summary['total']}", f"- 通过：{summary['passed']}", f"- 失败：{summary['failed']}", f"- 警告：{summary['warning']}", f"- 阻塞/未观测：{summary['blocked']}", f"- 已评估用例通过率：{summary['pass_rate']:.1%}", "", "| 用例 | 模型 | 结果 | 耗时(s) | 说明 |", "|---|---|---:|---:|---|"]
    icons = {"passed": "✅ 通过", "failed": "❌ 失败", "error": "❌ 错误", "warning": "⚠️ 警告", "blocked": "⏸ 未验证"}
    for r in summary["results"]:
        detail = str(r["details"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r['case_id']} {r['name']} | {r['model']} | {icons.get(r['status'], r['status'])} | {r['duration_seconds']:.2f} | {detail} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
