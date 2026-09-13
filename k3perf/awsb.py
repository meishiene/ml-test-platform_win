"""AWS Bedrock Claude Opus 5 上游/网关自检测试。

默认面向 Anthropic Messages 兼容接口（通常为 ``/v1/messages``），用于
验证一个对外 alias 是否确实具备 Claude Opus 5 / Bedrock 的行为特征。
本模块不创建 AWS 资源、不修改配额，只发送低成本请求。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from common.test_framework import HttpExecutor

MODEL = "claude-opus-5"

@dataclass(frozen=True)
class AWSBConfig:
    api_key: str
    base_url: str
    model: str = MODEL
    messages_path: str = "/v1/messages"
    timeout: float = 300.0
    verify_ssl: bool = True
    trust_env: bool = False

    @classmethod
    def from_env(cls) -> "AWSBConfig":
        key = os.getenv("AWSB_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("缺少 AWSB_API_KEY（或 ANTHROPIC_API_KEY）")
        base = (os.getenv("AWSB_BASE_URL") or "https://api.example.com").rstrip("/")
        return cls(key, base, os.getenv("AWSB_MODEL", MODEL), os.getenv("AWSB_MESSAGES_PATH", "/v1/messages"),
                   float(os.getenv("AWSB_REQUEST_TIMEOUT_SECONDS", "300")),
                   os.getenv("AWSB_VERIFY_SSL", "true").lower() not in {"0", "false", "no"},
                   os.getenv("AWSB_TRUST_ENV", "false").lower() in {"1", "true", "yes"})

@dataclass
class AWSBResponse:
    status: int
    body: dict[str, Any] | None
    text: str
    elapsed: float
    headers: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class AWSBCaseResult:
    case_id: str
    name: str
    model: str
    passed: bool
    details: str
    duration_seconds: float
    observations: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AWSBCaseSpec:
    case_id: str
    name: str
    priority: str
    handler: str

CASE_SPECS = (
    AWSBCaseSpec("AWSB-O5-01", "基础调用与模型映射", "P0", "basic"),
    AWSBCaseSpec("AWSB-O5-02", "Adaptive thinking 与 effort", "P0", "effort"),
    AWSBCaseSpec("AWSB-O5-03", "关闭 thinking 的 effort 约束", "P0", "thinking_constraints"),
    AWSBCaseSpec("AWSB-O5-04", "参数边界与输出上限", "P0", "parameter_limits"),
    AWSBCaseSpec("AWSB-O5-05", "Anthropic SSE 流式响应", "P0", "stream"),
    AWSBCaseSpec("AWSB-O5-06", "工具调用与多轮闭环", "P0", "tools"),
    AWSBCaseSpec("AWSB-O5-07", "未知模型 ID 的渠道行为", "P1", "model_ids"),
)

def _blocks(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list((body or {}).get("content") or [])

def _text(body: dict[str, Any] | None) -> str:
    return "".join(str(x.get("text", "")) for x in _blocks(body) if x.get("type") == "text")

def _thinking(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [x for x in _blocks(body) if x.get("type") == "thinking"]

class AWSBRunner:
    def __init__(self, config: AWSBConfig, client: httpx.AsyncClient | None = None):
        self.config, self.client, self._external = config, client, client is not None
        self.executor: HttpExecutor | None = None

    async def __aenter__(self) -> "AWSBRunner":
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.config.timeout, verify=self.config.verify_ssl,
                trust_env=self.config.trust_env, headers={"x-api-key": self.config.api_key,
                "Authorization": f"Bearer {self.config.api_key}", "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"})
        self.executor = HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
        await self.executor.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.executor: await self.executor.__aexit__(*args)
        if not self._external and self.client: await self.client.aclose()

    async def request(self, payload: dict[str, Any], stream: bool = False) -> AWSBResponse:
        if self.executor is None:
            if self.client is None: raise RuntimeError("runner is not entered")
            self.executor=HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
            await self.executor.__aenter__()
        headers={"X-Request-ID":f"K3-AWSB-{uuid.uuid4().hex[:16]}","x-api-key":self.config.api_key,"anthropic-version":"2023-06-01"}
        record=await self.executor.execute(case_id=payload.get("_case_id","K3-AWSB-request"),url=self.config.base_url+self.config.messages_path,payload={k:v for k,v in payload.items() if k != "_case_id"},headers=headers,stream=stream)
        if not stream: return AWSBResponse(record.http_status or 0,record.response_body if isinstance(record.response_body,dict) else None,record.response_text,record.timing.get("e2e_ms",0)/1000,record.response_headers)
        return AWSBResponse(record.http_status or 0,None,record.raw_response_body or record.response_text,record.timing.get("e2e_ms",0)/1000,record.response_headers,record.parsed_sse_events)

    def payload(self, **kwargs: Any) -> dict[str, Any]:
        p = {"model": self.config.model, "max_tokens": 128, "messages": [{"role": "user", "content": "只回复 OK"}]}
        p.update(kwargs); return p

    async def case_basic(self):
        r = await self.request(self.payload(thinking={"type": "disabled"}))
        model = (r.body or {}).get("model")
        return r.status == 200 and bool(_text(r.body)), "HTTP %s" % r.status, {"response_model": model}

    async def case_effort(self):
        statuses, sizes = {}, {}
        prompt = "计算 918273*817263，并简要说明计算过程。"
        for effort in ("low", "medium", "high", "xhigh", "max"):
            r = await self.request(self.payload(messages=[{"role":"user","content":prompt}], max_tokens=1024,
                thinking={"type":"adaptive"}, output_config={"effort": effort}))
            statuses[effort] = r.status; sizes[effort] = sum(len(str(x.get("thinking", ""))) for x in _thinking(r.body))
        return all(x == 200 for x in statuses.values()), "五档 effort 均应可调用", {"statuses": statuses, "thinking_chars": sizes}

    async def case_thinking_constraints(self):
        results = {}
        for label, extra in (("high", {"thinking":{"type":"disabled"},"output_config":{"effort":"high"}}),
                             ("xhigh", {"thinking":{"type":"disabled"},"output_config":{"effort":"xhigh"}}),
                             ("max", {"thinking":{"type":"disabled"},"output_config":{"effort":"max"}}),
                             ("budget64", {"thinking":{"type":"enabled","budget_tokens":64}})):
            r = await self.request(self.payload(**extra)); results[label] = r.status
        return results["high"] == 200 and all(results[x] >= 400 for x in ("xhigh", "max", "budget64")), "thinking/effort 约束符合 O5 预期", results

    async def case_parameter_limits(self):
        results = {}
        for label, extra in (("temperature0", {"temperature":0}), ("temperature1", {"temperature":1}),
                             ("top_p09", {"top_p":0.9}), ("max128000", {"max_tokens":128000}), ("max128001", {"max_tokens":128001})):
            r = await self.request(self.payload(**extra)); results[label] = r.status
        ok = (results["temperature0"] >= 400 and results["temperature1"] == 200
              and results["top_p09"] >= 400 and results["max128000"] == 200
              and results["max128001"] >= 400)
        return ok, "检查采样参数与 128000 输出上限", results

    async def case_stream(self):
        r = await self.request(self.payload(messages=[{"role":"user","content":"计算 37*83，并说明过程。"}], max_tokens=256,
            thinking={"type":"adaptive"}, output_config={"effort":"max"}, stream=True), stream=True)
        types = [e.get("type") for e in r.events]
        has_metrics = any("amazon-bedrock-invocationMetrics" in json.dumps(e) for e in r.events)
        ok = r.status == 200 and "message_start" in types and "message_stop" in types and any("thinking_delta" == e.get("type") for e in r.events)
        return ok, "SSE 应包含 thinking 与结束事件", {"event_types": types, "bedrock_metrics": has_metrics}

    async def case_tools(self):
        tool = {"name":"get_weather","description":"获取天气","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}
        r1 = await self.request(self.payload(messages=[{"role":"user","content":"调用天气工具查询上海。"}], tools=[tool], tool_choice={"type":"tool","name":"get_weather"}, thinking={"type":"disabled"}))
        calls = [x for x in _blocks(r1.body) if x.get("type") == "tool_use"]
        if r1.status != 200 or not calls: return False, "应返回 tool_use", {"status":r1.status}
        assistant = [{k:v for k,v in x.items() if k != "caller"} for x in _blocks(r1.body)]
        r2 = await self.request(self.payload(messages=[{"role":"user","content":"调用天气工具查询上海。"},{"role":"assistant","content":assistant},{"role":"user","content":[{"type":"tool_result","tool_use_id":calls[0].get("id"),"content":"上海晴，25°C"}]}], tools=[tool], thinking={"type":"disabled"}))
        return r2.status == 200 and "25" in _text(r2.body), "tool_use/tool_result 多轮闭环", {"first_status":r1.status,"second_status":r2.status}

    async def case_model_ids(self):
        results = {}
        for model in ("anthropic.claude-opus-5", "global.anthropic.claude-opus-5", "us.anthropic.claude-opus-5"):
            p = self.payload(model=model, thinking={"type":"disabled"}); r = await self.request(p); results[model] = r.status
        return all(x >= 400 for x in results.values()), "原生模型 ID 的渠道行为应有明确结果", results

    async def run(self, priorities=("P0", "P1"), case_ids: set[str] | None = None):
        out=[]
        for spec in CASE_SPECS:
            if spec.priority not in priorities or (case_ids and spec.case_id not in case_ids): continue
            started=time.perf_counter()
            try: passed, details, obs = await getattr(self, "case_" + spec.handler)()
            except Exception as exc: passed, details, obs = False, f"{type(exc).__name__}: {exc}", {}
            out.append(AWSBCaseResult(spec.case_id,spec.name,self.config.model,passed,details,time.perf_counter()-started,obs))
        return out

def summarize(results):
    passed=sum(x.passed for x in results)
    return {"model": results[0].model if results else None, "total":len(results), "passed":passed, "failed":len(results)-passed, "all_passed":passed==len(results), "results":[asdict(x) for x in results]}

def write_report(directory: Path, summary: dict[str, Any]):
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines=["# AWSB Claude Opus 5 测试报告", "", f"模型：`{summary['model']}`", f"通过：{summary['passed']}/{summary['total']}", "", "| 用例 | 结果 | 说明 |", "|---|---|---|"]
    for x in summary["results"]:
        details = x["details"].replace("|", "\\|")
        status = "PASS" if x["passed"] else "FAIL"
        lines.append(f"| {x['case_id']} {x['name']} | {status} | {details} |")
    (directory/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
