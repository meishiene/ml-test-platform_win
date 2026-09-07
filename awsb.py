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
    upstream_request_id: str | None = None
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
    concurrency: int
    requests: str
    standard: str


# 用例按测试目标归类；子项在 handler 内执行并单独记录观察结果。
# 设计依据：用户提供的 O5 自查报告 + Anthropic Messages/Bedrock 官方协议约束。
CASE_SPECS = (
    AWSBCaseSpec("AWSB-O5-01", "基础协议与模型身份", "P0", "identity", 2, "4", "HTTP 200、返回非空文本，模型映射/身份字段符合预期；未知原生 ID 应明确拒绝。"),
    AWSBCaseSpec("AWSB-O5-02", "Thinking 与 effort 能力/约束", "P0", "thinking", 2, "9", "adaptive 的 low/medium/high/xhigh/max 可调用；disabled 与 high/xhigh/max、低 budget 的组合符合 O5 约束。"),
    AWSBCaseSpec("AWSB-O5-03", "参数边界与多数据验证", "P0", "boundaries", 2, "15", "3 组不同业务语义数据均完成边界探针；合法边界接受，越界/非法参数拒绝，不能只依据单条样本判定。"),
    AWSBCaseSpec("AWSB-O5-04", "流式、工具与多轮闭环", "P0", "stream_tools", 2, "3", "SSE 事件顺序完整并含 thinking/结束事件；tool_use 和 tool_result 多轮均成功。"),
)

CACHE_CASE_SPECS = (
    AWSBCaseSpec("AWSB-O5-CACHE", "Context Cache 命中专项", "P0", "cache", 1, "9", "3 组不同数据各执行种子+2次探测；有效缓存读取率目标 80%，允许误差 ±10 个百分点，低于 70% 或 usage 缺失则不通过。"),
)


def _blocks(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list((body or {}).get("content") or [])

def _text(body: dict[str, Any] | None) -> str:
    return "".join(str(x.get("text", "")) for x in _blocks(body) if x.get("type") == "text")

def _thinking(body: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [x for x in _blocks(body) if x.get("type") == "thinking"]

def _cache_usage(body: dict[str, Any] | None) -> tuple[int, int, int]:
    usage = (body or {}).get("usage") or {}
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    total = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    read = int(usage.get("cache_read_input_tokens") or details.get("cache_read_input_tokens") or details.get("cached_tokens") or 0)
    created = int(usage.get("cache_creation_input_tokens") or details.get("cache_creation_input_tokens") or 0)
    return total, read, created

class AWSBRunner:
    def __init__(self, config: AWSBConfig, client: httpx.AsyncClient | None = None):
        self.config, self.client, self._external = config, client, client is not None

    async def __aenter__(self) -> "AWSBRunner":
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.config.timeout, verify=self.config.verify_ssl,
                trust_env=self.config.trust_env, headers={"x-api-key": self.config.api_key,
                "Authorization": f"Bearer {self.config.api_key}", "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"})
        return self

    async def __aexit__(self, *args: Any) -> None:
        if not self._external and self.client:
            await self.client.aclose()

    async def request(self, payload: dict[str, Any], stream: bool = False) -> AWSBResponse:
        assert self.client is not None
        headers = {"X-Request-ID": f"AWSB-{uuid.uuid4().hex[:16]}"}
        started = time.perf_counter()
        if not stream:
            r = await self.client.post(self.config.base_url + self.config.messages_path, json=payload, headers=headers)
            try: body = r.json()
            except (ValueError, json.JSONDecodeError): body = None
            return AWSBResponse(r.status_code, body, r.text, time.perf_counter()-started, dict(r.headers), r.headers.get("x-request-id") or r.headers.get("request-id") or r.headers.get("x-trace-id") or r.headers.get("x-tt-logid"))
        events: list[dict[str, Any]] = []
        text = ""
        async with self.client.stream("POST", self.config.base_url + self.config.messages_path, json=payload, headers=headers) as r:
            raw = "" if r.status_code < 300 else (await r.aread()).decode("utf-8", "replace")
            if r.status_code < 300:
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]": continue
                        try:
                            obj = json.loads(data); events.append(obj)
                            delta = (obj.get("delta") or {})
                            if isinstance(delta.get("text"), str): text += delta["text"]
                        except json.JSONDecodeError: pass
            return AWSBResponse(r.status_code, None, raw, time.perf_counter()-started, dict(r.headers), r.headers.get("x-request-id") or r.headers.get("request-id") or r.headers.get("x-trace-id") or r.headers.get("x-tt-logid"), events)

    def payload(self, **kwargs: Any) -> dict[str, Any]:
        p = {"model": self.config.model, "max_tokens": 128, "messages": [{"role": "user", "content": "只回复 OK"}]}
        p.update(kwargs); return p

    async def case_identity(self):
        results = {}
        r = await self.request(self.payload(thinking={"type": "adaptive"}))
        results["alias"] = {"status": r.status, "text": bool(_text(r.body)), "response_model": (r.body or {}).get("model")}
        for model in ("anthropic.claude-opus-5", "global.anthropic.claude-opus-5", "us.anthropic.claude-opus-5"):
            rr = await self.request(self.payload(model=model, thinking={"type": "disabled"}))
            results[model] = rr.status
        ok = results["alias"]["status"] == 200 and results["alias"]["text"] and all(results[x] >= 400 for x in ("anthropic.claude-opus-5", "global.anthropic.claude-opus-5", "us.anthropic.claude-opus-5"))
        return ok, "基础调用成功且原生模型 ID 行为符合预期", results

    async def case_thinking(self):
        statuses, sizes = {}, {}
        prompt = "计算 918273*817263，并简要说明计算过程。"
        for effort in ("low", "medium", "high", "xhigh", "max"):
            r = await self.request(self.payload(messages=[{"role":"user","content":prompt}], max_tokens=1024, thinking={"type":"adaptive"}, output_config={"effort": effort}))
            statuses[effort] = r.status; sizes[effort] = len(_thinking(r.body))
        constraints = {}
        for label, extra in (("disabled_high", {"thinking":{"type":"disabled"},"output_config":{"effort":"high"}}), ("disabled_xhigh", {"thinking":{"type":"disabled"},"output_config":{"effort":"xhigh"}}), ("disabled_max", {"thinking":{"type":"disabled"},"output_config":{"effort":"max"}}), ("budget64", {"thinking":{"type":"enabled","budget_tokens":64}})):
            r = await self.request(self.payload(**extra)); constraints[label] = r.status
        ok = all(x == 200 for x in statuses.values()) and constraints["disabled_high"] == 200 and all(constraints[x] >= 400 for x in ("disabled_xhigh", "disabled_max", "budget64"))
        return ok, "adaptive 五档与 thinking 约束检查完成", {"effort_status": statuses, "thinking_blocks": sizes, "constraints": constraints}

    async def case_boundaries(self):
        # 三种不同语义数据：计算、代码、中文业务；每种都覆盖采样和输出边界。
        datasets = [
            ("math", "计算 37*83，并说明过程。"),
            ("code", "写一个 Python 函数判断字符串是否为回文，并只给出函数。"),
            ("business", "请将‘订单已支付，预计明天发货’改写成一句正式通知。"),
        ]
        observations = {}
        all_ok = True
        for name, prompt in datasets:
            checks = {}
            for label, extra, expected in (("temperature0", {"temperature":0}, False), ("temperature1", {"temperature":1}, True), ("top_p09", {"top_p":0.9}, False), ("max128000", {"max_tokens":128000}, True), ("max128001", {"max_tokens":128001}, False)):
                r = await self.request(self.payload(messages=[{"role":"user","content":prompt}], **extra))
                checks[label] = r.status
                accepted = r.status == 200
                all_ok = all_ok and (accepted == expected)
            observations[name] = checks
        return all_ok, "三种不同数据均完成参数边界验证", observations

    async def case_stream_tools(self):
        r = await self.request(self.payload(messages=[{"role":"user","content":"计算 37*83，并说明过程。"}], max_tokens=256, thinking={"type":"adaptive"}, output_config={"effort":"max"}, stream=True), stream=True)
        types = [e.get("type") for e in r.events]
        has_metrics = any("amazon-bedrock-invocationMetrics" in json.dumps(e) for e in r.events)
        stream_ok = r.status == 200 and "message_start" in types and "message_stop" in types and "thinking_delta" in types
        tool = {"name":"get_weather","description":"获取天气","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}
        r1 = await self.request(self.payload(messages=[{"role":"user","content":"调用天气工具查询上海。"}], tools=[tool], tool_choice={"type":"tool","name":"get_weather"}, thinking={"type":"disabled"}))
        calls = [x for x in _blocks(r1.body) if x.get("type") == "tool_use"]
        tool_ok = r1.status == 200 and bool(calls)
        second_status = None
        if tool_ok:
            assistant = [{k:v for k,v in x.items() if k != "caller"} for x in _blocks(r1.body)]
            r2 = await self.request(self.payload(messages=[{"role":"user","content":"调用天气工具查询上海。"},{"role":"assistant","content":assistant},{"role":"user","content":[{"type":"tool_result","tool_use_id":calls[0].get("id"),"content":"上海晴，25°C"}]}], tools=[tool], thinking={"type":"disabled"}))
            second_status = r2.status; tool_ok = tool_ok and r2.status == 200 and "25" in _text(r2.body)
        return stream_ok and tool_ok, "SSE、工具调用及多轮闭环检查完成", {"stream_event_types": types, "bedrock_metrics": has_metrics, "tool_first_status": r1.status, "tool_second_status": second_status}

    async def case_cache(self):
        datasets = [("math", "计算 37*83。"), ("code", "写一个 Python 回文判断函数。"), ("business", "把订单已支付改写成正式通知。")]
        observations = {}; ratios = []; valid = 0
        for name, question in datasets:
            prefix = f"AWSB-CACHE-{name}-" + ("固定公共上下文。" * 700)
            base_messages = [{"role":"user", "content":[{"type":"text","text":prefix,"cache_control":{"type":"ephemeral"}},{"type":"text","text":question}]}]
            sequence = []
            for label in ("seed", "probe1", "probe2"):
                r = await self.request(self.payload(messages=base_messages, max_tokens=128))
                total, read, created = _cache_usage(r.body)
                ratio = read / total if total else None
                sequence.append({"label":label,"status":r.status,"input_tokens":total,"cache_read_input_tokens":read,"cache_creation_input_tokens":created,"cache_ratio":ratio})
                if r.status == 200 and total > 0:
                    valid += 1
                    if label != "seed" and ratio is not None: ratios.append(ratio)
            observations[name] = sequence
        target = sum(ratios) / len(ratios) if ratios else 0
        passed = valid == len(datasets) * 3 and len(ratios) == len(datasets) * 2 and target >= 0.70
        return passed, f"缓存读取率均值 {target:.1%}（目标 80%，允许误差 ±10 个百分点）", {"target_ratio":0.80,"tolerance":0.10,"observed_probe_ratio_mean":target,"observations":observations}

    async def run(self, priorities=("P0", "P1"), case_ids: set[str] | None = None):
        out=[]
        for spec in (*CASE_SPECS, *CACHE_CASE_SPECS):
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
    lines += [f"| {x['case_id']} {x['name']} | {'PASS' if x['passed'] else 'FAIL'} | {x['details'].replace('|','\\|')} |" for x in summary['results']]
    (directory/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
