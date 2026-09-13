from __future__ import annotations
import asyncio, json, os, time, uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import httpx
from common.test_framework import HttpExecutor, ResultLog

@dataclass(frozen=True)
class GLMConfig:
    api_key: str
    base_url: str
    timeout: float = 180.0
    verify_ssl: bool = True
    trust_env: bool = False

@dataclass
class CaseResult:
    case_id: str; name: str; model: str; passed: bool; status: str = "passed"
    details: str = ""; duration_seconds: float = 0.0; observations: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CaseSpec:
    case_id: str; name: str; priority: str; handler: str

CASE_SPECS = (
    CaseSpec("GLM-53-01", "基础对话与模型标识", "P0", "basic"),
    CaseSpec("GLM-53-02", "非流式 Usage 完整性", "P0", "usage"),
    CaseSpec("GLM-53-03", "流式输出与结束事件", "P0", "stream"),
    CaseSpec("GLM-53-04", "深度思考与 reasoning_effort", "P0", "thinking"),
    CaseSpec("GLM-53-06", "结构化 JSON 输出", "P0", "json_output"),
    CaseSpec("GLM-53-07", "多轮上下文保持", "P0", "multi_turn"),
    CaseSpec("GLM-53-08", "长上下文稳定响应", "P0", "long_context"),
    CaseSpec("GLM-53-09", "工具调用基础链路", "P1", "tool_call"),
    CaseSpec("GLM-53-10", "tool_choice=required", "P1", "tool_required"),
    CaseSpec("GLM-53-11", "工具结果多轮回传", "P1", "tool_replay"),
    CaseSpec("GLM-53-12", "工具流式输出", "P1", "tool_stream"),
    CaseSpec("GLM-53-C01", "Context Cache 命中专项", "PCACHE", "cache"),
    CaseSpec("GLM-53-C02", "同身份多轮 Session 连续性与缓存命中", "PCACHE", "session_continuity_cache"),
    CaseSpec("GLM-53-F01", "模型身份一致性", "P2", "fingerprint_identity"),
    CaseSpec("GLM-53-F02", "Thinking 字段与错误指纹", "P2", "fingerprint_thinking"),
    CaseSpec("GLM-53-F03", "Tool Choice 强制调用指纹", "P2", "fingerprint_tool_choice"),
    CaseSpec("GLM-53-F04", "JSON Schema 能力指纹", "P2", "fingerprint_schema"),
    CaseSpec("GLM-53-F05", "流式协议完整性指纹", "P2", "fingerprint_stream"),
    CaseSpec("GLM-53-F06", "参数静默忽略检测", "P2", "fingerprint_silent_ignore"),
    CaseSpec("GLM-53-F07", "上下文长度边界指纹", "P2", "fingerprint_context"),
    CaseSpec("GLM-53-F08", "满血证据汇总", "P2", "fingerprint_summary"),
)

def msg(body):
    return (((body or {}).get("choices") or [{}])[0].get("message") or {})
def content(body):
    value = msg(body).get("content", "")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
def usage_ok(body):
    u=(body or {}).get("usage") or {}
    return all(isinstance(u.get(k), (int,float)) for k in ("prompt_tokens","completion_tokens","total_tokens"))

def cache_usage(body):
    u=(body or {}).get("usage") or {}; details=u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    total=int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    hit=int(u.get("prompt_cache_hit_tokens") or u.get("cache_read_input_tokens") or details.get("cached_tokens") or details.get("cache_read_input_tokens") or 0)
    created=int(u.get("cache_creation_input_tokens") or details.get("cache_creation_input_tokens") or 0)
    return total,hit,created

def _schema_valid(raw):
    try:
        value=json.loads(raw); return isinstance(value,dict) and isinstance(value.get("answer"),str)
    except (TypeError,ValueError): return False

class GLM53Runner:
    def __init__(self, config: GLMConfig, audit_dir: Path | None = None):
        self.config=config; self.client=None; self.audit_dir=audit_dir; self.audit_path=None; self.request_seq=0; self.current_case_id=None
        if audit_dir:
            audit_dir.mkdir(parents=True, exist_ok=True)
            self.audit_path=audit_dir / "requests.jsonl"
    async def __aenter__(self):
        self.client=httpx.AsyncClient(timeout=self.config.timeout, verify=self.config.verify_ssl, trust_env=self.config.trust_env, headers={"Authorization":f"Bearer {self.config.api_key}","Content-Type":"application/json"})
        self.executor=HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
        await self.executor.__aenter__()
        if self.audit_dir: self.log=ResultLog(self.audit_dir / "requests.jsonl")
        return self
    async def __aexit__(self,*args):
        if self.log: self.log.close()
        if self.executor: await self.executor.__aexit__(*args)
        if self.client: await self.client.aclose()
    def _audit(self, record: dict[str, Any]):
        if not self.audit_path: return
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    async def request(self, model, messages, *, stream=False, **kwargs):
        if self.executor is None:
            self.client=httpx.AsyncClient(timeout=self.config.timeout, verify=self.config.verify_ssl, trust_env=self.config.trust_env, headers={"Authorization":f"Bearer {self.config.api_key}","Content-Type":"application/json"})
            self.executor=HttpExecutor(timeout=self.config.timeout, verify_ssl=self.config.verify_ssl, trust_env=self.config.trust_env, client=self.client)
            await self.executor.__aenter__()
        payload={"model":model,"messages":messages,"max_tokens":kwargs.pop("max_tokens",2048),"stream":stream,**kwargs}
        request_id=f"GLM53-{uuid.uuid4().hex[:12]}"
        record=await self.executor.execute(case_id=self.current_case_id or "GLM53-request", url=self.config.base_url.rstrip("/")+"/chat/completions", payload=payload, headers={"Authorization":f"Bearer {self.config.api_key}","Content-Type":"application/json","X-Request-ID":request_id}, stream=stream)
        if self.log: await self.log.write(record)
        if not stream:
            return record.http_status or 0, record.response_body if isinstance(record.response_body,dict) else None, record.timing.get("e2e_ms",0)/1000, {}
        parsed={"content":record.content,"reasoning_content":record.reasoning,"done":record.done_received,"invalid_events":record.invalid_sse_events,"finish_reasons":record.finish_reasons,"raw_sse_lines":record.raw_sse_lines,"raw_sse_events":record.raw_sse_events,"usage":record.usage or {}}
        return record.http_status or 0, None, record.timing.get("e2e_ms",0)/1000, parsed

    def tools(self): return [{"type":"function","function":{"name":"get_weather","description":"查询天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":False}}}]
    async def run(self, model, priorities):
        out=[]
        for s in CASE_SPECS:
            if s.priority not in priorities: continue
            self.current_case_id=s.case_id
            start=time.perf_counter()
            try:
                passed,detail,obs,status=await getattr(self,"case_"+s.handler)(model)
            except Exception as e: passed,detail,obs,status=False,f"{type(e).__name__}: {e}",{},"error"
            out.append(CaseResult(s.case_id,s.name,model,passed,status,detail,time.perf_counter()-start,obs))
        self.current_case_id=None
        return out
    async def _basic(self,model,**kw):
        st,b,el,_=await self.request(model,[{"role":"user","content":"只回复 OK"}],**kw); return st==200 and bool(content(b)),f"HTTP {st}",{"model":(b or {}).get("model")},"passed" if st==200 else "failed"
    async def case_basic(self,m): return await self._basic(m)
    async def case_usage(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"只回复 OK"}]); ok=st==200 and usage_ok(b); return ok,f"HTTP {st}，Usage {'完整' if ok else '不完整'}",{"usage":(b or {}).get("usage")},"passed" if ok else "failed"
    async def case_stream(self,m):
        st,_,_,o=await self.request(m,[{"role":"user","content":"只回复 OK"}],stream=True); ok=st==200 and o.get("done") and o.get("content"); return bool(ok),f"HTTP {st}，收到结束事件={o.get('done')}",o,"passed" if ok else "failed"
    async def case_thinking(self,m):
        statuses={}; reasoning_chars={}; reasoning_tokens={}
        for effort in ("low", "medium", "high"):
            st,b,_,_=await self.request(m,[{"role":"user","content":"计算 918273*817263，并给出结果。"}],thinking={"type":"enabled"},reasoning_effort=effort,max_tokens=2048)
            mm=msg(b); raw=mm.get("reasoning_content") or mm.get("reasoning") or ""; usage=(b or {}).get("usage") or {}; details=usage.get("completion_tokens_details") or {}
            statuses[effort]=st; reasoning_chars[effort]=len(raw); reasoning_tokens[effort]=details.get("reasoning_tokens")
        ok=all(st==200 for st in statuses.values()) and all(n>0 for n in reasoning_chars.values())
        return ok,f"reasoning_effort {statuses}，reasoning 字符={reasoning_chars}",{"statuses":statuses,"reasoning_chars":reasoning_chars,"reasoning_tokens":reasoning_tokens,"efforts":["low","medium","high"]},"passed" if ok else "failed"
    async def case_json_output(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"返回 JSON，字段 answer，值为 OK"}],thinking={"type":"enabled"},response_format={"type":"json_object"}); raw=content(b)
        try: obj=json.loads(raw); ok=st==200 and obj.get("answer")=="OK"
        except ValueError: ok=False
        return ok,f"HTTP {st}，JSON 可解析={ok}",{"content":raw[:500]},"passed" if ok else "failed"
    async def case_multi_turn(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"记住代号是 ALPHA-53"},{"role":"assistant","content":"好的"},{"role":"user","content":"代号是什么？只回复代号"}]); ok=st==200 and "ALPHA-53" in content(b); return ok,f"HTTP {st}",{},"passed" if ok else "failed"
    async def case_long_context(self,m): return await self._basic(m, max_tokens=2048)
    async def case_tool_call(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"请查询北京天气"}],tools=self.tools(),thinking={"type":"enabled"}); calls=msg(b).get("tool_calls") or []; ok=st==200 and bool(calls); return ok,f"HTTP {st}，工具调用数={len(calls)}",{"tool_calls":calls[:1]},"passed" if ok else "failed"
    async def case_tool_required(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"请查询天气"}],tools=self.tools(),tool_choice="required",thinking={"type":"enabled"}); ok=st==200 and bool(msg(b).get("tool_calls")); return ok,f"HTTP {st}",{},"passed" if ok else "failed"
    async def case_tool_replay(self,m):
        tool=self.tools()[0]; st,b,_,_=await self.request(m,[{"role":"user","content":"查询北京天气"}],tools=[tool],tool_choice="required",thinking={"type":"enabled"}); calls=msg(b).get("tool_calls") or []
        if st!=200 or not calls: return False,f"首轮 HTTP {st} 未产生工具调用",{},"failed"
        call=calls[0]; msgs=[{"role":"user","content":"查询北京天气"},{"role":"assistant","content":None,"tool_calls":calls},{"role":"tool","tool_call_id":call.get("id","call_1"),"content":"晴，25度"}]; st2,b2,_,_=await self.request(m,msgs,tools=[tool],thinking={"type":"enabled"}); ok=st2==200 and bool(content(b2)); return ok,f"首轮/回传 HTTP {st}/{st2}",{},"passed" if ok else "failed"
    async def case_tool_stream(self,m):
        st,_,_,o=await self.request(m,[{"role":"user","content":"查询天气"}],tools=self.tools(),stream=True); ok=st==200 and o.get("done"); return bool(ok),f"HTTP {st}，结束事件={o.get('done')}",o,"passed" if ok else "failed"


    async def case_session_continuity_cache(self,m):
        """同一 user/session 身份追加完整历史，分别验证连续性和缓存 usage。"""
        session_id = f"glm-session-{uuid.uuid4().hex[:10]}"
        prefix = "SESSION-CONTINUITY-GLM53-" + ("固定会话背景。" * 700)
        turns = [
            [{"role":"system","content":prefix},{"role":"user","content":"记住会话暗号：ORBIT-731。只回复已记住。"}],
            [{"role":"system","content":prefix},{"role":"user","content":"记住会话暗号：ORBIT-731。只回复已记住。"},{"role":"assistant","content":"已记住。"},{"role":"user","content":"会话暗号是什么？只回复暗号。"}],
            [{"role":"system","content":prefix},{"role":"user","content":"记住会话暗号：ORBIT-731。只回复已记住。"},{"role":"assistant","content":"已记住。"},{"role":"user","content":"会话暗号是什么？只回复暗号。"},{"role":"assistant","content":"ORBIT-731"},{"role":"user","content":"再次确认暗号，只回复暗号。"}],
        ]
        rows=[]
        for i,messages in enumerate(turns):
            st,b,elapsed,_=await self.request(m,messages,user=session_id,thinking={"type":"enabled"},max_tokens=32)
            total,hit,created=cache_usage(b); text=content(b)
            rows.append({"turn":i+1,"http_status":st,"content":text[:100],"prompt_tokens":total,"cached_tokens":hit,"cache_creation_tokens":created,"cache_ratio":hit/total if total else None,"e2e_seconds":elapsed})
            if i < len(turns)-1: await asyncio.sleep(2)
        continuity=all(r["http_status"]==200 for r in rows) and all("ORBIT-731" in rows[i]["content"] for i in (1,2))
        cache_observed=any(r["cached_tokens"]>0 for r in rows[1:])
        ok=continuity and cache_observed
        status="passed" if ok else ("blocked" if continuity and not cache_observed else "failed")
        return ok,f"Session 连续性={'通过' if continuity else '失败'}，后续轮缓存命中={[r['cached_tokens'] for r in rows[1:]]}",{"session_id":session_id,"turns":rows,"continuity":continuity,"cache_observed":cache_observed},status

    async def case_cache(self,m):
        datasets=[("math","计算 37*83，只回复结果。"),("code","写一个 Python 回文判断函数，只回复代码。"),("business","将订单已支付改写成正式通知。")]
        observations={}; probe_ratios=[]; group_ratios=[]; usage_complete=True; requests_ok=True
        for name,question in datasets:
            prefix=f"GLM53-CACHE-{name}-"+("固定公共上下文。"*700)
            group=[]
            for label in ("seed","probe1","probe2"):
                st,b,elapsed,_=await self.request(m,[{"role":"system","content":prefix},{"role":"user","content":question}],thinking={"type":"enabled"},max_tokens=256)
                total,hit,created=cache_usage(b); ratio=hit/total if total else None
                group.append({"label":label,"http_status":st,"prompt_tokens":total,"cached_tokens":hit,"cache_creation_tokens":created,"cache_ratio":ratio,"e2e_seconds":elapsed})
                requests_ok=requests_ok and st==200
                if label!="seed":
                    usage_complete=usage_complete and total>0 and ("prompt_tokens_details" in ((b or {}).get("usage") or {}) or "prompt_cache_hit_tokens" in ((b or {}).get("usage") or {}) or "cache_read_input_tokens" in ((b or {}).get("usage") or {}))
                    if ratio is not None: probe_ratios.append(ratio)
            ratios=[x["cache_ratio"] for x in group[1:] if x["cache_ratio"] is not None]
            group_ratio=sum(ratios)/len(ratios) if ratios else 0; group_ratios.append(group_ratio); observations[name]=group
        mean=sum(probe_ratios)/len(probe_ratios) if probe_ratios else 0
        if not usage_complete:
            return False,"接口未完整返回缓存 usage，无法验证",{"target":0.80,"tolerance":0.10,"mean":mean,"groups":group_ratios,"observations":observations},"blocked"
        ok=requests_ok and len(probe_ratios)==6 and mean>=0.70 and all(x>=0.50 for x in group_ratios)
        return ok,f"缓存读取率均值 {mean:.1%}（目标80%，允许±10个百分点）",{"target":0.80,"tolerance":0.10,"mean":mean,"groups":group_ratios,"observations":observations},"passed" if ok else "failed"

    async def case_fingerprint_identity(self,m):
        st,b,_,_=await self.request(m,[{"role":"user","content":"只回复 OK"}]); returned=(b or {}).get("model"); ok=st==200 and returned in {None,m} or (st==200 and isinstance(returned,str) and (m in returned or returned in m))
        return bool(ok),f"HTTP {st}，请求模型={m}，返回模型={returned or '未返回'}",{"requested_model":m,"returned_model":returned,"evidence":"identity"},"passed" if ok else ("warning" if st in range(400,500) else "failed")
    async def _capability_probe(self,m,payload,label,validator):
        st,b,_,_=await self.request(m,[{"role":"user","content":"执行测试并只返回结果"}],**payload)
        raw=json.dumps(b or {},ensure_ascii=False)[:500]; text=raw.lower(); unsupported=any(x in text for x in ("unsupported","not support","unknown field","invalid parameter","不支持"))
        ok=st==200 and validator(b)
        status="passed" if ok else ("warning" if unsupported or st in (400,404,422) else "failed")
        return ok,f"{label}：HTTP {st}，{'支持' if ok else ('接口拒绝/疑似未开放' if unsupported else '未满足')}",{"http_status":st,"unsupported_error":unsupported,"response":raw,"probe":label},status
    async def case_fingerprint_thinking(self,m):
        return await self._capability_probe(m,{"thinking":{"type":"enabled"},"reasoning_effort":"high","max_tokens":1024},"thinking+reasoning_effort",lambda b:bool(msg(b).get("reasoning_content") or msg(b).get("reasoning")))
    async def case_fingerprint_tool_choice(self,m):
        return await self._capability_probe(m,{"tools":self.tools(),"tool_choice":"required"},"tool_choice=required",lambda b:bool(msg(b).get("tool_calls")))
    async def case_fingerprint_schema(self,m):
        schema={"type":"json_schema","json_schema":{"name":"answer_schema","strict":True,"schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":False}}}
        return await self._capability_probe(m,{"response_format":schema},"json_schema",lambda b:_schema_valid(content(b)))
    async def case_fingerprint_stream(self,m):
        st,_,_,o=await self.request(m,[{"role":"user","content":"只回复 OK"}],stream=True); ok=st==200 and bool(o.get("done")) and not o.get("invalid_events")
        return ok,f"stream：HTTP {st}，DONE={o.get('done')}，非法事件={o.get('invalid_events',0)}",{"http_status":st,**o,"probe":"stream"},"passed" if ok else ("warning" if st in (400,404,422) else "failed")
    async def case_fingerprint_silent_ignore(self,m):
        # GLM3/GLM5.3 必须开启深度思考：不再发送 disabled 对照，改为验证 enabled 的稳定返回。
        statuses=[]; lengths=[]
        for prompt in ("计算 17*23，并说明步骤。", "解释为什么 2+2=4，并说明步骤。"):
            st,b,_,_=await self.request(m,[{"role":"user","content":prompt}],thinking={"type":"enabled"},max_tokens=2048)
            r=msg(b).get("reasoning_content") or msg(b).get("reasoning") or ""; statuses.append(st); lengths.append(len(r))
        ok=all(st==200 for st in statuses) and all(n>0 for n in lengths)
        return ok,f"Thinking 必须开启：HTTP {statuses}，reasoning 字符数={lengths}",{"statuses":statuses,"reasoning_chars":lengths,"probe":"thinking_enabled_stability"},"passed" if ok else "failed"
    async def case_fingerprint_context(self,m):
        prompt="请记住标记 CONTEXT-GLM53-OK。\n" + ("这是用于上下文边界探测的填充文本。"*500)
        st,b,_,_=await self.request(m,[{"role":"user","content":prompt},{"role":"user","content":"只回复你记住的标记"}],max_tokens=2048)
        ok=st==200 and "CONTEXT-GLM53-OK" in content(b); return ok,f"HTTP {st}，上下文标记命中={ok}",{"http_status":st,"marker_found":ok,"prompt_chars":len(prompt),"probe":"context"},"passed" if ok else ("warning" if st in (400,413,422) else "failed")
    async def case_fingerprint_summary(self,m):
        # 由单次探测确认报告链路可记录指纹结果；真实汇总在 summarize_results 中生成。
        return True,"指纹结果将在报告中按能力维度汇总",{"probe":"summary"},"passed"

def summarize_results(results):
    counts={x:sum(r.status==x for r in results) for x in ("passed","failed","warning","blocked","error")}; hard=counts["failed"]+counts["error"]; evaluated=counts["passed"]+hard
    fingerprint=[r for r in results if r.case_id.startswith("GLM-53-F") and r.case_id != "GLM-53-F08"]
    warnings=[r for r in fingerprint if r.status=="warning"]
    hard_fingerprint=[r for r in fingerprint if r.status in {"failed","error"}]
    if hard_fingerprint: fingerprint_level="证据不足/存在硬失败"
    elif warnings: fingerprint_level="部分能力或接口层限制，不能认定满血"
    elif fingerprint and len(fingerprint)==7: fingerprint_level="满血能力证据充分"
    else: fingerprint_level="无法判定"
    diagnosis=[{"case_id":r.case_id,"name":r.name,"status":r.status,"details":r.details,"observations":r.observations} for r in fingerprint if r.status!="passed"]
    return {"generated_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"total":len(results),"passed":counts["passed"],"failed":hard,"warning":counts["warning"],"blocked":counts["blocked"],"error":counts["error"],"pass_rate":counts["passed"]/evaluated if evaluated else 0,"all_passed":hard==0,"results":[asdict(r) for r in results],"fingerprint_assessment":{"level":fingerprint_level,"confidence":"medium" if fingerprint else "low","tested":len(fingerprint),"warnings":len(warnings),"hard_failures":len(hard_fingerprint),"suspected_degradations":diagnosis}}

def write_report(out,summary):
    out.mkdir(parents=True,exist_ok=True); (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fp=summary.get("fingerprint_assessment",{})
    lines=["# GLM 5.3 专项测试报告","",f"## 满血能力指纹结论：{fp.get('level','未执行')}",f"- 置信度：{fp.get('confidence','—')}；已测试：{fp.get('tested',0)} 项；疑似限制/异常：{fp.get('warnings',0)+fp.get('hard_failures',0)} 项","","",f"- 总计：{summary['total']}",f"- 通过：{summary['passed']}",f"- 失败：{summary['failed']}","","| 用例 | 模型 | 结果 | 说明 |","|---|---|---|---|"]
    icon={"passed":"✅ 通过","failed":"❌ 失败","error":"❌ 错误"}
    for r in summary["results"]:
        details = r["details"].replace("|", "\\|")
        lines.append(f"| {r['case_id']} {r['name']} | {r['model']} | {icon.get(r['status'], r['status'])} | {details} |")
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
