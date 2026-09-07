#!/usr/bin/env python3
"""运行 KVV/DeepSeek/GLM 共用契约探针；模型差异由 common.model_contracts.PROFILES 控制。"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from pathlib import Path
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.model_contracts import PROFILES, load_profile_data, validate_identity, validate_usage, text, message

async def main():
    ap=argparse.ArgumentParser(description="跨模型公共契约探针")
    ap.add_argument("family", choices=PROFILES); ap.add_argument("--model"); ap.add_argument("--base-url"); ap.add_argument("--api-key"); ap.add_argument("--output-root",type=Path,default=Path("results"))
    a=ap.parse_args(); env={"kvv":("VOLC_BASE_URL","VOLC_API_KEY","VOLC_MODEL"),"deepseek":("DS_BASE_URL","DS_API_KEY","DS_MODEL"),"glm":("GLM_BASE_URL","GLM_API_KEY","GLM_MODEL")}[a.family]
    base=a.base_url or os.getenv(env[0]); key=a.api_key or os.getenv(env[1]); model=a.model or os.getenv(env[2])
    if not base or not key or not model: raise SystemExit(f"缺少 {env[0]}/{env[1]}/{env[2]}")
    profile=PROFILES[a.family]; data=load_profile_data(a.family); results=[]
    out=a.output_root/f"{os.getenv('CONTRACT_CASE') or f'CONTRACT-{a.family.upper()}'}-{time.strftime('%Y%m%d-%H%M%S')}"; out.mkdir(parents=True,exist_ok=True)
    audit_path=out/'requests.jsonl'; request_seq=0
    def audit(x):
      with audit_path.open('a',encoding='utf-8') as f: f.write(json.dumps(x,ensure_ascii=False,default=str)+'\n')
    expected_tokens=tuple(data.get("model_tokens", profile.expected_model_tokens))
    marker=data.get("smoke_marker", profile.answer_marker)
    thinking_on=data.get("thinking_on", profile.thinking_on or {})
    thinking_off=data.get("thinking_off", profile.thinking_off or {})
    tool_name=data.get("tool_name", profile.tool_name)
    supports_reasoning=bool(data.get("supports_reasoning", profile.supports_reasoning))
    async with httpx.AsyncClient(timeout=180,headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}) as c:
      async def req(payload, stream=False):
        nonlocal request_seq
        request_seq += 1; rid=f"CONTRACT-{request_seq:03d}-{os.urandom(4).hex()}"; started=time.perf_counter()
        rec={"sequence":request_seq,"request_id":rid,"endpoint":base.rstrip('/')+'/chat/completions',
             "method":"POST","stream":stream,"timeout_seconds":180,
             "request_headers":{"Content-Type":"application/json","X-Request-ID":rid},"request":payload}
        if not stream:
          r=await c.post(base.rstrip('/')+'/chat/completions',json=payload,headers={"X-Request-ID":rid})
          try: b=r.json()
          except ValueError: b=None
          rec.update({"http_status":r.status_code,"response_headers":dict(r.headers),"upstream_request_id":r.headers.get("x-request-id") or r.headers.get("request-id") or r.headers.get("x-trace-id") or r.headers.get("x-tt-logid"),"response_object_id":(b or {}).get("id") if isinstance(b,dict) else None,"response":b,"elapsed_seconds":time.perf_counter()-started}); audit(rec)
          return r.status_code,b,{}
        chunks=[];reasoning=[];done=False;bad=0;raw_lines=[];raw_events=[];finishes=[]
        async with c.stream('POST',base.rstrip('/')+'/chat/completions',json=payload,headers={"X-Request-ID":rid}) as r:
          rec.update({"http_status":r.status_code,"response_headers":dict(r.headers),"upstream_request_id":r.headers.get("x-request-id") or r.headers.get("request-id") or r.headers.get("x-trace-id") or r.headers.get("x-tt-logid")})
          if r.status_code>=300:
            raw=await r.aread(); rec.update({"raw_response":raw.decode(errors='replace'),"elapsed_seconds":time.perf_counter()-started}); audit(rec)
            return r.status_code,None,{"error":rec["raw_response"][:300]}
          async for line in r.aiter_lines():
            raw_lines.append(line)
            if not line.startswith('data:'): continue
            raw=line[5:].strip();raw_events.append(raw)
            if raw=='[DONE]': done=True;continue
            try:o=json.loads(raw)
            except ValueError: bad+=1;continue
            for ch in o.get('choices') or []:
              d=ch.get('delta') or {}; chunks.append(d.get('content') or '');reasoning.append(d.get('reasoning_content') or '')
              if 'finish_reason' in ch: finishes.append(ch.get('finish_reason'))
        parsed={"content":''.join(chunks),"reasoning_content":''.join(reasoning),"done":done,"invalid_events":bad,"finish_reasons":finishes}
        rec.update({"raw_sse_lines":raw_lines,"raw_sse_events":raw_events,"parsed":parsed,"elapsed_seconds":time.perf_counter()-started});audit(rec)
        return r.status_code,None,parsed
      def payload(messages, **kw): return {"model":model,"messages":messages,"max_tokens":2048,**kw}
      st,b,o=await req(payload([{"role":"user","content":f"只回复 {marker}"}],**(thinking_off or {}))); ok,obs=validate_identity(st,b,expected_tokens); results.append({"id":"CONTRACT-01","name":"身份与路由","status":"passed" if ok else "failed","details":f"HTTP {st}","observations":obs})
      st,b,o=await req(payload([{"role":"user","content":"只回复 OK"}],**(thinking_off or {}),temperature=2.1)); ok=400<=st<500; results.append({"id":"CONTRACT-02","name":"参数错误契约","status":"passed" if ok else "failed","details":f"非法 temperature HTTP {st}","observations":{}})
      st,b,o=await req(payload([{"role":"user","content":"计算 37*83，并简要说明。"}],**(thinking_on or {}),stream=True),True); ok=st==200 and o.get('done') and o.get('invalid_events')==0 and bool(o.get('content')); results.append({"id":"CONTRACT-03","name":"流式协议完整性","status":"passed" if ok else "failed","details":f"HTTP {st}，DONE={o.get('done')}","observations":o})
      tool={"type":"function","function":{"name":tool_name,"description":"查询订单","parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"],"additionalProperties":False}}}
      st,b,o=await req(payload([{"role":"user","content":"查询订单 ORD-000731，必须调用工具。"}],tools=[tool],tool_choice="required",**(thinking_off or {}))); calls=message(b).get('tool_calls') or []; ok=st==200 and bool(calls) and calls[0].get('function',{}).get('name')==tool_name; results.append({"id":"CONTRACT-04","name":"工具调用多轮闭环","status":"passed" if ok else "failed","details":f"HTTP {st}，tool_calls={len(calls)}","observations":{"tool_calls":calls[:1]}})
      if a.family in {"deepseek", "kvv", "glm"}:
        efforts=data.get("reasoning_efforts", ["low", "medium", "high"]); statuses={}; chars={}
        for effort in efforts:
          stx,bx,ox=await req(payload([{"role":"user","content":"请多步推理，计算 918273*817263，并给出结果。"}],thinking={"type":"enabled"},reasoning_effort=effort,max_tokens=2048)); raw=message(bx).get('reasoning_content') or message(bx).get('reasoning') or ''; statuses[effort]=stx; chars[effort]=len(raw)
        ok=all(v==200 for v in statuses.values()) and all(v>0 for v in chars.values()); results.append({"id":"CONTRACT-05","name":"Thinking 与 reasoning_effort","status":"passed" if ok else "failed","details":f"档位={efforts}，HTTP={statuses}，reasoning 字符={chars}","observations":{"statuses":statuses,"reasoning_chars":chars}})

    summary={"family":a.family,"model":model,"total":len(results),"passed":sum(x['status']=='passed' for x in results),"failed":sum(x['status']=='failed' for x in results),"warning":sum(x['status']=='warning' for x in results),"results":results};(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2));print(f"结果：{out.resolve()}");return 0 if not summary['failed'] else 2
if __name__=='__main__': raise SystemExit(asyncio.run(main()))
