"""跨模型黑盒真实性/满血能力指纹探测。结果只能表示与官方行为的一致性，不能证明权重来源。"""
from __future__ import annotations
import asyncio, json, os, secrets, time, uuid
from pathlib import Path
import httpx

CASES={
 "DS_AUTH_FINGERPRINT": ("DeepSeek 满血能力指纹", "DeepSeek thinking、工具、JSON、流式、动态绑定与静默吞参探测", "deepseek"),
 "DS_V4_FINGERPRINT": ("DeepSeek V4 满血能力指纹", "DeepSeek Thinking、工具、JSON、流式、动态绑定与静默吞参探测", "deepseek"),
}

def token(): return secrets.token_hex(4).upper()
def msg(b): return (((b or {}).get('choices') or [{}])[0].get('message') or {})
def text(b):
 v=msg(b).get('content',''); return v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)
def unsupported(body):
 s=json.dumps(body or {},ensure_ascii=False).lower(); return any(x in s for x in ('unsupported','not support','unknown field','invalid parameter','不支持'))

async def run(case, base, key, model, out):
 rng=token(); headers={'Authorization':f'Bearer {key}','Content-Type':'application/json','X-Request-ID':f'AUTH-{uuid.uuid4().hex}'}
 results=[]
 async with httpx.AsyncClient(timeout=180,headers=headers) as c:
  async def req(payload, stream=False):
   if not stream:
    r=await c.post(base.rstrip('/')+'/chat/completions',json=payload)
    try:b=r.json()
    except ValueError:b=None
    return r.status_code,b,dict(r.headers)
   chunks=[]; done=False; bad=0
   async with c.stream('POST',base.rstrip('/')+'/chat/completions',json=payload) as r:
    if r.status_code>=300:return r.status_code,{'error':(await r.aread()).decode(errors='replace')[:500],'response_headers':dict(r.headers),'upstream_request_id':r.headers.get('x-request-id') or r.headers.get('request-id') or r.headers.get('x-trace-id') or r.headers.get('x-tt-logid')},dict(r.headers)
    async for line in r.aiter_lines():
     if not line.startswith('data:'):continue
     raw=line[5:].strip()
     if raw=='[DONE]':done=True;continue
     try:o=json.loads(raw)
     except ValueError:bad+=1;continue
     for ch in o.get('choices') or []: chunks.append((ch.get('delta') or {}).get('content') or '')
   return r.status_code,{'content':''.join(chunks),'done':done,'invalid_events':bad,'response_headers':dict(r.headers),'upstream_request_id':r.headers.get('x-request-id') or r.headers.get('request-id') or r.headers.get('x-trace-id') or r.headers.get('x-tt-logid')}
  async def probe(cid,name,payload,check):
   st,b,h=await req({'model':model,'messages':[{'role':'user','content':f'随机校验串 {rng}，请严格按要求执行'}],**payload})
   ok=st==200 and check(b); status='passed' if ok else ('warning' if st in (400,404,422) or unsupported(b) else 'failed')
   results.append({'case_id':cid,'name':name,'status':status,'passed':ok,'http_status':st,'details':f'HTTP {st}；随机串绑定={rng in json.dumps(b or {},ensure_ascii=False)}','observations':{'unsupported_error':unsupported(b),'response':json.dumps(b or {},ensure_ascii=False)[:500]}})
  tools=[{'type':'function','function':{'name':f'lookup_{rng.lower()}','description':'查询动态值','parameters':{'type':'object','properties':{'code':{'type':'string'}},'required':['code'],'additionalProperties':False}}}]
  await probe('AUTH-01','随机动态工具 required',{'tools':tools,'tool_choice':'required'},lambda b:bool(msg(b).get('tool_calls')))
  schema={'type':'json_schema','json_schema':{'name':f'answer_{rng.lower()}','strict':True,'schema':{'type':'object','properties':{'value':{'type':'string'}},'required':['value'],'additionalProperties':False}}}
  await probe('AUTH-02','随机 JSON Schema',{'response_format':schema},lambda b:_json_value(text(b)))
  await probe('AUTH-03','随机多轮记忆',{},lambda b:rng in text(b))
  st,b,h=await req({'model':model,'messages':[{'role':'user','content':f'计算 19*{len(rng)}，给出过程和结果'}],'thinking':{'type':'enabled'},'max_tokens':128})
  r=msg(b).get('reasoning_content'); results.append({'case_id':'AUTH-04','name':'Thinking 开启与 reasoning 字段','status':'passed' if st==200 and r else ('warning' if st in (400,404,422) or unsupported(b) else 'failed'),'passed':bool(st==200 and r),'http_status':st,'details':f'HTTP {st}；reasoning 字符数={len(r or "")}'})
  st,b,h=await req({'model':model,'messages':[{'role':'user','content':f'只回复 {rng}'}],'stream':True}); b=b or {}; ok=st==200 and b.get('done') and not b.get('invalid_events'); results.append({'case_id':'AUTH-05','name':'流式结束与事件完整性','status':'passed' if ok else ('warning' if st in (400,404,422) else 'failed'),'passed':ok,'http_status':st,'details':f'HTTP {st}；DONE={b.get("done")}；非法事件={b.get("invalid_events",0)}','observations':{'error':b.get('error'),'content_chars':len(b.get('content',''))}})
  # 对照开启/关闭，识别 200 但吞掉参数
  vals=[]
  for t in ('disabled','enabled'):
   st,b,h=await req({'model':model,'messages':[{'role':'user','content':f'请进行多步推理并回答 {rng}'}],'thinking':{'type':t},'max_tokens':128}); vals.append((st,len(msg(b).get('reasoning_content') or '')))
  silent=all(x[0]==200 for x in vals) and vals[1][1]==0
  results.append({'case_id':'AUTH-06','name':'Thinking 静默吞参对照','status':'warning' if silent else 'passed','passed':not silent,'details':f'禁用/开启 HTTP 与 reasoning 长度：{vals}','observations':{'silent_ignore':silent}})
 summary={'case':case,'test_kind':'authenticity','model':model,'random_nonce':rng,'total':len(results),'passed':sum(x['status']=='passed' for x in results),'failed':sum(x['status']=='failed' for x in results),'warning':sum(x['status']=='warning' for x in results),'results':results}
 hard=summary['failed']; summary['authenticity_assessment']='证据充分' if hard==0 and not summary['warning'] else ('存在接口限制或疑似降级' if hard==0 else '存在硬失败，不能认定满血')
 path=out/f'{case}-{time.strftime("%Y%m%d-%H%M%S")}';path.mkdir(parents=True,exist_ok=True);(path/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(path/'report.md').write_text('# 满血能力指纹报告\n\n结论：'+summary['authenticity_assessment']+'\n\n'+ '\n'.join(f"- {x['case_id']} {x['name']}：{x['status']}，{x['details']}" for x in results)+'\n')
 return path,summary

def _json_value(s):
 try:return isinstance(json.loads(s).get('value'),str)
 except (ValueError,AttributeError):return False
