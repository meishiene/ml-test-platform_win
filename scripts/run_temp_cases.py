#!/usr/bin/env python3
"""Run the K3 manifest with the shared transport/validation pipeline.

This runner deliberately has no pass/fail business logic.  Every verdict is
produced by validators and every request is written as an auditable record.
"""
from __future__ import annotations
import argparse, asyncio, json, os, time
from dataclasses import asdict
from pathlib import Path
from common.suite import ManifestCase, execute_manifest_case
from common.test_framework import HttpExecutor, ResultLog
from common.validators import status, status_2xx, status_error, content, usage, sse, json_content, tool_calls

ROOT=Path(__file__).resolve().parent.parent
MANIFEST=ROOT/'test_cases/kimi_k3_temp_cases.json'

def validators_for(case: ManifestCase):
    """Translate the legacy manifest's human assertions into explicit checks.

    New manifests should use ``expected`` directly.  This compatibility layer
    prevents old cases from silently becoming HTTP-200-only tests.
    """
    cid=case.id; out=[]
    if cid in {'K3-ERR-001','K3-ERR-002','K3-JSON-002'}:
        out.append(status(400))
        out.append(status_error(400, error_types=('invalid_request_error','invalid_request'),
                                message_contains=('temperature',) if cid=='K3-ERR-001' else ()))
    elif cid=='K3-AUTH-001':
        out += [status(401), status_error(401, message_contains=('auth','key','unauthorized'))]
    elif cid=='K3-AUTH-002':
        out += [status(429), status_error(429, message_contains=('rate','limit','quota','too many'))]
    else:
        out.append(status_2xx())
        if case.request.get('stream'): out.append(sse())
        if cid=='K3-CONN-001' or 'usage' in ' '.join(case.assertions).lower(): out.append(usage('prompt_tokens','completion_tokens'))
        if cid=='K3-JSON-001': out.append(json_content())
        if cid=='K3-THINK-002': out.append(content(exact='DISABLED-OK'))
        if cid=='K3-PARAM-001': out.append(content(exact='PARAM-OK'))
        if cid=='K3-TOOL-001': out.append(tool_calls(count=1,names=('lookup_order',),args={'order_id':'ORD-000731'}))
    return out

async def main_async(args):
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8')); cases=[ManifestCase(x) for x in manifest['cases']]
    out=ROOT/'results'/f"K3_TEMP_SUITE-{time.strftime('%Y%m%d-%H%M%S')}"; out.mkdir(parents=True)
    log=ResultLog(out/'execution.jsonl'); results=[]
    base_url=os.environ['VOLC_BASE_URL'].rstrip('/')+'/chat/completions'
    base_headers={'Authorization':f"Bearer {os.environ['VOLC_API_KEY']}",'Content-Type':'application/json','Accept':'text/event-stream'}
    async with HttpExecutor(timeout=300,max_connections=10) as executor:
        for case in cases:
            # Preserve the custom validator set for this legacy manifest.
            original=case.validators; case.validators=lambda v=validators_for(case): v
            result=await execute_manifest_case(executor,log,case,model=os.environ['VOLC_MODEL'],url=base_url,base_headers=base_headers)
            results.append(result); print(case.id,result['status'],flush=True)
    log.close()
    passed=sum(x['passed'] for x in results); evaluated=sum(x['status'] in {'passed','failed'} for x in results)
    summary={'schema_version':'volcperf.result.v2','record_id':os.getenv('VOLC_RECORD_ID'),'case':'K3_TEMP_SUITE','test_kind':'temporary','model':os.environ['VOLC_MODEL'],'base_url':os.environ['VOLC_BASE_URL'],'total':len(results),'passed':passed,'failed':sum(x['status']=='failed' for x in results),'not_evaluable':sum(x['status']=='not_evaluable' for x in results),'validator_errors':sum(x['status']=='error' for x in results),'pass_rate':passed/evaluated if evaluated else 0,'admission_passed':bool(results) and all(x['status']=='passed' for x in results),'results':results,'evidence':'execution.jsonl','note':'Transport status and case verdict are separate; not_evaluable never counts as pass.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# Kimi-K3 临时测试报告（v2 校验流水线）','',f'- 结论：{"通过" if summary["admission_passed"] else "未通过/数据不足"}',f'- 通过：{passed}/{len(results)}；失败：{summary["failed"]}；不可评估：{summary["not_evaluable"]}', '', '| 用例 | 状态 | 失败/不可评估原因 |','|---|---|---|']
    for x in results: lines.append(f"| `{x['case_id']}` | {x['status']} | {'；'.join(x['failure_reasons']) or '—'} |")
    lines += ['', '## 证据', '', '- `execution.jsonl`：请求、响应、原始 SSE、时间点和校验明细', '- `summary.json`：结构化汇总和准入判定']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(f'结果目录: {out.resolve()}'); return 0 if summary['admission_passed'] else 2

def main():
    p=argparse.ArgumentParser(description='运行 K3 manifest 测试')
    try: raise SystemExit(asyncio.run(main_async(p.parse_args())))
    except KeyError as e: raise SystemExit(f'缺少环境变量: {e}')
if __name__=='__main__': main()
