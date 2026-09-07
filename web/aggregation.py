from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "volcperf.result.v1"


def enrich_summary(summary_path: Path, *, record_id: str, batch_id: str | None, test_kind: str,
                   case_name: str, protocol: str, base_url: str, model: str) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({"schema_version": SCHEMA_VERSION, "record_id": record_id, "batch_id": batch_id,
                    "test_kind": test_kind, "case": case_name, "protocol": protocol,
                    "base_url": base_url, "model": model})
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary



def write_capability_analysis(out: Path, aggregate: dict[str, Any], items: list[dict[str, Any]]) -> None:
    """生成能力专项的性能与差异分析；不虚构缺失的 request-level latency。"""
    rows=[]
    for item in items:
        summary=item.get("summary") or {}
        results=summary.get("results") or []
        durations=[r.get("duration_seconds") for r in results if isinstance(r.get("duration_seconds"),(int,float))]
        passed=sum(1 for r in results if r.get("status")=="passed" or r.get("passed") is True)
        failed=sum(1 for r in results if r.get("status") in {"failed","error"})
        warnings=sum(1 for r in results if r.get("status") in {"warning","blocked"})
        rows.append({"case":item["case"],"record_id":item["record_id"],"total":len(results),"passed":passed,"failed":failed,"warnings":warnings,"duration":sum(durations),"p50":sorted(durations)[len(durations)//2] if durations else None})
    def f(v): return "—" if v is None else f"{v:.2f}"
    lines=["# 能力专项性能分析报告","",f"- 批次：`{aggregate['record_id']}`",f"- 模型：`{aggregate['model']}`",f"- 说明：这是测试用例执行耗时分析，不是吞吐/TTFT 压测报告；原始数据未提供请求级延迟时不推导虚构指标。","","## 套件执行概览","","| 套件 | 用例数 | 通过 | 失败 | 警告/阻断 | 总耗时(s) | 用例耗时P50(s) |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['case']} | {r['total']} | {r['passed']} | {r['failed']} | {r['warnings']} | {f(r['duration'])} | {f(r['p50'])} |")
    lines += ["","## 结论","",f"- 批次完成：{aggregate['completed']}/{aggregate['total']}；底层用例通过：{sum(r['passed'] for r in rows)}。", "- 失败主要应结合差异报告和子任务日志定位；不能将套件总耗时等同于模型响应延迟。", "- 若需要 TTFT、TPOT、E2E、吞吐等性能结论，应执行独立性能压测并保留请求级明细。"]
    (out/"performance-report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    lines=["# 能力专项差异分析报告","",f"- 批次：`{aggregate['record_id']}`",f"- 模型：`{aggregate['model']}`","","## 测试项差异","","| 测试项 | 子套件 | 状态 | 关键观察 | 初步判断 |","|---|---|---|---|---|"]
    for item in items:
        for r in (item.get("summary") or {}).get("results",[]):
            status=r.get("status","unknown"); obs=r.get("observations") or {}; detail=str(r.get("details") or "—").replace("|","\\|").replace("\n"," ")[:260]
            if status=="passed": judgment="符合当前断言"
            elif status in {"warning","blocked"}: judgment="接口/能力限制，需单独确认"
            else: judgment="未满足断言；需结合响应和日志定位"
            lines.append(f"| {r.get('case_id') or r.get('id','—')} {r.get('name','')} | {item['case']} | {status} | {detail} | {judgment} |")
    lines += ["","## 跨套件对比","","- GLM 与 DeepSeek 已取消完整聚合子项；多选后的批次报告只汇总用户实际选中的独立子项。", "- `GLM_CONTRACT` 是自研公共契约，不能与 GLM 官方/专项用例直接做一一对应的通过率比较。", "- 原批次生成于 GLM 关闭 Thinking 用例移除前；`GLM-53-05` 应标记为不适用，不作为模型失败依据。", "- 流式、工具和 JSON Schema 的失败若伴随 `finish_reason=length` 或空正文，应优先检查 Thinking 输出预算与 adapter，不宜直接判定模型能力不存在。", "","## 建议复测对照","","1. 所有 GLM 请求固定 `thinking.enabled`。", "2. 正常能力测试使用较大 `max_tokens`，并记录 reasoning token、finish_reason 和完整正文。", "3. 流式测试保存原始 SSE，分别核对正文 delta、reasoning delta 和结束事件。", "4. 将本报告与新的同模型、同参数、同套件结果进行对比，避免跨接口直接比较。"]
    (out/"difference-report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def aggregate_batch(out: Path, batch: dict[str, Any], children: list[dict[str, Any]]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    items=[]
    for child in children:
        summary={}
        path=child.get("summary_path")
        if path and Path(path).is_file():
            summary=json.loads(Path(path).read_text(encoding="utf-8"))
        def portable(value: str | None) -> str | None:
            if not value: return None
            path_value=Path(value)
            try: return str(path_value.relative_to(Path.cwd())) if path_value.is_absolute() else str(path_value)
            except ValueError: return str(path_value)
        items.append({"record_id":child["id"],"case":child["case_name"],"status":child["status"],
                      "passed":None if child.get("passed") is None else bool(child["passed"]),
                      "result_dir":portable(child.get("result_dir")),"report_path":portable(child.get("report_path")),
                      "summary_path":portable(child.get("summary_path")),"summary":summary,"error":child.get("error")})
    completed=[x for x in items if x["status"]=="completed"]
    passed=sum(x["passed"] is True for x in items)
    aggregate={"schema_version":SCHEMA_VERSION,"record_id":batch["id"],"record_type":"batch",
               "test_kind":batch["test_kind"],"protocol":batch["protocol"],"base_url":batch["base_url"],
               "model":batch["model"],"cases":[x["case"] for x in items],"child_record_ids":[x["record_id"] for x in items],
               "total":len(items),"completed":len(completed),"passed":passed,"failed":len(items)-passed,
               "admission_passed":bool(items) and passed==len(items),"generated_at":datetime.now(timezone.utc).isoformat(),
               "results":items}
    (out/"summary.json").write_text(json.dumps(aggregate,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # Self-contained manifest is optimized for later Codex aggregation by one or many record IDs.
    (out/"aggregation_manifest.json").write_text(json.dumps(aggregate,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # 能力/契约专项没有压测指标，聚合报告应展示每个子用例的真实断言和错误，而不是空的吞吐列。
    is_capability = batch.get("test_kind") in {"kvv", "deepseek", "glm", "authenticity"}
    if is_capability:
        lines=["# 能力专项批量测试报告","",f"- 记录 ID：`{batch['id']}`",f"- 协议：`{batch['protocol']}`",f"- 模型：`{batch['model']}`",
               f"- 用例数：{len(items)}",f"- 完成：{len(completed)}/{len(items)}",f"- 通过：{passed}/{len(items)}",f"- 总结论：{'通过' if aggregate['admission_passed'] else '未通过'}","",
               "## 子用例结果","","| 用例 | 子记录 | 状态 | 结论 | 测试结果/失败原因 |", "|---|---|---|---|---|"]
        for item in items:
            summary=item["summary"] or {}
            rows=summary.get("results") or []
            if rows:
                for row in rows:
                    status=row.get("status", "unknown"); mark="✅" if status=="passed" or row.get("passed") is True else "⚠️" if status in {"warning","blocked"} else "❌"
                    detail=str(row.get("details") or row.get("error") or "—").replace("|","\\|").replace("\n"," ")[:500]
                    lines.append(f"| {row.get('case_id', item['case'])} {row.get('name','')} | `{item['record_id']}` | {status} | {mark} | {detail} |")
            else:
                error = str(summary.get("error") or item.get("error") or item.get("status") or "未生成专项结果").replace("|","\\|")
                lines.append(f"| {item['case']} | `{item['record_id']}` | {item['status']} | ❌ | {error} |")
        lines += ["","## 子报告与日志","","| 子记录 | 报告 | 原始日志 |","|---|---|---|"]
        for item in items:
            report=f"`{item['report_path']}`" if item.get('report_path') else "—"; log=f"`results/web-{item['record_id']}.log`"
            lines.append(f"| `{item['record_id']}` | {report} | {log} |")
    else:
        lines=["# 批量测试聚合报告","",f"- 记录 ID：`{batch['id']}`",f"- 子记录 ID：{', '.join(f'`{x}' for x in aggregate['child_record_ids'])}",
           f"- 协议：{batch['protocol']}",f"- 模型：{batch['model']}",f"- 总结论：{'通过' if aggregate['admission_passed'] else '未通过'}",
           f"- 通过：{passed}/{len(items)}","","## 各轮结果","","| 轮次 | 子记录 ID | 状态 | 结论 | 正式请求 | 成功率 | TTFT P90(s) | TTFC P90(s) | TPOT P90(ms/token) |",
           "|---|---|---|---|---:|---:|---:|---:|---:|"]
        for item in items:
            s=item["summary"]; rate=s.get('success_rate'); rate_text=f"{rate:.2%}" if isinstance(rate,(int,float)) else "-"
            def f(v): return f"{v:.3f}" if isinstance(v,(int,float)) else "-"
            lines.append(f"| {item['case']} | `{item['record_id']}` | {item['status']} | {'通过' if item['passed'] else '未通过'} | {s.get('sent','-')} | {rate_text} | {f(s.get('ttft_p90_seconds'))} | {f(s.get('ttfc_p90_seconds'))} | {f(s.get('tpot_p90_ms_per_token'))} |")
    lines += ["","## 聚合数据说明","",f"机器可读数据：`aggregation_manifest.json`（schema `{SCHEMA_VERSION}`）。"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if is_capability:
        write_capability_analysis(out, aggregate, items)
    return aggregate
