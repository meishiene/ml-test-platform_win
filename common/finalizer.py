"""Central finalization for every test suite.

The web orchestrator calls this for load, strict, KVV, DeepSeek, GLM,
moderation and external benchmarks alike.  It preserves domain-specific
metrics while adding one common verdict/evidence contract.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .artifacts import SCHEMA_VERSION, write_json
from .result_adapter import normalize_summary

BLOCKING = {"failed", "error", "blocked", "not_evaluable", "interrupted"}

def _check(name: str, passed: bool | None, expected: Any, actual: Any, detail: str = "") -> dict[str, Any]:
    return {"name":name,"passed":passed,"expected":expected,"actual":actual,"detail":detail}

def _suite_checks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks=[]
    for name,item in (summary.get("checks") or {}).items():
        if isinstance(item,dict): checks.append(_check(name,item.get("passed"),{k:v for k,v in item.items() if k not in {"passed","value"}},item.get("value"),"领域指标校验"))
    results=summary.get("results") or []
    if results:
        statuses=[r.get("case_status",r.get("status")) for r in results if isinstance(r,dict)]
        checks.append(_check("case_verdicts_complete",all(x in {"passed","warning","skipped"} for x in statuses),"无 failed/error/blocked/not_evaluable",statuses,"存在阻断案例或不可评估案例"))
    suites=summary.get("suites") or []
    if suites:
        checks.append(_check("official_suite_exit",all(x.get("exit_code")==0 and not x.get("failed") and not x.get("errors") for x in suites),"所有官方套件 exit=0 且无失败/错误",[{k:x.get(k) for k in ("suite","exit_code","failed","errors")} for x in suites],"官方测试套件失败"))
    # External verifier suites must have actual scored/evaluated evidence.
    count_keys=("scored_tasks","evaluated_tasks","evaluated","total_problems")
    present=[(k,summary.get(k)) for k in count_keys if k in summary]
    for key,value in present:
        if key in {"scored_tasks","evaluated_tasks","evaluated"}: checks.append(_check(f"{key}_nonzero",isinstance(value,(int,float)) and value>0,"> 0",value,"没有可评分证据"))
    if "sent" in summary: checks.append(_check("formal_samples_nonzero",int(summary.get("sent") or 0)>0,"> 0",summary.get("sent"),"正式样本为 0；warmup 不能替代正式指标"))
    return checks

def finalize_result_dir(result_dir: Path) -> dict[str, Any]:
    path=result_dir/"summary.json"
    summary=normalize_summary(json.loads(path.read_text(encoding="utf-8")))
    checks=_suite_checks(summary)
    evidence=[]
    for name in ("execution.jsonl","requests.jsonl","requests.csv","pytest.log","run.log","connectivity.json","metadata.json"):
        p=result_dir/name
        if p.is_file(): evidence.append({"path":name,"bytes":p.stat().st_size,"nonempty":p.stat().st_size>0})
    evidence_ok=bool(evidence) and any(x["nonempty"] for x in evidence)
    checks.append(_check("evidence_present",evidence_ok,"至少一份非空请求/执行证据",evidence,"缺少执行证据"))
    blocking=[x for x in checks if x.get("passed") is not True]
    original=bool(summary.get("admission_passed"))
    admission=original and not blocking
    validation={"schema_version":SCHEMA_VERSION,"status":"passed" if admission else "failed" if any(x.get("passed") is False for x in checks) else "not_evaluable","passed":admission,"original_admission_passed":original,"checks":checks,"evidence":evidence,"rule":"只有领域校验与统一证据校验全部通过才允许准入；HTTP/进程退出成功不能单独构成案例通过"}
    summary["schema_version"]=SCHEMA_VERSION; summary["validation"]=validation; summary["admission_passed"]=admission
    write_json(path,summary); write_json(result_dir/"validation.json",validation)
    return summary
