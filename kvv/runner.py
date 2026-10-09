"""调用项目内置的官方 Kimi Vendor Verifier pytest；不改写官方 KVV 测试。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cases import CASES
from common.suite_runtime import finalize
from common.report_style import stats_lines
from volcperf.protocols import api_root

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KVV_HOME = ROOT / "vendor_verifier"
SUITE_PATHS = {
    "k3_features_core": ("tests/k3_features/test_response_format.py", "tests/k3_features/test_tokenization_groundtruth.py", "tests/k3_features/test_tool_choice.py"),
    "k3_features_extensions": ("tests/k3_features/test_dynamic_tools.py", "tests/k3_features/test_thinking_effort.py"),
    "params": ("tests/params",),
    "tool_call_json_schema": ("tests/tool_call_json_schema",),
}


def kvv_home() -> Path:
    return Path(os.getenv("KVV_HOME", str(DEFAULT_KVV_HOME))).expanduser().resolve()


def official_python(home: Path) -> Path:
    override = os.getenv("KVV_PYTHON")
    if override:
        return Path(override).expanduser().resolve()
    # 虚拟环境的解释器布局随平台不同：POSIX 在 bin/python，Windows 在 Scripts/python.exe。
    # Windows 上从别处拷来的 .venv 本身也不可用（路径已写死），此时请用 KVV_PYTHON 显式指定。
    for relative in (Path(".venv") / "bin" / "python", Path(".venv") / "Scripts" / "python.exe"):
        candidate = home / relative
        if candidate.is_file():
            return candidate
    return home / ".venv" / "bin" / "python"


def think_mode() -> str:
    value = os.getenv("KVV_THINK_MODE", "kimi")
    if value not in {"kimi", "opensource", "none"}:
        raise ValueError("KVV_THINK_MODE 仅支持 kimi、opensource、none")
    return value


def validate_runtime(home: Path, python: Path) -> None:
    if not (home / "tests" / "k3_features").is_dir() or not (home / "tests" / "params").is_dir():
        raise ValueError(f"未找到官方 KVV 测试目录：{home}（可用 KVV_HOME 指定官方套件目录）")
    if not python.is_file():
        raise ValueError(
            f"未找到官方 KVV Python：{python}。"
            "该套件使用自带的 vendor_verifier/.venv，若未随项目分发（例如在 Windows 上），"
            "请用 KVV_PYTHON 指向一个已安装官方套件依赖的解释器。"
        )


def build_command(home: Path, python: Path, suite: str, junit_path: Path, tool_report: Path) -> list[str]:
    base_url = api_root(os.getenv("VOLC_PROTOCOL", "volcengine"), os.environ["VOLC_BASE_URL"])
    command = [
        str(python), "-m", "pytest", *SUITE_PATHS[suite],
        "--base-url", base_url,
        "--smoke-model", os.environ["VOLC_MODEL"],
        "--think-mode", think_mode(),
        f"--junitxml={junit_path}", "-ra", "-v",
    ]
    if suite == "tool_call_json_schema":
        command += ["-n", os.getenv("KVV_TOOL_WORKERS", "4"), "--selection", "all",
                    "--reruns", os.getenv("KVV_RERUNS", "3"), "--reruns-delay", "2",
                    f"--tool-json-report={tool_report}"]
        if os.getenv("KVV_TOOL_THINKING", "true").lower() not in {"0", "false", "no"}:
            command.append("--thinking")
    return command


def parse_junit(path: Path, suite: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    nodes = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    total = sum(int(node.attrib.get("tests", 0)) for node in nodes)
    failures = sum(int(node.attrib.get("failures", 0)) for node in nodes)
    errors = sum(int(node.attrib.get("errors", 0)) for node in nodes)
    skipped = sum(int(node.attrib.get("skipped", 0)) for node in nodes)
    duration = sum(float(node.attrib.get("time", 0)) for node in nodes)
    failed_cases = []
    for case in root.iter("testcase"):
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is not None:
            failed_cases.append({"name": case.attrib.get("name"), "class": case.attrib.get("classname"), "message": (problem.attrib.get("message") or problem.text or "")[:1000]})
    return {"suite": suite, "total": total, "passed": total - failures - errors - skipped,
            "failed": failures, "errors": errors, "skipped": skipped, "duration_seconds": duration,
            "failed_cases": failed_cases}


def execute(case_id: str, output_root: Path) -> tuple[Path, dict[str, Any]]:
    for required in ("VOLC_API_KEY", "VOLC_BASE_URL", "VOLC_MODEL"):
        if not os.getenv(required):
            raise ValueError(f"缺少环境变量：{required}")
    home = kvv_home(); python = official_python(home); validate_runtime(home, python)
    meta = CASES[case_id]
    out = output_root / f"{case_id}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"KIMI_API_KEY": os.environ["VOLC_API_KEY"], "KIMI_BASE_URL": api_root(os.getenv("VOLC_PROTOCOL", "volcengine"), os.environ["VOLC_BASE_URL"]),
                "MODEL_NAME": os.environ["VOLC_MODEL"], "THINK_MODE": think_mode()})
    suite_results=[]; commands=[]
    with (out / "pytest.log").open("w", encoding="utf-8", buffering=1) as log:
        for suite in meta.suites:
            junit = out / f"junit-{suite}.xml"; tool_report = out / "tool-call-schema-report.json"
            command = build_command(home, python, suite, junit, tool_report)
            commands.append([part if part != os.environ.get("VOLC_API_KEY") else "***" for part in command])
            log.write(f"\n===== official KVV: {suite} =====\n$ {' '.join(command)}\n")
            started=time.perf_counter()
            process=subprocess.run(command, cwd=home, env=env, stdout=log, stderr=subprocess.STDOUT)
            if junit.exists(): result=parse_junit(junit,suite)
            else: result={"suite":suite,"total":0,"passed":0,"failed":0,"errors":1,"skipped":0,"duration_seconds":time.perf_counter()-started,"failed_cases":[{"name":"pytest","message":f"未生成 JUnit XML，退出码 {process.returncode}"}]}
            result["exit_code"]=process.returncode; suite_results.append(result)
    total=sum(x["total"] for x in suite_results); passed=sum(x["passed"] for x in suite_results)
    failed=sum(x["failed"]+x["errors"] for x in suite_results); skipped=sum(x["skipped"] for x in suite_results)
    admission=bool(suite_results) and all(x["exit_code"]==0 and x["failed"]==0 and x["errors"]==0 for x in suite_results)
    summary={"record_id":os.getenv("VOLC_RECORD_ID"),"test_kind":"kvv","case":case_id,
             "official_kvv":True,"kvv_home":str(home),"model":os.environ["VOLC_MODEL"],"base_url":os.environ["VOLC_BASE_URL"],
             "think_mode":think_mode(),"total":total,"passed":passed,"failed":failed,"skipped":skipped,
             "pass_rate":passed/total if total else 0.0,"admission_passed":admission,"suites":suite_results,
             "commands":commands,"generated_at":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write_report(out,summary)
    finalize(out)
    return out,summary


def write_report(out: Path, summary: dict[str, Any]) -> None:
    lines=["# 官方 Kimi Vendor Verifier 测试报告","",
           *stats_lines(summary['total'],summary['passed'],summary['failed'],0,summary['skipped'],summary.get('pass_rate'),
                        [f"结论：{'通过' if summary['admission_passed'] else '未通过'}",
                         f"模型：`{summary['model']}`；Base URL：`{summary['base_url']}`；Thinking 格式：`{summary['think_mode']}`",
                         f"官方工具目录：`{summary['kvv_home']}`"]),
           "","## 官方套件结果","","| 官方 pytest 套件 | 总数 | 通过 | 失败 | 错误 | 跳过 | 耗时(s) |","|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary["suites"]:
        lines.append(f"| `{' + '.join(SUITE_PATHS[item['suite']])}` | {item['total']} | {item['passed']} | {item['failed']} | {item['errors']} | {item['skipped']} | {item['duration_seconds']:.2f} |")
    failures=[(item["suite"],case) for item in summary["suites"] for case in item["failed_cases"]]
    if failures:
        lines += ["","## 失败用例",""]
        for suite,case in failures[:100]: lines.append(f"- `{suite}::{case.get('name')}`：{str(case.get('message','')).replace(chr(10),' ')[:500]}")
    lines += ["","## 产物","","- `pytest.log`：官方 pytest 完整输出","- `junit-*.xml`：官方 pytest JUnit 报告","- `tool-call-schema-report.json`：Tool JSON Schema 官方明细（执行该套件时生成）"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description="直接执行官方 Kimi Vendor Verifier pytest 套件")
    parser.add_argument("case",choices=sorted(CASES));parser.add_argument("--output-root",type=Path,default=Path("results"));args=parser.parse_args(argv)
    out,summary=execute(args.case,args.output_root);print(json.dumps(summary,ensure_ascii=False,indent=2));print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2

if __name__=="__main__":
    try: raise SystemExit(main())
    except ValueError as exc: print(f"配置错误: {exc}",file=os.sys.stderr);raise SystemExit(2)
