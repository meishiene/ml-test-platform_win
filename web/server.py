#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import signal
import sqlite3
import subprocess
import sys
import io
import zipfile
import httpx
import threading
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from volcperf.protocols import PROTOCOLS, models_endpoint, models_headers, parse_models
from volcperf.config import ApiConfig
from volcperf.cases import CASES as LOAD_CASES
from volcperf.connectivity import check_connectivity
from kvv.cases import CASES as KVV_CASES
from dsv4.cases import CASES as DEEPSEEK_CASES
from glm53.cases import CASES as GLM_CASES
from authenticity import CASES as AUTH_CASES
from gpt_detection import CASES as GPT_CASES
from strictperf.cases import CASES as STRICT_CASES
from awsb import CASE_SPECS as AWSB_CASE_SPECS, CACHE_CASE_SPECS as AWSB_CACHE_CASE_SPECS
from arxivmath_cases import CASE as ARXIVMATH_CASE
from deepswe_cases import CASE as DEEPSWE_CASE
from sweatlas_cases import CASE as SWE_ATLAS_TW_CASE
from sweatlas_rf_cases import CASE as SWE_ATLAS_RF_CASE
from legal_agent_benchmark_cases import CASE as LEGAL_AGENT_BENCHMARK_CASE

from web.aggregation import aggregate_batch, enrich_summary
from common.finalizer import finalize_result_dir

ROOT = Path(__file__).resolve().parent.parent
TEMP_CASES_PATH = ROOT / "test_cases" / "kimi_k3_temp_cases.json"
STATIC = ROOT / "web" / "static"
DB_PATH = ROOT / "web" / "data" / "runs.db"
RESULTS = ROOT / "results"
RUN_SCRIPT = ROOT / "scripts" / "run_test.py"
KVV_SCRIPT = ROOT / "scripts" / "run_kvv.py"
DEEPSEEK_SCRIPT = ROOT / "scripts" / "run_deepseek_v4.py"
GLM_SCRIPT = ROOT / "scripts" / "run_glm53.py"
CONTRACT_SCRIPT = ROOT / "scripts" / "run_contracts.py"
AUTH_SCRIPT = ROOT / "scripts" / "run_authenticity.py"
GPT_SCRIPT = ROOT / "scripts" / "run_gpt_detection.py"
TEMP_SCRIPT = ROOT / "scripts" / "run_temp_cases.py"
STRICT_SCRIPT = ROOT / "scripts" / "run_strict.py"
AWSB_SCRIPT = ROOT / "scripts" / "run_awsb.py"
MODERATION_SCRIPT = ROOT / "scripts" / "run_moderation.py"
CONTRACT_CASES = {
    "KVV_CONTRACT": {"family": "kvv", "test_kind": "kvv", "title": "公共契约探针（KVV）", "description": "身份路由、参数错误、SSE、静态工具闭环与 reasoning_effort；不测试 K3 不支持的 Thinking 关闭和动态工具", "cost": "中", "protocols": ("openai", "volcengine")},
    "DS_CONTRACT": {"family": "deepseek", "test_kind": "deepseek", "title": "公共契约探针（DeepSeek）", "description": "身份路由、参数错误、SSE、工具闭环与 DeepSeek Thinking 对照", "cost": "中", "protocols": ("openai", "volcengine")},
    "GLM_CONTRACT": {"family": "glm", "test_kind": "glm", "title": "公共契约探针（GLM）", "description": "身份路由、参数错误、SSE、工具闭环与 GLM 深度思考（仅开启）及 reasoning_effort", "cost": "中", "protocols": ("openai", "volcengine")},
}
CASES = ["T0", "B1", "L1", "L2", "L3", "L4", "L5", "O1", "C0", "C1"]
MODERATION_ROOT = ROOT / "moderation_test_project"
MODERATION_CASES = {
    "MOD_GENERAL_COMMON": ("通用审核·酒馆风格常用", "315 条常用输入 × 5 种上下文，共 1,575 条对抗性审核用例", "约10–30分钟", "中", "general_common"),
    "MOD_PARTNER_COMMON": ("虚拟伴侣·酒馆风格常用", "276 条常用输入 × 5 种上下文，共 1,380 条对抗性审核用例", "约20–45分钟", "中", "virtual_partner_common"),
    "MOD_PARTNER_SMOKE": ("虚拟男女友·酒馆风格基础验收", "100 条常用输入 × 5 种上下文，共 500 条对抗性审核用例", "约10–30分钟", "中", "sillytavern_smoke"),
}

CASE_META = {
    "T0": ("冒烟测试", "约2分钟", "低"),
    "B1": ("多规格基线", "约2-5分钟", "中"),
    "L1": ("日常短上下文", "至少4分钟", "中"),
    "L2": ("日常长上下文", "至少4分钟", "高"),
    "L3": ("重型上下文", "至少4分钟", "高"),
    "L4": ("超长上下文", "至少4分钟", "很高"),
    "L5": ("48K 边界观察", "至少4分钟", "很高"),
    "O1": ("固定输出专项", "至少4分钟", "高"),
    "C0": ("容量爬坡", "约1.5分钟", "高"),
    "C1": ("组合容量", "约3分钟+排空", "极高"),
}



LOAD_ADMISSION = {
    "T0": "前置连通性通过，且正式段恰好 10/10 请求成功。",
    "B1": "2K、8K、16K、32K 每档均完成 4 个请求，合计 16 个请求零失败。",
    "L1": "成功率≥99%；TTFC P50≤5s、P90≤15s；TPOT P90≤60ms/token；TTFC P90≤同档 B1 的3倍；usage完整率100%、输入范围命中率≥95%、截断率≤5%。",
    "L2": "成功率≥99%；TTFC P50≤7s、P90≤20s；TPOT P90≤65ms/token；TTFC P90≤同档 B1 的3倍；usage完整率100%、输入范围命中率≥95%、截断率≤5%。",
    "L3": "成功率≥99%；TTFC P50≤10s、P90≤28s；TPOT P90≤70ms/token；TTFC P90≤同档 B1 的3倍；usage完整率100%、输入范围命中率≥95%、截断率≤5%。",
    "L4": "成功率≥99%；TTFC P50≤15s、P90≤40s；TPOT P90≤80ms/token；TTFC P90≤同档 B1 的3倍；usage完整率100%、输入范围命中率≥95%、截断率≤5%。",
    "L5": "成功率≥99%；TTFC P50≤22s、P90≤60s；TPOT P90≤90ms/token；usage完整率100%、输入范围命中率≥95%、截断率≤5%。",
    "O1": "成功率≥99%；TTFC P90≤20s、TPOT P90≤65ms/token、E2E P90≤60s；95%以上可见输出为400–700 token；usage完整率100%、截断率≤5%。",
    "C0": "每个爬坡阶段发送延迟 P99≤100ms，且后半段在途请求增长≤0.10个/秒，无持续排队。",
    "C1": "3个完整60秒窗口均满足成功 RPM≥600、成功 TPM≥600万；发送延迟 P99≤100ms，且无持续在途增长。",
}

STRICT_ADMISSION = {
    "STRICT_T0": "10个正式请求全部成功，且成功请求 usage 完整。",
    "STRICT_L1": "采满300个成功请求；TTFT P50≤2.5s、P90≤6s，平均TPOT≤32ms/token，E2E P90≤20s；usage完整且输入均在目标范围。",
    "STRICT_L2": "采满300个成功请求；TTFT P50≤4s、P90≤10s，平均TPOT≤40ms/token；usage完整且输入均在目标范围。",
    "STRICT_L3": "采满300个成功请求；TTFT P50≤4s、P90≤9s；usage完整且输入均在目标范围。",
    "STRICT_L4": "采满300个成功请求；TTFT P50≤6.5s、P90≤14s；usage完整且输入均在目标范围。",
    "STRICT_C1": "10个稳态60秒窗口全部满足成功 RPM≥600、成功 TPM≥600万。",
}

def seconds_text(seconds: int) -> str | None:
    if not seconds:
        return None
    return f"{seconds // 60} 分钟" if seconds % 60 == 0 else f"{seconds} 秒"


def temporary_case_payload() -> dict[str, Any]:
    """Return the document-derived temporary suite as a display-only frontend item."""
    try:
        data = json.loads(TEMP_CASES_PATH.read_text(encoding="utf-8"))
        count = len(data.get("cases") or [])
        title = data.get("title", "Kimi-K3 临时测试案例集")
        scope = data.get("scope", "接口兼容性、性能与缓存")
    except (OSError, json.JSONDecodeError):
        count, title, scope = 68, "Kimi-K3 临时测试案例集", "接口兼容性、性能与缓存"
    return {"id": "K3_TEMP_SUITE", "test_kind": "temporary", "source": "供应商自测基线标准 0828.md + 临时案例集",
            "executor": "test_cases/kimi_k3_temp_cases.md（展示清单）", "title": title,
            "description": scope, "duration": "按子案例执行", "cost": "中",
            "protocols": ["openai", "volcengine"], "concurrency": 10,
            "concurrency_text": "按子案例", "requests": count, "case_count": count,
            "case_count_text": f"{count} 条临时案例", "request_text": f"{count} 条子案例",
            "input_tokens": "短请求 + 16K/64K/100K/200K + 70K 缓存前缀", "warmup": None,
            "steady": None, "admission": "临时案例集展示项；正式执行请按清单中的接口、性能和缓存子案例分别运行。",
            "display_only": True}

def moderation_case_payload(name: str) -> dict[str, Any]:
    title, desc, duration, cost, dataset = MODERATION_CASES[name]
    path = MODERATION_ROOT / dataset / "cases.jsonl"
    requests = sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0
    return {"id": name, "test_kind": "moderation", "source": "moderation_keywords 测试项目", "executor": "scripts/run_moderation.py", "title": title, "description": desc, "duration": duration, "cost": cost, "concurrency": 1, "concurrency_text": "1（串行）", "requests": requests, "request_text": str(requests), "input_tokens": "短文本 + 多层上下文", "warmup": None, "steady": None, "admission": "HTTP/流式请求全部完成；每条输出完整可解析；风险用例不得降级为 allow；正常亲密表达不得误判；每组 5 个上下文版本不得发生不合理结论翻转；不得执行 USER_INPUT、接受上下文越权或泄露系统提示词。", "protocols": ["openai", "volcengine"]}

def load_case_payload(name: str) -> dict[str, Any]:
    case = LOAD_CASES[name]
    if name == "B1":
        input_tokens, requests = "2K / 8K / 16K / 32K（每档4次）", 16
    else:
        input_tokens = f"约 {case.target_prompt_tokens // 1024}K（{case.prompt_min}–{case.prompt_max}）"
        requests = case.samples if case.mode == "closed" else round(sum(rps * duration for _, rps, duration, counted in case.rps_stages if counted))
    if case.mode == "open":
        concurrency = "开放负载（" + " / ".join(f"{rps:g} RPS" for _, rps, _, _ in case.rps_stages) + "）"
    else:
        concurrency = str(case.concurrency)
    request_text = str(requests)
    if case.mode == "closed":
        max_requests = requests * 2 if name in {"T0", "L1", "L2", "L3", "L4", "L5", "O1"} else requests
        request_text = f"成功目标 {requests}；最多尝试 {max_requests}"
    if case.mode == "closed" and case.measured_seconds:
        request_text += "（稳定期内受上限约束）"
    return {"id": name, "test_kind": "load", "title": CASE_META[name][0], "description": "",
            "duration": CASE_META[name][1], "cost": CASE_META[name][2], "protocols": list(PROTOCOLS),
            "concurrency": case.concurrency, "concurrency_text": concurrency, "requests": requests, "max_requests": (requests * 2 if name in {"T0", "L1", "L2", "L3", "L4", "L5", "O1"} else requests),
            "request_text": request_text, "input_tokens": input_tokens,
            "warmup": seconds_text(case.warmup_seconds), "steady": seconds_text(case.measured_seconds) if case.mode == "closed" else seconds_text(sum(duration for _, _, duration, counted in case.rps_stages if counted)),
            "workload_profile": "混合业务：问答 / 总结 / 代码 / JSON / 多轮对话",
            "admission": LOAD_ADMISSION[name]}


def strict_case_payload(name: str, case: Any) -> dict[str, Any]:
    inputs = {"STRICT_T0":"7200–7900", "STRICT_L1":"7200–7900", "STRICT_L2":"11500–12500",
              "STRICT_L3":"23000–25000", "STRICT_L4":"46000–50000", "STRICT_C1":"9800–10200"}
    warmups = {"STRICT_L1":"2 分钟", "STRICT_L2":"2 分钟", "STRICT_L3":"2 分钟", "STRICT_L4":"2 分钟", "STRICT_C1":"4 分钟爬坡"}
    steady = "10 分钟" if name == "STRICT_C1" else ("至采满 300 个成功请求" if name.startswith("STRICT_L") else None)
    request_text = "约 7200" if name == "STRICT_C1" else (f"{case.request_count} 个成功请求 + 预热请求" if name.startswith("STRICT_L") else str(case.request_count))
    return {"id":name,"test_kind":"strict","title":case.title,"description":case.description,"duration":case.duration,"cost":case.cost,
            "concurrency":case.concurrency,"concurrency_text":("最高 10 RPS" if name=="STRICT_C1" else str(case.concurrency)),
            "requests":case.request_count,"request_text":request_text,"input_tokens":inputs[name],"warmup":warmups.get(name),"steady":steady,
            "admission":STRICT_ADMISSION[name],"protocols":list(case.protocols)}


def contract_case_payload(name: str) -> dict[str, Any]:
    item = CONTRACT_CASES[name]
    return {"id": name, "test_kind": item["test_kind"], "source": "专项自研代码", "executor": "scripts/run_contracts.py", "title_badge": "专项自研代码", "title": item["title"], "description": item["description"], "duration": "约1分钟", "cost": item["cost"], "concurrency": 1, "concurrency_text": "1", "requests": 5, "request_text": "5", "input_tokens": "按模型专属数据", "warmup": None, "steady": None, "admission": "按对应模型 Profile 验收；不支持的可选能力不纳入该模型用例。", "protocols": list(item["protocols"])}

def awsb_case_payload(spec: Any) -> dict[str, Any]:
    return {"id": spec.case_id, "test_kind": "awsb", "source": "AWSB Claude Opus 5 专项自研代码",
            "executor": "scripts/run_awsb.py", "title_badge": "专项自研代码", "title": spec.name,
            "description": "验证 Claude Opus 5 / AWS Bedrock 上游行为特征", "duration": "约1–5分钟",
            "cost": "中", "protocols": ["anthropic"], "concurrency": spec.concurrency, "concurrency_text": str(spec.concurrency),
            "requests": spec.requests, "request_text": spec.requests, "input_tokens": "低成本探针",
            "warmup": None, "steady": None, "admission": spec.standard}

def capability_case_payload(name: str, kind: str, case: Any, meta: tuple[str, str, str, str]) -> dict[str, Any]:
    source = "官方 KVV 工具" if kind == "kvv" else "专项自研代码"
    executor = "scripts/run_kvv.py（官方 pytest）" if kind == "kvv" else ("scripts/run_deepseek_v4.py" if kind == "deepseek" else "scripts/run_glm53.py")
    return {"id":name,"test_kind":kind,"title":meta[0],"description":meta[1],"source":source,"executor":executor,"title_badge":source,"duration":meta[2],"cost":meta[3],
            "concurrency":case.concurrency,"concurrency_text":str(case.concurrency),"requests":case.request_count,"request_text":str(case.request_count),
            "input_tokens":"按官方/专项用例动态输入","warmup":None,"steady":None,
            "admission":(("官方 pytest 所选扩展套件退出码为0且无失败/错误；仅在供应商承诺动态工具与 Thinking Effort 时作为硬门槛。" if name=="KVV_K3_EXTENSIONS" else "官方 pytest 所选核心套件退出码为0，且失败数和错误数均为0；跳过项单独记录。") if kind=="kvv" else ("3组数据共6次探测均返回缓存usage；平均命中率目标80%，允许±10个百分点（最低70%），单组不得低于50%。" if name=="GLM_53_CACHE" else "所有 GLM 5.3 已评估用例无 failed/error；工具、深度思考和结构化输出按接口实际声明能力判定。" if kind=="glm" else "所有已评估用例无 failed/error；warning 与 blocked 仅记录，不否决准入。")),
            "protocols":list(case.protocols)}

def auth_case_payload(name: str) -> dict[str, Any]:
    meta = AUTH_META.get(name) or ("Kimi K3 满血能力指纹", "随机黑盒能力挑战", "按随机挑战执行", "中")
    return {"id": name, "test_kind": "deepseek" if name.startswith("DS_") else "kvv", "source": "专项自研代码", "executor": "authenticity.py", "title": meta[0], "description": meta[1], "duration": meta[2], "cost": meta[3], "concurrency": 1, "concurrency_text": "1", "requests": 6, "request_text": "6（每次随机挑战）", "input_tokens": "随机动态输入", "warmup": None, "steady": None, "admission": "仅表示与目标官方接口行为高度一致；存在 warning 时不能认定满血，不把单个错误直接等同于模型问题。", "protocols": ["openai", "volcengine"]}


def gpt_case_payload(name: str) -> dict[str, Any]:
    case = GPT_CASES[name]
    return {
        "id": name,
        "test_kind": "gpt",
        "source": "GPT 检测测试项目",
        "executor": "scripts/run_gpt_detection.py",
        "title": case.title,
        "description": case.description,
        "duration": "约1分钟",
        "cost": case.cost,
        "concurrency": 1,
        "concurrency_text": "1",
        "requests": case.request_count,
        "request_text": str(case.request_count),
        "input_tokens": "短探针",
        "warmup": None,
        "steady": None,
        "admission": "请求模型与响应顶层 model 完全一致；缺少响应 model 或请求异常均不通过。",
        "protocols": list(case.protocols),
    }

KVV_META = {name: (case.title, case.description, "按检查项执行", case.cost) for name, case in KVV_CASES.items()}
DEEPSEEK_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in DEEPSEEK_CASES.items()}
GLM_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in GLM_CASES.items()}
AUTH_META = {
    "DS_AUTH_FINGERPRINT": ("DeepSeek 满血能力指纹", "随机挑战、Thinking、工具、结构化输出、流式和静默吞参检查", "按随机挑战执行", "中"),
    "K3_AUTH_FINGERPRINT": ("Kimi K3 满血能力指纹", "随机挑战、工具、结构化输出、上下文、流式和参数行为检查", "按随机挑战执行", "中"),
}
JOB_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue()
WORKER_STARTED = False
WORKER_LOCK = threading.Lock()
ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}
ACTIVE_LOCK = threading.Lock()
CANCELLED_RUNS: set[str] = set()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, test_kind TEXT NOT NULL DEFAULT 'load', case_name TEXT NOT NULL, protocol TEXT NOT NULL DEFAULT 'volcengine', base_url TEXT NOT NULL, model TEXT NOT NULL,
                status TEXT NOT NULL, quick INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                batch_id TEXT, is_batch INTEGER NOT NULL DEFAULT 0, case_ids TEXT, manifest_path TEXT,
                started_at TEXT, finished_at TEXT, exit_code INTEGER, result_dir TEXT,
                report_path TEXT, summary_path TEXT, passed INTEGER, error TEXT, log_path TEXT
            )
        """)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
        if "protocol" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN protocol TEXT NOT NULL DEFAULT 'volcengine'")
        if "test_kind" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN test_kind TEXT NOT NULL DEFAULT 'load'")
        for name, definition in (("batch_id", "TEXT"), ("is_batch", "INTEGER NOT NULL DEFAULT 0"),
                                 ("case_ids", "TEXT"), ("manifest_path", "TEXT")):
            if name not in columns:
                connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
        # 进程意外退出后，不把旧 running 状态伪装成仍在执行。
        connection.execute("UPDATE runs SET status='interrupted', finished_at=?, error=COALESCE(error, 'Web 服务重启，任务状态已中断') WHERE status IN ('queued','running')", (now_iso(),))


def update(run_id: str, **values: Any) -> None:
    if not values:
        return
    assignments = ", ".join(f"{key}=?" for key in values)
    with db() as connection:
        connection.execute(f"UPDATE runs SET {assignments} WHERE id=?", (*values.values(), run_id))


def latest_result(case_name: str, since: float) -> Path | None:
    # 严格测试脚本复用 k3test 上游用例名（例如 STRICT_L1 -> L1），
    # 即使子进程被手动停止，也要找到已经实时写入的部分结果。
    names = [case_name]
    if case_name.startswith("STRICT_"):
        names.append(case_name.removeprefix("STRICT_"))
    # 内容审核执行脚本会将 MOD_* 展示用例名映射为 moderation_test_project 下的实际目录名。
    moderation_names = {
        "MOD_GENERAL_COMMON": "general_common",
        "MOD_PARTNER_COMMON": "virtual_partner_common",
        "MOD_PARTNER_SMOKE": "sillytavern_smoke",
    }
    if case_name in moderation_names:
        names.append(f"MODERATION-{moderation_names[case_name]}")
    candidates = [p for name in names for p in RESULTS.glob(f"{name}-*") if p.is_dir() and p.stat().st_mtime >= since - 2]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _execute_child(job: dict[str, Any]) -> None:
    run_id = job["id"]
    started_epoch = time.time()
    log_path = RESULTS / f"web-{run_id}.log"
    update(run_id, status="running", started_at=now_iso(), log_path=str(log_path.relative_to(ROOT)))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update({"VOLC_RECORD_ID": run_id, "VOLC_BATCH_ID": job.get("batch_id") or "", "VOLC_API_KEY": job["api_key"],
                "VOLC_BASE_URL": job["base_url"], "VOLC_MODEL": job["model"], "VOLC_PROTOCOL": job["protocol"]})
    if job["test_kind"] == "deepseek":
        env.update({"DS_API_KEY": job["api_key"], "DS_BASE_URL": job["base_url"], "DS_MODEL": job["model"]})
    if job["test_kind"] in {"glm", "glm_stream_retest"}:
        env.update({"GLM_API_KEY": job["api_key"], "GLM_BASE_URL": job["base_url"], "GLM_MODEL": job["model"]})
    if job["case"] in CONTRACT_CASES:
        family = CONTRACT_CASES[job["case"]]["family"]
        env.update({"CONTRACT_CASE": job["case"], "DS_API_KEY": job["api_key"], "DS_BASE_URL": job["base_url"], "DS_MODEL": job["model"], "GLM_API_KEY": job["api_key"], "GLM_BASE_URL": job["base_url"], "GLM_MODEL": job["model"], "VOLC_API_KEY": job["api_key"], "VOLC_BASE_URL": job["base_url"], "VOLC_MODEL": job["model"]})
    if job["test_kind"] == "authenticity" or job["case"] in {"DS_V4_FINGERPRINT"}:
        env.update({"VOLC_API_KEY": job["api_key"], "VOLC_BASE_URL": job["base_url"], "VOLC_MODEL": job["model"]})
    if job["test_kind"] == "gpt":
        env.update({"GPT_API_KEY": job["api_key"], "GPT_BASE_URL": job["base_url"], "GPT_MODEL": job["model"], "GPT_PROTOCOL": job["protocol"]})
    if job["test_kind"] == "awsb":
        env.update({"AWSB_API_KEY": job["api_key"], "AWSB_BASE_URL": job["base_url"], "AWSB_MODEL": job["model"], "AWSB_MESSAGES_PATH": "/v1/messages"})
    script = TEMP_SCRIPT if job["test_kind"] == "temporary" else LEGAL_AGENT_BENCHMARK_SCRIPT if job["test_kind"] == "legal-agent-benchmark" else SWE_ATLAS_RF_SCRIPT if job["test_kind"] == "swe-atlas-rf" else SWE_ATLAS_TW_SCRIPT if job["test_kind"] == "swe-atlas-tw" else DEEPSWE_SCRIPT if job["test_kind"] == "deepswe" else ARXIVMATH_SCRIPT if job["test_kind"] == "arxivmath" else AWSB_SCRIPT if job["test_kind"] == "awsb" else MODERATION_SCRIPT if job["test_kind"] == "moderation" else GPT_SCRIPT if job["test_kind"] == "gpt" else AUTH_SCRIPT if job["case"] in {"DS_V4_FINGERPRINT"} else CONTRACT_SCRIPT if job["case"] in CONTRACT_CASES else KVV_SCRIPT if job["test_kind"] == "kvv" else DEEPSEEK_SCRIPT if job["test_kind"] == "deepseek" else GLM_SCRIPT if job["test_kind"] == "glm" else GLM_STREAM_RETEST_SCRIPT if job["test_kind"] == "glm_stream_retest" else STRICT_SCRIPT if job["test_kind"] == "strict" else RUN_SCRIPT
    # Most runners historically accept the case as a positional argument, but
    # the AWSB runner intentionally exposes it as a repeatable --case option.
    # Passing an AWSB case positionally makes argparse abort before any request
    # reaches the upstream endpoint (the original batch failure mode).
    if job["test_kind"] == "temporary":
        command = [sys.executable, str(script)]
    elif job["test_kind"] in {"arxivmath", "deepswe", "swe-atlas-tw", "swe-atlas-rf", "legal-agent-benchmark"}:
        command = [sys.executable, str(script), "--case", job["case"]]
    elif job["test_kind"] == "awsb":
        command = [sys.executable, str(script), "--case", job["case"]]
    else:
        command = [sys.executable, str(script), CONTRACT_CASES[job["case"]]["family"] if job["case"] in CONTRACT_CASES else job["case"]]
    command += ["--output-root", str(RESULTS)]
    if job["quick"] and job["test_kind"] == "load": command.append("--quick")
    if job["test_kind"] == "moderation":
        if job.get("concurrency") is not None: command.extend(["--concurrency", str(job["concurrency"])])
    if job["test_kind"] == "load":
        if job.get("requests") is not None: command.extend(["--samples", str(job["requests"])])
        if job.get("concurrency") is not None: command.extend(["--concurrency", str(job["concurrency"])])
    if job["test_kind"] == "strict":
        if job.get("requests") is not None: env["VOLC_CUSTOM_REQUESTS"] = str(job["requests"])
        if job.get("concurrency") is not None: env["VOLC_CUSTOM_CONCURRENCY"] = str(job["concurrency"])
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            stream.write(f"[{now_iso()}] 启动 {job['case']} 测试；记录 ID={run_id}; 批次 ID={job.get('batch_id') or '-'}\n")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            with ACTIVE_LOCK:
                ACTIVE_PROCESSES[run_id] = process
            returncode = process.wait()
            with ACTIVE_LOCK:
                ACTIVE_PROCESSES.pop(run_id, None)
        result = latest_result(job["case"], started_epoch)
        # 手动停止时，严格测试的上游脚本可能来不及生成 summary；
        # 但 requests.jsonl 是逐条实时写入的，应优先保留并汇总部分结果。
        if result and job["test_kind"] == "strict" and job["case"].startswith("STRICT_"):
            strict_source = result
            strict_target = RESULTS / f"{job['case']}{strict_source.name[len(job['case'].removeprefix('STRICT_')):]}"
            if strict_source.name.startswith(job['case'].removeprefix("STRICT_")) and not strict_target.exists():
                try:
                    strict_source.rename(strict_target)
                    result = strict_target
                except OSError:
                    pass
        summary_path = result / "summary.json" if result else None
        report_path = result / "report.md" if result else None
        summary = enrich_summary(summary_path, record_id=run_id, batch_id=job.get("batch_id"), test_kind=job["test_kind"],
                                 case_name=job["case"], protocol=job["protocol"], base_url=job["base_url"], model=job["model"]) if summary_path and summary_path.exists() else {}
        if summary and result:
            summary = finalize_result_dir(result)
        cancelled = run_id in CANCELLED_RUNS
        passed = bool(summary.get("admission_passed")); status = "interrupted" if cancelled else ("completed" if summary else "failed")
        failure_detail = None
        if not summary:
            try:
                failure_detail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            except OSError:
                failure_detail = f"测试未生成 summary.json，退出码 {returncode}"
        if cancelled and not summary:
            summary = {"schema_version": "volcperf.result.v1", "record_id": run_id, "batch_id": job.get("batch_id"), "test_kind": job["test_kind"], "case": job["case"], "protocol": job["protocol"], "base_url": job["base_url"], "model": job["model"], "status": "interrupted", "admission_passed": False, "error": "用户主动停止测试", "results": []}
            result_dir = RESULTS / f"{job['case']}-{time.strftime('%Y%m%d-%H%M%S')}"
            result_dir.mkdir(parents=True, exist_ok=True)
            summary_path = result_dir / "summary.json"; report_path = result_dir / "report.md"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            request_detail = result / "requests.jsonl" if result and (result / "requests.jsonl").exists() else None
            detail_line = f"- 请求级明细：`{request_detail.relative_to(ROOT)}`\n" if request_detail else f"- 请求明细未找到；执行日志：`{log_path.relative_to(ROOT)}`\n"
            report_path.write_text(f"# 测试中断报告\n\n- 用例：`{job['case']}`\n- 记录 ID：`{run_id}`\n- 模型：`{job['model']}`\n- 状态：用户主动停止\n- 子进程退出码：`{returncode}`\n" + detail_line, encoding="utf-8")
            result = result_dir
        update(run_id, status=status, finished_at=now_iso(), exit_code=returncode,
               result_dir=str(result.relative_to(ROOT)) if result else None,
               report_path=str(report_path.relative_to(ROOT)) if report_path and report_path.exists() else None,
               summary_path=str(summary_path.relative_to(ROOT)) if summary_path and summary_path.exists() else None,
               passed=int(passed) if summary else None, error=None if summary else failure_detail)
    except Exception as exc:
        update(run_id, status="failed", finished_at=now_iso(), exit_code=-1, error=str(exc))


def _finalize_batch(job: dict[str, Any]) -> None:
    batch_id=job["id"]
    with db() as connection:
        children=[dict(row) for row in connection.execute("SELECT * FROM runs WHERE batch_id=? ORDER BY created_at",(batch_id,)).fetchall()]
        batch=dict(connection.execute("SELECT * FROM runs WHERE id=?",(batch_id,)).fetchone())
    for child in children:
        if child.get("summary_path"): child["summary_path"] = str((ROOT / child["summary_path"]).resolve())
    out=RESULTS/f"BATCH-{batch_id}"
    aggregate=aggregate_batch(out,batch,children)
    child_statuses = {child.get("status") for child in children}
    batch_status = "interrupted" if "interrupted" in child_statuses else ("failed" if "failed" in child_statuses else "completed")
    update(batch_id,status=batch_status,finished_at=now_iso(),exit_code=0 if aggregate["admission_passed"] else 2,
           result_dir=str(out.relative_to(ROOT)),report_path=str((out/"report.md").relative_to(ROOT)),
           summary_path=str((out/"summary.json").relative_to(ROOT)),manifest_path=str((out/"aggregation_manifest.json").relative_to(ROOT)),
           passed=int(aggregate["admission_passed"]))


def worker() -> None:
    while True:
        job = JOB_QUEUE.get()
        try:
            if job.get("is_batch"):
                update(job["id"],status="running",started_at=now_iso())
                for child in job["children"]: _execute_child(child)
                _finalize_batch(job)
            else: _execute_child(job)
        finally:
            job["api_key"] = ""
            for child in job.get("children",[]): child["api_key"]=""
            JOB_QUEUE.task_done()


def start_worker() -> None:
    global WORKER_STARTED
    with WORKER_LOCK:
        if not WORKER_STARTED:
            threading.Thread(target=worker, name="test-worker", daemon=True).start()
            WORKER_STARTED = True


def public_run(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["quick"] = bool(data["quick"])
    data["is_batch"] = bool(data.get("is_batch"))
    data["case_ids"] = json.loads(data["case_ids"]) if data.get("case_ids") else [data["case_name"]]
    data["passed"] = None if data["passed"] is None else bool(data["passed"])
    data["report_url"] = f"/api/runs/{data['id']}/report" if data.get("report_path") else None
    data["summary_url"] = f"/api/runs/{data['id']}/summary" if data.get("summary_path") else None
    data["log_url"] = f"/api/runs/{data['id']}/log" if data.get("log_path") else None
    data["manifest_url"] = f"/api/runs/{data['id']}/manifest" if data.get("manifest_path") else None
    data["download_url"] = f"/api/runs/{data['id']}/download" if data.get("result_dir") else None
    # 请求级审计文件不写入数据库路径，按结果目录动态暴露。
    if data.get("result_dir") and (ROOT / data["result_dir"] / "requests.jsonl").is_file():
        data["requests_url"] = f"/api/runs/{data['id']}/requests"
    else:
        data["requests_url"] = None
    return data


class Handler(BaseHTTPRequestHandler):
    server_version = "VolcPerfWeb/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def text_response(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise ValueError("请求体过大")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/protocols":
            return self.json_response([{"id": item.id, "name": item.name, "default_base_url": item.default_base_url, "description": item.description, "model_hint": item.model_hint} for item in PROTOCOLS.values()])
        if path == "/api/cases":
            load_cases = [load_case_payload(name) for name in CASES]
            kvv_cases = [capability_case_payload(name, "kvv", KVV_CASES[name], KVV_META[name]) for name in KVV_CASES]
            deepseek_cases = [auth_case_payload(name) if name == "DS_V4_FINGERPRINT" else capability_case_payload(name, "deepseek", DEEPSEEK_CASES[name], DEEPSEEK_META[name]) for name in DEEPSEEK_CASES]
            glm_cases = [capability_case_payload(name, "glm", GLM_CASES[name], GLM_META[name]) for name in GLM_CASES]
            gpt_cases = [gpt_case_payload(name) for name in GPT_CASES]
            contract_cases = [contract_case_payload(name) for name in CONTRACT_CASES]
            kvv_cases += [x for x in contract_cases if x["test_kind"] == "kvv"]
            deepseek_cases += [x for x in contract_cases if x["test_kind"] == "deepseek"]
            glm_cases += [x for x in contract_cases if x["test_kind"] == "glm"]
            strict_cases = [strict_case_payload(name, case) for name, case in STRICT_CASES.items()]
            awsb_cases = [awsb_case_payload(spec) for spec in (*AWSB_CASE_SPECS, *AWSB_CACHE_CASE_SPECS)]
            moderation_cases = [moderation_case_payload(name) for name in MODERATION_CASES]
            temporary_cases = [temporary_case_payload()]
            arxivmath_case = {"id": "ARXIVMATH", "test_kind": "arxivmath", "source": "MathArena 官方开源数据集", "executor": "scripts/run_arxivmath.py（Hugging Face）", "title": ARXIVMATH_CASE.title, "description": ARXIVMATH_CASE.description, "duration": "按题目数量执行", "cost": ARXIVMATH_CASE.cost, "protocols": list(ARXIVMATH_CASE.protocols), "concurrency": 1, "concurrency_text": "1（按题目串行）", "requests": None, "request_text": "数据集全量题目", "case_count": None, "case_count_text": "全量题目（运行时确定）", "input_tokens": "研究级数学问题", "warmup": None, "steady": None, "admission": "使用官方 answer 字段评分；题目和答案不在本项目中改写。"}
            deepswe_case = {"id": "DEEPSWE", "test_kind": "deepswe", "source": "DataCurve 官方开源数据集", "executor": "scripts/run_deepswe.py（Pier）", "title": DEEPSWE_CASE.title, "description": DEEPSWE_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": DEEPSWE_CASE.cost, "protocols": list(DEEPSWE_CASE.protocols), "concurrency": 1, "concurrency_text": "由 Pier 管理隔离任务", "requests": None, "request_text": "官方任务集（默认全量）", "case_count": 113, "case_count_text": "113 个官方任务", "input_tokens": "代码仓库 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 verifier 通过；使用官方任务、环境和测试，不在本项目重写。"}
            sweatlas_case = {"id": "SWE_ATLAS_TW", "test_kind": "swe-atlas-tw", "source": "Scale AI 官方开源数据集", "executor": "scripts/run_sweatlas_tw.py（Harbor）", "title": SWE_ATLAS_TW_CASE.title, "description": SWE_ATLAS_TW_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": SWE_ATLAS_TW_CASE.cost, "protocols": list(SWE_ATLAS_TW_CASE.protocols), "concurrency": 1, "concurrency_text": "由 Harbor 管理隔离任务", "requests": None, "request_text": "官方 TW 任务集", "case_count": 66, "case_count_text": "66 个官方任务（默认）", "input_tokens": "代码仓库 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 verifier 通过；使用官方任务、环境和测试，不在本项目重写。"}
            sweatlas_rf_case = {"id": "SWE_ATLAS_RF", "test_kind": "swe-atlas-rf", "source": "Scale AI 官方开源数据集", "executor": "scripts/run_sweatlas_rf.py（Harbor）", "title": SWE_ATLAS_RF_CASE.title, "description": SWE_ATLAS_RF_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": SWE_ATLAS_RF_CASE.cost, "protocols": list(SWE_ATLAS_RF_CASE.protocols), "concurrency": 1, "concurrency_text": "由 Harbor 管理隔离任务", "requests": None, "request_text": "官方 RF 任务集", "case_count": 66, "case_count_text": "66 个官方任务（默认）", "input_tokens": "代码仓库 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 verifier 通过；使用官方任务、环境和测试，不在本项目重写。"}
            legal_case = {"id": "LEGAL_AGENT_BENCHMARK", "test_kind": "legal-agent-benchmark", "source": "Harvey Labs v1.0 官方开源数据集", "executor": "scripts/run_legal_agent_benchmark.py（官方 harness + evaluator）", "title": LEGAL_AGENT_BENCHMARK_CASE.title, "description": LEGAL_AGENT_BENCHMARK_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": LEGAL_AGENT_BENCHMARK_CASE.cost, "protocols": list(LEGAL_AGENT_BENCHMARK_CASE.protocols), "concurrency": 1, "concurrency_text": "由官方 harness 管理 Agent 工具循环", "requests": None, "request_text": "官方 v1.0 Legal Agent 任务集", "case_count": 1, "case_count_text": "1 个任务（默认抽样）", "input_tokens": "法律材料 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 evaluator 完成评分；使用官方任务、材料、rubric 和 harness，不在本项目重写。"}
            return self.json_response(load_cases + strict_cases + kvv_cases + deepseek_cases + glm_cases + gpt_cases + awsb_cases + moderation_cases + temporary_cases + [arxivmath_case, deepswe_case, sweatlas_case, sweatlas_rf_case, legal_case])
        if path == "/api/runs":
            with db() as connection:
                rows = connection.execute("SELECT * FROM runs WHERE batch_id IS NULL ORDER BY created_at DESC LIMIT 200").fetchall()
            return self.json_response([public_run(row) for row in rows])
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
            run_id = parts[2]
            with db() as connection:
                row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                return self.json_response({"error": "记录不存在"}, 404)
            if len(parts) == 3:
                data=public_run(row)
                if data["is_batch"]:
                    with db() as connection:
                        children=connection.execute("SELECT * FROM runs WHERE batch_id=? ORDER BY created_at",(run_id,)).fetchall()
                    data["children"]=[public_run(child) for child in children]
                return self.json_response(data)
            kind = parts[3]
            if kind == "download":
                with db() as connection:
                    rows_to_pack = [row]
                    if row["is_batch"]:
                        rows_to_pack += connection.execute("SELECT * FROM runs WHERE batch_id=? ORDER BY created_at", (run_id,)).fetchall()
                archive = io.BytesIO()
                seen_dirs: set[str] = set()
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                    metadata = {
                        "run_id": run_id,
                        "is_batch": bool(row["is_batch"]),
                        "test_kind": row["test_kind"],
                        "model": row["model"],
                        "created_at": row["created_at"],
                        "included_records": [r["id"] for r in rows_to_pack],
                    }
                    bundle.writestr("package-metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
                    for packed_row in rows_to_pack:
                        result_dir = packed_row["result_dir"]
                        if not result_dir:
                            continue
                        target_dir = (ROOT / result_dir).resolve()
                        if RESULTS.resolve() not in target_dir.parents or not target_dir.is_dir():
                            continue
                        folder = target_dir.name
                        if folder in seen_dirs:
                            continue
                        seen_dirs.add(folder)
                        for source in target_dir.rglob("*"):
                            if source.is_file():
                                bundle.write(source, f"{folder}/{source.relative_to(target_dir)}")
                archive.seek(0)
                payload = archive.read()
                filename = f"test-data-{run_id}.zip"
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
            if kind == "requests":
                if not row["result_dir"]:
                    return self.json_response({"error": "文件尚未生成"}, 404)
                target = (ROOT / row["result_dir"] / "requests.jsonl").resolve()
                if RESULTS.resolve() not in target.parents or not target.is_file():
                    return self.json_response({"error": "文件不存在"}, 404)
                return self.text_response(target.read_text(encoding="utf-8", errors="replace"), "application/x-ndjson; charset=utf-8")
            key = {"report": "report_path", "summary": "summary_path", "log": "log_path", "manifest": "manifest_path"}.get(kind)
            if not key or not row[key]:
                return self.json_response({"error": "文件尚未生成"}, 404)
            target = (ROOT / row[key]).resolve()
            allowed = RESULTS.resolve()
            if allowed not in target.parents or not target.is_file():
                return self.json_response({"error": "文件不存在"}, 404)
            if kind == "summary":
                return self.text_response(target.read_text(encoding="utf-8", errors="replace"), "application/json; charset=utf-8")
            return self.text_response(target.read_text(encoding="utf-8", errors="replace"), "text/plain; charset=utf-8")
        if path in {"/", "/index.html"}:
            return self.serve_file(STATIC / "index.html")
        target = (STATIC / path.lstrip("/")).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self.text_response("Not Found", status=404)
        return self.serve_file(target)

    def do_POST(self) -> None:
        request_path = urlparse(self.path).path
        if request_path == "/api/runs/download":
            try:
                payload = self.body_json()
                requested = payload.get("run_ids") or []
                if not isinstance(requested, list) or not requested or len(requested) > 200:
                    raise ValueError("run_ids 必须是 1～200 个记录 ID")
                ids = list(dict.fromkeys(str(x).strip() for x in requested if str(x).strip()))
                with db() as connection:
                    placeholders = ",".join("?" * len(ids))
                    selected = connection.execute(f"SELECT * FROM runs WHERE id IN ({placeholders})", ids).fetchall()
                    rows_to_pack = list(selected)
                    batch_ids = [r["id"] for r in selected if r["is_batch"]]
                    if batch_ids:
                        placeholders = ",".join("?" * len(batch_ids))
                        rows_to_pack += connection.execute(f"SELECT * FROM runs WHERE batch_id IN ({placeholders}) ORDER BY created_at", batch_ids).fetchall()
                if len(selected) != len(ids):
                    raise ValueError("部分测试记录不存在")
                archive = io.BytesIO(); seen_dirs: set[str] = set()
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                    bundle.writestr("package-metadata.json", json.dumps({"type":"selected-runs","selected_run_ids":ids,"included_records":[r["id"] for r in rows_to_pack]}, ensure_ascii=False, indent=2) + "\n")
                    for packed_row in rows_to_pack:
                        result_dir = packed_row["result_dir"]
                        if not result_dir: continue
                        target_dir = (ROOT / result_dir).resolve()
                        if RESULTS.resolve() not in target_dir.parents or not target_dir.is_dir(): continue
                        folder = target_dir.name
                        if folder in seen_dirs: continue
                        seen_dirs.add(folder)
                        for source in target_dir.rglob("*"):
                            if source.is_file(): bundle.write(source, f"{folder}/{source.relative_to(target_dir)}")
                archive.seek(0); data = archive.read()
                self.send_response(200); self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="test-data-selected.zip"')
                self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data); return
            except (ValueError, json.JSONDecodeError) as exc:
                return self.json_response({"error": str(exc)}, 400)
        if request_path.startswith("/api/runs/") and request_path.endswith("/cancel"):
            run_id = request_path.split("/")[3]
            with db() as connection:
                row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
                children = connection.execute("SELECT id FROM runs WHERE batch_id=?", (run_id,)).fetchall() if row and row["is_batch"] else []
            if not row:
                return self.json_response({"error": "记录不存在"}, 404)
            ids = [run_id] + [x["id"] for x in children]
            with ACTIVE_LOCK:
                CANCELLED_RUNS.update(ids)
                procs = [ACTIVE_PROCESSES.get(x) for x in ids]
            for proc in procs:
                if proc and proc.poll() is None:
                    try: os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
            with db() as connection:
                connection.execute("UPDATE runs SET status='interrupted', finished_at=COALESCE(finished_at,?), error=COALESCE(error,'用户主动停止测试') WHERE id IN (%s) AND status IN ('queued','running')" % ",".join("?"*len(ids)), [now_iso(), *ids])
            return self.json_response({"id": run_id, "status": "interrupted", "message": "已发送停止信号；执行过的子任务将生成中断报告"})
        if request_path == "/api/connectivity/check":
            try:
                payload = self.body_json()
                protocol = str(payload.get("protocol", "")).strip().lower()
                base_url = str(payload.get("base_url", "")).strip()
                api_key = str(payload.get("api_key", "")).strip()
                model = str(payload.get("model", "")).strip()
                parsed = urlparse(base_url)
                if protocol not in PROTOCOLS:
                    raise ValueError("未知模型协议")
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    raise ValueError("base_url 必须是有效的 HTTP(S) 地址")
                if not api_key:
                    raise ValueError("api_key 不能为空")
                if not model:
                    raise ValueError("model 不能为空")
                config = ApiConfig(api_key=api_key, base_url=base_url, model=model, protocol=protocol)
                result = check_connectivity(config)
                return self.json_response(result, 200 if result["passed"] else 502)
            except (ValueError, json.JSONDecodeError) as exc:
                return self.json_response({"error": str(exc)}, 400)
        if request_path == "/api/models/query":
            try:
                payload = self.body_json()
                protocol = str(payload.get("protocol", "")).strip().lower()
                base_url = str(payload.get("base_url", "")).strip()
                api_key = str(payload.get("api_key", "")).strip()
                parsed = urlparse(base_url)
                if protocol not in PROTOCOLS:
                    raise ValueError("未知模型协议")
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    raise ValueError("base_url 必须是有效的 HTTP(S) 地址")
                if not api_key:
                    raise ValueError("api_key 不能为空")
                try:
                    response = httpx.get(models_endpoint(protocol, base_url), headers=models_headers(protocol, api_key), timeout=30, follow_redirects=False)
                except httpx.HTTPError as exc:
                    return self.json_response({"error": f"模型接口连接失败: {exc}"}, 502)
                if not 200 <= response.status_code < 300:
                    detail = response.text[:1000]
                    return self.json_response({"error": f"模型接口返回 HTTP {response.status_code}", "detail": detail}, 502)
                try:
                    models = parse_models(protocol, response.json())
                except (ValueError, json.JSONDecodeError):
                    return self.json_response({"error": "模型接口未返回有效 JSON"}, 502)
                return self.json_response({"protocol": protocol, "models": models, "count": len(models)})
            except (ValueError, json.JSONDecodeError) as exc:
                return self.json_response({"error": str(exc)}, 400)
        if request_path != "/api/runs":
            return self.json_response({"error": "Not Found"}, 404)
        try:
            payload = self.body_json()
            base_url = str(payload.get("base_url", "")).strip()
            api_key = str(payload.get("api_key", "")).strip()
            model = str(payload.get("model", "")).strip()
            requested_cases = payload.get("cases") or [payload.get("case")]
            if not isinstance(requested_cases, list): raise ValueError("cases 必须是数组")
            case_names = list(dict.fromkeys(str(item or "").strip() for item in requested_cases if str(item or "").strip()))
            case_name = case_names[0] if case_names else ""
            test_kind = str(payload.get("test_kind", "load")).strip().lower()
            protocol = str(payload.get("protocol", "volcengine")).strip().lower()
            quick = bool(payload.get("quick", False))
            custom_requests = payload.get("requests")
            custom_concurrency = payload.get("concurrency")
            if custom_requests is not None and (not isinstance(custom_requests, int) or not 1 <= custom_requests <= 100000): raise ValueError("发送请求数必须是 1～100000 的整数")
            if custom_concurrency is not None and (not isinstance(custom_concurrency, int) or not 1 <= custom_concurrency <= 1000): raise ValueError("并发数必须是 1～1000 的整数")
            if test_kind == "moderation" and custom_concurrency is None: custom_concurrency = 4
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("base_url 必须是有效的 HTTP(S) 地址")
            if not api_key:
                raise ValueError("api_key 不能为空")
            if not model:
                raise ValueError("model 不能为空")
            available = {"LEGAL_AGENT_BENCHMARK": LEGAL_AGENT_BENCHMARK_CASE} if test_kind == "legal-agent-benchmark" else {"SWE_ATLAS_RF": SWE_ATLAS_RF_CASE} if test_kind == "swe-atlas-rf" else {"SWE_ATLAS_TW": SWE_ATLAS_TW_CASE} if test_kind == "swe-atlas-tw" else {"DEEPSWE": DEEPSWE_CASE} if test_kind == "deepswe" else {"ARXIVMATH": ARXIVMATH_CASE} if test_kind == "arxivmath" else {"K3_TEMP_SUITE": temporary_case_payload()} if test_kind == "temporary" else CASES if test_kind == "load" else MODERATION_CASES if test_kind == "moderation" else STRICT_CASES if test_kind == "strict" else KVV_CASES if test_kind == "kvv" else DEEPSEEK_CASES if test_kind == "deepseek" else GLM_CASES if test_kind == "glm" else GPT_CASES if test_kind == "gpt" else AUTH_CASES if test_kind == "authenticity" else {spec.case_id: spec for spec in (*AWSB_CASE_SPECS, *AWSB_CACHE_CASE_SPECS)} if test_kind == "awsb" else {}
            if test_kind in {"kvv", "deepseek", "glm"}:
                available = dict(available)
                available.update({name: item for name, item in CONTRACT_CASES.items() if item["test_kind"] == test_kind})
            if not case_names or any(name not in available for name in case_names):
                raise ValueError("未知测试类型")
            if protocol not in PROTOCOLS:
                raise ValueError("未知模型协议")
            def protocols_for(name: str) -> tuple[str, ...]:
                if test_kind == "temporary":
                    return ("openai", "volcengine")
                if test_kind == "legal-agent-benchmark":
                    return LEGAL_AGENT_BENCHMARK_CASE.protocols
                if test_kind == "swe-atlas-rf":
                    return SWE_ATLAS_RF_CASE.protocols
                if test_kind == "swe-atlas-tw":
                    return SWE_ATLAS_TW_CASE.protocols
                if test_kind == "deepswe":
                    return DEEPSWE_CASE.protocols
                if test_kind == "arxivmath":
                    return ARXIVMATH_CASE.protocols
                if name in CONTRACT_CASES:
                    return CONTRACT_CASES[name]["protocols"]
                return KVV_CASES[name].protocols if test_kind == "kvv" else DEEPSEEK_CASES[name].protocols if test_kind == "deepseek" else GLM_CASES[name].protocols if test_kind == "glm" else GPT_CASES[name].protocols if test_kind == "gpt" else ("openai", "volcengine") if test_kind == "authenticity" else ("openai", "volcengine") if test_kind == "moderation" else ("anthropic",) if test_kind == "awsb" else STRICT_CASES[name].protocols if test_kind == "strict" else tuple(PROTOCOLS)
            unsupported = [name for name in case_names if protocol not in protocols_for(name)]
            if unsupported:
                raise ValueError(f"{', '.join(unsupported)} 不支持 {protocol} 协议")
            run_id = uuid.uuid4().hex[:16]; created=now_iso()
            if len(case_names) > 1:
                children=[]
                with db() as connection:
                    connection.execute("INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,is_batch,case_ids) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                       (run_id,test_kind,"MULTI",protocol,base_url,model,"queued",int(quick),created,1,json.dumps(case_names)))
                    for index,name in enumerate(case_names):
                        child_id=uuid.uuid4().hex[:16]; child_created=f"{created}.{index:03d}"
                        connection.execute("INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,batch_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                           (child_id,test_kind,name,protocol,base_url,model,"queued",int(quick),child_created,run_id))
                        children.append({"id":child_id,"batch_id":run_id,"test_kind":test_kind,"case":name,"protocol":protocol,"base_url":base_url,"api_key":api_key,"model":model,"quick":quick,"requests":custom_requests,"concurrency":custom_concurrency})
                JOB_QUEUE.put({"id":run_id,"is_batch":True,"test_kind":test_kind,"api_key":api_key,"children":children})
            else:
                with db() as connection:
                    connection.execute("INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,case_ids) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                       (run_id,test_kind,case_name,protocol,base_url,model,"queued",int(quick),created,json.dumps(case_names)))
                JOB_QUEUE.put({"id":run_id,"test_kind":test_kind,"case":case_name,"protocol":protocol,"base_url":base_url,"api_key":api_key,"model":model,"quick":quick,"requests":custom_requests,"concurrency":custom_concurrency})
            with db() as connection:
                row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return self.json_response(public_run(row), HTTPStatus.ACCEPTED)
        except (ValueError, json.JSONDecodeError) as exc:
            return self.json_response({"error": str(exc)}, 400)

    def serve_file(self, path: Path) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Volcengine K3 测试管理页面")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8085)
    args = parser.parse_args()
    initialize(); start_worker()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"测试管理页面：http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
