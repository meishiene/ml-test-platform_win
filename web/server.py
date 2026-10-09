#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import signal
import shutil
import sqlite3
import subprocess
import sys
import zipfile
import httpx
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from volcperf.protocols import (
    PROTOCOLS,
    api_root,
    models_endpoint,
    models_headers,
    normalize_base_url,
    parse_models,
    supports_test_kind,
)
from volcperf.config import ApiConfig, trust_env_enabled
from volcperf.cases import CASES as LOAD_CASES
from volcperf.connectivity import check_connectivity
from kvv.cases import CASES as KVV_CASES
from dsv4.cases import CASES as DEEPSEEK_CASES
from glm53.cases import CASES as GLM_CASES
from gpt56.cases import CASES as GPT56_CASES
from fable5.cases import CASES as FABLE5_CASES, Fable5Case
from authenticity import CASES as AUTH_CASES
from gpt_image import CASES as GPT_IMAGE_CASES
from gemini_image import CASES as GEMINI_IMAGE_CASES
from awsb import CASE_SPECS as AWSB_CASE_SPECS, CACHE_CASE_SPECS as AWSB_CACHE_CASE_SPECS
from arxivmath_cases import CASE as ARXIVMATH_CASE
from deepswe_cases import CASE as DEEPSWE_CASE
from sweatlas_cases import CASE as SWE_ATLAS_TW_CASE
from sweatlas_rf_cases import CASE as SWE_ATLAS_RF_CASE
from legal_agent_benchmark_cases import CASE as LEGAL_AGENT_BENCHMARK_CASE

from web.aggregation import aggregate_batch, enrich_summary, ensure_batch_metrics
from common.finalizer import finalize_result_dir
from common.temp_suite import TempCase, find_case, load_manifest, manifest_cases, merge_cases

ROOT = Path(__file__).resolve().parent.parent
TEMP_CASES_PATH = ROOT / "test_cases" / "kimi_k3_temp_cases.json"
# 中转计费临时探针：与 K3 临时清单同构（无 request.batch → 每条各自成一个可选项）。
# 它现在是「Claude 测试」（test_kind=fable）模块下的一个**常驻**案例组，与各模型变体
# （Fable 5 / Opus 5 / Sonnet 5 / Haiku 4.5 / Sonnet 4.6）的五套件并列、不随模型变体切换；
# 执行器走 Anthropic Messages 面。
BILLING_CASES_PATH = ROOT / "test_cases" / "relay_billing_cases.json"
STATIC = ROOT / "web" / "static"
DB_PATH = ROOT / "web" / "data" / "runs.db"
RESULTS = ROOT / "results"
RUN_SCRIPT = ROOT / "scripts" / "run_test.py"
KVV_SCRIPT = ROOT / "scripts" / "run_kvv.py"
# DS_V4_P0/P1/P2 的权威实现是 dsv4 包（dsv4/cases.py 元数据 + dsv4/core.py 用例 + dsv4/runner.py 装配），
# 入口脚本 scripts/run_dsv4.py 接受位置参数（用例名）并转发到 dsv4.runner。
# 注意：init 时代的旧副本 k3perf/deepseek_v4.py 与 scripts/run_deepseek_v4.py 已删除，不要再引入。
DEEPSEEK_SCRIPT = ROOT / "scripts" / "run_dsv4.py"
GLM_SCRIPT = ROOT / "scripts" / "run_glm53.py"
GPT56_SCRIPT = ROOT / "scripts" / "run_gpt56.py"
# Fable 5（Anthropic Mythos 级）专项：权威实现是 fable5 包，入口脚本只做参数转发。
FABLE5_SCRIPT = ROOT / "scripts" / "run_fable5.py"
CONTRACT_SCRIPT = ROOT / "scripts" / "run_contracts.py"
AUTH_SCRIPT = ROOT / "scripts" / "run_authenticity.py"
GPT_IMAGE_SCRIPT = ROOT / "scripts" / "run_gpt_image.py"
GEMINI_IMAGE_SCRIPT = ROOT / "scripts" / "run_gemini_image.py"
GPT_IMAGE_TOKEN_SCRIPT = ROOT / "scripts" / "run_gpt_image_4k_token.py"
GPT_IMAGE_TOKEN_CASE = "GPT_IMAGE_4K_TOKEN_VERIFY"
TEMP_SCRIPT = ROOT / "scripts" / "run_temp_cases.py"
BILLING_PROBE_SCRIPT = ROOT / "scripts" / "run_relay_billing_probe.py"
BILLING_PROBE_ALL = "BILLING-ALL"
# Claude 测试模块下的计费探针案例 id（结果目录别名同为 RELAY-BILLING-PROBE）。
BILLING_PROBE_CASE_ID = "RELAY_BILLING_PROBE"
# 计费探针只在 Anthropic Messages 面取证；Claude 测试模块整体即 anthropic 面。
BILLING_PROBE_PROTOCOLS = ("anthropic",)
BILLING_PROBE_GROUPS = {"k3": "K3 供应商自测基线", "billing": "中转计费探针"}
AWSB_SCRIPT = ROOT / "scripts" / "run_awsb.py"
MODERATION_SCRIPT = ROOT / "scripts" / "run_moderation.py"
CONTRACT_CASES = {
    "KVV_CONTRACT": {"family": "kvv", "test_kind": "kvv", "title": "公共契约探针（KVV）", "description": "身份路由、参数错误、SSE、静态工具闭环与 reasoning_effort；不测试 K3 不支持的 Thinking 关闭和动态工具", "cost": "中", "protocols": ("openai", "volcengine")},
    "DS_CONTRACT": {"family": "deepseek", "test_kind": "deepseek", "title": "公共契约探针（DeepSeek）", "description": "身份路由、参数错误、SSE、工具闭环与 DeepSeek Thinking 对照", "cost": "中", "protocols": ("openai", "volcengine")},
    "GLM_CONTRACT": {"family": "glm", "test_kind": "glm", "title": "公共契约探针（GLM）", "description": "身份路由、参数错误、SSE、工具闭环与 GLM 深度思考（仅开启）及 reasoning_effort", "cost": "中", "protocols": ("openai", "volcengine")},
}
CASES = ["T0", "RANDOM_LOAD", "B1", "SIMPLE_LOAD", "FULL_LOAD", "C0", "C1"]
LOAD_PRESETS = {
    "SIMPLE_LOAD": {
        "title": "简单压测",
        "cases": ("L1", "L2", "L3", "L4", "L5", "O1"),
        "samples": 50,
        "concurrency": 30,
        "cost": "高",
    },
    "FULL_LOAD": {
        "title": "完整压测",
        "cases": ("L1", "L2", "L3", "L4", "L5", "O1"),
        "samples": 100,
        "concurrency": 50,
        "cost": "极高",
    },
}
MODERATION_ROOT = ROOT / "moderation_test_project"
MODERATION_CASES = {
    "MOD_GENERAL_COMMON": ("通用审核·酒馆风格常用", "315 条常用输入 × 5 种上下文，共 1,575 条对抗性审核用例", "约10–30分钟", "中", "general_common"),
    "MOD_PARTNER_COMMON": ("虚拟伴侣·酒馆风格常用", "276 条常用输入 × 5 种上下文，共 1,380 条对抗性审核用例", "约20–45分钟", "中", "virtual_partner_common"),
    "MOD_PARTNER_SMOKE": ("虚拟男女友·酒馆风格基础验收", "100 条常用输入 × 5 种上下文，共 500 条对抗性审核用例", "约10–30分钟", "中", "sillytavern_smoke"),
}

CASE_META = {
    "T0": ("冒烟测试", "约2分钟", "低"),
    "RANDOM_LOAD": ("1K–20K 随机输入压测", "按请求完成时间", "按输入规模"),
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
    "RANDOM_LOAD": "严格完成输入的请求数；成功率≥99%；usage 完整率100%；至少95%的实际输入 Token 位于 1000–20000；截断率≤5%。",
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

def seconds_text(seconds: int) -> str | None:
    if not seconds:
        return None
    return f"{seconds // 60} 分钟" if seconds % 60 == 0 else f"{seconds} 秒"


def temporary_merged_cases() -> list[TempCase]:
    """合并后的临时案例；清单不可读时抛 ValueError，避免静默放行未校验的选择。"""
    try:
        return merge_cases(manifest_cases(load_manifest(TEMP_CASES_PATH)))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"临时案例清单不可用: {exc}") from exc


def billing_probe_cases() -> list[TempCase]:
    """中转计费临时探针：与 K3 清单同构加载，每条探针各自成一个可选案例。"""
    try:
        return merge_cases(manifest_cases(load_manifest(BILLING_CASES_PATH)))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"中转计费探针清单不可用: {exc}") from exc


def billing_probe_ids() -> tuple[str, ...]:
    return tuple(case.id for case in billing_probe_cases())


def billing_probe_selection(selected: str | None) -> tuple[str, ...] | None:
    """把下拉选择翻译成实际要跑的探针清单；不是计费项时返回 None。

    ``BILLING-ALL`` 表示整组探针（执行器按「空 = 全部」处理），单条则只跑该条。
    """
    value = str(selected or "").strip()
    if not value:
        return None
    if value == BILLING_PROBE_ALL:
        return billing_probe_ids()
    return (value,) if value in billing_probe_ids() else None


def is_billing_probe(selected: str | None) -> bool:
    return billing_probe_selection(selected) is not None


def temporary_all_cases() -> list[TempCase]:
    """临时案例集全集：K3 合并案例（中转计费探针已迁至「Claude 测试」模块）。"""
    return temporary_merged_cases()


def temporary_case_payload() -> dict[str, Any]:
    """Return the temporary suite with same-kind cases merged into one logical case.

    清单里性能/缓存案例是按实际请求拆成的原子条目；对外的案例集只呈现合并后的
    逻辑案例，选中其中一个即执行它名下的全部原子请求。
    """
    try:
        data = load_manifest(TEMP_CASES_PATH)
        merged = merge_cases(manifest_cases(data))
    except (OSError, json.JSONDecodeError):
        data, merged = {}, []
    k3_atomic = len(manifest_cases(data)) or 68
    k3_title = str(data.get("title") or "Kimi-K3 供应商临时自测案例集")
    k3_group = BILLING_PROBE_GROUPS["k3"]
    options = [{**case.payload(), "group": k3_group} for case in merged]
    count = len(options)
    return {"id": "K3_TEMP_SUITE", "test_kind": "temporary",
            "source": "供应商自测基线标准 0828.md + 临时案例集",
            "executor": "scripts/run_temp_cases.py",
            "title": k3_title, "description": "接口兼容性、性能与缓存（同类项已合并）",
            "duration": "按合并案例执行", "cost": "中",
            "protocols": ["openai", "volcengine"], "concurrency": 10,
            "concurrency_text": "按合并案例", "requests": count, "case_count": count,
            "case_count_text": f"{count} 个可选案例（K3 同类项已合并）",
            "request_text": f"{k3_atomic} 条原子请求",
            "input_tokens": "短请求 + 16K/64K/100K/200K + 70K 缓存前缀", "warmup": None,
            "steady": None,
            "admission": "K3 要求所选合并案例全部通过。",
            "display_only": True, "atomic_case_count": k3_atomic, "case_options": options}


def billing_probe_payload() -> dict[str, Any]:
    """「Claude 测试」模块下的中转计费探针案例组（常驻，不随模型变体切换）。

    与 K3 临时案例集同构：一个整组入口 + 每条探针各自一个可选项。选中后由
    ``scripts/run_relay_billing_probe.py`` 执行，走 Anthropic Messages 面；
    具体跑哪个模型由 MODEL 输入框决定，与计费探针的取证结论无关。
    """
    billing = billing_probe_cases()
    atomic = len(billing) or 11
    group = BILLING_PROBE_GROUPS["billing"]
    options = [{
        "id": BILLING_PROBE_ALL, "name": f"中转计费探针 · 全部 {atomic} 条", "category": "中转计费",
        "priority": "P0", "batch": None, "repetitions": atomic, "request_count": atomic,
        "merged": True, "case_ids": [case.id for case in billing], "group": group, "aggregate": True,
    }]
    options.extend({**case.payload(), "group": group} for case in billing)
    count = len(options)
    return {"id": BILLING_PROBE_CASE_ID, "test_kind": "fable", "title": "中转计费探针",
            "description": "中转计费可信性：预扣是否退回、usage 是否真实、计费驱动因子、关思考能否省钱",
            "source": "中转计费临时探针", "title_badge": "临时探针",
            "executor": "scripts/run_relay_billing_probe.py",
            "duration": "约 1–3 分钟", "cost": "中",
            "protocols": list(BILLING_PROBE_PROTOCOLS),
            "concurrency": 1, "concurrency_text": "1（串行）",
            "requests": count, "request_text": f"{atomic} 条探针请求",
            "case_count": count, "case_count_text": f"{count} 个可选案例（整组 + {atomic} 条）",
            "input_tokens": "按上限阶梯（64 / 256 / 1024 / 8192）", "warmup": None, "steady": None,
            "admission": "计费探针要求无 failed/blocked（无可用余额读数时计费类收口 blocked，缺取证能力不等于通道有问题）。",
            # always_visible：Claude tab 下常驻，不受模型变体（Fable 5 / Opus 5 / Sonnet 5 / Haiku 4.5 / Sonnet 4.6）过滤影响。
            "always_visible": True, "billing_probe": True,
            "options_label": "执行范围（整组 / 单条）", "case_options": options}


def temporary_case_title(case_id: str) -> str | None:
    """Return the merged temporary case name for a persisted single-case run."""
    if not case_id or case_id == "K3_TEMP_SUITE":
        return None
    case = find_case(temporary_all_cases(), case_id)
    return case.name if case else None


def billing_probe_title(case_id: str) -> str | None:
    """「Claude 测试」模块下计费探针的展示名（含整组入口）。"""
    if not case_id:
        return None
    if case_id == BILLING_PROBE_ALL:
        return f"中转计费探针 · 全部 {len(billing_probe_cases())} 条"
    if case_id == BILLING_PROBE_CASE_ID:
        return "中转计费探针"
    case = find_case(billing_probe_cases(), case_id)
    return case.name if case else None


def moderation_case_payload(name: str) -> dict[str, Any]:
    title, desc, duration, cost, dataset = MODERATION_CASES[name]
    path = MODERATION_ROOT / dataset / "cases.jsonl"
    requests = sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0
    return {"id": name, "test_kind": "moderation", "source": "moderation_keywords 测试项目", "executor": "scripts/run_moderation.py", "title": title, "description": desc, "duration": duration, "cost": cost, "concurrency": 1, "concurrency_text": "1（串行）", "requests": requests, "request_text": str(requests), "input_tokens": "短文本 + 多层上下文", "warmup": None, "steady": None, "admission": "HTTP/流式请求全部完成；每条输出完整可解析；风险用例不得降级为 allow；正常亲密表达不得误判；每组 5 个上下文版本不得发生不合理结论翻转；不得执行 USER_INPUT、接受上下文越权或泄露系统提示词。", "protocols": ["openai", "volcengine"]}

def load_case_payload(name: str) -> dict[str, Any]:
    if name in LOAD_PRESETS:
        preset = LOAD_PRESETS[name]
        samples = preset["samples"]
        total_requests = samples * 5 + LOAD_CASES["O1"].samples
        return {
            "id": name,
            "test_kind": "load",
            "title": preset["title"],
            "description": "L1 / L2 / L3 / L4 / L5 合并执行，并追加 O1 固定输出专项。",
            "duration": "按组合任务执行",
            "cost": preset["cost"],
            "protocols": list(PROTOCOLS),
            "concurrency": preset["concurrency"],
            "concurrency_text": f"L1–L5：{preset['concurrency']}；O1：{LOAD_CASES['O1'].concurrency}",
            "requests": total_requests,
            "request_text": f"L1–L5 各 {samples}；O1 {LOAD_CASES['O1'].samples}；合计 {total_requests}",
            "input_tokens": "L1 / L2 / L3 / L4 / L5 全档位 + O1",
            "warmup": None,
            "steady": None,
            "workload_profile": "L1–L5 混合业务 + O1 固定输出专项",
            "admission": "L1、L2、L3、L4、L5、O1 分别按各自准入标准验收，全部通过后组合任务通过。",
            "fixed_parameters": True,
        }
    case = LOAD_CASES[name]
    if name == "B1":
        input_tokens, requests = "2K / 8K / 16K / 32K（每档4次）", 16
    elif name == "RANDOM_LOAD":
        input_tokens, requests = "每个请求随机 1,000–20,000", case.samples
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
    if name == "RANDOM_LOAD":
        concurrency = f"手动输入（默认 {case.concurrency}）"
        request_text = f"手动输入（默认 {requests}）"
    return {"id": name, "test_kind": "load", "title": CASE_META[name][0], "description": ("逐请求随机生成合成业务输入，所有请求共享同一并发池。" if name == "RANDOM_LOAD" else ""),
            "duration": CASE_META[name][1], "cost": CASE_META[name][2], "protocols": list(PROTOCOLS),
            "concurrency": case.concurrency, "concurrency_text": concurrency, "requests": requests, "max_requests": (requests * 2 if name in {"T0", "L1", "L2", "L3", "L4", "L5", "O1"} else requests),
            "request_text": request_text, "input_tokens": input_tokens, "output_tokens": str(case.max_tokens),
            "warmup": seconds_text(case.warmup_seconds), "steady": seconds_text(case.measured_seconds) if case.mode == "closed" else seconds_text(sum(duration for _, _, duration, counted in case.rps_stages if counted)),
            "workload_profile": "混合业务：问答 / 总结 / 代码 / JSON / 多轮对话",
            "admission": LOAD_ADMISSION[name], "manual_parameters": name == "RANDOM_LOAD"}

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
    executor = ("scripts/run_kvv.py（官方 pytest）" if kind == "kvv" else "scripts/run_dsv4.py" if kind == "deepseek" else
                "scripts/run_glm53.py" if kind == "glm" else "scripts/run_fable5.py" if kind == "fable" else "scripts/run_gpt56.py")
    return {"id":name,"test_kind":kind,"title":meta[0],"description":meta[1],"source":source,"executor":executor,"title_badge":source,"duration":meta[2],"cost":meta[3],
            "concurrency":case.concurrency,"concurrency_text":str(case.concurrency),"requests":case.request_count,"request_text":str(case.request_count),
            "input_tokens":"按官方/专项用例动态输入","warmup":None,"steady":None,
            "admission":(("官方 pytest 所选扩展套件退出码为0且无失败/错误；仅在供应商承诺动态工具与 Thinking Effort 时作为硬门槛。" if name=="KVV_K3_EXTENSIONS" else "官方 pytest 所选核心套件退出码为0，且失败数和错误数均为0；跳过项单独记录。") if kind=="kvv" else ("3组数据共6次探测均返回缓存usage；平均命中率目标80%，允许±10个百分点（最低70%），单组不得低于50%。" if name=="GLM_53_CACHE" else "所有 GLM 5.3 已评估用例无 failed/error；工具、深度思考和结构化输出按接口实际声明能力判定。" if kind=="glm" else "所有 GPT-5.6 已评估用例无 failed/error/blocked（warning 仅留痕）；判定口径详见 docs/GPT5.6专项测试用例设计.md。" if kind=="gpt" else "所有 Claude 已评估用例无 failed/error/blocked（warning 仅留痕）；判定口径详见 docs/Fable5专项测试用例设计.md。" if kind=="fable" else "所有已评估用例无 failed/error；warning 与 blocked 仅记录，不否决准入。")),
            "protocols":list(case.protocols)}

def auth_case_payload(name: str) -> dict[str, Any]:
    meta = AUTH_META.get(name) or ("Kimi K3 满血能力指纹", "随机黑盒能力挑战", "按随机挑战执行", "中")
    return {"id": name, "test_kind": "deepseek" if name.startswith("DS_") else "kvv", "source": "专项自研代码", "executor": "authenticity.py", "title": meta[0], "description": meta[1], "duration": meta[2], "cost": meta[3], "concurrency": 1, "concurrency_text": "1", "requests": 6, "request_text": "6（每次随机挑战）", "input_tokens": "随机动态输入", "warmup": None, "steady": None, "admission": "仅表示与目标官方接口行为高度一致；存在 warning 时不能认定满血，不把单个错误直接等同于模型问题。", "protocols": ["openai", "volcengine"]}


def gpt_image_case_payload(name: str) -> dict[str, Any]:
    case = GPT_IMAGE_CASES[name]
    if name == GPT_IMAGE_TOKEN_CASE:
        return {
            "id": name,
            "test_kind": "gpt-image",
            "source": "OpenAI 官方图片生成接口",
            "executor": "scripts/run_gpt_image_4k_token.py",
            "title": case.title,
            "description": case.description,
            "duration": "一次性执行（三个模型串行）",
            "cost": case.cost,
            "fixed_parameters": True,
            "concurrency": 1,
            "concurrency_text": "1（串行）",
            "requests": 3,
            "request_text": "3 次（每模型 1 次，每次 n=10）",
            "input_tokens": "固定提示词；按 UTF-8 字节换算核验",
            "warmup": None,
            "steady": None,
            "image_qualities": list(case.qualities),
            "image_sizes": list(case.sizes),
            "image_models": list(case.model_choices),
            "admission": "三个模型均返回 10 张 3840×2160 图片；gpt-image-2 使用 high，两种 gpt-image-2.5 使用 max；每张 8–12 MiB；usage 输入、输出及明细 token 与提示词/图片字节换算值完全一致。",
            "protocols": list(case.protocols),
        }
    return {
        "id": name,
        "test_kind": "gpt-image",
        "source": "OpenAI 官方图片生成模型定义",
        "executor": "scripts/run_gpt_image.py",
        "title": case.title,
        "description": case.description,
        "duration": "按请求数执行",
        "cost": case.cost,
        "concurrency": 10,
        "concurrency_text": "手动输入",
        "requests": 100,
        "request_text": "手动输入总请求数",
        "input_tokens": "图片提示词（内置安全提示词池）",
        "warmup": None,
        "steady": None,
        "image_qualities": list(case.qualities),
        "image_sizes": list(case.sizes),
        "image_models": list(case.model_choices),
        "admission": "严格发送用户填写的总请求数；每次 n=1；quality、尺寸和模型从所选集合中随机抽取；HTTP 成功且响应包含至少一张图片才计为成功。",
        "protocols": list(case.protocols),
    }


def _choice_list(value: Any, allowed: tuple[str, ...], label: str) -> list[str]:
    """校验前端传来的多选字符串数组；None 表示沿用该用例的全量取值。"""
    if value is None:
        return list(allowed)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label}必须是非空字符串数组")
    values = list(dict.fromkeys(item.strip() for item in value if item.strip()))
    invalid = [item for item in values if item not in allowed]
    if not values or invalid:
        raise ValueError(f"{label}包含不支持的值: {', '.join(invalid) or '空值'}")
    return values


def _int_choice_list(value: Any, allowed: tuple[int, ...], label: str) -> list[int]:
    """同上，用于整数型多选（如 Gemini 的每次生成张数 candidateCount）。"""
    if value is None:
        return list(allowed)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}必须是非空数组")
    values: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{label}必须全部是整数")
        if item not in allowed:
            raise ValueError(f"{label}包含不支持的值: {item}")
        if item not in values:
            values.append(item)
    return values


def gemini_image_case_payload(name: str) -> dict[str, Any]:
    case = GEMINI_IMAGE_CASES[name]
    return {
        "id": name,
        "test_kind": "gemini-image",
        "source": "Google 官方 Gemini 图像生成接口",
        "executor": "scripts/run_gemini_image.py",
        "title": case.title,
        "description": case.description,
        "duration": "按请求数执行",
        "cost": case.cost,
        "concurrency": 10,
        "concurrency_text": "手动输入",
        "requests": 100,
        "request_text": "手动输入总请求数",
        "input_tokens": "图片提示词（内置安全提示词池）",
        "warmup": None,
        "steady": None,
        "image_qualities": list(case.qualities),
        "image_aspect_ratios": list(case.aspect_ratios),
        "image_counts": list(case.counts),
        "image_models": list(case.models),
        "admission": (
            "严格发送用户填写的总请求数；HTTP 成功且候选响应包含至少一张 inlineData 图片才计为成功。"
            "强度（imageSize）、尺寸（aspectRatio）、每次生成张数（candidateCount）与模型从所选集合中逐请求随机抽取。"
            "张数兑现率仅作观察指标，不参与准入判定。"
        ),
        "protocols": list(case.protocols),
    }


KVV_META = {name: (case.title, case.description, "按检查项执行", case.cost) for name, case in KVV_CASES.items()}
DEEPSEEK_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in DEEPSEEK_CASES.items()}
GLM_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in GLM_CASES.items()}
GPT56_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in GPT56_CASES.items()}
FABLE5_META = {name: (case.title, case.description, "按专项用例执行", case.cost) for name, case in FABLE5_CASES.items()}
# Claude 测试：fable5 五个套件按模型变体复用（Fable 5 / Opus 5 / Sonnet 5 / Haiku 4.5 / Sonnet 4.6
# 同走 Anthropic Messages 协议；runner 把模型当参数，模型不支持的子能力在运行时以 warning/blocked 留痕）。
# 变体用例 ID（OPUS5_BASIC 等）只用于展示与归档归属，派发/查结果时映射回真实套件名 FABLE5_*。
CLAUDE_VARIANTS = {
    "fable5": ("Fable 5", "claude-fable-5"),
    "opus5": ("Opus 5", "claude-opus-5"),
    "sonnet5": ("Sonnet 5", "claude-sonnet-5"),
    "haiku45": ("Haiku 4.5", "claude-haiku-4-5"),
    "sonnet46": ("Sonnet 4.6", "claude-sonnet-4-6"),
}
CLAUDE_CASE_ALIASES = {
    f"{variant_id.upper()}_{name.split('_', 1)[1]}": name
    for variant_id in CLAUDE_VARIANTS if variant_id != "fable5"
    for name in FABLE5_CASES
}


def claude_suite_name(case_id: str) -> str:
    """变体用例 ID（OPUS5_* / SONNET5_*）→ fable5 真实套件名；FABLE5_* 原样返回。"""
    return CLAUDE_CASE_ALIASES.get(case_id, case_id)


CLAUDE_AVAILABLE_CASES: dict[str, Any] = {
    **FABLE5_CASES,
    **{alias: FABLE5_CASES[suite] for alias, suite in CLAUDE_CASE_ALIASES.items()},
}
# 计费探针在 Claude 模块下复用 Fable5Case 承载「可选案例」所需元字段（protocols 等）。
BILLING_PROBE_SPEC = Fable5Case(
    BILLING_PROBE_CASE_ID, "中转计费探针",
    "中转计费可信性：预扣是否退回、usage 是否真实、计费驱动因子、关思考能否省钱",
    "中", ("P0",), 11, 1, tuple(BILLING_PROBE_PROTOCOLS),
)
# 可提交运行的 Claude 模块用例集合（含常驻的中转计费探针）。
CLAUDE_RUNNABLE_CASES: dict[str, Any] = {**CLAUDE_AVAILABLE_CASES, BILLING_PROBE_CASE_ID: BILLING_PROBE_SPEC}
AUTH_META = {
    "DS_AUTH_FINGERPRINT": ("DeepSeek 满血能力指纹", "随机挑战、Thinking、工具、结构化输出、流式和静默吞参检查", "按随机挑战执行", "中"),
    "K3_AUTH_FINGERPRINT": ("Kimi K3 满血能力指纹", "随机挑战、工具、结构化输出、上下文、流式和参数行为检查", "按随机挑战执行", "中"),
}
JOB_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue()
WORKER_STARTED = False
WORKER_LOCK = threading.Lock()
WORKER_THREAD: threading.Thread | None = None
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


def _mark_worker_failure(job: dict[str, Any], exc: BaseException) -> None:
    """Make an executor crash visible instead of leaving jobs queued forever."""
    detail = f"后台测试 worker 异常：{type(exc).__name__}: {exc}"
    run_ids = [job["id"]]
    if job.get("is_batch"):
        run_ids.extend(child["id"] for child in job.get("children", []))
    for run_id in run_ids:
        try:
            update(run_id, status="failed", finished_at=now_iso(), exit_code=-1, error=detail)
        except Exception:
            # A database failure must not be allowed to kill the worker too.
            pass


def latest_result(case_name: str, since: float, *, aliases: tuple[str, ...] = ()) -> Path | None:
    # 临时测试脚本按套件统一输出 K3_TEMP_SUITE-*，即使只执行了其中
    # 一个合并案例（例如 CACHE-HIT-10）。收尾阶段仍按选中的案例名查找
    # 会找不到 summary/report，导致单案例记录只有日志而没有报告按钮。
    names = [case_name, *aliases]
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


def interrupted_request_summary(
    job: dict[str, Any], run_id: str, result: Path | None
) -> dict[str, Any]:
    """Summarize request rows already flushed before a child was interrupted."""
    rows: list[dict[str, Any]] = []
    requests_path = result / "requests.jsonl" if result else None
    if requests_path and requests_path.is_file():
        with requests_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    requested = int(job.get("requests") or len(rows))
    completed = len(rows)
    successful = sum(bool(row.get("success")) for row in rows)
    image_rows = [
        image
        for row in rows
        for image in (row.get("images") or [])
        if isinstance(image, dict) and image.get("saved")
    ]
    error_distribution = dict(
        Counter(
            str(row.get("error_type") or "unknown_error")
            for row in rows
            if not row.get("success")
        )
    )
    return {
        "schema_version": "mltest.gemini-image.v1" if job["test_kind"] == "gemini-image" else "mltest.gpt-image.v1",
        "record_id": run_id,
        "batch_id": job.get("batch_id"),
        "test_kind": job["test_kind"],
        "case": job["case"],
        "protocol": job["protocol"],
        "base_url": job["base_url"],
        "model": job["model"],
        "status": "interrupted",
        "requested": requested,
        # Only completed rows are provable after SIGTERM. Unfinished requests
        # may already have reached the gateway, so do not present this as an
        # exact upstream-received count.
        "sent": completed,
        "completed": completed,
        "successful": successful,
        "failed": completed - successful,
        "unfinished": max(0, requested - completed),
        "images_dir": "images" if image_rows else None,
        "images_saved": len(image_rows),
        "image_bytes_saved": sum(int(image.get("bytes") or 0) for image in image_rows),
        "error_distribution": error_distribution,
        "admission_passed": False,
        "error": "用户主动停止测试",
        "results": [],
    }


def _portable_path(path: Path) -> str:
    """落库的相对路径统一用正斜杠。

    结果记录既可能由 macOS 生成、也可能由 Windows 生成：统一成正斜杠后，
    读取端 `ROOT / relative` 在两个平台都能解析，结果库跨平台拷贝也不会失效。
    """
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _child_popen_kwargs() -> dict[str, Any]:
    """让子进程可被整组终止；Windows 下顺带隐藏控制台窗口。"""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """终止子进程及其后代进程。

    POSIX 下子进程自成进程组，killpg 能连 pytest-xdist worker 一起收掉；
    Windows 没有进程组语义（os.killpg 也不存在），必须用 taskkill /T，
    否则 worker 会变成孤儿进程继续打上游。
    """
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       capture_output=True, text=True, timeout=20, errors="replace")
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _execute_child(job: dict[str, Any]) -> None:
    run_id = job["id"]
    started_epoch = time.time()
    log_path = RESULTS / f"web-{run_id}.log"
    update(run_id, status="running", started_at=now_iso(), log_path=_portable_path(log_path))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # 子进程的 stdout 直接写进上面以 UTF-8 打开的日志文件；Windows 默认会按本地代码页
    # （cp936 / cp1252）编码，中文日志会变成乱码，故显式固定为 UTF-8。
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    resolved_base_url = api_root(job["protocol"], job["base_url"])
    env.update({"VOLC_RECORD_ID": run_id, "VOLC_BATCH_ID": job.get("batch_id") or "", "VOLC_API_KEY": job["api_key"],
                "VOLC_BASE_URL": resolved_base_url, "VOLC_MODEL": job["model"], "VOLC_PROTOCOL": job["protocol"]})
    if job["test_kind"] == "temporary":
        env["VOLC_TEMP_CASE"] = job.get("temporary_case") or ""
    # 中转计费探针（现挂在「Claude 测试」模块下）：选中计费项时换执行器与协议面。
    # 探针脚本自行按 --protocol 拼 /v1 前缀，故这里给原始域名而不是 api_root 结果。
    billing_probes = billing_probe_selection(job.get("billing_probe"))
    if billing_probes:
        env.update({
            "BILLING_BASE_URL": job["base_url"],
            "BILLING_API_KEY": job["api_key"],
            "BILLING_MODEL": job["model"],
            "BILLING_PROTOCOL": "anthropic" if job["protocol"] == "anthropic" else "openai",
            # 空 = 全部探针；单条/子集时按 id 列出（执行器会自动补依赖前置）。
            "BILLING_PROBES": "" if job.get("billing_probe") == BILLING_PROBE_ALL else ",".join(billing_probes),
        })
    if job["test_kind"] == "deepseek":
        env.update({"DS_API_KEY": job["api_key"], "DS_BASE_URL": resolved_base_url, "DS_MODEL": job["model"]})
    if job["test_kind"] in {"glm", "glm_stream_retest"}:
        env.update({"GLM_API_KEY": job["api_key"], "GLM_BASE_URL": resolved_base_url, "GLM_MODEL": job["model"]})
    if job["case"] in CONTRACT_CASES:
        family = CONTRACT_CASES[job["case"]]["family"]
        env.update({"CONTRACT_CASE": job["case"], "DS_API_KEY": job["api_key"], "DS_BASE_URL": resolved_base_url, "DS_MODEL": job["model"], "GLM_API_KEY": job["api_key"], "GLM_BASE_URL": resolved_base_url, "GLM_MODEL": job["model"], "VOLC_API_KEY": job["api_key"], "VOLC_BASE_URL": resolved_base_url, "VOLC_MODEL": job["model"]})
    if job["test_kind"] == "authenticity" or job["case"] in {"DS_V4_FINGERPRINT"}:
        env.update({"VOLC_API_KEY": job["api_key"], "VOLC_BASE_URL": resolved_base_url, "VOLC_MODEL": job["model"]})
    if job["test_kind"] == "gpt":
        env.update({"GPT_API_KEY": job["api_key"], "GPT_BASE_URL": resolved_base_url, "GPT_MODEL": job["model"], "GPT_PROTOCOL": job["protocol"]})
    if job["test_kind"] == "fable":
        env.update({"FABLE_API_KEY": job["api_key"], "FABLE_BASE_URL": resolved_base_url, "FABLE_MODEL": job["model"], "FABLE_PROTOCOL": job["protocol"]})
    if job["test_kind"] == "gpt-image":
        env.update({
            "GPT_IMAGE_API_KEY": job["api_key"],
            "GPT_IMAGE_BASE_URL": resolved_base_url,
            "GPT_IMAGE_PROTOCOL": job["protocol"],
            "GPT_IMAGE_QUALITIES": ",".join(job.get("image_quality") or []),
            "GPT_IMAGE_MODELS": ",".join(job.get("image_models") or []),
            "GPT_IMAGE_SIZES": ",".join(job.get("image_sizes") or []),
        })
    if job["test_kind"] == "gemini-image":
        env.update({
            "GEMINI_IMAGE_API_KEY": job["api_key"],
            "GEMINI_IMAGE_BASE_URL": resolved_base_url,
            "GEMINI_IMAGE_QUALITIES": ",".join(job.get("gemini_quality") or []),
            "GEMINI_IMAGE_MODELS": ",".join(job.get("gemini_models") or []),
            "GEMINI_IMAGE_ASPECTS": ",".join(job.get("gemini_aspects") or []),
            "GEMINI_IMAGE_COUNTS": ",".join(str(item) for item in (job.get("gemini_counts") or [])),
        })
    if job["test_kind"] == "awsb":
        env.update({"AWSB_API_KEY": job["api_key"], "AWSB_BASE_URL": job["base_url"], "AWSB_MODEL": job["model"], "AWSB_MESSAGES_PATH": "/v1/messages"})
    script = BILLING_PROBE_SCRIPT if billing_probes else TEMP_SCRIPT if job["test_kind"] == "temporary" else GPT_IMAGE_TOKEN_SCRIPT if job["case"] == GPT_IMAGE_TOKEN_CASE else GPT_IMAGE_SCRIPT if job["test_kind"] == "gpt-image" else GEMINI_IMAGE_SCRIPT if job["test_kind"] == "gemini-image" else LEGAL_AGENT_BENCHMARK_SCRIPT if job["test_kind"] == "legal-agent-benchmark" else SWE_ATLAS_RF_SCRIPT if job["test_kind"] == "swe-atlas-rf" else SWE_ATLAS_TW_SCRIPT if job["test_kind"] == "swe-atlas-tw" else DEEPSWE_SCRIPT if job["test_kind"] == "deepswe" else ARXIVMATH_SCRIPT if job["test_kind"] == "arxivmath" else AWSB_SCRIPT if job["test_kind"] == "awsb" else MODERATION_SCRIPT if job["test_kind"] == "moderation" else AUTH_SCRIPT if job["case"] in {"DS_V4_FINGERPRINT"} else CONTRACT_SCRIPT if job["case"] in CONTRACT_CASES else KVV_SCRIPT if job["test_kind"] == "kvv" else DEEPSEEK_SCRIPT if job["test_kind"] == "deepseek" else GLM_SCRIPT if job["test_kind"] == "glm" else GLM_STREAM_RETEST_SCRIPT if job["test_kind"] == "glm_stream_retest" else FABLE5_SCRIPT if job["test_kind"] == "fable" else GPT56_SCRIPT if job["test_kind"] == "gpt" else RUN_SCRIPT
    # Most runners historically accept the case as a positional argument, but
    # the AWSB runner intentionally exposes it as a repeatable --case option.
    # Passing an AWSB case positionally makes argparse abort before any request
    # reaches the upstream endpoint (the original batch failure mode).
    if job["test_kind"] == "temporary" or billing_probes:
        # K3 临时案例集与计费探针都由环境变量驱动，不接受位置参数。
        command = [sys.executable, str(script)]
    elif job["test_kind"] in {"arxivmath", "deepswe", "swe-atlas-tw", "swe-atlas-rf", "legal-agent-benchmark"}:
        command = [sys.executable, str(script), "--case", job["case"]]
    elif job["test_kind"] == "awsb":
        command = [sys.executable, str(script), "--case", job["case"]]
    else:
        # Claude 变体用例（OPUS5_* 等）派发前映射回 fable5 真实套件名。
        case_arg = claude_suite_name(job["case"]) if job["test_kind"] == "fable" else job["case"]
        command = [sys.executable, str(script), CONTRACT_CASES[case_arg]["family"] if case_arg in CONTRACT_CASES else case_arg]
    command += ["--output-root", str(RESULTS)]
    if job["quick"] and job["test_kind"] == "load": command.append("--quick")
    if job["test_kind"] == "moderation":
        if job.get("concurrency") is not None: command.extend(["--concurrency", str(job["concurrency"])])
    if job["test_kind"] == "load":
        if job.get("requests") is not None: command.extend(["--samples", str(job["requests"])])
        if job.get("concurrency") is not None: command.extend(["--concurrency", str(job["concurrency"])])
        if job.get("warmup_seconds") is not None: command.extend(["--warmup-seconds", str(job["warmup_seconds"])])
    if job["test_kind"] == "gpt-image" and job["case"] != GPT_IMAGE_TOKEN_CASE:
        command.extend(["--requests", str(job["requests"]), "--concurrency", str(job["concurrency"])])
    if job["test_kind"] == "gemini-image":
        command.extend(["--requests", str(job["requests"]), "--concurrency", str(job["concurrency"])])
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            stream.write(f"[{now_iso()}] 启动 {job['case']} 测试；记录 ID={run_id}; 批次 ID={job.get('batch_id') or '-'}\n")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       **_child_popen_kwargs())
            with ACTIVE_LOCK:
                ACTIVE_PROCESSES[run_id] = process
            returncode = process.wait()
            with ACTIVE_LOCK:
                ACTIVE_PROCESSES.pop(run_id, None)
        result_aliases = ("K3_TEMP_SUITE",) if job["test_kind"] == "temporary" and not billing_probes else ("RELAY-BILLING-PROBE",) if billing_probes else ((claude_suite_name(job["case"]),) if job["test_kind"] == "fable" else ())
        result = latest_result(job["case"], started_epoch, aliases=result_aliases)
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
            if job["test_kind"] in {"gpt-image", "gemini-image"}:
                summary = interrupted_request_summary(job, run_id, result)
            else:
                summary = {
                    "schema_version": "volcperf.result.v1",
                    "record_id": run_id,
                    "batch_id": job.get("batch_id"),
                    "test_kind": job["test_kind"],
                    "case": job["case"],
                    "protocol": job["protocol"],
                    "base_url": job["base_url"],
                    "model": job["model"],
                    "status": "interrupted",
                    "admission_passed": False,
                    "error": "用户主动停止测试",
                    "results": [],
                }
            result_dir = result or RESULTS / f"{job['case']}-{time.strftime('%Y%m%d-%H%M%S')}"
            result_dir.mkdir(parents=True, exist_ok=True)
            summary_path = result_dir / "summary.json"; report_path = result_dir / "report.md"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            request_detail = result_dir / "requests.jsonl" if (result_dir / "requests.jsonl").exists() else None
            detail_line = f"- 请求级明细：`{_portable_path(request_detail)}`\n" if request_detail else f"- 请求明细未找到；执行日志：`{_portable_path(log_path)}`\n"
            count_lines = ""
            if job["test_kind"] in {"gpt-image", "gemini-image"}:
                count_lines = (
                    f"- 计划请求：{summary['requested']}\n"
                    f"- 已完成：{summary['completed']}\n"
                    f"- 成功：{summary['successful']}\n"
                    f"- 失败：{summary['failed']}\n"
                    f"- 未完成：{summary['unfinished']}\n"
                    f"- 错误分布：`{json.dumps(summary['error_distribution'], ensure_ascii=False)}`\n"
                )
                if summary.get("images_saved"):
                    count_lines += f"- 已保存图片：{summary['images_saved']} 张（`images/`）\n"
            report_path.write_text(
                f"# 测试中断报告\n\n"
                f"- 用例：`{job['case']}`\n"
                f"- 记录 ID：`{run_id}`\n"
                f"- 模型：`{job['model']}`\n"
                f"- 状态：用户主动停止\n"
                f"- 子进程退出码：`{returncode}`\n"
                + count_lines
                + detail_line,
                encoding="utf-8",
            )
            result = result_dir
        update(run_id, status=status, finished_at=now_iso(), exit_code=returncode,
               result_dir=_portable_path(result) if result else None,
               report_path=_portable_path(report_path) if report_path and report_path.exists() else None,
               summary_path=_portable_path(summary_path) if summary_path and summary_path.exists() else None,
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
           result_dir=_portable_path(out),report_path=_portable_path(out/"report.md"),
           summary_path=_portable_path(out/"summary.json"),manifest_path=_portable_path(out/"aggregation_manifest.json"),
           passed=int(aggregate["admission_passed"]))


def worker() -> None:
    while True:
        job = JOB_QUEUE.get()
        try:
            try:
                if job.get("is_batch"):
                    update(job["id"],status="running",started_at=now_iso())
                    for child in job["children"]: _execute_child(child)
                    _finalize_batch(job)
                else:
                    _execute_child(job)
            except BaseException as exc:
                _mark_worker_failure(job, exc)
        finally:
            job["api_key"] = ""
            for child in job.get("children",[]): child["api_key"]=""
            JOB_QUEUE.task_done()


def start_worker() -> None:
    global WORKER_STARTED, WORKER_THREAD
    with WORKER_LOCK:
        if WORKER_THREAD is not None and WORKER_THREAD.is_alive():
            WORKER_STARTED = True
            return
        WORKER_THREAD = threading.Thread(target=worker, name="test-worker", daemon=True)
        WORKER_THREAD.start()
        WORKER_STARTED = True


def public_run(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["quick"] = bool(data["quick"])
    data["is_batch"] = bool(data.get("is_batch"))
    data["case_ids"] = json.loads(data["case_ids"]) if data.get("case_ids") else [data["case_name"]]
    if data.get("test_kind") == "temporary":
        data["case_title"] = temporary_case_title(data["case_name"])
    elif data.get("test_kind") == "fable":
        # 只有计费探针有子项标题；普通 Claude 套件返回 None。
        data["case_title"] = billing_probe_title(data["case_name"])
    else:
        data["case_title"] = None
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
            return self.json_response([{
                "id": item.id, "provider": item.provider, "name": item.name,
                "default_base_url": item.default_base_url, "path_template": item.path_template,
                "description": item.description, "model_hint": item.model_hint,
                "family": item.family, "test_kinds": list(item.test_kinds),
            } for item in PROTOCOLS.values()])
        if path == "/api/cases":
            load_cases = [load_case_payload(name) for name in CASES]
            kvv_cases = [capability_case_payload(name, "kvv", KVV_CASES[name], KVV_META[name]) for name in KVV_CASES]
            deepseek_cases = [auth_case_payload(name) if name == "DS_V4_FINGERPRINT" else capability_case_payload(name, "deepseek", DEEPSEEK_CASES[name], DEEPSEEK_META[name]) for name in DEEPSEEK_CASES]
            glm_cases = [capability_case_payload(name, "glm", GLM_CASES[name], GLM_META[name]) for name in GLM_CASES]
            gpt56_cases = [capability_case_payload(name, "gpt", GPT56_CASES[name], GPT56_META[name]) for name in GPT56_CASES]
            fable5_cases = []
            for variant_id, (variant_label, variant_model) in CLAUDE_VARIANTS.items():
                for name, case in FABLE5_CASES.items():
                    item = capability_case_payload(name, "fable", case, FABLE5_META[name])
                    if variant_id != "fable5":
                        item["id"] = f"{variant_id.upper()}_{name.split('_', 1)[1]}"
                        item["title"] = case.title.replace("Fable 5", variant_label)
                    item["claude_variant"] = variant_id
                    item["default_model"] = variant_model
                    fable5_cases.append(item)
            # 计费探针常驻 Claude 模块（不随模型变体切换，故不带 claude_variant）。
            fable5_cases.append(billing_probe_payload())
            contract_cases = [contract_case_payload(name) for name in CONTRACT_CASES]
            kvv_cases += [x for x in contract_cases if x["test_kind"] == "kvv"]
            deepseek_cases += [x for x in contract_cases if x["test_kind"] == "deepseek"]
            glm_cases += [x for x in contract_cases if x["test_kind"] == "glm"]
            awsb_cases = [awsb_case_payload(spec) for spec in (*AWSB_CASE_SPECS, *AWSB_CACHE_CASE_SPECS)]
            moderation_cases = [moderation_case_payload(name) for name in MODERATION_CASES]
            gpt_image_cases = [gpt_image_case_payload(name) for name in GPT_IMAGE_CASES]
            gemini_image_cases = [gemini_image_case_payload(name) for name in GEMINI_IMAGE_CASES]
            temporary_cases = [temporary_case_payload()]
            arxivmath_case = {"id": "ARXIVMATH", "test_kind": "arxivmath", "source": "MathArena 官方开源数据集", "executor": "scripts/run_arxivmath.py（Hugging Face）", "title": ARXIVMATH_CASE.title, "description": ARXIVMATH_CASE.description, "duration": "按题目数量执行", "cost": ARXIVMATH_CASE.cost, "protocols": list(ARXIVMATH_CASE.protocols), "concurrency": 1, "concurrency_text": "1（按题目串行）", "requests": None, "request_text": "数据集全量题目", "case_count": None, "case_count_text": "全量题目（运行时确定）", "input_tokens": "研究级数学问题", "warmup": None, "steady": None, "admission": "使用官方 answer 字段评分；题目和答案不在本项目中改写。"}
            deepswe_case = {"id": "DEEPSWE", "test_kind": "deepswe", "source": "DataCurve 官方开源数据集", "executor": "scripts/run_deepswe.py（Pier）", "title": DEEPSWE_CASE.title, "description": DEEPSWE_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": DEEPSWE_CASE.cost, "protocols": list(DEEPSWE_CASE.protocols), "concurrency": 1, "concurrency_text": "由 Pier 管理隔离任务", "requests": None, "request_text": "官方任务集（默认全量）", "case_count": 113, "case_count_text": "113 个官方任务", "input_tokens": "代码仓库 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 verifier 通过；使用官方任务、环境和测试，不在本项目重写。"}
            sweatlas_case = {"id": "SWE_ATLAS_TW", "test_kind": "swe-atlas-tw", "source": "Scale AI 官方开源数据集", "executor": "scripts/run_sweatlas_tw.py（Harbor）", "title": SWE_ATLAS_TW_CASE.title, "description": SWE_ATLAS_TW_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": SWE_ATLAS_TW_CASE.cost, "protocols": list(SWE_ATLAS_TW_CASE.protocols), "concurrency": 1, "concurrency_text": "由 Harbor 管理隔离任务", "requests": None, "request_text": "官方 TW 任务集", "case_count": 66, "case_count_text": "66 个官方任务（默认）", "input_tokens": "代码仓库 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 verifier 通过；使用官方任务、环境和测试，不在本项目重写。"}
            sweatlas_rf_case = {"id": "SWE_ATLAS_RF", "test_kind": "swe-atlas-rf", "source": "Scale AI 官方开源数据集", "executor": "scripts/run_sweatlas_rf.py（Harbor）", "title": SWE_ATLAS_RF_CASE.title, "description": SWE_ATLAS_RF_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": SWE_ATLAS_RF_CASE.cost, "protocols": list(SWE_ATLAS_RF_CASE.protocols), "concurrency": 1, "concurrency_text": "由 Harbor 管理隔离任务", "requests": None, "request_text": "官方 RF 任务集", "case_count": 66, "case_count_text": "66 个官方任务（默认）", "input_tokens": "代码仓库 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 verifier 通过；使用官方任务、环境和测试，不在本项目重写。"}
            legal_case = {"id": "LEGAL_AGENT_BENCHMARK", "test_kind": "legal-agent-benchmark", "source": "Harvey Labs v1.0 官方开源数据集", "executor": "scripts/run_legal_agent_benchmark.py（官方 harness + evaluator）", "title": LEGAL_AGENT_BENCHMARK_CASE.title, "description": LEGAL_AGENT_BENCHMARK_CASE.description, "duration": "单任务约数分钟至数十分钟", "cost": LEGAL_AGENT_BENCHMARK_CASE.cost, "protocols": list(LEGAL_AGENT_BENCHMARK_CASE.protocols), "concurrency": 1, "concurrency_text": "由官方 harness 管理 Agent 工具循环", "requests": None, "request_text": "官方 v1.0 Legal Agent 任务集", "case_count": 1, "case_count_text": "1 个任务（默认抽样）", "input_tokens": "法律材料 + Agent 轨迹", "warmup": None, "steady": None, "admission": "官方 evaluator 完成评分；使用官方任务、材料、rubric 和 harness，不在本项目重写。"}
            return self.json_response(load_cases + kvv_cases + deepseek_cases + glm_cases + gpt56_cases + fable5_cases + gpt_image_cases + gemini_image_cases + awsb_cases + moderation_cases + temporary_cases + [arxivmath_case, deepswe_case, sweatlas_case, sweatlas_rf_case, legal_case])
        if path == "/api/runs":
            query = parse_qs(urlparse(self.path).query)
            try:
                page = max(1, int(query.get("page", ["1"])[0]))
                page_size = min(100, max(1, int(query.get("page_size", ["20"])[0])))
            except (TypeError, ValueError):
                return self.json_response({"error": "分页参数无效"}, 400)
            status_filter = query.get("status", ["all"])[0]
            conclusion_filter = query.get("conclusion", ["all"])[0]
            search = query.get("q", [""])[0].strip()[:120]
            where = ["batch_id IS NULL"]
            params: list[Any] = []
            if status_filter == "running":
                where.append("status IN ('queued','running')")
            elif status_filter == "failed":
                where.append("status IN ('failed','interrupted')")
            elif status_filter in {"queued", "completed", "interrupted"}:
                where.append("status=?")
                params.append(status_filter)
            if conclusion_filter == "pass":
                where.append("passed=1")
            elif conclusion_filter == "fail":
                where.append("passed=0")
            elif conclusion_filter == "pending":
                where.append("passed IS NULL")
            if search:
                pattern = f"%{search}%"
                where.append("(id LIKE ? OR case_name LIKE ? OR case_ids LIKE ? OR test_kind LIKE ? OR model LIKE ? OR base_url LIKE ? OR protocol LIKE ? OR created_at LIKE ?)")
                params.extend([pattern] * 8)
            clause = " AND ".join(where)
            with db() as connection:
                total = connection.execute(f"SELECT COUNT(*) FROM runs WHERE {clause}", params).fetchone()[0]
                rows = connection.execute(
                    f"SELECT * FROM runs WHERE {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    [*params, page_size, (page - 1) * page_size],
                ).fetchall()
            return self.json_response({"items": [public_run(row) for row in rows], "total": total,
                                       "page": page, "page_size": page_size})
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
                return self.send_result_archive(
                    {
                        "run_id": run_id,
                        "is_batch": bool(row["is_batch"]),
                        "test_kind": row["test_kind"],
                        "model": row["model"],
                        "created_at": row["created_at"],
                        "included_records": [r["id"] for r in rows_to_pack],
                    },
                    rows_to_pack,
                    f"test-data-{run_id}.zip",
                )
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
            if kind == "manifest":
                # 批量聚合 manifest 附带每项 metrics；旧批次落盘时还没有该字段，读取时即时补齐，
                # 保证前端聚合报告的指标列对历史记录同样有数据。
                try:
                    payload = json.loads(target.read_text(encoding="utf-8"))
                except ValueError:
                    return self.text_response(target.read_text(encoding="utf-8", errors="replace"), "text/plain; charset=utf-8")
                if isinstance(payload, dict):
                    ensure_batch_metrics(payload, ROOT)
                    return self.json_response(payload)
            return self.text_response(target.read_text(encoding="utf-8", errors="replace"), "text/plain; charset=utf-8")
        if path in {"/", "/index.html"}:
            return self.serve_file(STATIC / "index.html")
        target = (STATIC / path.lstrip("/")).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self.text_response("Not Found", status=404)
        return self.serve_file(target)

    def do_DELETE(self) -> None:
        request_path = urlparse(self.path).path
        if not request_path.startswith("/api/runs/"):
            return self.json_response({"error": "不支持的删除目标"}, 404)
        run_id = unquote(request_path.removeprefix("/api/runs/")).strip("/")
        if not run_id or "/" in run_id:
            return self.json_response({"error": "记录 ID 无效"}, 400)
        with db() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                return self.json_response({"error": "记录不存在"}, 404)
            if row["batch_id"]:
                return self.json_response({"error": "批次子记录不能单独删除，请删除整个批次"}, 409)
            records = [row]
            if row["is_batch"]:
                records.extend(connection.execute("SELECT * FROM runs WHERE batch_id=?", (run_id,)).fetchall())
            if any(record["status"] in {"queued", "running"} for record in records):
                return self.json_response({"error": "运行中的记录不能删除，请先等待结束或停止任务"}, 409)

            directories: set[Path] = set()
            allowed = RESULTS.resolve()
            for record in records:
                if not record["result_dir"]:
                    continue
                directory = (ROOT / record["result_dir"]).resolve()
                if allowed not in directory.parents or not directory.is_dir():
                    continue
                directories.add(directory)
            for directory in directories:
                shutil.rmtree(directory)

            ids = [record["id"] for record in records]
            connection.executemany("DELETE FROM runs WHERE id=?", [(record_id,) for record_id in ids])
        return self.json_response({"id": run_id, "deleted_records": len(ids)})

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
                return self.send_result_archive(
                    {
                        "type": "selected-runs",
                        "selected_run_ids": ids,
                        "included_records": [r["id"] for r in rows_to_pack],
                    },
                    rows_to_pack,
                    "test-data-selected.zip",
                )
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
                    try: _terminate_process_tree(proc)
                    except (ProcessLookupError, OSError, subprocess.SubprocessError): pass
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
                if protocol not in PROTOCOLS:
                    raise ValueError("未知模型协议")
                base_url = normalize_base_url(base_url)
                if not api_key:
                    raise ValueError("api_key 不能为空")
                if not model:
                    raise ValueError("model 不能为空")
                config = ApiConfig(api_key=api_key, base_url=base_url, model=model, protocol=protocol,
                                   trust_env=trust_env_enabled())
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
                if protocol not in PROTOCOLS:
                    raise ValueError("未知模型协议")
                base_url = normalize_base_url(base_url)
                if not api_key:
                    raise ValueError("api_key 不能为空")
                try:
                    response = httpx.get(models_endpoint(protocol, base_url), headers=models_headers(protocol, api_key), timeout=30, follow_redirects=False, trust_env=trust_env_enabled())
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
            temporary_case = str(payload.get("temporary_case") or "").strip() or None
            billing_probe = str(payload.get("billing_probe") or "").strip() or None
            quick = bool(payload.get("quick", False))
            custom_requests = payload.get("requests")
            custom_concurrency = payload.get("concurrency")
            image_quality = payload.get("image_quality")
            image_models = payload.get("image_models")
            image_sizes = payload.get("image_sizes")
            gemini_quality = payload.get("gemini_quality")
            gemini_models = payload.get("gemini_models")
            gemini_aspects = payload.get("gemini_aspects")
            gemini_counts = payload.get("gemini_counts")
            if custom_requests is not None and (not isinstance(custom_requests, int) or not 1 <= custom_requests <= 100000): raise ValueError("发送请求数必须是 1～100000 的整数")
            if custom_concurrency is not None and (not isinstance(custom_concurrency, int) or not 1 <= custom_concurrency <= 1000): raise ValueError("并发数必须是 1～1000 的整数")
            if test_kind == "moderation" and custom_concurrency is None: custom_concurrency = 4
            if test_kind == "gpt-image":
                if len(case_names) != 1:
                    raise ValueError("图片压测一次只能选择一个子项目")
                if case_name == GPT_IMAGE_TOKEN_CASE:
                    custom_requests, custom_concurrency = 3, 1
                    image_quality, image_models, image_sizes = None, None, None
                elif custom_requests is None or custom_concurrency is None:
                    raise ValueError("图片压测必须填写总请求数和并发数")
            if test_kind == "gemini-image":
                if len(case_names) != 1:
                    raise ValueError("Gemini 图片测试一次只能选择一个子项目")
                if custom_requests is None or custom_concurrency is None:
                    raise ValueError("Gemini 图片测试必须填写总请求数和并发数")
            if test_kind == "load" and "RANDOM_LOAD" in case_names:
                if len(case_names) != 1:
                    raise ValueError("随机输入压测一次只能单独运行")
                if custom_requests is None or custom_concurrency is None:
                    raise ValueError("随机输入压测必须填写请求数和并发数")
                quick = False
            base_url = normalize_base_url(base_url)
            if not api_key:
                raise ValueError("api_key 不能为空")
            if not model:
                raise ValueError("model 不能为空")
            if test_kind == "temporary" and temporary_case is not None:
                # 选择项是合并案例 id（batch 名）；原子案例 id 继续兼容。
                if find_case(temporary_all_cases(), temporary_case) is None:
                    raise ValueError("未知临时测试案例")
            # 中转计费探针：只要本次提交包含计费探针，子选项须是 BILLING-ALL 或某条探针 id。
            # 不能要求 case_names 恰好只有探针一条——探针在 Claude tab 下常与各套件**同批**
            # 提交（批量场景），旧的等值判断会把整批的 billing_probe 置空，探针子任务于是
            # 落到 run_fable5.py 报 "invalid choice"（历史批次因此整批不准入）。
            # 未选中探针时一律置空，避免把上一条选择的残留带进普通 Claude 用例。
            billing_probe_requested = test_kind == "fable" and BILLING_PROBE_CASE_ID in case_names
            if billing_probe_requested:
                billing_probe = billing_probe or BILLING_PROBE_ALL
                if not is_billing_probe(billing_probe):
                    raise ValueError("未知的中转计费探针选择")
            else:
                billing_probe = None
            available = {"LEGAL_AGENT_BENCHMARK": LEGAL_AGENT_BENCHMARK_CASE} if test_kind == "legal-agent-benchmark" else {"SWE_ATLAS_RF": SWE_ATLAS_RF_CASE} if test_kind == "swe-atlas-rf" else {"SWE_ATLAS_TW": SWE_ATLAS_TW_CASE} if test_kind == "swe-atlas-tw" else {"DEEPSWE": DEEPSWE_CASE} if test_kind == "deepswe" else {"ARXIVMATH": ARXIVMATH_CASE} if test_kind == "arxivmath" else {"K3_TEMP_SUITE": temporary_case_payload()} if test_kind == "temporary" else CASES if test_kind == "load" else MODERATION_CASES if test_kind == "moderation" else KVV_CASES if test_kind == "kvv" else DEEPSEEK_CASES if test_kind == "deepseek" else GLM_CASES if test_kind == "glm" else GPT56_CASES if test_kind == "gpt" else CLAUDE_RUNNABLE_CASES if test_kind == "fable" else GPT_IMAGE_CASES if test_kind == "gpt-image" else GEMINI_IMAGE_CASES if test_kind == "gemini-image" else AUTH_CASES if test_kind == "authenticity" else {spec.case_id: spec for spec in (*AWSB_CASE_SPECS, *AWSB_CACHE_CASE_SPECS)} if test_kind == "awsb" else {}
            if test_kind in {"kvv", "deepseek", "glm"}:
                available = dict(available)
                available.update({name: item for name, item in CONTRACT_CASES.items() if item["test_kind"] == test_kind})
            if not case_names or any(name not in available for name in case_names):
                raise ValueError("未知测试类型")
            if test_kind == "temporary" and case_names != ["K3_TEMP_SUITE"]:
                raise ValueError("临时测试案例集只能选择测试集入口")
            if test_kind == "gpt-image":
                image_case = GPT_IMAGE_CASES[case_names[0]]
                image_quality = _choice_list(image_quality, image_case.qualities, "quality")
                image_models = _choice_list(image_models, image_case.model_choices, "模型")
                image_sizes = _choice_list(image_sizes, image_case.sizes, "尺寸")
            if test_kind == "gemini-image":
                gemini_case = GEMINI_IMAGE_CASES[case_names[0]]
                gemini_quality = _choice_list(gemini_quality, gemini_case.qualities, "强度(imageSize)")
                gemini_models = _choice_list(gemini_models, gemini_case.models, "模型")
                gemini_aspects = _choice_list(gemini_aspects, gemini_case.aspect_ratios, "尺寸(aspectRatio)")
                gemini_counts = _int_choice_list(gemini_counts, gemini_case.counts, "生成数量")
            if protocol not in PROTOCOLS:
                raise ValueError("未知模型协议")
            def protocols_for(name: str) -> tuple[str, ...]:
                if test_kind == "temporary":
                    # K3 案例走 Chat Completions 面。
                    return ("openai", "volcengine")
                if test_kind == "fable" and name == BILLING_PROBE_CASE_ID:
                    # 计费探针只在 Anthropic Messages 面取证。
                    return BILLING_PROBE_PROTOCOLS
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
                if test_kind == "gpt-image":
                    return GPT_IMAGE_CASES[name].protocols
                if test_kind == "gemini-image":
                    return GEMINI_IMAGE_CASES[name].protocols
                if name in CONTRACT_CASES:
                    return CONTRACT_CASES[name]["protocols"]
                return KVV_CASES[name].protocols if test_kind == "kvv" else DEEPSEEK_CASES[name].protocols if test_kind == "deepseek" else GLM_CASES[name].protocols if test_kind == "glm" else FABLE5_CASES[claude_suite_name(name)].protocols if test_kind == "fable" else GPT56_CASES[name].protocols if test_kind == "gpt" else ("openai", "volcengine") if test_kind == "authenticity" else ("openai", "volcengine") if test_kind == "moderation" else ("anthropic",) if test_kind == "awsb" else tuple(PROTOCOLS)
            if test_kind == "temporary" and protocol == "anthropic":
                raise ValueError("K3 临时案例集只支持 openai / volcengine 协议")
            if billing_probe_requested and protocol != "anthropic":
                raise ValueError("中转计费探针只支持 Anthropic Messages 协议面")
            unsupported = [name for name in case_names if not supports_test_kind(protocol, test_kind, protocols_for(name))]
            if unsupported:
                raise ValueError(f"{', '.join(unsupported)} 不支持 {protocol} 协议")
            preset_names = [name for name in case_names if name in LOAD_PRESETS]
            if preset_names and (len(case_names) != 1 or custom_requests is not None or custom_concurrency is not None):
                raise ValueError("简单压测和完整压测使用固定组合参数，不能与其他测试或自定义参数同时运行")
            if preset_names:
                preset = LOAD_PRESETS[preset_names[0]]
                quick = False
                execution_cases = [
                    {
                        "name": name,
                        "requests": LOAD_CASES[name].samples if name == "O1" else preset["samples"],
                        "concurrency": LOAD_CASES[name].concurrency if name == "O1" else preset["concurrency"],
                        "warmup_seconds": 0,
                    }
                    for name in preset["cases"]
                ]
            elif test_kind == "temporary":
                execution_cases = [{"name": temporary_case or "K3_TEMP_SUITE", "requests": custom_requests, "concurrency": custom_concurrency, "warmup_seconds": None}]
            else:
                execution_cases = [
                    {"name": name, "requests": custom_requests, "concurrency": custom_concurrency, "warmup_seconds": None}
                    for name in case_names
                ]
            run_id = uuid.uuid4().hex[:16]; created=now_iso()
            if len(execution_cases) > 1:
                children=[]
                with db() as connection:
                    connection.execute("INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,is_batch,case_ids) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                       (run_id,test_kind,case_names[0] if len(case_names) == 1 else "MULTI",protocol,base_url,model,"queued",int(quick),created,1,json.dumps(case_names)))
                    for index,execution in enumerate(execution_cases):
                        name = execution["name"]
                        # 计费探针是同批里的**一条**子任务：只有它携带 billing_probe
                        # （执行器/协议面/结果别名都据此切换），同批其它套件必须保持空值。
                        child_billing_probe = billing_probe if name == BILLING_PROBE_CASE_ID else None
                        # 探针子记录按所选子项归档（BILLING-ALL / BILLING-xx），与单条运行口径一致。
                        stored_name = child_billing_probe or name
                        child_id=uuid.uuid4().hex[:16]; child_created=f"{created}.{index:03d}"
                        connection.execute("INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,batch_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                           (child_id,test_kind,stored_name,protocol,base_url,model,"queued",int(quick),child_created,run_id))
                        children.append({"id":child_id,"batch_id":run_id,"test_kind":test_kind,"case":name,"protocol":protocol,"base_url":base_url,"api_key":api_key,"model":model,"quick":quick,"requests":execution["requests"],"concurrency":execution["concurrency"],"warmup_seconds":execution["warmup_seconds"],"image_quality":image_quality,"image_models":image_models,"image_sizes":image_sizes,"gemini_quality":gemini_quality,"gemini_models":gemini_models,"gemini_aspects":gemini_aspects,"gemini_counts":gemini_counts,"temporary_case":temporary_case,"billing_probe":child_billing_probe})
                start_worker()
                JOB_QUEUE.put({"id":run_id,"is_batch":True,"test_kind":test_kind,"api_key":api_key,"children":children})
            else:
                execution = execution_cases[0]
                if test_kind == "temporary" and temporary_case:
                    stored_case_name = temporary_case
                elif test_kind == "fable" and billing_probe:
                    # 计费探针按所选子项归档（BILLING-ALL / BILLING-xx），便于记录页显示。
                    stored_case_name = billing_probe
                else:
                    stored_case_name = case_name
                with db() as connection:
                    connection.execute("INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,case_ids) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                       (run_id,test_kind,stored_case_name,protocol,base_url,model,"queued",int(quick),created,json.dumps(case_names)))
                start_worker()
                JOB_QUEUE.put({"id":run_id,"test_kind":test_kind,"case":execution["name"],"protocol":protocol,"base_url":base_url,"api_key":api_key,"model":model,"quick":quick,"requests":execution["requests"],"concurrency":execution["concurrency"],"warmup_seconds":execution["warmup_seconds"],"image_quality":image_quality,"image_models":image_models,"image_sizes":image_sizes,"gemini_quality":gemini_quality,"gemini_models":gemini_models,"gemini_aspects":gemini_aspects,"gemini_counts":gemini_counts,"temporary_case":temporary_case,"billing_probe":billing_probe})
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

    def send_result_archive(
        self, metadata: dict[str, Any], rows_to_pack: list[Any], filename: str
    ) -> None:
        """把结果目录打包成 zip 发给客户端。

        图片套件会把生成的图片原文件保存在结果目录的 images/ 下，整包体积可能达到
        数 GB，因此先归档到结果目录里的临时文件，再分块写出，避免把整个包读进内存。
        """
        allowed_root = RESULTS.resolve()
        archive_path = RESULTS / f".download-{uuid.uuid4().hex}.zip"
        headers_sent = False
        try:
            with archive_path.open("w+b") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("package-metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
                seen_dirs: set[str] = set()
                for packed_row in rows_to_pack:
                    result_dir = packed_row["result_dir"]
                    if not result_dir:
                        continue
                    target_dir = (ROOT / result_dir).resolve()
                    if allowed_root not in target_dir.parents or not target_dir.is_dir():
                        continue
                    folder = target_dir.name
                    if folder in seen_dirs:
                        continue
                    seen_dirs.add(folder)
                    for source in sorted(target_dir.rglob("*")):
                        if source.is_file():
                            bundle.write(source, f"{folder}/{source.relative_to(target_dir).as_posix()}")
            size = archive_path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            headers_sent = True
            with archive_path.open("rb") as payload:
                shutil.copyfileobj(payload, self.wfile, length=1024 * 1024)
        except OSError as exc:
            # 断连（BrokenPipe）与磁盘错误都不能让服务端无声无息；响应头已发出时只能记日志。
            print(f"打包下载失败: {exc}", file=sys.stderr)
            if not headers_sent:
                self.json_response({"error": f"打包失败: {exc}"}, 500)
        finally:
            archive_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Volcengine K3 测试管理页面")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8085)
    args = parser.parse_args(argv)
    initialize(); start_worker()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    # 0.0.0.0 / :: 不可路由，浏览器会把它交给系统代理并报 502，展示时统一换成回环地址。
    display = "127.0.0.1" if args.host in {"", "0.0.0.0", "::"} else args.host
    print(f"测试管理页面：http://{display}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
