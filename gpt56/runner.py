from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from common.suite_runtime import finalize
from volcperf.protocols import api_root

from .cases import CASES
from .core import GPT56Config, GPT56Runner, summarize_results, write_report

FAMILIES = {"openai", "openai_responses"}


def config_from_env() -> GPT56Config:
    api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("GPT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("缺少 GPT_API_KEY/GPT_BASE_URL（Web 启动时由界面配置自动注入）")
    # Web 传入的是 host-only 地址；openai 家族按 api_root() 规范化成 ".../v1"。
    protocol = os.getenv("GPT_PROTOCOL", "openai")
    if protocol not in FAMILIES:
        protocol = "openai"
    return GPT56Config(
        api_key=api_key,
        base_url=api_root(protocol, base_url),
        timeout=float(os.getenv("GPT_REQUEST_TIMEOUT_SECONDS", "300")),
        verify_ssl=os.getenv("GPT_VERIFY_SSL", "true").lower() not in {"0", "false", "no"},
        trust_env=os.getenv("GPT_TRUST_ENV", "false").lower() in {"1", "true", "yes"},
    )


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 GPT-5.6 专项测试")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    config = config_from_env()
    model = os.getenv("GPT_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("缺少 GPT_MODEL（Web 启动时由界面配置自动注入）")
    meta = CASES[args.case]
    # 目录必须先于请求创建：runner 会把每个请求的证据实时写入该目录下的 requests.jsonl。
    out = args.output_root / f"{args.case}-{time.strftime('%Y%m%d-%H%M%S')}"
    async with GPT56Runner(config, audit_dir=out) as runner:
        results = await runner.run((model,), meta.priorities)
    summary = summarize_results(results)
    summary.update({
        "record_id": os.getenv("VOLC_RECORD_ID"),
        "test_kind": "gpt",
        "case": args.case,
        "model": model,
        "endpoint": config.base_url,
        "admission_passed": summary["all_passed"],
    })
    write_report(out, summary)
    # write_report 已写入结果；补写 Web 聚合需要的扩展字段。
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finalize(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(argv))
