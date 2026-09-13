from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from common.suite_runtime import finalize

from .cases import CASES
from .core import DeepSeekConfig, DeepSeekV4Runner, summarize_results, write_report


def config_from_env() -> DeepSeekConfig:
    api_key = os.getenv("DS_API_KEY") or os.getenv("VOLC_API_KEY")
    base_url = os.getenv("DS_BASE_URL") or os.getenv("VOLC_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("缺少 DS_API_KEY/DS_BASE_URL（Web 启动时由界面配置自动注入）")
    return DeepSeekConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        beta_base_url=os.getenv("DS_BETA_BASE_URL"),
        anthropic_base_url=os.getenv("DS_ANTHROPIC_BASE_URL"),
        timeout=float(os.getenv("DS_REQUEST_TIMEOUT_SECONDS", "300")),
        verify_ssl=os.getenv("DS_VERIFY_SSL", "true").lower() not in {"0", "false", "no"},
        trust_env=os.getenv("DS_TRUST_ENV", "false").lower() in {"1", "true", "yes"},
    )


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 DeepSeek V4 专项测试")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    config = config_from_env()
    model = os.getenv("DS_MODEL") or os.getenv("VOLC_MODEL")
    if not model:
        raise ValueError("缺少 DS_MODEL（Web 启动时由界面配置自动注入）")
    meta = CASES[args.case]
    async with DeepSeekV4Runner(config) as runner:
        results = await runner.run((model,), meta.priorities)
    summary = summarize_results(results)
    summary.update({
        "record_id": os.getenv("VOLC_RECORD_ID"),
        "test_kind": "deepseek",
        "case": args.case,
        "model": model,
        "endpoint": config.base_url,
        "admission_passed": summary["all_passed"],
    })
    out = args.output_root / f"{args.case}-{time.strftime('%Y%m%d-%H%M%S')}"
    write_report(out, summary)
    # write_report 已写入结果；补写 Web 聚合需要的扩展字段。
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finalize(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(argv))
