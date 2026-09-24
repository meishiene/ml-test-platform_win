from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from common.suite_runtime import finalize
from common.model_contracts import model_matches_family
from volcperf.protocols import api_root

from .cases import CASES
from .core import DeepSeekConfig, DeepSeekV4Runner, summarize_results, write_report


def config_from_env() -> DeepSeekConfig:
    api_key = os.getenv("DS_API_KEY") or os.getenv("VOLC_API_KEY")
    base_url = os.getenv("DS_BASE_URL") or os.getenv("VOLC_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("缺少 DS_API_KEY/DS_BASE_URL（Web 启动时由界面配置自动注入）")
    return DeepSeekConfig(
        api_key=api_key,
        base_url=api_root(os.getenv("VOLC_PROTOCOL", "volcengine"), base_url),
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
    # 模型门禁：误配其他家族的模型（如 gpt-*）会让正向用例被上游 400 拒绝，
    # 而负向用例的 4xx 断言恰被该 400 满足产生假通过。发请求前直接拦截。
    allowed, gate_reason = model_matches_family("deepseek", model)
    if not allowed:
        print(json.dumps({"error": "model_gate_rejected", "message": gate_reason, "model": model, "case": args.case}, ensure_ascii=False))
        return 3
    meta = CASES[args.case]
    # 目录必须先于请求创建：runner 会把每个请求的证据实时写入该目录下的 requests.jsonl。
    out = args.output_root / f"{args.case}-{time.strftime('%Y%m%d-%H%M%S')}"
    async with DeepSeekV4Runner(config, audit_dir=out) as runner:
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
    write_report(out, summary)
    # write_report 已写入结果；补写 Web 聚合需要的扩展字段。
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finalize(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(argv))
