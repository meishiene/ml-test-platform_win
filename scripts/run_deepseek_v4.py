#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from k3perf.deepseek_v4 import CASE_SPECS, MODELS, DeepSeekConfig, DeepSeekV4Runner, summarize_results, write_report


async def main() -> int:
    parser = argparse.ArgumentParser(description="运行 DeepSeek V4 Pro/Flash 专项测试")
    parser.add_argument("--model", action="append", choices=MODELS, help="可重复指定；默认测试 Pro 和 Flash")
    parser.add_argument("--priority", action="append", choices=("P0", "P1"), help="默认执行 P0 和 P1")
    parser.add_argument("--case", action="append", dest="cases", help="只执行指定用例，如 DS-V4-06；可重复")
    parser.add_argument("--list", action="store_true", help="列出用例但不请求 API")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.list:
        for case in CASE_SPECS:
            print(f"{case.case_id}\t{case.priority}\t{case.name}{' [Beta]' if case.beta else ''}")
        return 0
    try:
        config = DeepSeekConfig.from_env()
    except ValueError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1
    models = tuple(args.model or MODELS)
    priorities = tuple(args.priority or ("P0", "P1"))
    case_ids = set(args.cases) if args.cases else None
    unknown = case_ids - {x.case_id for x in CASE_SPECS} if case_ids else set()
    if unknown:
        print(f"未知用例：{', '.join(sorted(unknown))}", file=sys.stderr)
        return 1
    async with DeepSeekV4Runner(config) as runner:
        results = await runner.run(models, priorities, case_ids)
    summary = summarize_results(results)
    out = args.output_root / f"DEEPSEEK-V4-{time.strftime('%Y%m%d-%H%M%S')}"
    write_report(out, summary)
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'} {r.case_id} {r.model}: {r.details}")
    print(f"结果：{out.resolve()}")
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
