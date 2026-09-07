#!/usr/bin/env python3
from __future__ import annotations
import argparse, asyncio, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from k3perf.awsb import AWSBConfig, AWSBRunner, CASE_SPECS, summarize, write_report

async def main():
    ap=argparse.ArgumentParser(description="AWS Bedrock Claude Opus 5 上游/网关自检")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--case", action="append", dest="cases")
    ap.add_argument("--priority", action="append", choices=("P0","P1")); ap.add_argument("--output-root", type=Path, default=Path("results"))
    args=ap.parse_args()
    if args.list:
        for x in CASE_SPECS: print(f"{x.case_id}\t{x.priority}\t{x.name}")
        return 0
    try: cfg=AWSBConfig.from_env()
    except ValueError as e: print(f"配置错误：{e}", file=sys.stderr); return 1
    ids=set(args.cases) if args.cases else None
    known={x.case_id for x in CASE_SPECS}
    if ids and ids-known: print("未知用例："+", ".join(sorted(ids-known)), file=sys.stderr); return 1
    async with AWSBRunner(cfg) as runner: results=await runner.run(tuple(args.priority or ("P0","P1")), ids)
    summary=summarize(results); out=args.output_root/f"AWSB-O5-{time.strftime('%Y%m%d-%H%M%S')}"; write_report(out,summary)
    for x in results: print(f"{'PASS' if x.passed else 'FAIL'} {x.case_id}: {x.details}")
    print(f"结果：{out.resolve()}"); return 0 if summary["all_passed"] else 2
if __name__ == "__main__": raise SystemExit(asyncio.run(main()))
