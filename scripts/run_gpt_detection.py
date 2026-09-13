#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpt_detection.cases import CASES
from gpt_detection.runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="GPT 模型身份一致性检测")
    parser.add_argument("case", nargs="?", choices=sorted(CASES), default="GPT_MODEL_MATCH")
    parser.add_argument("--base-url", default=os.getenv("GPT_BASE_URL") or os.getenv("VOLC_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("GPT_API_KEY") or os.getenv("VOLC_API_KEY"))
    parser.add_argument("--model", default=os.getenv("GPT_MODEL") or os.getenv("VOLC_MODEL"))
    parser.add_argument("--protocol", default=os.getenv("GPT_PROTOCOL") or os.getenv("VOLC_PROTOCOL", "openai"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    missing = [name for name, value in (("base_url", args.base_url), ("api_key", args.api_key), ("model", args.model)) if not value]
    if missing:
        parser.error("缺少参数或环境变量：" + ", ".join(missing))
    result_dir, summary = run(
        case_name=args.case,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        protocol=args.protocol,
        output_root=args.output_root,
        timeout=args.timeout,
    )
    print(f"结果目录: {result_dir.resolve()}")
    print(f"请求模型: {summary.get('expected_model')}")
    print(f"响应模型: {summary.get('actual_model') or '缺失'}")
    print(f"结论: {'通过' if summary.get('admission_passed') else '未通过'}")
    return 0 if summary.get("admission_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
