from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from itertools import cycle
from pathlib import Path
from typing import Any

import httpx

from .cases import CASES, Case
from .client import stream_chat
from .config import ApiConfig
from .connectivity import check_connectivity
from .data import PromptRecord, load_prompts
from .metrics import summarize
from .reporting import write_artifacts


B1_DATASETS = ("P2K", "P8K", "P16K", "P32K")


def load_case_prompts(case: Case, data_override: Path | None = None, *, quick: bool = False) -> list[PromptRecord]:
    """B1 每档取固定数量并交错排列，确保并发 2 时各规格均匀执行。"""
    if case.name != "B1" or data_override is not None:
        return load_prompts(data_override or case.data_file)
    per_size = 1 if quick else 4
    groups = [load_prompts(Path(__file__).resolve().parent.parent / "data" / f"{dataset}.jsonl.gz")[:per_size]
              for dataset in B1_DATASETS]
    return [groups[size][index] for index in range(per_size) for size in range(len(groups))]


class ResultWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", encoding="utf-8", buffering=1)

    def write(self, row: dict[str, Any]) -> None:
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.stream.close()


def session(config: ApiConfig, *, connection_limit: int | None = None) -> httpx.AsyncClient:
    # 连接数按实际并发设置上限，避免默认 1200 个连接在高并发时造成本机
    # socket/端口竞争和中转层突发建连；连接可复用，保持连接数略高于并发即可。
    limit = connection_limit or config.max_connections
    timeout = httpx.Timeout(config.request_timeout_seconds, connect=config.connect_timeout_seconds, pool=10.0)
    limits = httpx.Limits(max_connections=limit, max_keepalive_connections=limit, keepalive_expiry=60.0)
    return httpx.AsyncClient(timeout=timeout, limits=limits, verify=config.verify_ssl, trust_env=config.trust_env,
                             headers={"Accept": "text/event-stream"}, http2=False)


async def run_closed(case: Case, prompts: list[PromptRecord], writer: ResultWriter, samples: int, warmup: int, duration: int) -> list[dict[str, Any]]:
    config, rows, prompt_iter = ApiConfig.from_env(), [], cycle(prompts)
    sequence, issued = 0, 0
    lock, stop = asyncio.Lock(), asyncio.Event()
    warmup_end = time.monotonic() + warmup
    measured_start: float | None = None
    async with session(config, connection_limit=max(case.concurrency * 2, 20)) as client:
        async def worker(worker_id: int) -> None:
            nonlocal sequence, issued, measured_start
            while not stop.is_set():
                now = time.monotonic()
                async with lock:
                    # samples 是本轮“总请求上限”，预热和正式请求共用，不再额外追加预热请求。
                    if issued >= samples:
                        stop.set(); return
                    counted = now >= warmup_end
                    if counted and measured_start is None:
                        measured_start = now
                    record, sequence = next(prompt_iter), sequence + 1
                    issued += 1
                    current = issued
                workload = record.workload(case.name == "O1")
                row = await stream_chat(client, config, request_id=f"{case.name}-{worker_id}-{sequence}-{uuid.uuid4().hex[:8]}",
                                        messages=workload.messages, workload_type=workload.workload_type,
                                        target_visible_tokens=workload.target_visible_tokens, max_tokens=case.max_tokens,
                                        case_name=case.name, dataset_id=record.id,
                                        phase="measured" if counted else "warmup", counted=counted)
                writer.write(row); rows.append(row)
                if counted:
                    print(f"\r{case.name}: completed {sum(r.get('counted') for r in rows)}/{samples}, total {current}/{samples}", end="", flush=True)
        await asyncio.gather(*(asyncio.create_task(worker(i)) for i in range(case.concurrency)))
    print()
    return rows


async def run_open(case: Case, prompts: list[PromptRecord], writer: ResultWriter, stages: tuple[tuple[str, float, int, bool], ...]) -> list[dict[str, Any]]:
    config, rows, prompt_iter = ApiConfig.from_env(), [], cycle(prompts)
    tasks: set[asyncio.Task] = set()
    semaphore = asyncio.Semaphore(config.max_connections)
    async with session(config, connection_limit=max(config.max_connections, 20)) as client:
        async def send(record: PromptRecord, phase: str, counted: bool, planned: float, seq: int) -> None:
            async with semaphore:
                workload = record.workload()
                row = await stream_chat(client, config, request_id=f"{case.name}-{seq}-{uuid.uuid4().hex[:8]}",
                                        messages=workload.messages, workload_type=workload.workload_type,
                                        target_visible_tokens=workload.target_visible_tokens, max_tokens=case.max_tokens,
                                        case_name=case.name, dataset_id=record.id, phase=phase,
                                        counted=counted, planned_epoch=planned)
            writer.write(row); rows.append(row)
        seq = 0
        for phase, rps, duration, counted in stages:
            print(f"{case.name}: {phase}, {rps:g} RPS, {duration}s")
            start_mono, start_epoch, interval = time.monotonic(), time.time(), 1 / rps
            for index in range(round(rps * duration)):
                delay = start_mono + index * interval - time.monotonic()
                if delay > 0: await asyncio.sleep(delay)
                task = asyncio.create_task(send(next(prompt_iter), phase, counted, start_epoch + index * interval, seq))
                tasks.add(task); task.add_done_callback(tasks.discard); seq += 1
            deadline = start_mono + duration
            if deadline > time.monotonic(): await asyncio.sleep(deadline - time.monotonic())
        if tasks:
            print(f"等待 {len(tasks)} 个在途请求归零……")
            await asyncio.gather(*tasks)
    return rows


def metadata(path: Path, case: Case, args: argparse.Namespace) -> None:
    cfg = ApiConfig.from_env()
    obj = {"case": case.name, "started_epoch": time.time(), "endpoint": cfg.endpoint, "model": cfg.model,
           "max_tokens": case.max_tokens, "stream": True,
           "thinking": {"provider_default": cfg.reasoning_effort is None, "reasoning_effort": cfg.reasoning_effort},
           "workload_profile": "mixed-business-v1", "mode": case.mode,
           "arguments": vars(args), "python": os.sys.version, "pid": os.getpid()}
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="火山引擎 Kimi K3 短时准入性能测试")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--warmup-seconds", type=int)
    parser.add_argument("--measured-seconds", type=int)
    parser.add_argument("--baseline-summary", type=Path, help="对应 B1 summary.json，用于相对退化检查")
    parser.add_argument("--quick", action="store_true", help="仅验证脚本，不能形成准入结论")
    parser.add_argument("--concurrency", type=int, help="覆盖默认并发数（仅闭环性能测试）")
    args = parser.parse_args(argv)
    case = CASES[args.case]
    if args.concurrency is not None and (args.concurrency < 1 or args.concurrency > 1000):
        parser.error("--concurrency 必须在 1～1000 之间")
    if args.samples is not None and (args.samples < 1 or args.samples > 100000):
        parser.error("--samples 必须在 1～100000 之间")
    prompts = load_case_prompts(case, args.data, quick=args.quick)
    out = args.output_root / f"{case.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False); metadata(out / "metadata.json", case, args)
    connectivity = await asyncio.to_thread(check_connectivity, ApiConfig.from_env())
    (out / "connectivity.json").write_text(json.dumps(connectivity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not connectivity["passed"]:
        summary = {"case": case.name, "sent": 0, "successful": 0, "success_rate": 0,
                   "connectivity": connectivity, "checks": {"connectivity": connectivity},
                   "errors": {"connectivity_failed": 1}, "admission_passed": False,
                   "scope_note": "前置连通性测试失败，正式测试未启动。"}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_artifacts(out, [], summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2)); print(f"结果目录: {out.resolve()}")
        return 2
    writer = ResultWriter(out / "requests.jsonl")
    try:
        if case.mode == "closed":
            samples = len(prompts) if case.name == "B1" and args.samples is None else (min(args.samples or case.samples, 4) if args.quick else (args.samples or case.samples))
            warmup = 0 if args.quick else (case.warmup_seconds if args.warmup_seconds is None else args.warmup_seconds)
            # 用户显式指定 --samples 时按“精确发送请求数”执行，不再叠加默认 measured_seconds。
            # 否则 L1/L2 等默认轮次会继续运行到时长结束，导致请求数超过界面填写值。
            duration = 0 if args.quick or args.samples is not None else (case.measured_seconds if args.measured_seconds is None else args.measured_seconds)
            if args.concurrency is not None:
                case = case.__class__(**{**case.__dict__, "concurrency": args.concurrency})
            rows = await run_closed(case, prompts, writer, samples, warmup, duration)
        else:
            stages = (("quick", 1.0, 3, True),) if args.quick else case.rps_stages
            rows = await run_open(case, prompts, writer, stages)
    finally:
        writer.close()
    baseline = None
    if args.baseline_summary:
        baseline_summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        dataset_baseline = (baseline_summary.get("baselines_by_dataset") or {}).get(case.dataset, {})
        baseline = dataset_baseline.get("ttfc_p90_seconds", baseline_summary.get("ttfc_p90_seconds"))
    summary = summarize(case, rows, baseline)
    summary["attempt_limit"] = samples if case.mode == "closed" else None
    summary["request_limit_scope"] = "预热与正式请求合计" if case.mode == "closed" else "按固定速率阶段"
    summary["success_target"] = samples if case.mode == "closed" else None
    summary.update({"schema_version": "volcperf.result.v1", "record_id": os.getenv("VOLC_RECORD_ID"),
                    "batch_id": os.getenv("VOLC_BATCH_ID") or None, "test_kind": "load",
                    "protocol": ApiConfig.from_env().protocol, "base_url": ApiConfig.from_env().base_url,
                    "model": ApiConfig.from_env().model,
                    "workload_profile": "mixed-business-v1",
                    "reasoning_effort": ApiConfig.from_env().reasoning_effort})
    summary["connectivity"] = connectivity
    summary.setdefault("checks", {})["connectivity"] = connectivity
    summary["admission_passed"] = summary["admission_passed"] and connectivity["passed"]
    if args.quick:
        summary["quick_run"] = True; summary["admission_passed"] = False; summary["scope_note"] = "quick 模式仅验证脚本，不得作为准入结论。"
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_artifacts(out, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2)); print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def cli() -> None:
    try:
        raise SystemExit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except ValueError as exc:
        print(f"配置错误: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
