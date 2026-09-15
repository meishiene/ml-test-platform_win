from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from common.audit import redact
from common.finalizer import finalize_result_dir

from .cases import CASES, Case


PROMPTS = (
    "A clean editorial product photograph of a translucent glass teapot on a pale stone table, soft morning light, no text, no logo.",
    "A cinematic wide landscape of a red train crossing a high bridge above a misty forest valley at sunrise, detailed natural colors.",
    "A modern architectural poster showing a quiet concrete pavilion beside a reflecting pool, geometric composition, no words or logos.",
    "A friendly orange cat wearing a tiny blue raincoat walking through a neon-lit Tokyo side street at night, polished illustration.",
    "A top-down food photograph of a colorful seasonal vegetable platter on a dark ceramic plate, studio lighting, realistic texture.",
    "A hand-painted science-fiction observatory on an icy moon, distant planet in the sky, atmospheric perspective, highly detailed.",
    "A warm lifestyle photograph of a wooden desk with a notebook, dried flowers, and a brass lamp beside a sunny window.",
    "A whimsical miniature coastal town built inside a seashell, tiny boats and people, soft storybook lighting.",
)


def image_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/images/generations"


def request_headers(protocol: str, api_key: str, request_id: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Request-ID": request_id,
    }
    if protocol in {"openai", "volcengine"}:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        raise ValueError(f"图片压测暂不支持 {protocol} 协议")
    return headers


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = int(index), min(len(ordered) - 1, int(index) + 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _parse_positive_list(raw: str | None, allowed: tuple[str, ...], label: str) -> list[str]:
    values = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not values:
        return list(allowed)
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"{label}包含不支持的值: {', '.join(invalid)}")
    return list(dict.fromkeys(values))


def _model_choices(case: Case, raw: str | None) -> list[str]:
    values = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if not values:
        return list(case.model_choices)
    invalid = [value for value in values if value not in case.model_choices]
    if invalid:
        raise ValueError(f"模型选择包含不支持的值: {', '.join(invalid)}")
    return list(dict.fromkeys(values))


def _safe_response_headers(response: httpx.Response) -> dict[str, str]:
    keep = {
        "content-type",
        "server",
        "via",
        "x-request-id",
        "request-id",
        "x-trace-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
    }
    return {
        key.lower(): value
        for key, value in response.headers.items()
        if key.lower() in keep
    }


def _response_summary(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {"image_count": 0, "result_types": [], "response_id": None}
    items = body.get("data")
    if not isinstance(items, list):
        items = []
    result_types = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("b64_json"):
            result_types.append("b64_json")
        elif item.get("url"):
            result_types.append("url")
        elif item:
            result_types.append("object")
    return {
        "image_count": len(items),
        "result_types": result_types,
        "response_id": body.get("id"),
        "created": body.get("created"),
        "usage": body.get("usage"),
        "response_model": body.get("model"),
    }


def build_request_plan(
    *,
    case: Case,
    requests: int,
    qualities: list[str],
    sizes: list[str],
    models: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Generate every request definition before any HTTP request starts."""
    rng = random.Random(seed)
    plan: list[dict[str, Any]] = []
    for index in range(requests):
        model = rng.choice(models)
        quality = rng.choice(qualities)
        size = rng.choice(sizes)
        prompt = rng.choice(PROMPTS)
        plan.append(
            {
                "case": case.case_id,
                "index": index + 1,
                "model": model,
                "quality": quality,
                "size": size,
                "prompt": prompt,
                "payload": {
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "quality": quality,
                    "n": 1,
                },
            }
        )
    return plan


async def _request_one(
    client: httpx.AsyncClient,
    *,
    request_plan: dict[str, Any],
    protocol: str,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    case_name = request_plan["case"]
    index = int(request_plan["index"]) - 1
    model = request_plan["model"]
    quality = request_plan["quality"]
    size = request_plan["size"]
    prompt = request_plan["prompt"]
    payload = dict(request_plan["payload"])
    request_id = f"{case_name}-{index + 1}-{uuid.uuid4().hex[:10]}"
    started_epoch = time.time()
    started_mono = time.perf_counter()
    response: httpx.Response | None = None
    body: Any = None
    error_type: str | None = None
    error_message: str | None = None
    try:
        response = await client.post(
            image_endpoint(base_url),
            json=payload,
            headers=request_headers(protocol, api_key, request_id),
        )
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError):
            body = None
        if not 200 <= response.status_code < 300:
            error_type = f"http_{response.status_code}"
            error_message = response.text[:2000]
        else:
            details = _response_summary(body)
            if details["image_count"] < 1:
                error_type = "invalid_image_response"
                error_message = "HTTP 成功但响应未包含 data 图片结果"
    except httpx.TimeoutException as exc:
        error_type, error_message = "timeout", str(exc)
    except httpx.HTTPError as exc:
        error_type, error_message = "client_error", str(exc)
    except Exception as exc:  # keep one failed request from aborting the batch
        error_type, error_message = type(exc).__name__, str(exc)

    elapsed = time.perf_counter() - started_mono
    details = _response_summary(body)
    return {
        "case": case_name,
        "index": index + 1,
        "request_id": request_id,
        "started_at": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
        "started_epoch": started_epoch,
        "completed_epoch": time.time(),
        "protocol": protocol,
        "endpoint": image_endpoint(base_url),
        "model": model,
        "quality": quality,
        "size": size,
        "prompt": prompt,
        "http_status": response.status_code if response is not None else None,
        "success": error_type is None,
        "error_type": error_type,
        "error_message": error_message,
        "elapsed_seconds": elapsed,
        "response_headers": _safe_response_headers(response) if response is not None else {},
        "image_count": details["image_count"],
        "result_types": details["result_types"],
        "response_id": details["response_id"],
        "response_model": details["response_model"],
        "created": details["created"],
        "usage": details["usage"],
        "transport_status": "completed" if response is not None else "error",
        # Do not persist b64_json image payloads in the load-test artifact.
        "response_body_shape": {
            "keys": sorted(body.keys()) if isinstance(body, dict) else [],
            "image_count": details["image_count"],
            "result_types": details["result_types"],
        },
    }


async def _run_requests(
    *,
    request_plan: list[dict[str, Any]],
    concurrency: int,
    protocol: str,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(len(request_plan)):
        queue.put_nowait(index)
    rows: list[dict[str, Any]] = []
    rows_lock = asyncio.Lock()
    timeout = httpx.Timeout(
        float(os.getenv("GPT_IMAGE_REQUEST_TIMEOUT_SECONDS", "900")),
        connect=30.0,
        pool=30.0,
    )
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
        keepalive_expiry=60.0,
    )
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        verify=True,
        trust_env=False,
    ) as client:
        async def worker() -> None:
            while True:
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    row = await _request_one(
                        client,
                        request_plan=request_plan[index],
                        protocol=protocol,
                        base_url=base_url,
                        api_key=api_key,
                    )
                    async with rows_lock:
                        rows.append(row)
                        completed = len(rows)
                    if (
                        completed % max(1, min(100, len(request_plan) // 20 or 1)) == 0
                        or completed == len(request_plan)
                    ):
                        print(
                            f"\r{request_plan[0]['case']}: "
                            f"{completed}/{len(request_plan)}",
                            end="",
                            flush=True,
                        )
                finally:
                    queue.task_done()

        await asyncio.gather(
            *(
                asyncio.create_task(worker())
                for _ in range(min(concurrency, len(request_plan)))
            )
        )
    print()
    return sorted(rows, key=lambda row: row["index"])


def _write_report(out: Path, summary: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        f"# {summary['case']} 图片压测报告",
        "",
        f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        f"- 总请求数：{summary['requested']}",
        f"- 成功：{summary['successful']}；失败：{summary['failed']}；成功率：{summary['success_rate']:.2%}",
        f"- 并发数：{summary['concurrency']}",
        f"- 实际模型分布：{json.dumps(summary['model_distribution'], ensure_ascii=False)}",
        f"- 尺寸分布：{json.dumps(summary['size_distribution'], ensure_ascii=False)}",
        f"- 质量分布：{json.dumps(summary['quality_distribution'], ensure_ascii=False)}",
        "- 执行方式：先生成完整 `generated_requests.jsonl`，再按预生成顺序以配置并发执行。",
        "",
        "## 延迟指标",
        "",
        "| 指标 | 数值（秒） |",
        "|---|---:|",
        f"| P50 | {fmt(summary['latency_p50_seconds'])} |",
        f"| P90 | {fmt(summary['latency_p90_seconds'])} |",
        f"| P99 | {fmt(summary['latency_p99_seconds'])} |",
        f"| 平均 | {fmt(summary['latency_avg_seconds'])} |",
        "",
        "> 请求明细只保存响应结构和元数据，不保存返回的 base64 图片内容，避免结果目录被图片数据放大。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPT Image 并发图片压测")
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--quality", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--sizes", default="")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    if not 1 <= args.requests <= 100000:
        parser.error("--requests 必须在 1～100000 之间")
    if not 1 <= args.concurrency <= 1000:
        parser.error("--concurrency 必须在 1～1000 之间")

    case = CASES[args.case]
    qualities = _parse_positive_list(
        args.quality or os.getenv("GPT_IMAGE_QUALITIES"),
        case.qualities,
        "quality",
    )
    sizes = _parse_positive_list(
        args.sizes or os.getenv("GPT_IMAGE_SIZES"),
        case.sizes,
        "size",
    )
    models = _model_choices(
        case,
        args.models or os.getenv("GPT_IMAGE_MODELS"),
    )
    api_key = os.getenv("GPT_IMAGE_API_KEY") or os.getenv("VOLC_API_KEY") or ""
    base_url = os.getenv("GPT_IMAGE_BASE_URL") or os.getenv("VOLC_BASE_URL") or ""
    protocol = (os.getenv("GPT_IMAGE_PROTOCOL") or os.getenv("VOLC_PROTOCOL") or "openai").lower()
    if not api_key or not base_url:
        parser.error("缺少 GPT_IMAGE_API_KEY/VOLC_API_KEY 或 GPT_IMAGE_BASE_URL/VOLC_BASE_URL")
    if protocol not in {"openai", "volcengine"}:
        parser.error("图片压测仅支持 openai 或 volcengine 协议")

    run_id = os.getenv("VOLC_RECORD_ID") or uuid.uuid4().hex[:16]
    out = args.output_root / f"{case.case_id}-{time.strftime('%Y%m%d-%H%M%S')}-{run_id[:6]}"
    out.mkdir(parents=True, exist_ok=False)
    resolved_seed = (
        args.seed
        if args.seed is not None
        else random.SystemRandom().randrange(0, 2**63)
    )
    request_plan = build_request_plan(
        case=case,
        requests=args.requests,
        qualities=qualities,
        sizes=sizes,
        models=models,
        seed=resolved_seed,
    )
    metadata = {
        "schema_version": "mltest.gpt-image.v1",
        "suite": "gpt_image",
        "test_kind": "gpt-image",
        "case": case.case_id,
        "record_id": run_id,
        "protocol": protocol,
        "base_url": base_url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "qualities": qualities,
        "sizes": sizes,
        "models": models,
        "seed": resolved_seed,
        "prompt_count": len(PROMPTS),
        "spec_source": "OpenAI official image generation model definitions",
        "generation_phase": "complete_before_execution",
    }
    (out / "metadata.json").write_text(
        json.dumps(redact(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "generated_requests.jsonl").write_text(
        "\n".join(
            json.dumps(redact(item), ensure_ascii=False) for item in request_plan
        )
        + "\n",
        encoding="utf-8",
    )

    rows = await _run_requests(
        request_plan=request_plan,
        concurrency=args.concurrency,
        protocol=protocol,
        base_url=base_url,
        api_key=api_key,
    )
    (out / "requests.jsonl").write_text(
        "\n".join(json.dumps(redact(row), ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    successful = [row for row in rows if row["success"]]
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    successful_latencies = [float(row["elapsed_seconds"]) for row in successful]
    summary = {
        "schema_version": "mltest.gpt-image.v1",
        "record_id": run_id,
        "suite": "gpt_image",
        "test_kind": "gpt-image",
        "case": case.case_id,
        "protocol": protocol,
        "base_url": base_url,
        "models": models,
        "qualities": qualities,
        "sizes": sizes,
        "requested": args.requests,
        "sent": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "concurrency": args.concurrency,
        "latency_p50_seconds": percentile(successful_latencies, 0.50),
        "latency_p90_seconds": percentile(successful_latencies, 0.90),
        "latency_p99_seconds": percentile(successful_latencies, 0.99),
        "latency_avg_seconds": sum(latencies) / len(latencies) if latencies else None,
        "model_distribution": dict(Counter(row["model"] for row in rows)),
        "quality_distribution": dict(Counter(row["quality"] for row in rows)),
        "size_distribution": dict(Counter(row["size"] for row in rows)),
        "error_distribution": dict(
            Counter(row["error_type"] for row in rows if row["error_type"])
        ),
        "checks": {
            "exact_request_count": {
                "value": len(rows),
                "expected": args.requests,
                "passed": len(rows) == args.requests,
            },
            "all_successful": {
                "value": len(successful),
                "expected": args.requests,
                "passed": len(successful) == args.requests,
            },
        },
        "admission_passed": len(rows) == args.requests and len(successful) == args.requests,
        "scope_note": "图片生成压测按请求完成时间统计；每次 n=1，不下载或保存 base64 图片内容。",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(out, summary)
    finalize_result_dir(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录: {out.resolve()}")
    return 0 if summary["admission_passed"] else 2


def cli() -> None:
    try:
        raise SystemExit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except ValueError as exc:
        print(f"配置错误: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
