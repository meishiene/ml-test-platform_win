from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from common.finalizer import finalize_result_dir
from common.verdict import check, verdict
from volcperf.protocols import endpoint, headers

from .cases import CASES, GPT_MODEL_MATCH


def extract_response_model(response_body: Any) -> str | None:
    """Return the model reported by a Chat Completions response."""
    if not isinstance(response_body, dict):
        return None
    value = response_body.get("model")
    return value.strip() if isinstance(value, str) and value.strip() else None


def model_match_checks(*, expected_model: str, response_body: Any, http_status: int | None) -> list[Any]:
    """Build validator checks without treating HTTP 200 as a domain pass."""
    actual_model = extract_response_model(response_body)
    return [
        check(
            "http_status",
            http_status == 200,
            200,
            http_status,
            f"期望 HTTP 200，实际为 {http_status}",
        ),
        check(
            "response_json",
            isinstance(response_body, dict),
            "JSON object",
            type(response_body).__name__,
            "响应不是 JSON 对象，无法读取 model 字段",
        ),
        check(
            "response_model_present",
            actual_model is not None,
            "非空字符串",
            actual_model,
            "响应缺少顶层 model 字段，无法确认实际模型",
        ),
        check(
            "model_match",
            actual_model == expected_model if actual_model is not None else None,
            expected_model,
            actual_model,
            f"请求模型 {expected_model!r} 与响应模型 {actual_model!r} 不一致",
        ),
    ]


def _safe_headers(response: httpx.Response) -> dict[str, str]:
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
    return {key.lower(): value for key, value in response.headers.items() if key.lower() in keep}


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _request_payload(model: str) -> dict[str, Any]:
    # Keep the probe small and avoid temperature/reasoning-specific parameters so
    # it can be sent through OpenAI-compatible gateways with different defaults.
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK only."}],
        "max_completion_tokens": 16,
        "stream": False,
    }


def run(
    *,
    case_name: str,
    base_url: str,
    api_key: str,
    model: str,
    protocol: str = "openai",
    output_root: Path = Path("results"),
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> tuple[Path, dict[str, Any]]:
    if case_name not in CASES:
        raise ValueError(f"未知 GPT 检测用例: {case_name}")
    if protocol not in CASES[case_name].protocols:
        raise ValueError(f"{case_name} 不支持 {protocol} 协议")
    if not base_url.strip() or not api_key.strip() or not model.strip():
        raise ValueError("base_url、api_key 和 model 均不能为空")

    run_id = os.getenv("GPT_RECORD_ID") or os.getenv("VOLC_RECORD_ID") or uuid.uuid4().hex[:16]
    request_id = f"GPT-{run_id}-{uuid.uuid4().hex[:10]}"
    payload = _request_payload(model)
    request_headers = headers(protocol, api_key, request_id)
    request_headers["Accept"] = "application/json"
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    response: httpx.Response | None = None
    response_body: Any = None
    transport_error: dict[str, str] | None = None

    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout, verify=True, trust_env=False)
    try:
        try:
            response = http_client.post(
                endpoint(protocol, base_url, model),
                headers=request_headers,
                json=payload,
            )
            try:
                response_body = response.json()
            except (ValueError, json.JSONDecodeError):
                response_body = None
        except Exception as exc:  # preserve a structured result for network failures
            transport_error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if owns_client:
            http_client.close()

    http_status = response.status_code if response is not None else None
    checks = model_match_checks(expected_model=model, response_body=response_body, http_status=http_status)
    if transport_error:
        result_status = "error"
        result_passed = False
        reasons = [f"请求异常：{transport_error['type']}: {transport_error['message']}"]
    else:
        result = verdict(checks)
        result_status, result_passed, reasons = result.status, result.passed, result.reasons
    actual_model = extract_response_model(response_body)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response_text = response.text if response is not None else ""
    execution = {
        "request_id": request_id,
        "started_at": started_at,
        "endpoint": endpoint(protocol, base_url, model),
        "http_status": http_status,
        "request": payload,
        "request_headers": {k: v for k, v in request_headers.items() if k.lower() != "authorization"},
        "response_headers": _safe_headers(response) if response is not None else {},
        "response_body": _jsonable(response_body),
        "response_text": response_text[:10000],
        "expected_model": model,
        "actual_model": actual_model,
        "elapsed_ms": elapsed_ms,
        "transport_error": transport_error,
        "case_status": result_status,
        "case_passed": result_passed,
        "checks": [item.__dict__ for item in checks],
        "failure_reasons": reasons,
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_dir = output_root / f"{case_name}-{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "mltest.result.v3",
                "suite": "gpt_detection",
                "test_kind": "gpt",
                "case": case_name,
                "record_id": run_id,
                "protocol": protocol,
                "base_url": base_url,
                "model": model,
                "probe": "Chat Completions response.model exact match",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (result_dir / "requests.jsonl").write_text(
        json.dumps(execution, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    summary = {
        "schema_version": "mltest.result.v3",
        "record_id": run_id,
        "suite": "gpt_detection",
        "test_kind": "gpt",
        "case": case_name,
        "protocol": protocol,
        "base_url": base_url,
        "model": model,
        "expected_model": model,
        "actual_model": actual_model,
        "status": "completed" if not transport_error else "failed",
        "admission_passed": result_passed,
        "request_count": 1,
        "results": [
            {
                "case_id": case_name,
                "case_status": result_status,
                "case_passed": result_passed,
                "checks": [item.__dict__ for item in checks],
                "failure_reasons": reasons,
                "execution": execution,
            }
        ],
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    report = [
        "# GPT 模型身份一致性检测报告",
        "",
        f"- 用例：`{case_name}`",
        f"- 请求模型：`{model}`",
        f"- 响应模型：`{actual_model or '缺失'}`",
        f"- HTTP 状态：`{http_status if http_status is not None else '请求异常'}`",
        f"- 结论：**{'通过' if result_passed else '未通过'}**",
        "",
        "## 检查项",
        "",
    ]
    for item in checks:
        report.append(f"- `{item.name}`：{'通过' if item.passed is True else '未通过' if item.passed is False else '无法判定'}；{item.detail}")
    if transport_error:
        report.extend(["", f"请求异常：`{transport_error['type']}: {transport_error['message']}`"])
    report.extend(["", "说明：该检查只验证接口返回的 model 字段与请求值是否一致，不能单独证明网络一定是官方直连。"])
    (result_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    finalized = finalize_result_dir(result_dir)
    return result_dir, finalized
