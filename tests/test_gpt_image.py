import asyncio
import base64
import io
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from gpt_image.cases import (
    GPT_IMAGE_2_QUALITIES,
    GPT_IMAGE_25_QUALITIES,
    GPT_IMAGE_2_SIZES,
    CASES,
)
from gpt_image.runner import _failed_request_row, _request_one, _run_requests, image_endpoint
from gpt_image.runner import build_request_plan
from gpt_image.token_verify import (
    CASE_ID as TOKEN_VERIFY_CASE_ID,
    IMAGE_BYTES_PER_TOKEN,
    IMAGE_COUNT,
    PROMPT as TOKEN_VERIFY_PROMPT,
    SIZE as TOKEN_VERIFY_SIZE,
    expected_input_tokens,
    expected_output_tokens,
    run_model,
    verify_usage,
)


def test_official_quality_matrix_is_split_between_image_generations():
    assert GPT_IMAGE_2_QUALITIES == ("auto", "low", "medium", "high")
    assert GPT_IMAGE_25_QUALITIES == (
        "auto",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    assert set(GPT_IMAGE_2_SIZES) == {
        "1024x1024",
        "2048x2048",
        "1536x1024",
        "1024x1536",
        "1536x1152",
        "2048x1536",
        "1152x1536",
        "1536x2048",
        "1536x864",
        "2048x1152",
        "2560x1440",
        "3840x2160",
        "864x1536",
        "1152x2048",
        "1440x2560",
        "2160x3840",
    }
    assert CASES["GPT_IMAGE_25"].model_choices == (
        "gpt-image-2.5-flare",
        "gpt-image-2.5-sunburst",
    )
    assert CASES[TOKEN_VERIFY_CASE_ID].model_choices == (
        "gpt-image-2",
        "gpt-image-2.5-flare",
        "gpt-image-2.5-sunburst",
    )


def test_image_request_uses_generation_endpoint_and_does_not_store_base64():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/images/generations"
        assert request.headers["authorization"] == "Bearer test-secret"
        body = request.read().decode()
        assert '"model":"gpt-image-2"' in body
        assert '"quality":"high"' in body
        assert '"size":"1024x1024"' in body
        return httpx.Response(
            200,
            json={
                "created": 1,
                "data": [{"b64_json": "a-very-large-image-payload"}],
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _request_one(
                client,
                request_plan={
                    "case": "GPT_IMAGE_2",
                    "index": 1,
                    "model": "gpt-image-2",
                    "quality": "high",
                    "size": "1024x1024",
                    "prompt": "test prompt",
                    "payload": {
                        "model": "gpt-image-2",
                        "prompt": "test prompt",
                        "size": "1024x1024",
                        "quality": "high",
                        "n": 1,
                    },
                },
                protocol="openai",
                base_url="https://example.test/v1",
                api_key="test-secret",
            )

    row = asyncio.run(run())
    assert row["success"] is True
    assert row["image_count"] == 1
    assert "a-very-large-image-payload" not in str(row)
    assert image_endpoint("https://api.openai.com/v1/") == "https://api.openai.com/v1/images/generations"
    assert image_endpoint("https://mux.la", "openai_images") == "https://mux.la/v1/images/generations"
    assert image_endpoint("https://ark.example", "volcengine") == "https://ark.example/api/v3/images/generations"


def test_request_plan_is_complete_and_reproducible_before_execution():
    case = CASES["GPT_IMAGE_25"]
    first = build_request_plan(
        case=case,
        requests=20,
        qualities=["low", "high"],
        sizes=["1024x1024", "1536x1024"],
        models=list(case.model_choices),
        seed=20260915,
    )
    second = build_request_plan(
        case=case,
        requests=20,
        qualities=["low", "high"],
        sizes=["1024x1024", "1536x1024"],
        models=list(case.model_choices),
        seed=20260915,
    )
    assert first == second
    assert len(first) == 20
    assert [item["index"] for item in first] == list(range(1, 21))
    assert all(item["payload"]["n"] == 1 for item in first)


def test_transport_timeout_returns_complete_failure_row():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("forced timeout", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _request_one(
                client,
                request_plan={
                    "case": "GPT_IMAGE_2",
                    "index": 1,
                    "model": "gpt-image-2",
                    "quality": "low",
                    "size": "1024x1024",
                    "prompt": "test",
                    "payload": {"model": "gpt-image-2", "prompt": "test"},
                },
                protocol="openai",
                base_url="https://example.test/v1",
                api_key="secret",
            )

    row = asyncio.run(run())
    assert row["success"] is False
    assert row["error_type"] == "timeout"
    assert row["response_model"] is None


def test_absolute_timeout_does_not_block_following_requests_and_flushes_rows():
    plans = [
        {
            "case": "GPT_IMAGE_2",
            "index": index,
            "model": "gpt-image-2",
            "quality": "low",
            "size": "1024x1024",
            "prompt": "test",
            "payload": {"model": "gpt-image-2", "prompt": "test"},
        }
        for index in range(1, 4)
    ]

    async def fake_request(client, *, request_plan, protocol, base_url, api_key):
        if request_plan["index"] == 1:
            await asyncio.sleep(0.05)
        row = _failed_request_row(
            request_plan,
            protocol=protocol,
            base_url=base_url,
            started_epoch=time.time(),
            elapsed_seconds=0.001,
            error_type="http_503",
            error_message="test",
        )
        return row

    async def run(path: Path):
        with patch("gpt_image.runner._request_one", side_effect=fake_request):
            return await _run_requests(
                request_plan=plans,
                concurrency=1,
                protocol="openai",
                base_url="https://example.test/v1",
                api_key="secret",
                request_timeout_seconds=0.01,
                requests_path=path,
            )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "requests.jsonl"
        rows = asyncio.run(run(path))
        persisted = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(rows) == 3
    assert len(persisted) == 3
    assert rows[0]["error_type"] == "request_timeout"
    assert [row["index"] for row in rows] == [1, 2, 3]


def test_4k_token_verifier_matches_prompt_and_each_decoded_image():
    image_sizes = [10 * 1024 * 1024 + index for index in range(IMAGE_COUNT)]
    converted_output = expected_output_tokens(image_sizes)
    converted_input = expected_input_tokens(TOKEN_VERIFY_PROMPT)
    images = [{"bytes": size} for size in image_sizes]
    usage = {
        "generated_images": IMAGE_COUNT,
        "input_tokens": converted_input,
        "input_tokens_details": {"text_tokens": converted_input},
        "output_tokens": converted_output,
        "output_tokens_details": {"image_tokens": converted_output},
        "total_tokens": converted_input + converted_output,
    }

    checks = verify_usage(usage, TOKEN_VERIFY_PROMPT, images)

    assert all(check["passed"] for check in checks.values())
    assert expected_output_tokens([IMAGE_BYTES_PER_TOKEN + 1]) == 2


def test_4k_token_verifier_sends_one_n10_request_and_saves_images(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode()
    output_tokens = expected_output_tokens([len(raw)] * IMAGE_COUNT)
    input_tokens = expected_input_tokens(TOKEN_VERIFY_PROMPT)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        assert payload == {
            "model": "gpt-image-2",
            "prompt": TOKEN_VERIFY_PROMPT,
            "size": TOKEN_VERIFY_SIZE,
            "quality": "high",
            "output_format": "png",
            "n": IMAGE_COUNT,
        }
        return httpx.Response(
            200,
            json={
                "model": "gpt-image-2",
                "data": [{"b64_json": encoded} for _ in range(IMAGE_COUNT)],
                "usage": {
                    "generated_images": IMAGE_COUNT,
                    "input_tokens": input_tokens,
                    "input_tokens_details": {"text_tokens": input_tokens},
                    "output_tokens": output_tokens,
                    "output_tokens_details": {"image_tokens": output_tokens},
                    "total_tokens": input_tokens + output_tokens,
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_model(
                client,
                model="gpt-image-2",
                quality="high",
                protocol="openai",
                base_url="https://example.test/v1",
                api_key="secret",
                image_dir=tmp_path,
            )

    row = asyncio.run(run())

    assert row["image_count"] == IMAGE_COUNT
    assert len(list(tmp_path.glob("*.png"))) == IMAGE_COUNT
    assert row["checks"]["input_tokens_match"]["passed"] is True
    assert row["checks"]["output_tokens_match"]["passed"] is True
    assert row["checks"]["all_dimensions_4k"]["passed"] is False
    assert row["checks"]["all_images_around_10m"]["passed"] is False
    assert row["case_passed"] is False


def _probe_png_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def _run_probe(handler, save_image=None):
    from scripts.probe_gpt_image import run_probe

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await run_probe(
            client,
            model="gpt-image-2.5-sunburst",
            protocol="openai",
            base_url="https://example.test/v1",
            api_key="secret",
            save_image=save_image,
        )


def test_probe_reports_callable_for_valid_b64_response():
    from scripts.probe_gpt_image import DEFAULT_PROMPT, probe_exit_code

    seen: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("authorization"),
                "payload": json.loads(request.content),
            }
        )
        return httpx.Response(
            200,
            json={
                "model": "gpt-image-2.5-sunburst",
                "data": [{"b64_json": _probe_png_b64()}],
                "usage": {"input_tokens": 21, "output_tokens": 6, "total_tokens": 27},
            },
        )

    row = asyncio.run(_run_probe(handler))

    assert seen[0]["url"] == "https://example.test/v1/images/generations"
    assert seen[0]["auth"] == "Bearer secret"
    assert seen[0]["payload"] == {
        "model": "gpt-image-2.5-sunburst",
        "prompt": DEFAULT_PROMPT,
        "size": "1024x1024",
        "quality": "low",
        "n": 1,
    }
    assert row["callable"] is True
    assert row["image_count"] == 1
    assert row["result_types"] == ["b64_json"]
    assert row["image_bytes"] > 0
    assert probe_exit_code(row) == 0


def test_probe_saves_url_result_and_flags_unusable_response():
    from scripts.probe_gpt_image import probe_exit_code

    async def url_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/images/generations"):
            return httpx.Response(
                200,
                json={"model": "gpt-image-2.5-sunburst", "data": [{"url": "https://cdn.test/a.png"}]},
            )
        return httpx.Response(200, content=base64.b64decode(_probe_png_b64()))

    async def empty_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "gpt-image-2.5-sunburst", "data": []})

    async def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "probe.png"
        url_row = asyncio.run(_run_probe(url_handler, target))
        assert url_row["callable"] is True
        assert url_row["result_types"] == ["url"]
        assert target.exists() and target.stat().st_size > 0
        assert url_row["saved_image"] == str(target)

    empty_row = asyncio.run(_run_probe(empty_handler))
    assert empty_row["callable"] is False
    assert empty_row["error_type"] == "invalid_image_response"
    assert empty_row["error_message"] == "HTTP 成功但响应未包含 data 图片结果"
    assert probe_exit_code(empty_row) == 2

    error_row = asyncio.run(_run_probe(error_handler))
    assert error_row["callable"] is False
    assert error_row["error_type"] == "http_404"
    assert "model not found" in error_row["error_message"]
    assert probe_exit_code(error_row) == 1
