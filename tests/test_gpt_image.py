import asyncio

import httpx

from gpt_image.cases import (
    GPT_IMAGE_2_QUALITIES,
    GPT_IMAGE_25_QUALITIES,
    GPT_IMAGE_2_SIZES,
    CASES,
)
from gpt_image.runner import _request_one, image_endpoint
from gpt_image.runner import build_request_plan


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
