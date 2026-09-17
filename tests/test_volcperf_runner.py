import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from volcperf.cases import CASES
from volcperf.config import ApiConfig
from volcperf.prompts import Workload
from volcperf.runner import (
    ResultWriter,
    build_closed_request_plan,
    build_open_request_plan,
    load_case_prompts,
    run_closed,
)


class TrackingRecord:
    def __init__(self, record_id: str, built: list[str], *, broken: bool = False):
        self.id = record_id
        self.built = built
        self.broken = broken

    def workload(self, fixed_output: bool = False) -> Workload:
        self.built.append(self.id)
        if self.broken:
            raise ValueError("broken workload")
        return Workload(
            "fixed_report" if fixed_output else "qa",
            [{"role": "user", "content": self.id}],
            16,
        )


def test_closed_load_prebuilds_all_requests_and_continues_after_case_errors():
    built: list[str] = []
    records = [
        TrackingRecord("I1-1", built),
        TrackingRecord("I1-2", built, broken=True),
        TrackingRecord("I1-3", built),
    ]
    config = ApiConfig(
        api_key="secret",
        base_url="https://example.test/v1",
        model="test-model",
        request_timeout_seconds=1,
    )
    executed: list[str] = []

    async def fake_stream_chat(client, config, **kwargs):
        # All workload definitions, including the broken one, must have been
        # materialized before the first HTTP execution starts.
        assert built == ["I1-1", "I1-2", "I1-3"]
        executed.append(kwargs["dataset_id"])
        if kwargs["dataset_id"] == "I1-1":
            raise RuntimeError("one request failed")
        return {
            "case": kwargs["case_name"],
            "counted": kwargs["counted"],
            "success": True,
            "error_type": None,
            "dataset_id": kwargs["dataset_id"],
        }

    async def run(path: Path):
        writer = ResultWriter(path)
        try:
            with (
                patch("volcperf.runner.ApiConfig.from_env", return_value=config),
                patch("volcperf.runner.stream_chat", side_effect=fake_stream_chat),
            ):
                return await run_closed(
                    CASES["T0"],
                    records,
                    writer,
                    samples=3,
                    warmup=0,
                    duration=0,
                )
        finally:
            writer.close()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "requests.jsonl"
        rows = asyncio.run(run(path))
        persisted = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert [row["error_type"] for row in rows] == [
        "internal_RuntimeError",
        "plan_ValueError",
        None,
    ]
    assert executed == ["I1-1", "I1-3"]
    assert len(persisted) == 3


def test_open_load_prebuilds_the_complete_relative_schedule():
    built: list[str] = []
    records = [TrackingRecord(f"C1-{index}", built) for index in range(5)]
    plans = build_open_request_plan(
        CASES["C0"],
        records,
        (("ramp", 5.0, 1, True),),
    )

    assert built == [f"C1-{index}" for index in range(5)]
    assert [plan["sequence"] for plan in plans] == [1, 2, 3, 4, 5]
    assert [plan["planned_offset_seconds"] for plan in plans] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert all(plan["phase"] == "ramp" and plan["counted"] for plan in plans)


def test_random_load_generates_exact_bounded_request_set_with_2048_output_limit():
    case = CASES["RANDOM_LOAD"]
    prompts = load_case_prompts(case, samples=37)
    plans = build_closed_request_plan(case, prompts, 37)

    assert case.max_tokens == 2048
    assert len(prompts) == len(plans) == 37
    assert all(1000 <= record.target_prompt_tokens <= 20000 for record in prompts)
    assert len({record.target_prompt_tokens for record in prompts}) > 1
    assert [plan["target_prompt_tokens"] for plan in plans] == [record.target_prompt_tokens for record in prompts]
    assert all(plan["max_tokens"] == 2048 for plan in plans)
