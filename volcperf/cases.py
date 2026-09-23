from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass(frozen=True)
class Case:
    name: str
    dataset: str
    target_prompt_tokens: int
    prompt_min: int
    prompt_max: int
    max_tokens: int
    mode: str
    concurrency: int = 1
    samples: int = 0
    warmup_seconds: int = 0
    measured_seconds: int = 0
    rps_stages: tuple[tuple[str, float, int, bool], ...] = ()
    ttfc_p50_limit: float | None = None
    ttfc_p90_limit: float | None = None
    tpot_p90_limit: float | None = None
    e2e_p90_limit: float | None = None
    min_visible_tokens: int = 128
    max_visible_tokens: int | None = None
    core: bool = False

    @property
    def data_file(self) -> Path:
        return DATA / f"{self.dataset}.jsonl.gz"


CASES = {
    # T0 冒烟：thinking 默认开启时推理会吃掉大部分预算，1024 对 code 负载
    # 系统性不足（推理未结束即 length 截断 → 零可见正文）。提到 4096 留出余量。
    "T0": Case("T0", "I1", 7500, 7200, 7900, 4096, "closed", concurrency=2, samples=10, core=True),
    "RANDOM_LOAD": Case("RANDOM_LOAD", "RANDOM_1K_20K", 10500, 1000, 20000, 2048, "closed", concurrency=10, samples=100),
    # B1 合并执行 2K/8K/16K/32K，每档固定 4 次，合计 16 次；并发统一为 2。
    "B1": Case("B1", "I1", 7500, 7200, 50000, 1024, "closed", concurrency=2, samples=16),
    "L1": Case("L1", "I1", 7500, 7200, 7900, 1024, "closed", 50, 100, 60, 180, ttfc_p50_limit=5, ttfc_p90_limit=15, tpot_p90_limit=60, core=True),
    "L2": Case("L2", "I2", 12000, 11500, 12500, 1024, "closed", 40, 100, 60, 180, ttfc_p50_limit=7, ttfc_p90_limit=20, tpot_p90_limit=65, core=True),
    "L3": Case("L3", "I3", 24000, 23000, 25000, 1024, "closed", 25, 100, 60, 180, ttfc_p50_limit=10, ttfc_p90_limit=28, tpot_p90_limit=70, core=True),
    "L4": Case("L4", "I4", 48000, 46000, 50000, 1024, "closed", 15, 80, 60, 180, ttfc_p50_limit=15, ttfc_p90_limit=40, tpot_p90_limit=80, core=True),
    "L5": Case("L5", "I4", 48000, 46000, 50000, 1024, "closed", 8, 50, 60, 180, ttfc_p50_limit=22, ttfc_p90_limit=60, tpot_p90_limit=90),
    "O1": Case("O1", "I2", 12000, 11500, 12500, 2048, "closed", 10, 50, 60, 180, ttfc_p90_limit=20, tpot_p90_limit=65, e2e_p90_limit=60, min_visible_tokens=400, max_visible_tokens=700, core=True),
    "C0": Case("C0", "C1", 10000, 9800, 10200, 1024, "open", rps_stages=(("ramp_4", 4.0, 30, True), ("ramp_7", 7.0, 30, True), ("ramp_10_5", 10.5, 30, True))),
    "C1": Case("C1", "C1", 10000, 9800, 10200, 1024, "open", rps_stages=(("steady", 10.5, 180, True),), core=True),
}
