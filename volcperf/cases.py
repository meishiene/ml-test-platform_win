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
    "T0": Case("T0", "P2K", 2048, 1900, 2200, 1024, "closed", concurrency=2, samples=10, core=True),
    # B1 合并执行 2K/8K/16K/32K，每档固定 4 次，合计 16 次；并发统一为 2。
    "B1": Case("B1", "P2K", 2048, 1900, 35200, 1024, "closed", concurrency=2, samples=16),
    "L1": Case("L1", "P2K", 2048, 1900, 2200, 1024, "closed", 50, 100, 60, 180, ttfc_p50_limit=5, ttfc_p90_limit=15, tpot_p90_limit=60, core=True),
    "L2": Case("L2", "P8K", 8192, 7600, 8800, 1024, "closed", 40, 100, 60, 180, ttfc_p50_limit=7, ttfc_p90_limit=20, tpot_p90_limit=65, core=True),
    "L3": Case("L3", "P16K", 16384, 15200, 17600, 1024, "closed", 25, 100, 60, 180, ttfc_p50_limit=10, ttfc_p90_limit=28, tpot_p90_limit=70, core=True),
    "L4": Case("L4", "P32K", 32768, 30400, 35200, 1024, "closed", 15, 80, 60, 180, ttfc_p50_limit=15, ttfc_p90_limit=40, tpot_p90_limit=80, core=True),
    "L5": Case("L5", "P48K", 49152, 45600, 52800, 1024, "closed", 8, 50, 60, 180, ttfc_p50_limit=22, ttfc_p90_limit=60, tpot_p90_limit=90),
    "O1": Case("O1", "P8K", 8192, 7600, 8800, 2048, "closed", 10, 50, 60, 180, ttfc_p90_limit=20, tpot_p90_limit=65, e2e_p90_limit=60, min_visible_tokens=400, max_visible_tokens=700, core=True),
    "C0": Case("C0", "P10K", 10240, 9500, 11000, 1024, "open", rps_stages=(("ramp_4", 4.0, 30, True), ("ramp_7", 7.0, 30, True), ("ramp_10_5", 10.5, 30, True))),
    "C1": Case("C1", "P10K", 10240, 9500, 12000, 1024, "open", rps_stages=(("steady", 10.5, 180, True),), core=True),
}
