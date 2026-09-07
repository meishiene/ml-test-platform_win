from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from .prompts import Workload, render_prompt, render_workload


@dataclass(frozen=True)
class PromptRecord:
    id: str
    dataset: str
    target_prompt_tokens: int
    marker: str
    word_count: int
    word_offset: int

    def content(self, fixed_output: bool = False) -> str:
        return render_prompt(self.marker, self.word_count, self.word_offset, "fixed" if fixed_output else "normal")

    def workload(self, fixed_output: bool = False) -> Workload:
        try:
            variant = int(self.id.rsplit("-", 1)[-1])
        except ValueError:
            variant = int(self.marker[:8], 16)
        return render_workload(self.marker, self.word_count, self.word_offset, variant, fixed_output=fixed_output)


def load_prompts(path: Path) -> list[PromptRecord]:
    if not path.exists():
        raise FileNotFoundError(f"测试数据不存在: {path}；请先运行 python scripts/prepare_data.py")
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[PromptRecord] = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                records.append(PromptRecord(raw["id"], raw["dataset"], int(raw["target_prompt_tokens"]), raw["marker"], int(raw["word_count"]), int(raw["word_offset"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"无效数据 {path}:{number}: {exc}") from exc
    if not records:
        raise ValueError(f"测试数据为空: {path}")
    return records
