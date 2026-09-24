"""统一的测试报告 markdown 样式（对齐 dsv4/core.py write_report 的样式基线）。

所有测试项的 report.md 都应使用这里的助手生成，保证 Web 报告弹窗样式一致：
标题 → 统计要点（总计/通过/失败/警告/阻塞/未观测/已评估用例通过率）→ 用例表格。
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

STATUS_ICONS = {
    "passed": "✅ 通过",
    "failed": "❌ 失败",
    "error": "❌ 错误",
    "warning": "⚠️ 警告",
    "blocked": "⏸ 未验证",
}


def status_icon(status: str) -> str:
    return STATUS_ICONS.get(status, status)


def _escape(cell: Any) -> str:
    return str(cell).replace("|", "\\|").replace("\n", " ")


def stats_lines(
    total: int,
    passed: int,
    failed: int,
    warning: int = 0,
    blocked: int = 0,
    pass_rate: float | None = None,
    extra: Iterable[str] = (),
) -> list[str]:
    """统计要点块。pass_rate 缺省按 (passed)/(passed+failed) 计算。"""
    if pass_rate is None:
        evaluated = passed + failed
        pass_rate = passed / evaluated if evaluated else 0.0
    lines = [
        f"- 总计：{total}",
        f"- 通过：{passed}",
        f"- 失败：{failed}",
        f"- 警告：{warning}",
        f"- 阻塞/未观测：{blocked}",
        f"- 已评估用例通过率：{pass_rate:.1%}",
    ]
    lines.extend(f"- {x}" for x in extra)
    return lines


def case_table_report(
    title: str,
    *,
    total: int,
    passed: int,
    failed: int,
    warning: int = 0,
    blocked: int = 0,
    pass_rate: float | None = None,
    extra_stats: Iterable[str] = (),
    columns: Sequence[str] = ("用例", "模型", "结果", "耗时(s)", "说明"),
    status_column: int | None = 2,
    rows: Sequence[Sequence[Any]],
    trailing: Iterable[str] = (),
) -> str:
    """渲染与 DeepSeek V4 专项测试报告一致的 report.md 全文。

    rows 单元格自动转义；若指定 status_column，该列的状态词（passed/failed/...）
    会自动替换为图标文案。
    """
    lines = [
        f"# {title}",
        "",
        *stats_lines(total, passed, failed, warning, blocked, pass_rate, extra_stats),
        "",
        "| " + " | ".join(columns) + " |",
    ]
    align = ["---"] * len(columns)
    if len(columns) >= 4:
        align[3] = "---:"
    lines.append("|" + "|".join(align) + "|")
    for row in rows:
        cells = [_escape(c) for c in row]
        if status_column is not None and status_column < len(cells):
            cells[status_column] = status_icon(str(cells[status_column]))
        lines.append("| " + " | ".join(cells) + " |")
    tail = list(trailing)
    if tail:
        lines.extend(["", *tail])
    return "\n".join(lines) + "\n"
