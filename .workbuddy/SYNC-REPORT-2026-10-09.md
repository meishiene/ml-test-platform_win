# 源码同步报告：`ml-test-platform` → `ml-test-platform_win`

日期：2026-10-09 16:5x　执行范围：单向同步（源仓 → Windows 仓）+ 启动链路验证

## 一、结论

`_win` 现已同时具备 **源仓全部最新特性** 与 **全部 Windows 平台层改造**，可在 Windows 上启动。
回归 **470 passed / 6 subtests passed**，真实链路冒烟通过（`/` 200、`/api/cases` 的 fable 卡 = 26）。

## 二、关键前提：两仓是**双向分叉**，不是单向落后

`_win` 于 15:53 从源仓拷出，之后两仓各自演进。因此**不能直接对拷**，否则每侧都会丢东西：

| 文件 | 源仓领先（特性） | `_win` 领先（Windows 平台层） |
|---|---|---|
| `web/server.py` | `CLAUDE_VARIANTS` 增 `haiku45`/`sonnet46`；相关注释 | `_portable_path` / `_child_popen_kwargs` / `_terminate_process_tree` / `PYTHONIOENCODING`+`PYTHONUTF8` / `main(argv)` / 展示地址换回环 |
| `tests/test_web.py` | 26 卡与 5 变体断言、`test_create_haiku_and_sonnet46_variant_runs_...` | `import os/subprocess/sys`、`CrossPlatformLaunchTests`、`WindowsStartupEntryTests` |
| `web/static/app.js` | `CLAUDE_VARIANT_MODELS` 增两项 | —（无特有改动） |
| `web/static/index.html` | 下拉增 `haiku45`/`sonnet46`、tab 副标题 | —（无特有改动） |
| `docs/Fable5专项测试用例设计.md` | 新增「模型变体复用」章节（5 变体表 + 判定口径） | —（无特有改动） |
| `docs/Haiku4.5与Sonnet4.6专项测试用例设计.md` | 新建（§一~§七） | 不存在 |
| `tests/test_fable5.py` | 变体族令牌推导与交叉不互认断言 | — |
| `scripts/start_web.py` | —（仍是 3 行旧桩） | 跨平台启动入口全文 |
| `start.sh` | — | 瘦身为「找解释器/建 venv/补依赖」后 exec 入口 |
| `kvv/runner.py` | — | `official_python()` 探测 `Scripts/python.exe`、报错指向 `KVV_PYTHON`/`KVV_HOME` |
| `README.md` | — | Windows 启动段、代理端口换机提醒、已知问题第 2/11 条 |
| `start.bat` / `.gitattributes` | 不存在 | 新建 |

## 三、变更清单

**A. 直接取源版本（5 个文件）** — 源侧较新且 `_win` 无特有改动
`docs/Fable5专项测试用例设计.md`、`docs/Haiku4.5与Sonnet4.6专项测试用例设计.md`（新增）、`tests/test_fable5.py`、`web/static/app.js`、`web/static/index.html`

**B. hunk 级合并（2 个文件）** — 先覆盖为源版本，再打回 Windows 补丁
- `web/server.py`：保留源的 11 个 hunk 中 1–3（变体相关），打回 4–11（Windows 相关）→ 现 1504 行
- `tests/test_web.py`：保留源的 5 个 hunk 中 2–4（变体断言），打回 1（导入）与 5（两个跨平台测试类）→ 现 935 行

**C. 原样保留（平台层，勿被源覆盖）**
`start.bat`、`.gitattributes`、`scripts/start_web.py`、`start.sh`、`kvv/runner.py`、`README.md`

**D. 未同步（非代码 / 运行时数据，避免误覆盖）**
`results/`、`web/data/runs.db`、`web-restart.log`、`.pytest_cache/`。如需把最新报告也带到 Windows，请单独拷贝对应 `results/<批次>/`。

**备份**：改动前的全部受影响文件已存到 `workspace/.backup-win-merge-<时间戳>/`。

## 四、验证证据

| 项 | 命令 / 方式 | 结果 |
|---|---|---|
| 语法编译 | `compileall common kvv dsv4 glm53 fable5 gemini_image gpt_image gpt56 volcperf web scripts tests` | 全绿 |
| 全量回归 | `TMPDIR=$PWD/.pytest-tmp .venv/bin/python -m pytest tests/ -q --basetemp=$PWD/.pytest-tmp/bt --ignore=vendor_verifier` | **470 passed, 6 subtests passed**（31.6s） |
| 跨平台用例 | `pytest tests/test_web.py -k "CrossPlatform or WindowsStartup" -v` | 6 passed |
| 启动冒烟 | `.venv/bin/python scripts/start_web.py --host 127.0.0.1 --port 8477 --no-browser` | 日志 `已加载配置：.env（1 项）` → `http://127.0.0.1:8477`；`/` **200** |
| 接口校验 | `GET /api/cases` | 69 张卡；fable **26** 张 = 5 变体 × 5 套件 + 1 常驻探针（`always_visible=True`）；变体集 = `{fable5, haiku45, opus5, sonnet46, sonnet5}`，每变体 5 套件 |
| `start.bat` 形态 | 二进制检查 | CRLF 113 行 / **裸 LF 0** / **无 BOM** / 首行 `@echo off` / 含 `chcp 65001` |
| `start.bat` 启动行 | `start.bat:65` | `"%VENV_PY%" "%ROOT_DIR%scripts\start_web.py" --open-browser %*`——**不传 `--host`**，落到 `start_web.py` 默认 `127.0.0.1`（符合「Windows 默认回环、避免防火墙弹窗」设计） |
| `start.sh` | `bash -n start.sh` | 通过（LF） |

## 五、仍未验证 / 风险

1. **Windows 未真机验证**：`start.bat` 的批处理语法与 `web/server.py` / `kvv/runner.py` 里 `os.name == "nt"` 分支在 macOS 上跑不到。当前保障 = 静态审阅 + 打桩单测（`parse_windows_listeners`、`display_host`、`load_dotenv`、`_child_popen_kwargs`、`_portable_path`）＋ `CrossPlatformLaunchTests` 在 POSIX 侧对进程树终止的实测判别力。**首次在 Windows 上跑时请优先看 `start.bat` 的 `:probe` 解释器探测与 `.venv` 重建提示。**
2. **`vendor_verifier/.venv` 不可跨平台**：Windows 上须重建该 venv，或用 `KVV_PYTHON` 指向已装官方依赖的解释器；`KVV_HOME` 指定官方套件目录。
3. **`.env` 代理端口**：随仓拷到 Windows 后代理端口通常不同（Clash for Windows 常见 `7890`），照抄会全部连接失败——先改 `.env`，或把 `VOLC_TRUST_ENV` 改回 `false` 试直连。
4. **已知平台缺陷（本次未修，源仓同样存在）**：批量运行时若 `case_names` 非"仅探针一条"，`billing_probe` 会被置空，探针派发回落 `run_fable5.py` 报 `invalid choice`。

## 六、根因治理建议

本次之所以要做 hunk 级合并，是因为 **Windows 平台层只存在于 `_win`，源仓没有**。建议把平台层反向提交进源仓（`start.bat`、`.gitattributes`、`scripts/start_web.py`、`start.sh` 的平台分支、`kvv/runner.py:official_python()`、`web/server.py` 的平台助手），使 `_win` 退回「纯粹副本」——之后每次同步就是一次目录拷贝 + 可选重建 venv。本次未做（用户只要求单向同步）。

## 七、复现：合并脚本（本次用的一次性工具）

```python
#!/usr/bin/env python3
"""从完整 diff 中抽出「Windows 移植」相关的 hunk，重写成可直接 git apply 的补丁。

用法: build_win_patch.py <diff> <keep-hunks 逗号分隔 1-based> <相对路径> <输出>
"""
import sys

diff_path, keeps, rel, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
keep = {int(x) for x in keeps.split(",") if x.strip()}

lines = open(diff_path, encoding="utf-8").read().splitlines(keepends=True)

body_start = 0
for i, ln in enumerate(lines):
    if ln.startswith("@@"):
        body_start = i
        break

hunks, cur = [], None
for ln in lines[body_start:]:
    if ln.startswith("@@"):
        if cur is not None:
            hunks.append(cur)
        cur = [ln]
    else:
        if cur is None:
            continue
        cur.append(ln)
if cur is not None:
    hunks.append(cur)

selected = [h for idx, h in enumerate(hunks, 1) if idx in keep]
missing = keep - set(range(1, len(hunks) + 1))
if missing:
    sys.exit(f"hunk 序号超出范围: {sorted(missing)} (共 {len(hunks)} 个)")

with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(f"--- a/{rel}\n+++ b/{rel}\n")
    for h in selected:
        fh.writelines(h)

print(f"{rel}: 选中 {len(selected)}/{len(hunks)} 个 hunk -> {out_path}")
```

复现步骤（在 `_win` 内执行）：

```bash
diff -u ../ml-test-platform/web/server.py web/server.py > /tmp/d_server.diff
diff -u ../ml-test-platform/tests/test_web.py tests/test_web.py > /tmp/d_webtest.diff
python3 build_win_patch.py /tmp/d_server.diff 4,5,6,7,8,9,10,11 web/server.py /tmp/win_server.patch
python3 build_win_patch.py /tmp/d_webtest.diff 1,5 tests/test_web.py /tmp/win_webtest.patch
cp ../ml-test-platform/web/server.py web/server.py
cp ../ml-test-platform/tests/test_web.py tests/test_web.py
git apply -p1 /tmp/win_server.patch && git apply -p1 /tmp/win_webtest.patch
```

> 注意：hunk 序号会随两仓后续演进变化，每次都要重新 `diff` 并按内容（而非序号）确认哪些 hunk 属于 Windows。
