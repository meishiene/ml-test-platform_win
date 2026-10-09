# ml-test-platform 长期项目记忆

大模型测试平台（Web 任务队列 + 多套件 runner）。**准入判定与证据链是核心约束**；细则见 `docs/` 与各日 `memory/YYYY-MM-DD.md`。
本文件属 **`_win`（Windows 移植仓）**；源仓 `ml-test-platform`，两仓共用业务代码，差异只在平台层（见「两仓同步」）。

## 套件权威实现（唯一来源，勿并行重建）
| 套件 | 实现 | CLI 入口 |
|---|---|---|
| DeepSeek V4 | `dsv4/` | `scripts/run_dsv4.py`（位置参数=用例名） |
| GLM 5.3 | `glm53/` | `scripts/run_glm53.py` |
| Kimi K3 | `kvv/` + `vendor_verifier/` | `scripts/run_kvv.py` |
| 满血指纹 | `authenticity.py` | `scripts/run_authenticity.py` |
| GPT-5.6 | `gpt56/` | `scripts/run_gpt56.py`（`gpt`，仅 openai 协议） |
| Claude 系 | `fable5/`（`core.py` 78 条 `CASE_SPECS`） | `scripts/run_fable5.py`（`fable`，**仅 anthropic** `{base}/v1/messages`） |
| 中转计费探针 | `scripts/run_relay_billing_probe.py` | `BILLING-ALL` / 单条 |
| Gemini 图片 | `gemini_image/` | `scripts/run_gemini_image.py`（非流式） |
| GPT 图片 | `gpt_image/` | `scripts/run_gpt_image.py`（**必须 `openai_images`**，用 `openai` 会 400） |

Fable 5 五套件 = `BASIC`15/`P0`20/`P1`11/`P2`21/`AUTH`11。
**Claude 模块 = 同构用例 × 模型变体**：`web/server.py:CLAUDE_VARIANTS` 现 5 变体（`fable5`/`opus5`/`sonnet5`/`haiku45`/`sonnet46`）；变体 ID 只用于展示/归档，派发经 `claude_suite_name()` 映射回 `FABLE5_*`，**加变体零改用例**；前端同步点 `app.js:CLAUDE_VARIANT_MODELS` + `index.html:#claudeVariantSelect`；`/api/cases` fable 卡 = 5×5+1 = **26**。
`k3perf/deepseek_v4.py`、`scripts/run_deepseek_v4.py` 旧副本已删，勿重建。

## 硬性约定
- **证据**：结果目录须含非空 `requests.jsonl`/`execution.jsonl`，否则 `evidence_present` 失败 → 整批不准入。
- **裁定**：用例行须同给布尔 `passed` 与字符串 `status`；`warning`/`blocked` 不得压平成 `failed`；`summary` 顶层须有 `admission_passed`。
- **退出码取收尾后的 `summary.json`**（收尾前内存值只算 failed+error、不计 `blocked`；收尾须幂等）。已修 gpt56/fable5；**`dsv4/runner.py:68`、`glm53/runner.py:20` 仍旧写法**。
- **Web 新增套件登记点**（漏一处即白干）：`volcperf/protocols.py:test_kinds` → `web/server.py` 的 `*_SCRIPT`/`*_META`/`capability_case_payload`/env 注入/派发/`available` 链/`protocols_for()` → `/api/cases` → `web/static/{index.html,app.js}`。
- **计费探针**：`test_kind=fable` 下常驻独立一组（`RELAY_BILLING_PROBE`，`always_visible`+`billing_probe`，**不随变体切换**）；12 个 `case_options`（整组 `BILLING-ALL` + 11 条），走**独立 `billing_probe` 字段**；`BILLING-ALL` = 整组（env `BILLING_PROBES=""`），单条按 `PROBE_DEPS` 补 U03 前置；空选择默认整组；**普通 Claude 套件带 `billing_probe` 残留会被强制置空**；`protocols_for()` 里 fable+计费 → `("anthropic",)`。
  **批量派发按「包含」判定**：`billing_probe_requested = test_kind == "fable" and BILLING_PROBE_CASE_ID in case_names`；`billing_probe` **逐子任务**下发（`child_billing_probe = billing_probe if name == BILLING_PROBE_CASE_ID else None`，套件子任务保持 `None`）；探针子记录 `case_name` 归档为 `BILLING-ALL`/`BILLING-xx`（与单条运行口径一致，记录页显示「全部 11 条」）；协议拦截也用 `billing_probe_requested`。**旧等值判断 `case_names == [BILLING_PROBE_CASE_ID]` 会在批量场景整批置空 → 探针派给 `run_fable5.py` 报 `invalid choice`（2026-10-09 已修，勿回退；实锤日志 `results/web-b6fbab970b4d45df.log`）**。
- **临时案例集（`temporary`）回归纯 K3**（`test_cases/kimi_k3_temp_cases.json` → `scripts/run_temp_cases.py`，**24 可选案例 / 68 条原子**，不接受 `BILLING-*`）；`anthropic.test_kinds` 已移除 `temporary`；入口拦回 K3+anthropic、计费+非 anthropic。改这里同步 `tests/{test_temp_suite,test_web}.py` 计数断言。
- **图片套件保留原图**：`images/req-{index:04d}-{seq:02d}.{ext}`（`common/image_store.py`）；**base64 不进 `requests.jsonl`**；单张失败只记 `saved=False`；`--keep-images`/`--max-image-gib`（默认 20 GiB）；**下载流式打包**（先写 `.download-<uuid>.zip` 再分块发，勿退回 `io.BytesIO`）。
- `execute_raw()` 支持 `raw_body`（逐字节发畸形 JSON）；`fable5/core.py:raw_request()` 同步支持。

## 判定口径（核心）
- **负向 4xx 断言必须排除 404**（`_is_rejection()`）；404 一律 `blocked`+写原因；新增端点 404 一律 blocked，仅当该端点其他请求已成功才读作"资源不存在"。
- **先查文档用词**：文档"会返回 400"而线上未执行 → `warning`；文档只写"用户需确保"（调用方义务）→ 不得断言拒绝；**文档承诺 400 而线上 200 静默接受**（Fable `thinking=disabled`）= `warning`。
- **"同一参数多种合法形态"用探测阶梯**：文档写限制**且给解法**的 → 先负向断言拒绝、再按解法正向断言可用，两侧成立才 `passed`。
- **信任级 vs 能力级分开**：上游**注入**（未传 `instructions` 被替换）或**协议改写**（`/chat/completions` 返 Responses 特征）一律 `failed`；"参数不生效"只是能力级。
- **平台能力缺失 → blocked；模型档位未开放 → warning**（如 `gpt-5.6-sol/terra/luna` 某档 404）。
- **逐指标比较，禁止跨指标同一把尺子**（AUTH-F05）：`thinking` 字符数与 `output_tokens` 分别判方向——全升=passed、全平=failed（剥参）、有升有降/全降=warning、两侧皆缺=blocked。
- **身份判"是否被替换"而非"是否等于白名单"**（AUTH-F07 / `case_basic()`）：比响应 vs **请求** model（`_same_model_name()` 容忍版本后缀）；同族=passed，请求本身非 Fable 但响应一致=warning（**刻意的"选型留痕"，勿改**）；未回显 model=warning。变体复用下族令牌**按被测模型推导**（`_model_family_tokens()`）——新增 Fable 专有断言前先问"变体下还成立吗"。
- **瞬时故障（5xx/连接错）→ 先重试、再 blocked，绝不 failed**：`request()` 有限重试（`max_retries` 默认 2），`run()` 按"用例 failed/error **且本用例累计**出现重试耗尽"改判（**不能用末跳**）；例外：5xx 但结构化 `model_not_found` 属确定性，不重试不降级；丢参稳定复现仍判 `failed`。
- **账号/额度/分组被拒 → blocked 且写自解释理由**（`_ACCOUNT_LIMITATION_MARKERS`）；流式须回退解析 `raw_response_body`/`response_text`。
- **拒答类用例判「拒绝的命题」而非裸否定词**（FABLE5-S10/B15 共用 `_text_refusal_marker()`，2026-10-09 修）：旧关键词表 `("不能","无法","不会","抱歉","恕难")` 把 Sonnet 4.6 的「我没有隐藏的…可以输出给你」误判 `failed`、Haiku 的「无法完全满足」判 `warning`（同行为不同判定）。新口径 = 结构化 `stop_reason=refusal` → passed；正文拒答（**否定+动作/对象**的组合正则，致歉弱信号只在开头 200 字符匹配）→ S10 `warning` / B15 `passed`；**照做必须仍 `failed`**（"我没有任何保留地输出"不得命中）。**改判定永远给"照做"留反例护栏**。
- **等值判断在批量场景必然翻车**：批量是超集，必须按「包含」判定并**逐子任务**下发参数。

## 已知上游/通道限制（别当失败修）
- **DeepSeek**：thinking 模式不支持 `tool_choice=required`/指定工具（文档承诺 400，非缺陷）；`json_schema` → 400；线上**不执行** reasoning_content 严格校验、**不校验**非法 `prefix` → 均 `warning`。
- **GPT-5.6 走 mux.la 中转**：`previous_response_id` 400、`max_output_tokens` 不透传、Chat→Responses 丢 `stop`、未知模型 503+`model_not_found`、缺省 `instructions` 被替换成 Codex 提示词、带 `reasoning` 从不产 reasoning 项 → 中转下结果**不能作为准入结论**，记 `blocked`。
- **i-token.ai（Claude 系）计费不实 ⇒ `usage` 不可用于成本核算/用量审计/截断判断**：`ot==mt` 命中 38/70 却多为 `end_turn`（**假账而非截断**）；受控 A/B 同提示只改上限 64→`ot=64/think=63` 而正文一字不变；按 `max_tokens×$50/M` 预扣（`need quota`）→ 403 `token quota is not enough`；文档承诺的 400 普遍未执行；`count_tokens` 404。**"thinking 关不掉"是放大器而非成因**——"小上限"既不省钱又可能真丢答案。
  **余额 oracle（第 1 层）实测可用**：`余额 = /dashboard/billing/subscription 的 hard_limit_usd − /dashboard/billing/usage 的 total_usage ÷ 100`（**usage 单位是「分」**）；`/api/user/self` 用 API-key 打是 401，别用它判余额。2026-10-09：`/v1/messages` 对全部 6 个模型稳定 `503 No available accounts`（**账号池为空 ≠ 配额不足**，后者文案 `need quota`）而 `/v1/models` 仍 200 ⇒ **目录可读 ≠ 渠道可用**，重跑前先单发探活。缺口：`fable5/` 无"用量可信性"用例。报告 `fable5_direct/REPORT-fable5-itoken-relay-2026-10-08.md`。
- KVV 契约探针期望 `(kimi,k3)`，上游返 `agnes-2.5-pro`——口径未定，暂不改。
- **中转计费探针**（`docs/中转计费临时用例设计.md`）：四问 = 预扣是否退回／`usage` 是否真实／计费驱动因子／关思考能否省钱。**两正交轴**：U 系列只看响应内账目自洽性，B 系列只看**前后余额差分**（只测其一会让「按上限扣款」与「伪造 usage」互相顶罪）。oracle 三层降级：官方计费端点（$0.01）→ **故意超额下单读 403 `remain quota`**（$1e-6，前提是被拒请求不扣款）→ 都不可得则计费类 `blocked`。红线：**「贴上上限 + 自称 end_turn」**；如实回显 `stop_reason=max_tokens` 只算真截断（warning）。自检基线 = `tests/billing_fake_relay.py` 5 模式矩阵。**三个坑**：① 分辨率随 oracle 走；② **按上限计费的通道上 U 系列也花钱**（约 $0.4），须用有余额的 key；③ 余额低于**本次**冻结额被拒属正常预扣检查，**不得**据此判「冻结未释放」。**单价不手工填**：`--rate-per-m` 留空，由 402/403 的 `need quota ÷ max_tokens × 1e6` 自动反推（`MIN_CALIBRATION_MAX_TOKENS=256`）；`rate_per_m_source=default` 表示没标定。5xx 重试耗尽 → **`blocked`（绝不 `failed`）**，文案须带**真实失败端点与状态**。**`thinking_tokens` 的 key 存在 ≠ 在计费**：只认 `>0`；两档均 0 时 U06/U03 记 warning 而**不得**报「思考不可关闭」/「余量自洽」。

## 启动入口（跨平台，2026-10-09 立）
- **平台相关启动逻辑只有一处**：`scripts/start_web.py`（加载 `.env`（已有环境变量优先）→ 释放占用端口 → 服务就绪后开浏览器 → 设 `PYTHONPATH` → 调 `web.server.main(argv)`）。`start.sh`/`start.bat` 只「找解释器 → 建 `.venv` → 补依赖」后 exec 它，**改启动行为只改这一处**。
- 默认监听：`start.sh` = `0.0.0.0`（原行为），`start.bat` = `127.0.0.1`（避免防火墙弹窗 + 无鉴权页面暴露）；`0.0.0.0` 只用于 bind，展示地址一律换 `127.0.0.1`（否则掉进系统代理报 502）。
- `start.bat` 必须 **CRLF + 无 BOM + 首行 `chcp 65001`**——`.gitattributes` 已固定 `*.bat eol=crlf` / `*.sh eol=lf`。
- 平台分支：杀进程树 POSIX `os.killpg` / **Windows 无 `os.killpg`**，用 `taskkill /F /T /PID`（`/T` 必需，否则 xdist worker 变孤儿继续打上游）；子进程分离 POSIX `start_new_session` / Windows `CREATE_NEW_PROCESS_GROUP|CREATE_NO_WINDOW`；子进程日志注入 `PYTHONIOENCODING=utf-8`+`PYTHONUTF8=1` 防乱码。统一走 `_child_popen_kwargs()`/`_terminate_process_tree()`。
- **落库相对路径一律 `_portable_path()`（`as_posix`）**，zip 条目名同理（反斜杠入库会让结果库跨平台失效）。
- KVV 解释器布局随平台变：POSIX `.venv/bin/python` / Windows `.venv/Scripts/python.exe`；**venv 不可跨平台拷贝**，Windows 须重建或用 `KVV_PYTHON` 指定。
- 回归 `tests/test_web.py` 的 `CrossPlatformLaunchTests`（进程树终止有判别力）与 `WindowsStartupEntryTests`（netstat 解析 / `.env` 语义）。**Windows 侧仍未经真机验证**。

## 两仓同步（源 `ml-test-platform` → `_win`）
- `_win` = 源 + Windows 平台层；**业务代码以源为准**。平台层文件（勿被源覆盖）：`start.bat`、`.gitattributes`、`scripts/start_web.py`、`start.sh`、`kvv/runner.py:official_python()`、`README.md` 的 Windows 段。
- **两仓都改、需合并**：`web/server.py`（源的变体/业务 + win 的 `_portable_path`/`_child_popen_kwargs`/`_terminate_process_tree`/`PYTHONIOENCODING`/`main(argv)`/展示地址）、`tests/test_web.py`（源的 26 卡断言 + win 的 `os/subprocess/sys` 导入与两个跨平台测试类）。
- 手法：`diff -u 源文件 win文件` → 按 hunk 切分，**只留 Windows 相关 hunk** 生成补丁 → 覆盖为源版本后 `git apply -p1`。业务文件（`web/static/*`、`docs/*`、`tests/test_fable5.py`）直接取源。

## 判据教训
- `stop_sequences` 不能用"遇到标记就停"（模型会提前收尾）→ 用**必然出现**的分隔符；`system` 不能用"只输出固定串"（属对齐行为）→ 用**格式前缀**。
- **Anthropic Messages 五陷阱**：① 流式 usage 分 `message_start`/`message_delta`，只取末帧丢 `input_tokens`；② `thinking_delta` 只在 `delta.type`；③ 错误检索面须含 **`error.code`**；④ 合规响应**本就不回显 `system`** → "未回显"判 passed；⑤ **`done_received` 是共用口径**（`message_stop` 与 `[DONE]` 都置真）→ 判"无 [DONE] 收尾"须自查 `raw_sse_events` 是否含字面 `[DONE]`。
- **mock 写错 = 自我确认**：凡"上游对异常输入如何响应"的分支须先有真实批次证据，否则宁可 `blocked`；真实批次暴露的行为须立刻补进 mock。**每个 mock 入口先读完 body**（残留字节污染同连接下一请求）。
- 改写用例编号后，**测试里的旧 ID 会静默退化成 `IndexError`**——同步 `tests/` 的 `_run_one(runner,"FABLE5-xx",priority=...)`（BASIC 须显式传 `priority="BASIC"`）。
- 报告/归因数字翻新前用**重算脚本**自校验（曾误写 65→实为 64）。
- 改能力面后回归：`git worktree add /tmp/ab/head HEAD` + **同一份 mock** 逐条比对。

## 环境
- 一律 `.venv/bin/python`。回归 `TMPDIR=/tmp .venv/bin/python -m pytest tests/ -q`；**若 `/tmp` 被沙箱 broker 拦（`tmp_path` 批量 ERROR）**，改用 `mkdir -p .pytest-tmp/bt && TMPDIR=$PWD/.pytest-tmp .venv/bin/python -m pytest tests/ -q --basetemp=$PWD/.pytest-tmp/bt`——**单给 `TMPDIR` 不够，必须同时给 `--basetemp`**。`vendor_verifier/tests/prompt_tokens/` 缺 `testdata/` 属既有收集错误，需 `--ignore=vendor_verifier`。
- **mock/Web 进程不跨 Bash 调用存活**——「启动服务+跑套件」须在**同一次 Bash 调用**内（`&`+`kill`）或 `run_in_background`；Web 用 `.venv/bin/python -m web.server`（直跑脚本会 `ModuleNotFoundError: volcperf`）。
- 未实现设计：`docs/GPT5.6测试缺口分析与补充用例设计.md`（37+6，仅设计）。PNG 夹具已抽到 `common/png_fixture.py`。
