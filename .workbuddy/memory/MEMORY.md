# ml-test-platform 长期项目记忆

> 项目：大模型测试平台（Web 任务队列 + 多套件 runner）。准入判定与证据链是核心约束。

## 套件权威实现（唯一来源，勿并行重建）

| 模型/套件 | 权威实现 | CLI 入口 |
|---|---|---|
| DeepSeek V4 能力专项 | `dsv4/core.py`（`MODELS`/`CASE_SPECS`/handler）+ `dsv4/cases.py`（元数据）+ `dsv4/runner.py`（装配） | `scripts/run_dsv4.py`（位置参数传用例名） |
| GLM 5.3 | `glm53/` | `scripts/run_glm53.py` |
| Kimi K3 | `kvv/` + 官方 `vendor_verifier/` | `scripts/run_kvv.py` |
| 满血指纹 | `authenticity.py` | `scripts/run_authenticity.py` |
| GPT-5.6 | `gpt56/`（cases/core/runner） | `scripts/run_gpt56.py`（test_kind=`gpt`，协议仅 openai，effort 六档 none→max，Responses API 主推） |

- `k3perf/deepseek_v4.py` 与 `scripts/run_deepseek_v4.py` 是 init 时代旧副本，**2026-09-24 已按用户指示删除**，不要再重建同类并行实现（旧副本缺 DS-V4-18 与整个 P2 段，且只接受 `--priority/--case`）。
- 教训：同一个模型出现两套 runner 时，Web 容易长期指向过时那一套，症状是 `unrecognized arguments` 或用例数对不上。

## 硬性约定

- **执行证据**：结果目录必须含非空 `requests.jsonl`/`execution.jsonl`，否则 `validation.json` 的 `evidence_present` 失败 → 整批不准入（比单用例失败更严重）。新套件要自己写证据。
- **裁定语义**：用例行必须同时给布尔 `passed` 与字符串 `status`；`warning`/`blocked` 不得被压平成 `failed`（`common/result_adapter.py` 按 `VERDICT_STATUSES` 白名单识别）。
- **准入字段**：`summary` 顶层要有 `admission_passed`，否则适配层判 `not_evaluable`。
- **模型身份校验**：令牌白名单放 `common/data/<family>.json` 的 `model_tokens`，避免上游别名口径差异（如 `deepseek-flash` vs `deepseek-v4-flash`）被误判为身份不符。
- 重复收尾（套件 `finalize()` + Web `finalize_result_dir()`）必须幂等，`source_schema_version` 首次记录优先。
- **端点地址派生**：`volcperf/protocols.py:api_root()` 会把 base_url 规范化成 `.../v1`，但 DeepSeek 的 beta / anthropic 端点挂在**根域**下——beta = `https://api.deepseek.com/beta`（chat prefix、FIM、strict tool），anthropic = `https://api.deepseek.com/anthropic`（端点 `{base}/v1/messages`）。派生这类端点前必须剥掉版本后缀（`dsv4/core.py` 的 `_api_parent()`），否则会得到 `.../v1/beta/chat/completions` 这类不存在的路径（线上 404，响应体为空、带 `x-ds-trace-id`）。
- **负向用例的 4xx 断言必须排除 404**：`400 <= status < 500` 会把"端点不存在"误判成"参数被正确拒绝"（DS-V4-20/22 曾因此假通过）。统一用 `dsv4/core.py` 的 `_is_rejection()`；遇 404 一律按 `blocked` 收口并写清原因（`_endpoint_missing_reason()`）。2026-09-24 已覆盖 beta 域（20/22）与 openai 域（17/30/31）。
- **负向用例的判定口径必须先查文档用词**：文档写"API 会返回 400"（承诺报错）→ 线上没执行记 `warning`；文档只写"用户需确保 / users must ensure"（调用方义务）→ **不得断言拒绝**，线上宽松容忍也记 `warning`，硬判 failed 等于自造契约。定口径前先拉 `api-docs.deepseek.com` 原文核对。
- **模拟上游（mock）不得编码"想当然的上游行为"**：mock 是验证工具，写错就等于自我确认。凡"上游对某种异常输入如何响应"的分支，必须先有真实批次证据（`results/<批次>/requests.jsonl`），否则该用例在 mock 下宁可 blocked。2026-09-24 踩过两次：mock 里假设"非法放置的 prefix → 400"，导致 DS-V4-22 端到端一路显绿，而线上实为 200；mock 里"只要有 function 工具就返回 function_call"，导致 DS-V4-61 端到端显绿，而线上 `tool_choice=required` 是 400。**反向同样成立**：真实批次暴露的行为必须立刻补进 mock。
- **"同一参数有多种合法形态"时用探测阶梯，并在两侧都做断言**：文档写了限制**且给出解法**的（典型：`tool_choice=required` 在思考模式下 400，文档说"请先关闭思考模式"），用例应把限制当**负向断言**验证拒绝本身，再按文档解法补**正向断言**验证关思考后能力可用——两侧都成立即 `passed`，不记 warning。**解法因协议面而异**（Chat 用 `thinking.disabled`，Responses 用 `reasoning.effort=none`）。写死单一形态、或把"文档已给解法的限制"当成缺陷，都会误报（DS-V4-61，2026-09-24）。
- **Responses 端点的 output 断言要给足 `max_output_tokens`**：思考模式的推理内容也占用 output 额度（DS-V4-34 用 32 时 output 只剩一个 `reasoning` 项、`status=incomplete`），额度太小会把"被截断"误判成"能力缺失"。DS-V4-61 已提到 1024。
- **"信任级失败"与"能力级失败"要分开**（2026-09-24 立）：上游**注入**（未传 `instructions` 却被替换成系统提示词）或**协议改写**（`/chat/completions` 返回 Responses 特征：`object=="response"`、`id` 以 `resp_` 开头、`usage.billing_usage.source=="oai_responses"`）一律记 `failed`——此时响应内容不由本次请求单独决定，整套指纹结论不可信；而"某个参数不生效"只是能力级问题。非标准字段（`billing_usage`/`access_programs`/`claude_cache_creation_*`）同属信任级。
- **平台能力缺失 → blocked；模型档位未开放 → warning**（2026-09-24 立）：内置工具、`background`、`conversation`、`multi_agent`、显式缓存、`compaction`、`service_tier=ultrafast` 被 4xx 拒绝 → `blocked`（平台前提不满足）；而 `gpt-5.6-sol/terra/luna` 某档 404 → `warning`（档位属**模型面**，且 404 在此语义为"别名不存在"，对齐 DS-V4-37 的 404 语义三分法）。
- **新增端点 404 一律 blocked**：仅当该端点上其他请求已成功，才把 404 解释为"资源/模型不存在"。
- **退出码必须取收尾后的 `summary.json`**：套件 `main_async` 结尾不能用收尾**前**内存里的 `admission_passed`——那个值来自 `all_passed`（只算 failed+error），`blocked` 不计入；收尾的统一校验（`case_verdicts_complete`）才会把 blocked 算作阻断。否则"整批只有 blocked、没有 failed"的批次 `exit_code=0`，而磁盘 summary / Web 显示不准入（CLI/CI 误判）。gpt56 已修（`gpt56/runner.py` 的 `read_finalized_summary()`，2026-09-24）；**`dsv4/runner.py:68`、`glm53/runner.py:20` 仍是旧写法**。Web 侧不受影响（`web/server.py` 只读磁盘 summary 并会再跑一次 `finalize_result_dir`）。
- **瞬时故障（5xx / 连接错误）→ 先重试、再 blocked，绝不记 failed**（2026-09-24 立）：网关/网络抖动不由被测模型决定，记 failed 会污染"模型能力"结论。GPT-5.6 套件已在唯一的请求出口 `request()` 做有限重试（`GPT56Config.max_retries` 默认 2 次 + 指数退避，`GPT_MAX_RETRIES=0` 可关），并在 `run()` 里按"用例判 failed/error **且本用例累计出现过重试耗尽的瞬时故障**"统一改判 `blocked`（`_transient_reason()` 写自解释理由）。**必须用"本用例累计"而非"末跳"判断**——多跳用例（如 `case_effort_levels`）故障发生在中途时末跳已恢复 200。例外：HTTP 5xx 但响应体是结构化 `model_not_found` 属确定性结论，**既不重试也不降级**（GPT56-23 依赖该响应本身）。识别特征：`requests.jsonl` 响应头带 `x-new-api-version`、响应体 `{"code":"system_memory_overloaded","current":91.5%,"threshold":90%","type":"new_api_error"}`（mux.la 网关自身故障，实测于 `GPT56_P0-20260924-153058`）。**与"网关丢参"分清**：丢参（未透传 `max_output_tokens`、Chat→Responses 翻译丢 `stop`）稳定复现 → 仍判 `failed`。
- **新增能力面/新用例后必须回归既有 39 条零漂移**：用 `git worktree add /tmp/ab_gpt56/head HEAD` 取改动前代码 + **同一份 mock** 跑同一套件逐条比对。注意改动 mock 后要重跑基线，否则 mock 自身噪声会被误记成代码漂移。
- GPT-5.6 缺口盘点与 37+6 条补充用例设计见 `docs/GPT5.6测试缺口分析与补充用例设计.md`（**仅设计，尚未实现**）。要点：三档模型未分别验证、多模态零覆盖、Responses 参数面（verbosity/truncation/parallel_tool_calls/max_tool_calls/metadata/service_tier/include/prompt_cache_key）几乎全零、`common/model_contracts.py` 的 PROFILES **缺 gpt 家族**、归因出的中转特征未固化成指纹用例。复用点：`dsv4/core.py:233-251` 的确定性 PNG 生成器应抽到 `common/png_fixture.py` 共用；`execute_raw` 已支持 GET/DELETE；`scripts/run_contracts.py` 现固定打 `/chat/completions`，gpt 家族需按 family 走 Responses 面。

## 已知上游真实限制（非平台 bug，复核时别当失败修）

- **GPT-5.6 批次走的是中转网关**（`https://mux.la/v1`，New API，group `muxla_company`，Codex/ChatGPT 账号通道）。中转硬限制：`previous_response_id` 返回 400 `requires an OpenAI API-key account for HTTP requests`（Responses 多轮链不可用）；`max_output_tokens` 不被透传（要 16 给 207）；Chat→Responses 翻译（`usage.billing_usage.source=oai_responses`）导致 `stop` 等 Chat 专有参数丢失；未知模型返回 503 + `model_not_found`（非 4xx）；缺省 `instructions` 被替换成 Codex 系统提示词（prompt 膨胀 4400+，其中 3840 自带缓存命中）。带 `reasoning` 参数的请求从不产出 reasoning 输出项（疑似剥参），不带参数的偶尔产出。
  → **结论**：中转下的 GPT-5.6 结果不能作为模型准入结论；判定这类平台级限制应记 `blocked` 而非 `failed`。

- DeepSeek thinking 模式**不支持** `tool_choice=required` 与「指定工具」→ 400 `Thinking mode does not support this tool_choice`。**这是文档承诺的负向行为，不是缺陷**（《Chat Completions API》tool_choice 段：「思考模式下不支持 required 和指定具体 tool 的用法，API 会返回 400 错误。请先关闭思考模式。」）。关思考方式**因协议面而异**：Chat/beta 用 `thinking={"type":"disabled"}`（DS-V4-33 判 passed）；**Responses 用 `reasoning={"effort":"none"}`**（《Thinking Mode》控制参数表，思考默认开启档位 high）。2026-09-24 真实批次 DS_V4_P2-20260924-151006 实测 `required + reasoning.effort=none` → 200 且 `output=['function_call']`。**不要因为这条限制本身给用例记 warning 或 failed**——DS-V4-61 曾误记 warning，当天已纠正。
- DeepSeek：`response_format.type=json_schema` 返回 400 `This response type is unavailable now`。
- KVV 契约探针期望令牌 `(kimi, k3)`，实际上游返回 `agnes-2.5-pro` —— 口径未定，暂不改。
- **DeepSeek 线上不执行文档承诺的 reasoning_content 严格校验**：官方文档（Thinking Mode → Tool Calls）称"携带 tools 的请求，后续未正确回传 reasoning_content 会返回 400"，但线上（`deepseek-flash`，2026-09 实测）返回 **200 并正常作答**。判定口径：这种情况记 `warning`（文档/线上行为差异），**不判模型失败**——先例是 DS-V4-10，DS-V4-09 于 2026-09-24 按同一口径改判。
- **DeepSeek 线上不校验非法放置的 `prefix`**：官方《Chat Prefix Completion (Beta)》只写"用户需确保最后一条消息 role=assistant 且 prefix=True"（调用方义务，未承诺报错）；线上对放在 user 消息上的 `prefix=True` 返回 **200 并静默忽略该参数**。判定口径：`warning`（2026-09-24 DS-V4-22 由 failed 改判，标题改为"Chat Prefix 非法放置的处理行为"）。

## 环境

- 一律用项目虚拟环境：`.venv/bin/python`。回归命令：`.venv/bin/python -m pytest tests/ -q`。
- 沙箱内 `test_gpt_image` 的 1 个 error 是 pytest tmpdir `PermissionError`，与代码无关。
- 验证新链路用本地模拟上游（`/tmp/dsv4_mock_upstream.py` 模式），不要打真实接口。
- 该 mock 现已**按路径严格路由**（`/v1/chat/completions`、`/beta/chat/completions`、`/beta/completions`、`/anthropic/v1/messages`、`/v1/responses`，其余一律 404），以对齐真实上游。改 mock 时不要退回 `endswith()` 模糊匹配，否则地址派生类缺陷复现不出来。
