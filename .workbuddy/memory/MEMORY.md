# ml-test-platform 长期项目记忆

> 项目：大模型测试平台（Web 任务队列 + 多套件 runner）。准入判定与证据链是核心约束。

## 套件权威实现（唯一来源，勿并行重建）

| 模型/套件 | 权威实现 | CLI 入口 |
|---|---|---|
| DeepSeek V4 能力专项 | `dsv4/core.py`（`MODELS`/`CASE_SPECS`/handler）+ `dsv4/cases.py`（元数据）+ `dsv4/runner.py`（装配） | `scripts/run_dsv4.py`（位置参数传用例名） |
| GLM 5.3 | `glm53/` | `scripts/run_glm53.py` |
| Kimi K3 | `kvv/` + 官方 `vendor_verifier/` | `scripts/run_kvv.py` |
| 满血指纹 | `authenticity.py` | `scripts/run_authenticity.py` |

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
- **模拟上游（mock）不得编码"想当然的上游行为"**：mock 是验证工具，写错就等于自我确认。凡"上游对某种异常输入如何响应"的分支，必须先有真实批次证据（`results/<批次>/requests.jsonl`），否则该用例在 mock 下宁可 blocked。2026-09-24 踩过：mock 里假设"非法放置的 prefix → 400"，导致 DS-V4-22 端到端一路显绿，而线上实为 200。

## 已知上游真实限制（非平台 bug，复核时别当失败修）

- DeepSeek thinking 模式：`tool_choice=required` 返回 400 `Thinking mode does not support this tool_choice`。
- DeepSeek：`response_format.type=json_schema` 返回 400 `This response type is unavailable now`。
- KVV 契约探针期望令牌 `(kimi, k3)`，实际上游返回 `agnes-2.5-pro` —— 口径未定，暂不改。
- **DeepSeek 线上不执行文档承诺的 reasoning_content 严格校验**：官方文档（Thinking Mode → Tool Calls）称"携带 tools 的请求，后续未正确回传 reasoning_content 会返回 400"，但线上（`deepseek-flash`，2026-09 实测）返回 **200 并正常作答**。判定口径：这种情况记 `warning`（文档/线上行为差异），**不判模型失败**——先例是 DS-V4-10，DS-V4-09 于 2026-09-24 按同一口径改判。
- **DeepSeek 线上不校验非法放置的 `prefix`**：官方《Chat Prefix Completion (Beta)》只写"用户需确保最后一条消息 role=assistant 且 prefix=True"（调用方义务，未承诺报错）；线上对放在 user 消息上的 `prefix=True` 返回 **200 并静默忽略该参数**。判定口径：`warning`（2026-09-24 DS-V4-22 由 failed 改判，标题改为"Chat Prefix 非法放置的处理行为"）。

## 环境

- 一律用项目虚拟环境：`.venv/bin/python`。回归命令：`.venv/bin/python -m pytest tests/ -q`。
- 沙箱内 `test_gpt_image` 的 1 个 error 是 pytest tmpdir `PermissionError`，与代码无关。
- 验证新链路用本地模拟上游（`/tmp/dsv4_mock_upstream.py` 模式），不要打真实接口。
- 该 mock 现已**按路径严格路由**（`/v1/chat/completions`、`/beta/chat/completions`、`/beta/completions`、`/anthropic/v1/messages`、`/v1/responses`，其余一律 404），以对齐真实上游。改 mock 时不要退回 `endswith()` 模糊匹配，否则地址派生类缺陷复现不出来。
