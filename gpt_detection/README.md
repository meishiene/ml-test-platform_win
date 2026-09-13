# GPT 检测测试项目

当前仅包含一个测试子项：`GPT_MODEL_MATCH`。

该子项向 OpenAI/火山云 OpenAI 兼容的 Chat Completions 接口发送一次非流式请求，比较请求中的 `model` 与响应顶层 `model`：

- 完全一致：测试通过；
- 响应模型不同：测试失败，视为疑似模型路由/降级；
- 响应缺少 `model`、响应不是 JSON 或请求异常：无法通过，结果会保留原始证据。

## 命令行

```bash
GPT_BASE_URL=https://api.openai.com/v1 \
GPT_API_KEY="$OPENAI_API_KEY" \
GPT_MODEL=gpt-5.6-sol \
GPT_PROTOCOL=openai \
.venv/bin/python scripts/run_gpt_detection.py
```

也可以直接传参：

```bash
.venv/bin/python scripts/run_gpt_detection.py GPT_MODEL_MATCH \
  --base-url https://api.openai.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --model gpt-5.6-sol \
  --protocol openai
```

结果目录会生成在 `results/GPT_MODEL_MATCH-*`，包含 `summary.json`、`requests.jsonl`、`metadata.json` 和 `report.md`。请求明细会去掉 Authorization，不会保存 API Key。

该测试只验证接口返回的模型字段，不能单独证明网络一定是官方直连；网关可以改写响应字段。
