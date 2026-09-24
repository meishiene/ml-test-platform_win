import json
import unittest

import httpx

from gpt56.cases import CASES
from gpt56.core import (
    CASE_SPECS,
    GPT56Config,
    GPT56Runner,
    MODEL_TOKENS,
    _cache_write_tokens,
    _cached_input_tokens,
    _output_text,
    _reasoning_tokens,
    summarize_results,
)
from gpt56.runner import config_from_env


class GPT56StaticTests(unittest.TestCase):
    def test_ui_suites(self):
        self.assertEqual(set(CASES), {"GPT56_P0", "GPT56_P1", "GPT56_P2", "GPT56_FINGERPRINT"})
        self.assertEqual(CASES["GPT56_P0"].request_count, 16)
        self.assertEqual(CASES["GPT56_P1"].request_count, 7)
        self.assertEqual(CASES["GPT56_P2"].request_count, 10)
        self.assertEqual(CASES["GPT56_FINGERPRINT"].request_count, 6)
        self.assertEqual({spec.priority for spec in CASE_SPECS}, {"P0", "P1", "P2", "AUTH"})

    def test_config_urls(self):
        # GPT-5.6 无 beta/anthropic 特殊端点，全部挂在 openai 家族根（.../v1）下。
        runner = GPT56Runner(GPT56Config("key", "https://api.openai.com/v1"))
        self.assertEqual(runner._url("chat/completions"), "https://api.openai.com/v1/chat/completions")
        self.assertEqual(runner._url("responses"), "https://api.openai.com/v1/responses")

    def test_config_from_env_normalizes_host_only_base_url(self, ):
        import os
        env = {"GPT_API_KEY": "key", "GPT_BASE_URL": "https://gw.example.com", "GPT_MODEL": "gpt-5.6"}
        original = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            config = config_from_env()
            self.assertEqual(config.base_url, "https://gw.example.com/v1")
        finally:
            for k, v in original.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_model_tokens_cover_alias_and_tiers(self):
        for token in ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            self.assertIn(token, MODEL_TOKENS)

    def test_extractors(self):
        body = {"output": [
            {"type": "reasoning", "summary": [{"text": "s"}], "encrypted_content": "enc"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "答案"}]},
        ], "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "output_tokens_details": {"reasoning_tokens": 7}, "input_tokens_details": {"cached_tokens": 3}}}
        self.assertEqual(_output_text(body), "答案")
        self.assertEqual(_reasoning_tokens(body), 7)
        self.assertEqual(_cached_input_tokens(body), 3)
        self.assertEqual(_cache_write_tokens({"usage": {"input_tokens_details": {"cache_write_tokens": 9}}}), 9)
        self.assertEqual(_cache_write_tokens({"usage": {}}), 0)

    def test_summary_shape(self):
        from gpt56.core import CaseResult
        summary = summarize_results([CaseResult("1", "a", "m", True)])
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["results"][0]["case_id"], "1")


class GPT56OutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def _runner(self, responses):
        queue = list(responses)
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            status, body = queue.pop(0)
            return httpx.Response(status, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return GPT56Runner(GPT56Config("key"), client), client, requests

    @staticmethod
    def _normalize(outcome):
        """handler 直接调用返回 3/4 元组；统一成 (passed, details, observations, status)。"""
        if len(outcome) == 4:
            return outcome
        passed, details, observations = outcome
        return passed, details, observations, ("passed" if passed else "failed")

    async def _one(self, responses, handler_name, *args):
        runner, client, requests = await self._runner(responses)
        try:
            return self._normalize(await getattr(runner, handler_name)(*args)), requests
        finally:
            await client.aclose()

    async def test_basic_identity_token_mismatch_fails(self):
        ok = {"model": "gpt-5.6-sol", "choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        ghost = {"model": "gpt-4o-mini", "choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        outcome, _ = await self._one([(200, ok)], "case_basic", "gpt-5.6")
        self.assertTrue(outcome[0])
        outcome, _ = await self._one([(200, ghost)], "case_basic", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_default_effort_missing_field_is_warning(self):
        runner, client, _ = await self._runner([(200, {"output": [{"type": "message", "content": [{"type": "output_text", "text": "391"}]}]})])
        try:
            passed, _, _, status = await runner.case_default_effort("gpt-5.6")
        finally:
            await client.aclose()
        self.assertTrue(passed)
        self.assertEqual(status, "warning")

    async def test_default_effort_medium_passes(self):
        body = {"reasoning": {"effort": "medium", "context": "auto"}, "output": [], "usage": {}}
        outcome, _ = await self._one([(200, body)], "case_default_effort", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_effort_none_missing_reasoning_tokens_is_warning(self):
        body = {"output": [], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
        outcome, _ = await self._one([(200, body)], "case_effort_none", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_effort_none_with_reasoning_items_fails(self):
        body = {"output": [{"type": "reasoning", "summary": []}], "usage": {"output_tokens_details": {"reasoning_tokens": 10}}}
        outcome, _ = await self._one([(200, body)], "case_effort_none", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_cache_hit_blocked_when_not_observed(self):
        empty_usage = {"choices": [{"message": {"role": "assistant", "content": "OK"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        runner, client, _ = await self._runner([(200, empty_usage), (200, empty_usage)])
        try:
            passed, _, _, status = await runner.case_cache_hit("gpt-5.6")
        finally:
            await client.aclose()
        self.assertFalse(passed)
        self.assertEqual(status, "blocked")

    @staticmethod
    def _chat_body(cached: int, written: int = 0):
        return {"choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101,
                          "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written}}}

    async def test_cache_hit_passes_on_increment_over_baseline(self):
        # 首跳自带基线命中（如上游注入的固定前缀），但第二跳增量显著 → 构造前缀确实被缓存
        outcome, _ = await self._one([(200, self._chat_body(3840)), (200, self._chat_body(11000))], "case_cache_hit", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertEqual(outcome[2]["delta_cached_tokens"], 7160)

    async def test_cache_hit_baseline_only_is_blocked_not_passed(self):
        # 真实批次回归（mux.la）：两跳 cached_tokens 同为注入前缀的 3840、增量为 0，不得判通过
        outcome, _ = await self._one([(200, self._chat_body(3840)), (200, self._chat_body(3840))], "case_cache_hit", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertEqual(outcome[2]["delta_cached_tokens"], 0)
        self.assertEqual(outcome[2]["baseline_cached_tokens"], 3840)

    async def test_cache_hit_tiny_increment_is_warning(self):
        outcome, _ = await self._one([(200, self._chat_body(0)), (200, self._chat_body(128))], "case_cache_hit", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "warning")

    async def test_unknown_model_routed_success_is_failed(self):
        outcome, _ = await self._one([(200, {"output": [], "usage": {}})], "case_unknown_model", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_unknown_model_structured_404_passes(self):
        body = {"error": {"message": "model not found", "type": "invalid_request_error"}}
        outcome, _ = await self._one([(404, body)], "case_unknown_model", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")

    async def test_unknown_model_gateway_503_model_not_found_passes(self):
        # 真实批次回归（mux.la）：New API 用 503 + code=model_not_found 承载"模型不存在"
        body = {"error": {"code": "model_not_found", "type": "new_api_error",
                          "message": "No available channel for model gpt-5.6-nonexistent-deadbeef under group muxla_company"}}
        outcome, _ = await self._one([(503, body)], "case_unknown_model", "gpt-5.6")
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3], "passed")
        self.assertIn("503", outcome[1])

    async def test_unknown_model_5xx_without_model_evidence_fails(self):
        # 渠道整体故障 / 非结构化错误不得被当成"模型校验成立"
        for body in ({"error": {"message": "upstream timeout", "type": "server_error"}},
                     {"error": {"code": "channel_error", "message": "No available channel for model gpt-5.6-sol"}}):
            outcome, _ = await self._one([(503, body)], "case_unknown_model", "gpt-5.6")
            self.assertFalse(outcome[0])
            self.assertEqual(outcome[3], "failed")

    _ACCOUNT_LIMIT = {"error": {"message": "previous_response_id requires an OpenAI API-key account for HTTP requests",
                                "type": "invalid_request_error"}}

    async def test_previous_response_account_limit_is_blocked(self):
        first = {"id": "resp_1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "已记住"}]}]}
        outcome, _ = await self._one([(200, first), (400, self._ACCOUNT_LIMIT)], "case_previous_response_state", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertIn("账号层", outcome[1])

    async def test_previous_response_plain_400_is_failed(self):
        # 普通 400（无账号层限制特征）不得被兜底成 blocked
        first = {"id": "resp_1", "output": []}
        outcome, _ = await self._one([(200, first), (400, {"error": {"message": "bad request"}})], "case_previous_response_state", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_context_all_turns_account_limit_is_blocked(self):
        first = {"id": "resp_1", "output": []}
        outcome, _ = await self._one([(200, first), (400, self._ACCOUNT_LIMIT)], "case_context_all_turns", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_random_memory_account_limit_is_blocked(self):
        first = {"id": "resp_1", "output": []}
        outcome, _ = await self._one([(200, first), (400, self._ACCOUNT_LIMIT)], "case_auth_random_memory", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")

    async def test_auth_reasoning_summary_without_items_is_blocked(self):
        # 与 GPT56-05 统一口径：200 但拿不到 reasoning 产物 → 指纹取证前提不满足（blocked），不判模型失败
        body = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "391"}]}], "usage": {}}
        outcome, _ = await self._one([(200, body)], "case_auth_reasoning_summary", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "blocked")
        self.assertIn("GPT56-05", outcome[1])

    async def test_negative_cases_block_on_gateway_404(self):
        # 404 是链路/配置问题：非法 messages 的"参数被拒绝"不能由 404 冒充
        body = {"error": {"message": "no route", "type": "not_found_error"}}
        runner, client, _ = await self._runner([(404, body)])
        try:
            passed, _, _, status = await runner.case_messages_invalid("gpt-5.6")
        finally:
            await client.aclose()
        self.assertFalse(passed)
        self.assertEqual(status, "blocked")

    async def test_temperature_lenient_200_is_warning(self):
        runner, client, _ = await self._runner([(200, {"choices": [{"message": {"role": "assistant", "content": "OK"}}]})])
        try:
            passed, _, _, status = await runner.case_temperature_invalid("gpt-5.6")
        finally:
            await client.aclose()
        self.assertTrue(passed)
        self.assertEqual(status, "warning")

    async def test_encrypted_replay_blocked_without_encrypted_content(self):
        body = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "已记住"}]}], "usage": {}}
        runner, client, _ = await self._runner([(200, body)])
        try:
            passed, _, _, status = await runner.case_encrypted_replay("gpt-5.6")
        finally:
            await client.aclose()
        self.assertFalse(passed)
        self.assertEqual(status, "blocked")

    async def test_effort_contrast_silent_swallow_is_failed(self):
        body = {"output": [], "usage": {"output_tokens_details": {"reasoning_tokens": 0}}}
        outcome, _ = await self._one([(200, body), (200, body)], "case_auth_effort_contrast", "gpt-5.6")
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3], "failed")

    async def test_tool_loop_blocked_when_upstream_rejects(self):
        runner, client, _ = await self._runner([(400, {"error": {"message": "bad"}})])
        try:
            passed, _, _, status = await runner.case_tool_loop("gpt-5.6")
        finally:
            await client.aclose()
        self.assertFalse(passed)
        self.assertEqual(status, "blocked")

    async def test_truncation_detects_incomplete(self):
        body = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        runner, client, _ = await self._runner([(200, body)])
        try:
            passed, _, _ = await runner.case_truncation("gpt-5.6")
        finally:
            await client.aclose()
        self.assertTrue(passed)

    async def test_pro_mode_rejection_is_blocked(self):
        runner, client, _ = await self._runner([(400, {"error": {"message": "pro not available"}})])
        try:
            passed, _, _, status = await runner.case_pro_mode("gpt-5.6")
        finally:
            await client.aclose()
        self.assertFalse(passed)
        self.assertEqual(status, "blocked")


if __name__ == "__main__":
    unittest.main()
