import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

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
    async def _runner(self, responses, max_retries=0):
        """handler 级判定测试：默认关闭瞬时故障重试，保持"一问一答"的脚本语义。

        重试与 blocked 收口（run 层）由 GPT56TransientTests 专门覆盖。
        """
        queue = list(responses)
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            status, body = queue.pop(0)
            return httpx.Response(status, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        config = GPT56Config("key", max_retries=max_retries, retry_backoff_seconds=0.0)
        return GPT56Runner(config, client), client, requests

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


# 真实批次证据（2026-09-24，results/GPT56_P0-20260924-153058/requests.jsonl）：
# mux.la 中转对 GPT56-12 / GPT56-16 返回的网关自身故障。
MEMORY_OVERLOADED_503 = {"error": {"message": "system memory overloaded (current: 91.5%, threshold: 90%)",
                                   "type": "new_api_error", "param": "", "code": "system_memory_overloaded"}}
OK_CHAT = {"model": "gpt-5.6-sol", "choices": [{"message": {"role": "assistant", "content": "OK"}}]}
MODEL_NOT_FOUND_503 = {"error": {"code": "model_not_found", "type": "new_api_error",
                                 "message": "No available channel for model gpt-5.6-nonexistent-deadbeef under group muxla_company"}}


class GPT56TransientTests(unittest.IsolatedAsyncioTestCase):
    """瞬时故障（上游 5xx / 连接错误）：先重试，重试耗尽后按 blocked 收口。"""

    async def _runner(self, queue, *, max_retries=2, raise_errors=None):
        queue = list(queue)
        errors = list(raise_errors or [])
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            if errors:
                raise errors.pop(0)
            status, body = queue.pop(0)
            return httpx.Response(status, json=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        config = GPT56Config("key", max_retries=max_retries, retry_backoff_seconds=0.0)
        return GPT56Runner(config, client), client, requests

    async def _run_one(self, runner, case_id, priority="P0"):
        return (await runner.run(("gpt-5.6",), (priority,), {case_id}))[0]

    async def test_5xx_is_retried_then_succeeds(self):
        """首跳 503、重试命中 200：用例应正常判定，且不留下瞬时故障记录。"""
        runner, client, requests = await self._runner([(503, MEMORY_OVERLOADED_503), (200, OK_CHAT)])
        try:
            result = await self._run_one(runner, "GPT56-01")
            self.assertEqual(len(requests), 2, "503 应触发一次重试")
            self.assertEqual(runner.transient_events, [])
        finally:
            await client.aclose()
        self.assertEqual(result.status, "passed")

    async def test_5xx_exhausted_retries_becomes_blocked_not_failed(self):
        """重试耗尽仍 503：原判 failed 的用例改判 blocked，理由需自解释。"""
        runner, client, requests = await self._runner([(503, MEMORY_OVERLOADED_503)] * 3)
        try:
            result = await self._run_one(runner, "GPT56-16")
            self.assertEqual(len(requests), 3, "max_retries=2 → 共 3 次尝试")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.passed)
        self.assertIn("瞬时故障", result.details)
        self.assertIn("已重试 2 次仍失败", result.details)
        self.assertIn("system memory overloaded", result.details)
        self.assertEqual(result.observations["transient_failures"][0]["http_status"], 503)

    async def test_connection_error_exhausted_becomes_blocked(self):
        """连接错误（ConnectError）同样重试，耗尽后记 blocked。"""
        error = httpx.ConnectError("connection reset")
        runner, client, requests = await self._runner([], raise_errors=[error, error, error])
        try:
            result = await self._run_one(runner, "GPT56-01")
            self.assertEqual(len(requests), 3)
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("ConnectError", result.details)

    async def test_retries_disabled_records_single_attempt(self):
        runner, client, requests = await self._runner([(503, MEMORY_OVERLOADED_503)] * 3, max_retries=0)
        try:
            result = await self._run_one(runner, "GPT56-16")
            self.assertEqual(len(requests), 1)
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("未启用重试", result.details)

    async def test_definitive_model_not_found_is_not_retried(self):
        """503 + code=model_not_found 是确定性结论，不是网关抖动：不重试，用例照常判通过。"""
        runner, client, requests = await self._runner([(503, MODEL_NOT_FOUND_503)] * 3)
        try:
            result = await self._run_one(runner, "GPT56-23", priority="P1")
            self.assertEqual(len(requests), 1, "确定性错误不得触发重试")
            self.assertEqual(runner.transient_events, [])
        finally:
            await client.aclose()
        self.assertEqual(result.status, "passed")

    async def test_4xx_is_not_retried_and_not_downgraded(self):
        runner, client, requests = await self._runner([(400, {"error": {"message": "bad request"}})] * 3)
        try:
            result = await self._run_one(runner, "GPT56-01")
            self.assertEqual(len(requests), 1, "4xx 是确定性判定，不重试")
        finally:
            await client.aclose()
        self.assertEqual(result.status, "failed")

    async def test_mid_case_transient_downgrades_failed_case(self):
        """故障发生在用例中途（最后一跳已 200）也要按 blocked 收口。"""
        runner, client, _ = await self._runner([(503, MEMORY_OVERLOADED_503)] * 3 + [(200, OK_CHAT)] * 2)
        try:
            result = await self._run_one(runner, "GPT56-04")  # effort_levels：5 档循环
        finally:
            await client.aclose()
        self.assertEqual(result.status, "blocked")
        self.assertIn("瞬时故障", result.details)

    async def test_passed_case_is_never_downgraded(self):
        """只有 failed/error 才降级：通过用例即便出现过瞬时故障也不得被改成 blocked。"""
        runner, client, _ = await self._runner([(503, MEMORY_OVERLOADED_503)])

        async def fake_basic(model):
            await runner.request("chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]})
            return True, "通过", {}

        runner.case_basic = fake_basic  # 模拟"末跳瞬时故障但用例仍通过"的极端组合
        try:
            result = await self._run_one(runner, "GPT56-01")
        finally:
            await client.aclose()
        self.assertEqual(runner.transient_events and len(runner.transient_events), 1)
        self.assertEqual(result.status, "passed")

    async def test_transient_reason_is_self_explanatory(self):
        from gpt56.core import _transient_reason
        text = _transient_reason({"path": "responses", "attempts": 3, "http_status": 503,
                                  "transport_error": None, "message": "boom"}, 1)
        for fragment in ("HTTP 503", "已重试 2 次仍失败", "boom", "blocked", "重跑"):
            self.assertIn(fragment, text)


class GPT56AdmissionExitCodeTests(unittest.IsolatedAsyncioTestCase):
    """退出码必须以 finalize() 之后的磁盘结论为准。

    回归：收尾前内存里的 summary 由 ``all_passed``（只算 failed+error）推出，
    "整批只有 blocked、没有 failed"时是 True，于是 main_async 曾返回 0，
    而磁盘 summary / Web 显示不准入（本轮瞬时故障场景实测 exit=0）。
    """

    async def _run_main(self, statuses):
        from gpt56 import runner as runner_module
        from gpt56.core import CaseResult

        created = {}

        class FakeRunner:
            def __init__(self, config, audit_dir=None):
                self.audit_dir = audit_dir
                created["dir"] = audit_dir

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def run(self, models, priorities):
                # 证据契约：结果目录必须有非空请求证据，否则整批不准入（与本用例无关，但要满足）。
                self.audit_dir.mkdir(parents=True, exist_ok=True)
                (self.audit_dir / "requests.jsonl").write_text(
                    json.dumps({"case_id": "GPT56-01", "http_status": 200}) + "\n", encoding="utf-8")
                return [CaseResult("GPT56-01", "基础对话", models[0], False, status, "说明")
                        for status in statuses]

        env = {"GPT_API_KEY": "key", "GPT_BASE_URL": "https://gw.example.com", "GPT_MODEL": "gpt-5.6-sol"}
        saved = {k: os.environ.get(k) for k in env}
        original = runner_module.GPT56Runner
        with tempfile.TemporaryDirectory() as tmp:
            try:
                runner_module.GPT56Runner = FakeRunner
                os.environ.update(env)
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = await runner_module.main_async(["GPT56_P0", "--output-root", tmp])
            finally:
                runner_module.GPT56Runner = original
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            disk = json.loads((Path(created["dir"]) / "summary.json").read_text(encoding="utf-8"))
        return rc, disk

    async def test_blocked_only_batch_is_not_admitted(self):
        """只有 blocked、没有 failed：磁盘不准入，退出码必须是 2（修复前误为 0）。"""
        rc, disk = await self._run_main(["blocked", "blocked"])
        self.assertFalse(disk["admission_passed"])
        self.assertEqual(disk["failed"], 0)  # 收尾前 all_passed 正是被这一项骗过
        self.assertEqual(rc, 2)

    async def test_all_passed_batch_still_admits(self):
        rc, disk = await self._run_main(["passed"])
        self.assertTrue(disk["admission_passed"])
        self.assertEqual(rc, 0)

    async def test_failed_batch_keeps_exit_code_2(self):
        rc, disk = await self._run_main(["failed"])
        self.assertFalse(disk["admission_passed"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
