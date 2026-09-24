import json
import unittest

import httpx

from dsv4.cases import CASES
from dsv4.core import CaseResult, DeepSeekConfig, DeepSeekV4Runner, _cache_tokens, summarize_results


class DeepSeekV4Tests(unittest.TestCase):
    def test_ui_suites(self):
        self.assertEqual(set(CASES), {"DS_V4_P0", "DS_V4_P1", "DS_V4_P2", "DS_V4_FINGERPRINT"})
        self.assertEqual(CASES["DS_V4_P0"].request_count,18)
        self.assertEqual(CASES["DS_V4_P2"].request_count,10)

    def test_config_urls(self):
        config=DeepSeekConfig("key", "https://api.deepseek.com/")
        self.assertEqual(config.url("chat/completions"), "https://api.deepseek.com/chat/completions")
        self.assertEqual(config.url("chat/completions", "beta"), "https://api.deepseek.com/beta/chat/completions")
        self.assertEqual(config.url("v1/messages", "anthropic"), "https://api.deepseek.com/anthropic/v1/messages")

    def test_config_urls_strip_api_version_for_beta_and_anthropic(self):
        # 运行时 base_url 已被 api_root() 规范化成 ".../v1"（web/server.py + dsv4/runner.py），
        # 而 beta/anthropic 端点挂在根域下（官方文档：api.deepseek.com/beta、api.deepseek.com/anthropic），
        # 不能再带版本后缀，否则会打到不存在的路径（线上 404）。
        config=DeepSeekConfig("key", "https://api.deepseek.com/v1")
        self.assertEqual(config.url("chat/completions"), "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(config.url("chat/completions", "beta"), "https://api.deepseek.com/beta/chat/completions")
        self.assertEqual(config.url("completions", "beta"), "https://api.deepseek.com/beta/completions")
        self.assertEqual(config.url("v1/messages", "anthropic"), "https://api.deepseek.com/anthropic/v1/messages")
        self.assertEqual(config.url("responses"), "https://api.deepseek.com/v1/responses")

    def test_explicit_beta_and_anthropic_override_wins(self):
        config=DeepSeekConfig("key", "https://api.deepseek.com/v1",
                              beta_base_url="https://gw.example.com/beta",
                              anthropic_base_url="https://gw.example.com/anthropic")
        self.assertEqual(config.url("chat/completions","beta"), "https://gw.example.com/beta/chat/completions")
        self.assertEqual(config.url("v1/messages","anthropic"), "https://gw.example.com/anthropic/v1/messages")

    def test_cache_usage_shapes(self):
        self.assertEqual(_cache_tokens({"usage":{"prompt_cache_hit_tokens":731}}),731)
        self.assertEqual(_cache_tokens({"usage":{"prompt_tokens_details":{"cached_tokens":123}}}),123)

    def test_summary_has_result_details(self):
        summary=summarize_results([CaseResult("1","a","m",True)])
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["results"][0]["case_id"],"1")


class DeepSeekV4MockTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_separates_reasoning_and_content(self):
        events=[{"choices":[{"delta":{"reasoning_content":"思考"}}]},{"choices":[{"delta":{"content":"答案"}}]}]
        body="".join(f"data: {json.dumps(x,ensure_ascii=False)}\n\n" for x in events)+"data: [DONE]\n\n"
        async def handler(request): return httpx.Response(200,text=body,headers={"content-type":"text/event-stream"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            runner=DeepSeekV4Runner(DeepSeekConfig("key"),client)
            response=await runner.chat("deepseek-v4-pro",[{"role":"user","content":"x"}],stream=True)
        self.assertEqual(response.stream["reasoning_content"],"思考")
        self.assertEqual(response.stream["content"],"答案")
        self.assertTrue(response.stream["done"])


if __name__=="__main__": unittest.main()

class DeepSeekV4OutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def _runner(self, responses):
        queue=list(responses)
        requests=[]
        async def handler(request):
            requests.append(json.loads(request.content))
            status,body=queue.pop(0)
            return httpx.Response(status,json=body)
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return DeepSeekV4Runner(DeepSeekConfig("key"),client),client,requests

    async def test_tool_priming_uses_auto_because_thinking_rejects_required(self):
        first={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        second={"choices":[{"message":{"role":"assistant","content":"数字是 731","reasoning_content":"done"}}]}
        runner,client,requests=await self._runner([(200,first),(200,second)])
        try: outcome=await runner.case_tool_replay("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        # thinking 模式不支持 tool_choice=required（上游 400），前置必须用 auto 诱导
        self.assertEqual(requests[0]["tool_choice"],"auto")
        self.assertEqual(requests[0]["thinking"],{"type":"enabled"})
        self.assertNotIn("required",[r.get("tool_choice") for r in requests])
        self.assertEqual(requests[1]["messages"][1]["reasoning_content"],"need tool")

    async def test_tool_priming_retries_with_stronger_prompt(self):
        no_call={"choices":[{"message":{"role":"assistant","content":"731","reasoning_content":"直接回答"}}]}
        with_call={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-2","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        final={"choices":[{"message":{"role":"assistant","content":"数字是 731","reasoning_content":"done"}}]}
        runner,client,requests=await self._runner([(200,no_call),(200,with_call),(200,final)])
        try: outcome=await runner.case_tool_replay("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(requests),3)
        self.assertNotEqual(requests[0]["messages"][0]["content"],requests[1]["messages"][0]["content"])
        self.assertEqual(requests[2]["messages"][1]["reasoning_content"],"need tool")

    async def test_tool_precondition_failure_is_blocked_not_failed(self):
        error={"error":{"message":"Thinking mode does not support this tool_choice"}}
        runner,client,_=await self._runner([(400,error)])
        try: outcome=await runner.case_tool_replay("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3],"blocked")
        self.assertEqual(outcome[2]["http_status"],400)
        self.assertIn("HTTP 400",outcome[1])

    async def test_tool_missing_reasoning_400_is_passed(self):
        first={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        second={"error":{"message":"reasoning_content is required when the assistant message carries tool_calls"}}
        runner,client,requests=await self._runner([(200,first),(400,second)])
        try: outcome=await runner.case_tool_missing_reasoning("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome),3)   # 400 分支不给显式 status，按 passed 收口
        self.assertEqual(requests[0]["tool_choice"],"auto")
        self.assertNotIn("reasoning_content",requests[1]["messages"][1])

    async def test_tool_missing_reasoning_200_is_warning(self):
        # 线上实际行为：漏传 reasoning_content 不报 400，容忍并正常作答（文档承诺 400）
        first={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        second={"choices":[{"message":{"role":"assistant","content":"数字是 731","reasoning_content":""}}]}
        runner,client,_=await self._runner([(200,first),(200,second)])
        try: outcome=await runner.case_tool_missing_reasoning("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertEqual(outcome[2]["http_status"],200)

    async def test_tool_missing_reasoning_unexpected_status_is_failed(self):
        first={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        runner,client,_=await self._runner([(200,first),(500,{"error":{"message":"boom"}})])
        try: outcome=await runner.case_tool_missing_reasoning("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(len(outcome),3)   # 既非 400 也非 200 → 无显式裁定，由 run() 归为 failed

    async def test_plain_tool_turn_200_is_warning(self):
        first={"choices":[{"message":{"role":"assistant","content":"OK","reasoning_content":"answer directly"}}]}
        second={"choices":[{"message":{"role":"assistant","content":"OK","reasoning_content":"again"}}]}
        runner,client,_=await self._runner([(200,first),(200,second)])
        try: outcome=await runner.case_tools_plain_turn_replay("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

    async def test_invalid_user_id_accepted_is_warning(self):
        response={"choices":[{"message":{"content":"OK"}}]}
        runner,client,_=await self._runner([(200,response)])
        try: outcome=await runner.case_invalid_user_id("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")

    def test_warning_and_blocked_do_not_fail_admission(self):
        summary=summarize_results([
            CaseResult("1","warning","m",True,"warning"),
            CaseResult("2","blocked","m",False,"blocked"),
            CaseResult("3","passed","m",True,"passed"),
        ])
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["warning"],1)
        self.assertEqual(summary["blocked"],1)

    async def test_beta_negative_case_404_is_blocked_not_passed(self):
        # 端点不存在（404）不能被当成"参数被正确拒绝"——原先 400<=status<500 会把 404 误判为通过
        not_found={"error":{"message":"not found"}}
        for handler in ("case_strict_invalid","case_prefix_invalid"):
            runner,client,_=await self._runner([(404,not_found)])
            try: outcome=await getattr(runner,handler)("deepseek-v4-pro")
            finally: await client.aclose()
            self.assertFalse(outcome[0],handler)
            self.assertEqual(outcome[3],"blocked",handler)
            self.assertIn("404",outcome[1],handler)

    async def test_openai_negative_cases_404_is_blocked_not_passed(self):
        # 同类缺陷（DS-V4-22 曾因此在两端点 404 的情况下显示"通过"）：temperature 越界 /
        # 非法 messages / 非法 user_id 原先用宽松的 400<=status<500，404 会被误读为
        # "参数被正确拒绝"。端点不存在一律记 blocked，不能算拒绝。
        not_found={"error":{"message":"not found"}}
        for handler in ("case_temperature_invalid","case_messages_invalid","case_invalid_user_id"):
            runner,client,_=await self._runner([(404,not_found)])
            try: outcome=await getattr(runner,handler)("deepseek-v4-pro")
            finally: await client.aclose()
            self.assertFalse(outcome[0],handler)
            self.assertEqual(outcome[3],"blocked",handler)
            self.assertIn("404",outcome[1],handler)

    async def test_beta_negative_case_400_is_passed_and_200_is_failed(self):
        runner,client,_=await self._runner([(400,{"error":{"message":"unsupported strict schema"}})])
        try: outcome=await runner.case_strict_invalid("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome),3)   # 正常拒绝 → 无显式裁定，按 passed 收口

        runner,client,_=await self._runner([(200,{"choices":[{"message":{"content":"ok"}}]})])
        try: outcome=await runner.case_strict_invalid("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(len(outcome),3)   # 未拒绝 → 由 run() 归为 failed

    async def test_prefix_misplacement_4xx_passed_200_warning_5xx_failed(self):
        # 官方《Chat Prefix Completion (Beta)》只要求"调用方自行确保"prefix 位于最后一条
        # assistant 消息，从未承诺对错误放置返回 4xx。线上实测：静默忽略并正常作答（200）。
        # 因此 4xx（严格校验）→ passed；200（宽松忽略）→ warning（不判模型失败、不阻断准入）。
        runner,client,_=await self._runner([(400,{"error":{"message":"prefix must be the last assistant message"}})])
        try: outcome=await runner.case_prefix_invalid("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(len(outcome),3)            # 校验型拒绝 → 无显式裁定，按 passed 收口

        ok={"choices":[{"message":{"role":"assistant","content":"写代码\n\n你好！"}}]}
        runner,client,_=await self._runner([(200,ok)])
        try: outcome=await runner.case_prefix_invalid("deepseek-flash")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(outcome[3],"warning")
        self.assertIn("未承诺拒绝",outcome[1])

        runner,client,_=await self._runner([(503,{"error":{"message":"server overloaded"}})])
        try: outcome=await runner.case_prefix_invalid("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(len(outcome),3)            # 5xx → 由 run() 归为 failed

    async def test_beta_positive_cases_404_are_blocked(self):
        not_found={"error":{"message":"not found"}}
        for handler in ("case_strict_tool","case_prefix","case_fim"):
            runner,client,_=await self._runner([(404,not_found)])
            try: outcome=await getattr(runner,handler)("deepseek-v4-pro")
            finally: await client.aclose()
            self.assertFalse(outcome[0],handler)
            self.assertEqual(outcome[3],"blocked",handler)

    async def test_anthropic_endpoint_404_is_blocked(self):
        runner,client,_=await self._runner([(404,{"error":{"message":"not found"}})])
        try: outcome=await runner.case_anthropic_mapping("deepseek-flash")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3],"blocked")
