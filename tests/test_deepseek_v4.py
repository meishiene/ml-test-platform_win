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

    async def test_tool_replay_uses_required_and_passes_complete_history(self):
        first={"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"need tool","tool_calls":[{"id":"call-1","type":"function","function":{"name":"get_number","arguments":"{}"}}]}}]}
        second={"choices":[{"message":{"role":"assistant","content":"数字是 731","reasoning_content":"done"}}]}
        runner,client,requests=await self._runner([(200,first),(200,second)])
        try: outcome=await runner.case_tool_replay("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertTrue(outcome[0])
        self.assertEqual(requests[0]["tool_choice"],"required")
        self.assertEqual(requests[1]["messages"][1]["reasoning_content"],"need tool")

    async def test_tool_precondition_failure_is_blocked_not_failed(self):
        error={"error":{"message":"unsupported tool_choice"}}
        runner,client,_=await self._runner([(400,error)])
        try: outcome=await runner.case_tool_replay("deepseek-v4-pro")
        finally: await client.aclose()
        self.assertFalse(outcome[0])
        self.assertEqual(outcome[3],"blocked")
        self.assertEqual(outcome[2]["http_status"],400)

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
