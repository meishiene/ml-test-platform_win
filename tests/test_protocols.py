import unittest
from volcperf.client import parse_event
from volcperf.protocols import endpoint, headers, payload, models_endpoint, models_headers, parse_models


class ProtocolTests(unittest.TestCase):
    def test_openai(self):
        self.assertEqual(endpoint("openai","https://api.openai.com/v1","gpt-x"), "https://api.openai.com/v1/chat/completions")
        self.assertEqual(headers("openai","key","id")["Authorization"], "Bearer key")
        body=payload("openai","gpt-x","hi",100,True); self.assertTrue(body["stream"]); self.assertIn("stream_options",body); self.assertNotIn("reasoning_effort",body)
        event=parse_event("openai",{"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}})
        self.assertEqual(event["content"],"hello"); self.assertEqual(event["prompt_tokens"],2)

    def test_volcengine(self):
        self.assertEqual(endpoint("volcengine","https://ark.example/api/v3","ep"), "https://ark.example/api/v3/chat/completions")

    def test_anthropic(self):
        self.assertEqual(endpoint("anthropic","https://api.anthropic.com/v1","claude"), "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers("anthropic","key","id")["x-api-key"],"key")
        event=parse_event("anthropic",{"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}})
        self.assertEqual(event["content"],"hello")
        usage=parse_event("anthropic",{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":50}})
        self.assertEqual(usage["finish_reason"],"length"); self.assertEqual(usage["completion_tokens"],50)

    def test_gemini(self):
        self.assertEqual(endpoint("gemini","https://generativelanguage.googleapis.com/v1beta","gemini/x"), "https://generativelanguage.googleapis.com/v1beta/models/gemini%2Fx:streamGenerateContent?alt=sse")
        self.assertEqual(headers("gemini","key","id")["x-goog-api-key"],"key")
        event=parse_event("gemini",{"candidates":[{"content":{"parts":[{"text":"think","thought":True},{"text":"answer"}]},"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":2}})
        self.assertEqual(event["content"],"answer"); self.assertEqual(event["reasoning"],"think"); self.assertEqual(event["finish_reason"],"length")

    def test_model_endpoints_and_parsing(self):
        self.assertEqual(models_endpoint("openai", "https://api.openai.com/v1"), "https://api.openai.com/v1/models")
        self.assertEqual(models_endpoint("anthropic", "https://api.anthropic.com/v1"), "https://api.anthropic.com/v1/models")
        self.assertEqual(models_endpoint("gemini", "https://example/v1beta"), "https://example/v1beta/models?pageSize=1000")
        self.assertEqual(models_headers("gemini", "k")["x-goog-api-key"], "k")
        self.assertEqual(parse_models("openai", {"data": [{"id":"gpt-b"},{"id":"gpt-a"}]}), [{"id":"gpt-a","name":"gpt-a"},{"id":"gpt-b","name":"gpt-b"}])
        gemini = parse_models("gemini", {"models": [{"name":"models/gemini-pro","displayName":"Gemini Pro","supportedGenerationMethods":["generateContent"]},{"name":"models/embed","supportedGenerationMethods":["embedContent"]}]})
        self.assertEqual(gemini, [{"id":"gemini-pro","name":"Gemini Pro"}])

if __name__ == "__main__": unittest.main()
