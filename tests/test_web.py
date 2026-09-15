import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from web import server


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp.name) / "runs.db"
        cls.db_patch = patch.object(server, "DB_PATH", cls.db_path)
        cls.db_patch.start()
        server.initialize()
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown(); cls.httpd.server_close(); cls.db_patch.stop(); cls.temp.cleanup()

    def get_json(self, path):
        with urlopen(self.base + path) as response:
            return json.load(response)

    def test_cases_and_index(self):
        cases=self.get_json("/api/cases")
        self.assertGreaterEqual(len(cases), 30)
        self.assertEqual({x["id"] for x in cases if x["id"].endswith("CONTRACT")}, {"KVV_CONTRACT", "DS_CONTRACT", "GLM_CONTRACT"})
        self.assertEqual(next(x for x in cases if x["id"]=="KVV_CONTRACT")["test_kind"], "kvv")
        self.assertEqual(next(x for x in cases if x["id"]=="DS_CONTRACT")["test_kind"], "deepseek")
        self.assertFalse(any(x["test_kind"] == "strict" for x in cases))
        load_ids={x["id"] for x in cases if x["test_kind"] == "load"}
        self.assertEqual(load_ids,{"T0","B1","SIMPLE_LOAD","FULL_LOAD","C0","C1"})
        simple=next(x for x in cases if x["id"]=="SIMPLE_LOAD")
        full=next(x for x in cases if x["id"]=="FULL_LOAD")
        self.assertEqual((simple["concurrency"],simple["requests"]),(30,300))
        self.assertEqual((full["concurrency"],full["requests"]),(50,550))
        self.assertIn("L1–L5 各 50",simple["request_text"])
        self.assertIn("O1 50",full["request_text"])
        required={"concurrency_text","input_tokens","request_text","warmup","steady","admission"}
        self.assertTrue(all(required.issubset(item) for item in cases))
        b1=next(x for x in cases if x["id"]=="B1")
        self.assertEqual((b1["concurrency_text"],b1["request_text"]),("2","成功目标 16；最多尝试 16"))
        self.assertIn("2K / 8K / 16K / 32K",b1["input_tokens"])
        self.assertIn("L1–L5",simple["workload_profile"])
        self.assertTrue(simple["fixed_parameters"])
        self.assertEqual(len(self.get_json("/api/protocols")), 4)
        with urlopen(self.base + "/") as response:
            self.assertIn("性能测试台", response.read().decode())

    def test_create_run_does_not_persist_key(self):
        payload = json.dumps({"protocol":"anthropic", "test_kind":"load", "base_url":"https://example.com/v1", "api_key":"super-secret", "model":"ep-test", "case":"T0", "quick":True}).encode()
        request = Request(self.base + "/api/runs", data=payload, method="POST", headers={"Content-Type":"application/json"})
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(request) as response: created=json.load(response)
            self.assertEqual(created["status"], "queued")
            self.assertNotIn("api_key", created)
            self.assertEqual(created["protocol"], "anthropic")
            self.assertEqual(created["test_kind"], "load")
            self.assertEqual(put.call_args.args[0]["api_key"], "super-secret")
        self.assertNotIn("super-secret", self.db_path.read_bytes().decode("utf-8", "ignore"))

    def test_create_load_presets(self):
        payload=json.dumps({"protocol":"volcengine", "test_kind":"load", "base_url":"https://example.com/v1", "api_key":"secret", "model":"m", "case":"SIMPLE_LOAD", "quick":True}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["case_ids"], ["SIMPLE_LOAD"])
        self.assertTrue(result["is_batch"])
        children=put.call_args.args[0]["children"]
        self.assertEqual([child["case"] for child in children], ["L1","L2","L3","L4","L5","O1"])
        self.assertEqual([(child["requests"],child["concurrency"]) for child in children], [(50,30)]*5+[(50,10)])
        self.assertTrue(all(child["warmup_seconds"] == 0 for child in children))
        self.assertTrue(all(not child["quick"] for child in children))

        payload=json.dumps({"protocol":"volcengine", "test_kind":"load", "base_url":"https://example.com/v1", "api_key":"secret", "model":"m", "case":"FULL_LOAD"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})):
                pass
        children=put.call_args.args[0]["children"]
        self.assertEqual([(child["requests"],child["concurrency"]) for child in children], [(100,50)]*5+[(50,10)])

    def test_reject_removed_strict_suite(self):
        payload=json.dumps({"protocol":"anthropic", "test_kind":"strict", "base_url":"https://example.com/v1", "api_key":"secret", "model":"m", "case":"STRICT_T0"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_create_kvv_run(self):
        payload=json.dumps({"protocol":"volcengine", "test_kind":"kvv", "base_url":"https://example.com/v1", "api_key":"secret", "model":"m", "case":"KVV_K3_FEATURES"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "kvv")
        self.assertEqual(put.call_args.args[0]["case"], "KVV_K3_FEATURES")

    def test_create_gpt_image_run_keeps_randomization_choices(self):
        payload=json.dumps({
            "protocol":"openai",
            "test_kind":"gpt-image",
            "base_url":"https://api.openai.com/v1",
            "api_key":"secret",
            "model":"gpt-image-2",
            "case":"GPT_IMAGE_2",
            "requests":7,
            "concurrency":3,
            "image_quality":["low","high"],
            "image_models":["gpt-image-2"],
            "image_sizes":["1024x1024","1536x1024"],
        }).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "gpt-image")
        job=put.call_args.args[0]
        self.assertEqual((job["requests"],job["concurrency"]),(7,3))
        self.assertEqual(job["image_quality"],["low","high"])
        self.assertEqual(job["image_sizes"],["1024x1024","1536x1024"])

    def test_create_deepseek_run(self):
        payload=json.dumps({"protocol":"openai", "test_kind":"deepseek", "base_url":"https://api.deepseek.com", "api_key":"secret", "model":"deepseek-v4-pro", "case":"DS_V4_P0"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "deepseek")
        job=put.call_args.args[0]
        self.assertEqual(job["case"], "DS_V4_P0")
        self.assertEqual(job["model"], "deepseek-v4-pro")

    def test_create_multi_kvv_batch(self):
        payload=json.dumps({"protocol":"volcengine","test_kind":"kvv","base_url":"https://example.com/v1","api_key":"secret","model":"m","cases":["KVV_K3_FEATURES","KVV_PARAMS"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                batch=json.load(response)
        self.assertTrue(batch["is_batch"])
        self.assertEqual([x["case"] for x in put.call_args.args[0]["children"]],["KVV_K3_FEATURES","KVV_PARAMS"])

    def test_create_multi_deepseek_batch(self):
        payload=json.dumps({"protocol":"openai","test_kind":"deepseek","base_url":"https://api.deepseek.com","api_key":"secret","model":"deepseek-v4-pro","cases":["DS_V4_P0","DS_V4_P1"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                batch=json.load(response)
        self.assertTrue(batch["is_batch"])
        self.assertEqual([x["case"] for x in put.call_args.args[0]["children"]],["DS_V4_P0","DS_V4_P1"])

    def test_accept_versioned_deepseek_model(self):
        payload=json.dumps({"protocol":"openai", "test_kind":"deepseek", "base_url":"https://api.deepseek.com", "api_key":"secret", "model":"deepseek-v4-pro-ga-260813", "case":"DS_V4_P0"}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["model"],"deepseek-v4-pro-ga-260813")
        self.assertEqual(put.call_args.args[0]["model"],"deepseek-v4-pro-ga-260813")

    def test_accept_gateway_model_alias_for_deepseek_suite(self):
        payload=json.dumps({"protocol":"volcengine", "test_kind":"deepseek", "base_url":"https://example.com/v1", "api_key":"secret", "model":"ep-customer-ds-prod", "case":"DS_V4_P0"}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["model"],"ep-customer-ds-prod")

    def test_reject_unsupported_kvv_protocol(self):
        payload=json.dumps({"protocol":"anthropic", "test_kind":"kvv", "base_url":"https://example.com/v1", "api_key":"secret", "model":"m", "case":"KVV_PARAMS"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_create_multi_load_batch(self):
        payload=json.dumps({"protocol":"volcengine","test_kind":"load","base_url":"https://example.com/v1","api_key":"secret","model":"m","cases":["T0","B1"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                batch=json.load(response)
        self.assertTrue(batch["is_batch"])
        self.assertEqual(batch["case_ids"],["T0","B1"])
        job=put.call_args.args[0]
        self.assertTrue(job["is_batch"])
        self.assertEqual([x["case"] for x in job["children"]],["T0","B1"])
        detail=self.get_json(f"/api/runs/{batch['id']}")
        self.assertEqual(len(detail["children"]),2)

    def test_connectivity_check(self):
        payload=json.dumps({"protocol":"volcengine", "base_url":"https://example.com/v1", "api_key":"secret", "model":"m"}).encode()
        result={"name":"connectivity","passed":True,"http_status":200,"duration_ms":12.3}
        with patch.object(server, "check_connectivity", return_value=result) as checked:
            with urlopen(Request(self.base+"/api/connectivity/check",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                actual=json.load(response)
        self.assertTrue(actual["passed"])
        config=checked.call_args.args[0]
        self.assertEqual(config.protocol,"volcengine")
        self.assertEqual(config.api_key,"secret")

    def test_query_models(self):
        payload=json.dumps({"protocol":"openai", "base_url":"https://example.com/v1", "api_key":"secret"}).encode()
        fake = type("Response", (), {"status_code": 200, "text": "", "json": lambda self: {"data": [{"id":"model-b"},{"id":"model-a"}]}})()
        with patch.object(server.httpx, "get", return_value=fake) as get:
            with urlopen(Request(self.base+"/api/models/query",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual([m["id"] for m in result["models"]], ["model-a", "model-b"])
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer secret")

    def test_reject_bad_url(self):
        payload=json.dumps({"base_url":"bad", "api_key":"x", "model":"m", "case":"T0"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

if __name__ == "__main__": unittest.main()
