import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
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

    def test_latest_result_finds_temporary_suite_for_single_case(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            suite_result = results / "K3_TEMP_SUITE-20260922-172637"
            suite_result.mkdir()
            with patch.object(server, "RESULTS", results):
                # 与 web/server.py 收尾阶段的真实调用一致：单案例执行时
                # 产物目录是 K3_TEMP_SUITE-*，通过 aliases 命中。
                found = server.latest_result(
                    "CACHE-HIT-10", time.time() - 2, aliases=("K3_TEMP_SUITE",)
                )
            self.assertEqual(found, suite_result)

    def test_worker_failure_marks_job_failed_instead_of_dying_silently(self):
        run_id = f"worker-failure-{time.time_ns()}"
        with server.db() as connection:
            connection.execute(
                "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, "load", "T0", "openai", "https://example.com", "m", "queued", 0,
                 time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())),
            )
        try:
            server._mark_worker_failure({"id": run_id}, RuntimeError("boom"))
            with server.db() as connection:
                row = connection.execute("SELECT status,error,exit_code FROM runs WHERE id=?", (run_id,)).fetchone()
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["exit_code"], -1)
            self.assertIn("RuntimeError: boom", row["error"])
        finally:
            with server.db() as connection:
                connection.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def test_single_temporary_run_persists_selected_case_name(self):
        payload = json.dumps({
            "protocol": "volcengine", "test_kind": "temporary",
            "base_url": "https://example.com", "api_key": "secret", "model": "kimi-k3",
            "case": "K3_TEMP_SUITE", "temporary_case": "CACHE-HIT-10",
        }).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "CACHE-HIT-10")
        self.assertEqual(result["case_title"], "90% 稳定前缀命中探测")
        self.assertEqual(put.call_args.args[0]["temporary_case"], "CACHE-HIT-10")

    def test_interrupted_image_summary_keeps_partial_request_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)
            rows = [
                {"success": True, "error_type": None},
                {"success": False, "error_type": "http_503"},
                {"success": False, "error_type": "request_timeout"},
            ]
            (result / "requests.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary = server.interrupted_request_summary(
                {
                    "test_kind": "gpt-image",
                    "case": "GPT_IMAGE_2",
                    "protocol": "openai",
                    "base_url": "https://example.test/v1",
                    "model": "gpt-image-2",
                    "requests": 5,
                },
                "run-id",
                result,
            )
        self.assertEqual(summary["completed"], 3)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["unfinished"], 2)
        self.assertEqual(
            summary["error_distribution"],
            {"http_503": 1, "request_timeout": 1},
        )

    def test_cases_and_index(self):
        cases=self.get_json("/api/cases")
        self.assertGreaterEqual(len(cases), 30)
        self.assertEqual({x["id"] for x in cases if x["id"].endswith("CONTRACT")}, {"KVV_CONTRACT", "DS_CONTRACT", "GLM_CONTRACT"})
        self.assertEqual(next(x for x in cases if x["id"]=="KVV_CONTRACT")["test_kind"], "kvv")
        self.assertEqual(next(x for x in cases if x["id"]=="DS_CONTRACT")["test_kind"], "deepseek")
        self.assertFalse(any(x["test_kind"] == "strict" for x in cases))
        load_ids={x["id"] for x in cases if x["test_kind"] == "load"}
        self.assertEqual(load_ids,{"T0","RANDOM_LOAD","B1","SIMPLE_LOAD","FULL_LOAD","C0","C1"})
        random_load=next(x for x in cases if x["id"]=="RANDOM_LOAD")
        self.assertEqual((random_load["concurrency"],random_load["requests"]),(10,100))
        self.assertEqual(random_load["output_tokens"],"2048")
        self.assertTrue(random_load["manual_parameters"])
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
        protocols=self.get_json("/api/protocols")
        self.assertEqual(len(protocols), 7)
        self.assertEqual(next(x for x in protocols if x["id"]=="openai_responses")["path_template"], "/v1/responses")
        gemini_images=next(x for x in protocols if x["id"]=="gemini_images")
        self.assertEqual(gemini_images["path_template"], "/v1beta/models/{model}:generateContent")
        self.assertEqual(gemini_images["test_kinds"], ["gemini-image"])
        with urlopen(self.base + "/") as response:
            self.assertIn("性能测试台", response.read().decode())

    def test_runs_api_returns_filtered_pages_and_total(self):
        now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        ids = [f"pagination-{uuid}" for uuid in ("a", "b", "c")]
        with server.db() as connection:
            for index, run_id in enumerate(ids):
                connection.execute(
                    "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,passed) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (run_id, "load", "T0", "openai", "https://pagination.example.test", "m", "completed", 0,
                     f"{now[:-6]}{index:02d}+00:00", int(index != 1)),
                )
        try:
            first = self.get_json("/api/runs?page=1&page_size=2&conclusion=pass")
            second = self.get_json("/api/runs?page=2&page_size=2&conclusion=pass")
            searched = self.get_json("/api/runs?page=1&page_size=2&q=pagination.example.test")
            self.assertEqual((first["total"], len(first["items"])), (2, 2))
            self.assertEqual((second["total"], len(second["items"])), (2, 0))
            self.assertEqual(searched["total"], 3)
        finally:
            with server.db() as connection:
                connection.executemany("DELETE FROM runs WHERE id=?", [(run_id,) for run_id in ids])

    def test_delete_run_removes_record_and_result_directory(self):
        run_id = f"delete-test-{time.time_ns()}"
        result_dir = server.RESULTS / run_id
        result_dir.mkdir(parents=True)
        (result_dir / "report.md").write_text("test report", encoding="utf-8")
        with server.db() as connection:
            connection.execute(
                "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,result_dir) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (run_id, "load", "T0", "openai", "https://example.com", "m", "completed", 0,
                 time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), str(result_dir.relative_to(server.ROOT))),
            )
        request = Request(self.base + f"/api/runs/{run_id}", method="DELETE")
        with urlopen(request) as response:
            result = json.load(response)
        self.assertEqual(result["deleted_records"], 1)
        self.assertFalse(result_dir.exists())
        with server.db() as connection:
            self.assertIsNone(connection.execute("SELECT id FROM runs WHERE id=?", (run_id,)).fetchone())

    def test_create_run_does_not_persist_key(self):
        payload = json.dumps({"protocol":"anthropic", "test_kind":"load", "base_url":"https://example.com", "api_key":"super-secret", "model":"ep-test", "case":"T0", "quick":True}).encode()
        request = Request(self.base + "/api/runs", data=payload, method="POST", headers={"Content-Type":"application/json"})
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(request) as response: created=json.load(response)
            self.assertEqual(created["status"], "queued")
            self.assertNotIn("api_key", created)
            self.assertEqual(created["protocol"], "anthropic")
            self.assertEqual(created["test_kind"], "load")
            self.assertEqual(put.call_args.args[0]["api_key"], "super-secret")
        self.assertNotIn("super-secret", self.db_path.read_bytes().decode("utf-8", "ignore"))

    def test_create_openai_responses_load_run(self):
        payload=json.dumps({"protocol":"openai_responses", "test_kind":"load", "base_url":"https://mux.la", "api_key":"secret", "model":"gpt-5-mini", "case":"T0", "quick":True}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["protocol"], "openai_responses")
        self.assertEqual(result["base_url"], "https://mux.la")
        self.assertEqual(put.call_args.args[0]["protocol"], "openai_responses")

    def test_create_load_presets(self):
        payload=json.dumps({"protocol":"volcengine", "test_kind":"load", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"SIMPLE_LOAD", "quick":True}).encode()
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

        payload=json.dumps({"protocol":"volcengine", "test_kind":"load", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"FULL_LOAD"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})):
                pass
        children=put.call_args.args[0]["children"]
        self.assertEqual([(child["requests"],child["concurrency"]) for child in children], [(100,50)]*5+[(50,10)])

    def test_create_random_load_uses_exact_manual_request_and_concurrency_values(self):
        payload=json.dumps({"protocol":"volcengine", "test_kind":"load", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"RANDOM_LOAD", "quick":True, "requests":37, "concurrency":6}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        job=put.call_args.args[0]
        self.assertEqual(result["case_ids"], ["RANDOM_LOAD"])
        self.assertEqual((job["requests"],job["concurrency"]),(37,6))
        self.assertFalse(job["quick"])

    def test_reject_removed_strict_suite(self):
        payload=json.dumps({"protocol":"anthropic", "test_kind":"strict", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"STRICT_T0"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_create_kvv_run(self):
        payload=json.dumps({"protocol":"volcengine", "test_kind":"kvv", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"KVV_K3_FEATURES"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "kvv")
        self.assertEqual(put.call_args.args[0]["case"], "KVV_K3_FEATURES")

    def test_create_gpt_image_run_keeps_randomization_choices(self):
        payload=json.dumps({
            "protocol":"openai_images",
            "test_kind":"gpt-image",
            "base_url":"https://api.openai.com",
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

    def test_create_fixed_gpt_image_4k_token_verification(self):
        payload=json.dumps({
            "protocol":"openai_images",
            "test_kind":"gpt-image",
            "base_url":"https://api.openai.com",
            "api_key":"secret",
            "model":"gpt-image-2",
            "case":"GPT_IMAGE_4K_TOKEN_VERIFY",
        }).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "gpt-image")
        job=put.call_args.args[0]
        self.assertEqual((job["requests"],job["concurrency"]),(3,1))
        self.assertEqual(job["image_quality"],["high","max"])
        self.assertEqual(job["image_models"],["gpt-image-2","gpt-image-2.5-flare","gpt-image-2.5-sunburst"])
        self.assertEqual(job["image_sizes"],["3840x2160"])

    def test_create_gemini_image_run_keeps_randomization_choices(self):
        payload=json.dumps({
            "protocol":"gemini_images",
            "test_kind":"gemini-image",
            "base_url":"https://generativelanguage.googleapis.com",
            "api_key":"secret",
            "model":"gemini-3-pro-image",
            "case":"GEMINI_IMAGE_GEN",
            "requests":6,
            "concurrency":2,
            "gemini_quality":["1K","4K"],
            "gemini_models":["gemini-3-pro-image"],
            "gemini_aspects":["1:1","16:9"],
            "gemini_counts":[1,2],
        }).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"], "gemini-image")
        job=put.call_args.args[0]
        self.assertEqual((job["requests"],job["concurrency"]),(6,2))
        self.assertEqual(job["gemini_quality"],["1K","4K"])
        self.assertEqual(job["gemini_models"],["gemini-3-pro-image"])
        self.assertEqual(job["gemini_aspects"],["1:1","16:9"])
        self.assertEqual(job["gemini_counts"],[1,2])

    def test_gemini_image_requires_manual_requests_and_concurrency(self):
        payload=json.dumps({
            "protocol":"gemini_images","test_kind":"gemini-image","base_url":"https://example.com",
            "api_key":"secret","model":"gemini-3-pro-image","case":"GEMINI_IMAGE_GEN",
        }).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_gemini_image_rejects_unsupported_choices_and_protocol(self):
        bad_count=json.dumps({
            "protocol":"gemini_images","test_kind":"gemini-image","base_url":"https://example.com",
            "api_key":"secret","model":"gemini-3-pro-image","case":"GEMINI_IMAGE_GEN",
            "requests":3,"concurrency":1,"gemini_counts":[9],
        }).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=bad_count,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

        bad_protocol=json.dumps({
            "protocol":"openai_images","test_kind":"gemini-image","base_url":"https://example.com",
            "api_key":"secret","model":"gemini-3-pro-image","case":"GEMINI_IMAGE_GEN",
            "requests":3,"concurrency":1,
        }).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=bad_protocol,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_gemini_image_case_is_listed_with_its_parameter_pools(self):
        cases=self.get_json("/api/cases")
        gemini=[item for item in cases if item["test_kind"]=="gemini-image"]
        self.assertEqual([item["id"] for item in gemini],["GEMINI_IMAGE_GEN"])
        self.assertEqual(gemini[0]["image_qualities"],["1K","2K","4K"])
        self.assertEqual(gemini[0]["image_counts"],[1,2,3,4])
        self.assertIn("16:9", gemini[0]["image_aspect_ratios"])

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
        payload=json.dumps({"protocol":"volcengine","test_kind":"kvv","base_url":"https://example.com","api_key":"secret","model":"m","cases":["KVV_K3_FEATURES","KVV_PARAMS"]}).encode()
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
        payload=json.dumps({"protocol":"volcengine", "test_kind":"deepseek", "base_url":"https://example.com", "api_key":"secret", "model":"ep-customer-ds-prod", "case":"DS_V4_P0"}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["model"],"ep-customer-ds-prod")

    def test_reject_unsupported_kvv_protocol(self):
        payload=json.dumps({"protocol":"anthropic", "test_kind":"kvv", "base_url":"https://example.com", "api_key":"secret", "model":"m", "case":"KVV_PARAMS"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_create_fable_run_accepts_anthropic_protocol(self):
        # 回归：/api/run 的 test_kind→用例集派发链曾漏掉 fable，available 变成空字典，
        # 于是任何 Fable 5 提交都直接 400「未知测试类型」，套件根本发不出去。
        payload=json.dumps({"protocol":"anthropic","test_kind":"fable","base_url":"https://i-token.ai","api_key":"secret","model":"claude-fable-5","cases":["FABLE5_FINGERPRINT"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"],"fable")
        job=put.call_args.args[0]
        self.assertEqual(job["case"],"FABLE5_FINGERPRINT")
        self.assertEqual(job["protocol"],"anthropic")
        self.assertEqual(job["model"],"claude-fable-5")

    def test_fable_suite_rejects_non_anthropic_protocol(self):
        payload=json.dumps({"protocol":"openai","test_kind":"fable","base_url":"https://example.com","api_key":"secret","model":"claude-fable-5","cases":["FABLE5_P0"]}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_fable_cases_are_listed_with_anthropic_protocol(self):
        cases=self.get_json("/api/cases")
        fable={x["id"]:x for x in cases if x["test_kind"]=="fable"}
        # Claude 测试：fable5 五套件按模型变体各发一份（FABLE5_* / OPUS5_* / SONNET5_* /
        # HAIKU45_* / SONNET46_*），外加一个常驻的中转计费探针（不带 claude_variant，不随变体切换）。
        self.assertEqual(len(fable),26)
        self.assertEqual({x["claude_variant"] for x in fable.values() if x.get("claude_variant")},{"fable5","opus5","sonnet5","haiku45","sonnet46"})
        self.assertNotIn("claude_variant", fable[server.BILLING_PROBE_CASE_ID])
        self.assertTrue(fable[server.BILLING_PROBE_CASE_ID]["always_visible"])
        # 每个变体都必须提供同一组五套件（BASIC / P0 / P1 / P2 / FINGERPRINT），仅模型不同。
        for variant_id, suffix in (("fable5","FABLE5"),("opus5","OPUS5"),("sonnet5","SONNET5"),
                                   ("haiku45","HAIKU45"),("sonnet46","SONNET46")):
            self.assertEqual({x["id"] for x in fable.values() if x.get("claude_variant")==variant_id},
                             {f"{suffix}_{name}" for name in ("BASIC","P0","P1","P2","FINGERPRINT")})
        self.assertTrue(all(x["protocols"]==["anthropic"] for x in fable.values()))
        self.assertEqual(fable["FABLE5_BASIC"]["requests"],15)
        self.assertEqual(fable["FABLE5_P0"]["requests"],20)
        self.assertEqual(fable["FABLE5_P1"]["requests"],11)
        self.assertEqual(fable["FABLE5_P2"]["requests"],21)
        self.assertEqual(fable["FABLE5_FINGERPRINT"]["requests"],11)
        # 变体套件与 fable5 套件请求数一致（同构用例，仅模型不同）。
        self.assertEqual(fable["OPUS5_FINGERPRINT"]["requests"],11)
        self.assertEqual(fable["SONNET5_P2"]["requests"],21)
        self.assertEqual(fable["HAIKU45_BASIC"]["requests"],15)
        self.assertEqual(fable["HAIKU45_FINGERPRINT"]["requests"],11)
        self.assertEqual(fable["SONNET46_P0"]["requests"],20)
        self.assertEqual(fable["SONNET46_P2"]["requests"],21)
        self.assertEqual(fable["OPUS5_BASIC"]["default_model"],"claude-opus-5")
        self.assertEqual(fable["SONNET5_FINGERPRINT"]["default_model"],"claude-sonnet-5")
        self.assertEqual(fable["HAIKU45_BASIC"]["default_model"],"claude-haiku-4-5")
        self.assertEqual(fable["SONNET46_FINGERPRINT"]["default_model"],"claude-sonnet-4-6")
        self.assertIn("docs/Fable5专项测试用例设计.md", fable["FABLE5_P0"]["admission"])
        self.assertIn("Opus 5", fable["OPUS5_BASIC"]["title"])
        self.assertIn("Haiku 4.5", fable["HAIKU45_BASIC"]["title"])
        self.assertIn("Sonnet 4.6", fable["SONNET46_BASIC"]["title"])
        # 提示词、并发等展示字段与 fable5 本体一致（只换模型与标题前缀）。
        self.assertEqual(fable["HAIKU45_P1"]["concurrency"], fable["FABLE5_P1"]["concurrency"])

    def test_create_claude_variant_run_maps_to_fable5_suite(self):
        # 各模型变体用例：入队记录保留变体 ID（归档归属），派发命令映射回 FABLE5_* 套件。
        payload=json.dumps({"protocol":"anthropic","test_kind":"fable","base_url":"https://i-token.ai","api_key":"secret","model":"claude-opus-5","cases":["OPUS5_FINGERPRINT"]}).encode()
        with patch.object(server.JOB_QUEUE,"put") as put:
            with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual(result["test_kind"],"fable")
        job=put.call_args.args[0]
        self.assertEqual(job["case"],"OPUS5_FINGERPRINT")
        self.assertEqual(job["model"],"claude-opus-5")
        self.assertEqual(server.claude_suite_name(job["case"]),"FABLE5_FINGERPRINT")

    def test_create_haiku_and_sonnet46_variant_runs_map_to_fable5_suites(self):
        # Haiku 4.5 / Sonnet 4.6 与既有变体同构：变体 ID → FABLE5_* 套件名，模型按提交值下发。
        for case_id, model, suite in (("HAIKU45_P2","claude-haiku-4-5","FABLE5_P2"),
                                      ("HAIKU45_BASIC","claude-haiku-4-5","FABLE5_BASIC"),
                                      ("SONNET46_FINGERPRINT","claude-sonnet-4-6","FABLE5_FINGERPRINT")):
            payload=json.dumps({"protocol":"anthropic","test_kind":"fable","base_url":"https://i-token.ai",
                                "api_key":"secret","model":model,"cases":[case_id]}).encode()
            with patch.object(server.JOB_QUEUE,"put") as put:
                with urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                    result=json.load(response)
            self.assertEqual(result["test_kind"],"fable")
            job=put.call_args.args[0]
            self.assertEqual(job["case"],case_id)
            self.assertEqual(job["model"],model)
            self.assertEqual(job["protocol"],"anthropic")
            self.assertEqual(server.claude_suite_name(job["case"]),suite)

    def test_create_multi_load_batch(self):
        payload=json.dumps({"protocol":"volcengine","test_kind":"load","base_url":"https://example.com","api_key":"secret","model":"m","cases":["T0","B1"]}).encode()
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
        payload=json.dumps({"protocol":"volcengine", "base_url":"https://example.com", "api_key":"secret", "model":"m"}).encode()
        result={"name":"connectivity","passed":True,"http_status":200,"duration_ms":12.3}
        with patch.object(server, "check_connectivity", return_value=result) as checked:
            with urlopen(Request(self.base+"/api/connectivity/check",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                actual=json.load(response)
        self.assertTrue(actual["passed"])
        config=checked.call_args.args[0]
        self.assertEqual(config.protocol,"volcengine")
        self.assertEqual(config.api_key,"secret")

    def test_connectivity_check_shares_trust_env_switch_with_load_tests(self):
        """探针必须与正式压测共用 VOLC_TRUST_ENV。

        否则会出现"模型列表查得到、连通性检查却走直连"的前后不一致：
        探针忽略 .env 里的代理设置，直连在境外网关上失败，正式测试根本起不来。
        """
        payload=json.dumps({"protocol":"anthropic", "base_url":"https://mux.la", "api_key":"secret", "model":"kimi-k3"}).encode()
        result={"name":"connectivity","passed":True,"http_status":200,"duration_ms":12.3}
        for flag, expected in ((True, True), (False, False)):
            with patch.object(server, "trust_env_enabled", return_value=flag):
                with patch.object(server, "check_connectivity", return_value=result) as checked:
                    with urlopen(Request(self.base+"/api/connectivity/check",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                        json.load(response)
            self.assertIs(checked.call_args.args[0].trust_env, expected)

    def test_query_models(self):
        payload=json.dumps({"protocol":"openai", "base_url":"https://example.com", "api_key":"secret"}).encode()
        fake = type("Response", (), {"status_code": 200, "text": "", "json": lambda self: {"data": [{"id":"model-b"},{"id":"model-a"}]}})()
        with patch.object(server.httpx, "get", return_value=fake) as get:
            with urlopen(Request(self.base+"/api/models/query",data=payload,method="POST",headers={"Content-Type":"application/json"})) as response:
                result=json.load(response)
        self.assertEqual([m["id"] for m in result["models"]], ["model-a", "model-b"])
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(get.call_args.args[0], "https://example.com/v1/models")
        # 与探针保持一致：是否走环境代理由 VOLC_TRUST_ENV 决定，不再隐式使用。
        self.assertIn("trust_env", get.call_args.kwargs)

    def test_reject_bad_url(self):
        payload=json.dumps({"base_url":"bad", "api_key":"x", "model":"m", "case":"T0"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def test_reject_base_url_with_path(self):
        payload=json.dumps({"protocol":"openai", "base_url":"https://mux.la/v1", "api_key":"x", "model":"m", "case":"T0"}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base+"/api/runs",data=payload,method="POST",headers={"Content-Type":"application/json"}))
        self.assertEqual(caught.exception.code,400)

    def _make_image_run(self, results: Path, run_id: str) -> Path:
        """造一条带 images/ 的图片测试记录，返回结果目录。"""
        result_dir = results / f"GEMINI_IMAGE_GEN-20261008-170000-{run_id[:6]}"
        (result_dir / "images").mkdir(parents=True)
        (result_dir / "summary.json").write_text('{"admission_passed": true}', encoding="utf-8")
        (result_dir / "requests.jsonl").write_text('{"index": 1, "images": []}\n', encoding="utf-8")
        (result_dir / "images" / "req-0001-01.png").write_bytes(b"\x89PNG\r\n\x1a\n-fake-image")
        with server.db() as connection:
            connection.execute(
                "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,result_dir) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (run_id, "gemini-image", "GEMINI_IMAGE_GEN", "gemini_images", "https://mux.la",
                 "gemini-3-pro-image", "completed", 0,
                 time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), result_dir.name),
            )
        return result_dir

    def test_download_bundle_keeps_generated_images(self):
        run_id = f"download-images-{time.time_ns()}"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve()
            target = self._make_image_run(results, run_id)
            try:
                # ROOT 与 RESULTS 一起指向临时目录，result_dir 用目录名（相对 ROOT）。
                with patch.object(server, "RESULTS", results), patch.object(server, "ROOT", results):
                    with urlopen(f"{self.base}/api/runs/{run_id}/download") as response:
                        self.assertEqual(response.headers["Content-Type"], "application/zip")
                        payload = response.read()
                names = zipfile.ZipFile(io.BytesIO(payload)).namelist()
                # 服务端在响应体写完后才删除临时归档，客户端可能先读到 EOF。
                for _ in range(50):
                    leftovers = list(results.glob(".download-*.zip"))
                    if not leftovers:
                        break
                    time.sleep(0.02)
            finally:
                with server.db() as connection:
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))
        self.assertIn("package-metadata.json", names)
        self.assertIn(f"{target.name}/images/req-0001-01.png", names)
        self.assertIn(f"{target.name}/requests.jsonl", names)
        # 临时归档必须清理，不能留在结果目录里。
        self.assertEqual(leftovers, [])

    def test_selected_download_bundle_keeps_generated_images(self):
        run_id = f"selected-images-{time.time_ns()}"
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve()
            target = self._make_image_run(results, run_id)
            try:
                payload = json.dumps({"run_ids": [run_id]}).encode()
                with patch.object(server, "RESULTS", results), patch.object(server, "ROOT", results):
                    with urlopen(Request(self.base + "/api/runs/download", data=payload, method="POST",
                                         headers={"Content-Type": "application/json"})) as response:
                        data = response.read()
                bundle = zipfile.ZipFile(io.BytesIO(data))
                names = bundle.namelist()
                metadata = json.loads(bundle.read("package-metadata.json"))
            finally:
                with server.db() as connection:
                    connection.execute("DELETE FROM runs WHERE id=?", (run_id,))
        self.assertEqual(metadata["type"], "selected-runs")
        self.assertEqual(metadata["selected_run_ids"], [run_id])
        self.assertIn(f"{target.name}/images/req-0001-01.png", names)

    def _temporary_payload(self, *, protocol="volcengine", temporary_case="CACHE-HIT-10", model="kimi-k3"):
        return json.dumps({
            "protocol": protocol, "test_kind": "temporary", "base_url": "https://i-token.ai",
            "api_key": "secret", "model": model, "case": "K3_TEMP_SUITE", "temporary_case": temporary_case,
        }).encode()

    def _billing_payload(self, *, protocol="anthropic", billing_probe="BILLING-ALL", model="claude-fable-5"):
        return json.dumps({
            "protocol": protocol, "test_kind": "fable", "base_url": "https://i-token.ai",
            "api_key": "secret", "model": model, "case": server.BILLING_PROBE_CASE_ID,
            "billing_probe": billing_probe,
        }).encode()

    def test_create_billing_probe_run_under_claude_module(self):
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=self._billing_payload(),
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["test_kind"], "fable")
        self.assertEqual(result["case_name"], "BILLING-ALL")
        self.assertEqual(result["case_title"], "中转计费探针 · 全部 11 条")
        job = put.call_args.args[0]
        self.assertEqual(job["case"], server.BILLING_PROBE_CASE_ID)
        self.assertEqual(job["billing_probe"], "BILLING-ALL")
        self.assertEqual(job["protocol"], "anthropic")

    def test_create_billing_probe_single_case(self):
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs",
                                 data=self._billing_payload(billing_probe="BILLING-B02"),
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "BILLING-B02")
        self.assertEqual(result["case_title"], "预扣结算口径与释放时延（是否退回）")
        self.assertEqual(put.call_args.args[0]["billing_probe"], "BILLING-B02")

    def test_billing_probe_defaults_to_whole_group(self):
        # 选中计费卡片但未指定子项时按整组处理，避免「选了卡片却跑不动」。
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-fable-5",
                              "cases": [server.BILLING_PROBE_CASE_ID]}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "BILLING-ALL")
        self.assertEqual(put.call_args.args[0]["billing_probe"], "BILLING-ALL")

    def test_billing_probe_rejects_unknown_selection(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs",
                            data=self._billing_payload(billing_probe="BILLING-NOPE"),
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("中转计费探针", json.loads(caught.exception.read())["error"])

    def test_billing_probe_rejects_other_protocol(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs",
                            data=self._billing_payload(protocol="gemini", model="gemini-2.5-flash"),
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("Anthropic", json.loads(caught.exception.read())["error"])

    def test_billing_probe_ignored_for_plain_claude_suite(self):
        # 普通 Claude 套件即便带了 billing_probe 残留，也必须置空并走 fable5 执行器。
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-fable-5", "cases": ["FABLE5_P0"],
                              "billing_probe": "BILLING-ALL"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "FABLE5_P0")
        self.assertIsNone(put.call_args.args[0]["billing_probe"])

    def test_billing_probe_in_batch_only_probe_child_carries_selection(self):
        # 计费探针在 Claude tab 下常与各套件同批提交。旧逻辑要求 case_names 恰好只有探针
        # 一条才保留 billing_probe，批量时被整批置空 → 探针子任务落到 run_fable5.py 报
        # invalid choice（历史批次因此整批不准入）。批量下必须逐子任务区分。
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-haiku-4-5",
                              "cases": ["HAIKU45_BASIC", server.BILLING_PROBE_CASE_ID],
                              "billing_probe": "BILLING-ALL"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertTrue(result["is_batch"])
        children = put.call_args.args[0]["children"]
        self.assertEqual([child["case"] for child in children], ["HAIKU45_BASIC", server.BILLING_PROBE_CASE_ID])
        self.assertIsNone(children[0]["billing_probe"])
        self.assertEqual(children[1]["billing_probe"], "BILLING-ALL")
        # 探针子记录的显示名按所选子项归档，与单条运行口径一致（记录页显示"全部 11 条"）。
        detail = self.get_json(f"/api/runs/{result['id']}")
        titles = {row["case_name"]: row.get("case_title") for row in detail["children"]}
        self.assertEqual(titles["BILLING-ALL"], "中转计费探针 · 全部 11 条")
        self.assertIsNone(titles["HAIKU45_BASIC"])

    def test_billing_probe_in_batch_passes_single_probe_id(self):
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-fable-5",
                              "cases": ["FABLE5_P1", server.BILLING_PROBE_CASE_ID],
                              "billing_probe": "BILLING-U03"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                json.load(response)
        children = {child["case"]: child for child in put.call_args.args[0]["children"]}
        self.assertEqual(children[server.BILLING_PROBE_CASE_ID]["billing_probe"], "BILLING-U03")
        self.assertIsNone(children["FABLE5_P1"]["billing_probe"])

    def test_batch_with_billing_probe_rejects_non_anthropic_protocol(self):
        # 探针只走 Anthropic Messages 面；同批混入时也必须拦回，而不是把探针派给错的面。
        payload = json.dumps({"protocol": "openai", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-fable-5",
                              "cases": ["FABLE5_P1", server.BILLING_PROBE_CASE_ID]}).encode()
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs", data=payload,
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("Anthropic", json.loads(caught.exception.read())["error"])

    def test_batch_without_billing_probe_keeps_selection_empty(self):
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-fable-5",
                              "cases": ["FABLE5_BASIC", "FABLE5_P0"], "billing_probe": "BILLING-ALL"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                json.load(response)
        children = put.call_args.args[0]["children"]
        self.assertTrue(all(child["billing_probe"] is None for child in children))

    def test_temporary_anthropic_is_rejected(self):
        # anthropic 面下 K3 案例不可执行（K3 走 Chat Completions）；计费探针已迁至 Claude 模块。
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs",
                            data=self._temporary_payload(protocol="anthropic", model="claude-fable-5"),
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("openai", json.loads(caught.exception.read())["error"])

    def test_temporary_rejects_billing_selection(self):
        # 计费探针搬走后，临时案例集不再接受 BILLING-* 选择。
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.base + "/api/runs",
                            data=self._temporary_payload(temporary_case="BILLING-ALL"),
                            method="POST", headers={"Content-Type": "application/json"}))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("未知临时测试案例", json.loads(caught.exception.read())["error"])

    def test_temporary_k3_selection_still_accepted_on_chat_completions(self):
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=self._temporary_payload(),
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                result = json.load(response)
        self.assertEqual(result["case_name"], "CACHE-HIT-10")
        self.assertEqual(put.call_args.args[0]["temporary_case"], "CACHE-HIT-10")

    def test_billing_probe_is_listed_under_claude_module(self):
        cases = self.get_json("/api/cases")
        billing = [item for item in cases if item["id"] == server.BILLING_PROBE_CASE_ID]
        self.assertEqual(len(billing), 1)
        item = billing[0]
        self.assertEqual(item["test_kind"], "fable")
        self.assertTrue(item["always_visible"])
        self.assertTrue(item["billing_probe"])
        self.assertEqual(item["protocols"], ["anthropic"])
        self.assertEqual({option.get("group") for option in item["case_options"]}, {"中转计费探针"})
        self.assertEqual(item["case_count"], 12)
        # 临时案例集不再暴露计费分组。
        temporary = [entry for entry in cases if entry["test_kind"] == "temporary"][0]
        self.assertEqual({option.get("group") for option in temporary["case_options"]}, {"K3 供应商自测基线"})
        self.assertEqual(temporary["case_count"], 24)

    def _execute_child_capture(self, job_fields):
        run_id = f"child-{time.time_ns()}"
        stored = job_fields.get("billing_probe") or job_fields.get("temporary_case") or job_fields["case"]
        with server.db() as connection:
            connection.execute(
                "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, job_fields["test_kind"], stored, job_fields["protocol"], "https://i-token.ai",
                 "claude-fable-5", "queued", 0, time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())),
            )
        job = {"id": run_id, "base_url": "https://i-token.ai", "api_key": "sk-x",
               "model": "claude-fable-5", "quick": False, **job_fields}
        captured = {}

        class FakeProcess:
            def wait(self):
                return 0

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs.get("env")
            return FakeProcess()

        try:
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(server, "RESULTS", Path(directory)), \
                     patch.object(server, "latest_result", return_value=None), \
                     patch.object(server.subprocess, "Popen", fake_popen):
                    server._execute_child(job)
        finally:
            with server.db() as connection:
                connection.execute("DELETE FROM runs WHERE id=?", (run_id,))
        return captured

    def _run_billing_child(self, billing_probe, protocol="anthropic"):
        return self._execute_child_capture({
            "test_kind": "fable", "case": server.BILLING_PROBE_CASE_ID,
            "protocol": protocol, "billing_probe": billing_probe,
        })

    def test_batch_children_dispatch_to_matching_runners(self):
        # 端到端：批量里同一批子任务按各自 case 派到不同执行器（探针→计费脚本，套件→fable5）。
        payload = json.dumps({"protocol": "anthropic", "test_kind": "fable", "base_url": "https://i-token.ai",
                              "api_key": "secret", "model": "claude-haiku-4-5",
                              "cases": ["HAIKU45_BASIC", server.BILLING_PROBE_CASE_ID],
                              "billing_probe": "BILLING-ALL"}).encode()
        with patch.object(server.JOB_QUEUE, "put") as put:
            with urlopen(Request(self.base + "/api/runs", data=payload,
                                 method="POST", headers={"Content-Type": "application/json"})) as response:
                json.load(response)
        children = {child["case"]: child for child in put.call_args.args[0]["children"]}
        probe = self._execute_child_capture({
            "test_kind": "fable", "case": server.BILLING_PROBE_CASE_ID, "protocol": "anthropic",
            "billing_probe": children[server.BILLING_PROBE_CASE_ID]["billing_probe"]})
        self.assertEqual(Path(probe["command"][1]).name, "run_relay_billing_probe.py")
        self.assertEqual(probe["env"]["BILLING_PROBES"], "")
        suite = self._execute_child_capture({
            "test_kind": "fable", "case": "HAIKU45_BASIC", "protocol": "anthropic", "billing_probe": None})
        self.assertEqual(Path(suite["command"][1]).name, "run_fable5.py")
        # 套件子任务用 fable5 真实套件名做位置参数，且不带任何计费环境变量。
        self.assertEqual(suite["command"][2], "FABLE5_BASIC")
        self.assertNotIn("BILLING_PROBES", suite["env"])

    def _run_temp_child(self, temporary_case, protocol="volcengine"):
        return self._execute_child_capture({
            "test_kind": "temporary", "case": temporary_case,
            "protocol": protocol, "temporary_case": temporary_case,
        })

    def test_billing_child_uses_probe_runner_and_billing_env(self):
        captured = self._run_billing_child("BILLING-ALL")
        self.assertEqual(Path(captured["command"][1]).name, "run_relay_billing_probe.py")
        # 计费探针靠环境变量驱动，命令里不得出现位置参数。
        self.assertNotIn("BILLING-ALL", [str(part) for part in captured["command"]])
        env = captured["env"]
        self.assertEqual(env["BILLING_BASE_URL"], "https://i-token.ai")
        self.assertEqual(env["BILLING_API_KEY"], "sk-x")
        self.assertEqual(env["BILLING_MODEL"], "claude-fable-5")
        self.assertEqual(env["BILLING_PROTOCOL"], "anthropic")
        # 整组 = 空（执行器按「空即全部 11 条」处理）。
        self.assertEqual(env["BILLING_PROBES"], "")

    def test_billing_child_passes_selected_probe_ids(self):
        captured = self._run_billing_child("BILLING-U03")
        self.assertEqual(captured["env"]["BILLING_PROBES"], "BILLING-U03")
        self.assertEqual(captured["env"]["BILLING_PROTOCOL"], "anthropic")

    def test_billing_child_maps_openai_protocol_to_chat_completions(self):
        # 探针脚本只认 anthropic / openai 两种面；其它协议在入口层已被拒绝。
        captured = self._run_billing_child("BILLING-S01", protocol="openai")
        self.assertEqual(captured["env"]["BILLING_PROTOCOL"], "openai")
        self.assertEqual(captured["env"]["BILLING_BASE_URL"], "https://i-token.ai")

    def test_k3_child_keeps_manifest_runner_without_billing_env(self):
        captured = self._run_temp_child("CACHE-HIT-10")
        self.assertEqual(Path(captured["command"][1]).name, "run_temp_cases.py")
        self.assertEqual(captured["env"]["VOLC_TEMP_CASE"], "CACHE-HIT-10")
        self.assertNotIn("BILLING_PROBES", captured["env"])


class CrossPlatformLaunchTests(unittest.TestCase):
    """Windows 启动/停止路径的回归。

    Windows 专有分支在 macOS 上跑不到（os.name、taskkill、netstat），但一旦写错，
    症状都是「页面能用、停止测试报 500」或「日志乱码」这种静默故障，所以这里把
    可判定的部分抽成纯函数/按平台断言打桩覆盖。
    """

    def test_stored_paths_are_posix_and_round_trip(self):
        target = server.RESULTS / "CASE-X-20260101-010203" / "summary.json"
        stored = server._portable_path(target)
        self.assertEqual(stored, "results/CASE-X-20260101-010203/summary.json")
        # 统一成正斜杠后，两个平台都能用 ROOT / relative 解析回来。
        self.assertEqual(server.ROOT / stored, target)
        # 落在项目根之外（例如被临时目录替换过的 RESULTS）不应抛异常。
        self.assertEqual(server._portable_path(Path("/tmp/outside")), "/tmp/outside")

    def test_popen_kwargs_match_platform(self):
        kwargs = server._child_popen_kwargs()
        if os.name == "nt":
            # Windows 没有 start_new_session；必须换成进程组 + 隐藏控制台。
            self.assertIn("creationflags", kwargs)
            self.assertNotIn("start_new_session", kwargs)
        else:
            self.assertEqual(kwargs, {"start_new_session": True})

    @unittest.skipIf(os.name == "nt", "Windows 走 taskkill /T 分支，无进程组可断言")
    def test_terminate_process_tree_kills_descendants(self):
        # 父进程只负责拉起一个孙进程并等待；只 terminate 父进程会留下孙进程继续打上游。
        parent = subprocess.Popen(["/bin/sh", "-c", "sleep 30 & wait"], start_new_session=True)
        try:
            deadline = time.time() + 5
            grandchild = None
            while time.time() < deadline and grandchild is None:
                listing = subprocess.run(["pgrep", "-P", str(parent.pid)], capture_output=True, text=True)
                if listing.stdout.strip():
                    grandchild = int(listing.stdout.split()[0])
                time.sleep(0.1)
            self.assertIsNotNone(grandchild, "未能创建孙进程，用例本身失效")
            server._terminate_process_tree(parent)
            # 必须回收：僵尸进程仍在进程表里，os.kill(pid, 0) 不会报错，会误判成「没杀掉」。
            parent.wait(timeout=5)
            self._assert_gone(grandchild)
        finally:
            if parent.poll() is None:
                parent.kill()

    def _assert_gone(self, pid):
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        self.fail(f"进程 {pid} 仍存活")


class WindowsStartupEntryTests(unittest.TestCase):
    """start.bat 依赖的启动入口：netstat 解析与 .env 解析必须与平台无关地可验证。"""

    @classmethod
    def setUpClass(cls):
        for candidate in (str(server.ROOT), str(server.ROOT / "scripts")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
        import start_web
        cls.entry = start_web

    def test_netstat_listeners_pick_only_the_bound_port(self):
        output = "\n".join([
            "活动连接",
            "",
            "  协议  本地地址          外部地址         状态           PID",
            "  TCP    0.0.0.0:8085       0.0.0.0:0        LISTENING       4321",
            "  TCP    127.0.0.1:8085     127.0.0.1:52001  ESTABLISHED     4321",
            "  TCP    [::]:8085          [::]:0           LISTENING       4321",
            "  TCP    0.0.0.0:8086       0.0.0.0:0        LISTENING       9999",
            # 8085 上被 System(PID 4) 占着时不能去 kill，否则会拿到拒绝访问还把端口问题掩盖掉。
            "  TCP    0.0.0.0:8085       0.0.0.0:0        LISTENING       4",
        ])
        self.assertEqual(self.entry.parse_windows_listeners(output, 8085), [4321])
        self.assertEqual(self.entry.parse_windows_listeners(output, 8086), [9999])
        self.assertEqual(self.entry.parse_windows_listeners(output, 8087), [])
        self.assertEqual(self.entry.parse_windows_listeners("", 8085), [])

    def test_display_host_replaces_wildcard(self):
        for wildcard in ("", "0.0.0.0", "::"):
            self.assertEqual(self.entry.display_host(wildcard), "127.0.0.1")
        self.assertEqual(self.entry.display_host("192.168.1.9"), "192.168.1.9")

    def test_dotenv_does_not_override_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "# 注释行\n"
                "\n"
                "MLTEST_KEEP=from-file\n"
                "export MLTEST_EXPORTED='带引号的值'\n"
                "MLTEST_INLINE=value # 行内注释\n"
                "MLTEST_HASH=ab#cd\n"
                "NOT A KEY\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"MLTEST_KEEP": "from-env"}, clear=False):
                loaded = self.entry.load_dotenv(env_file)
                self.assertNotIn("MLTEST_KEEP", loaded)
                self.assertEqual(os.environ["MLTEST_KEEP"], "from-env")
                self.assertEqual(os.environ["MLTEST_EXPORTED"], "带引号的值")
                self.assertEqual(os.environ["MLTEST_INLINE"], "value")
                # 行内 # 只有前面是空白才算注释，避免误伤含 # 的密钥。
                self.assertEqual(os.environ["MLTEST_HASH"], "ab#cd")
                self.assertNotIn("NOT A KEY", os.environ)
            for key in ("MLTEST_EXPORTED", "MLTEST_INLINE", "MLTEST_HASH"):
                os.environ.pop(key, None)
        self.assertEqual(self.entry.load_dotenv(Path("/nonexistent/.env")), [])


if __name__ == "__main__": unittest.main()
