import json
import tempfile
import threading
import time
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
        self.assertEqual(len(protocols), 6)
        self.assertEqual(next(x for x in protocols if x["id"]=="openai_responses")["path_template"], "/v1/responses")
        with urlopen(self.base + "/") as response:
            self.assertIn("性能测试台", response.read().decode())

    def test_runs_api_returns_filtered_pages_and_total(self):
        now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        ids = [f"pagination-{uuid}" for uuid in ("a", "b", "c")]
        with server.db() as connection:
            for index, run_id in enumerate(ids):
                connection.execute(
                    "INSERT INTO runs(id,test_kind,case_name,protocol,base_url,model,status,quick,created_at,passed) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (run_id, "load", "T0", "openai", "https://example.com", "m", "completed", 0,
                     f"{now[:-6]}{index:02d}+00:00", index % 2),
                )
        try:
            first = self.get_json("/api/runs?page=1&page_size=2&conclusion=pass")
            second = self.get_json("/api/runs?page=2&page_size=2&conclusion=pass")
            searched = self.get_json("/api/runs?page=1&page_size=2&q=example.com")
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

if __name__ == "__main__": unittest.main()
