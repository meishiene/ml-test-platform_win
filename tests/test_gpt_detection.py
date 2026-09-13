import httpx

from gpt_detection.runner import extract_response_model, model_match_checks, run


def test_extract_response_model_reads_top_level_model_only():
    assert extract_response_model({"model": "gpt-5.6-sol"}) == "gpt-5.6-sol"
    assert extract_response_model({"choices": [{"model": "gpt-5.6-sol"}]}) is None
    assert extract_response_model({"model": "  "}) is None


def test_model_match_checks_pass_when_requested_and_returned_model_are_equal():
    checks = model_match_checks(
        expected_model="gpt-5.6-sol",
        response_body={"model": "gpt-5.6-sol"},
        http_status=200,
    )
    assert all(item.passed is True for item in checks)


def test_model_match_checks_fail_when_response_model_is_different():
    checks = model_match_checks(
        expected_model="gpt-5.6-sol",
        response_body={"model": "gpt-5.6-terra"},
        http_status=200,
    )
    result = {item.name: item for item in checks}
    assert result["response_model_present"].passed is True
    assert result["model_match"].passed is False


def test_runner_archives_match_result_without_recording_api_key(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-secret"
        return httpx.Response(200, json={"id": "chatcmpl-test", "model": "gpt-test", "choices": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result_dir, summary = run(
        case_name="GPT_MODEL_MATCH",
        base_url="https://example.test/v1",
        api_key="test-secret",
        model="gpt-test",
        protocol="openai",
        output_root=tmp_path,
        client=client,
    )
    client.close()

    assert summary["admission_passed"] is True
    assert summary["actual_model"] == "gpt-test"
    assert (result_dir / "summary.json").is_file()
    assert "test-secret" not in (result_dir / "requests.jsonl").read_text(encoding="utf-8")
