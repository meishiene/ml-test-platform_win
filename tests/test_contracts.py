import json
import unittest
from common.model_contracts import PROFILES, json_object, validate_identity, validate_usage

class ContractTests(unittest.TestCase):
    def test_profiles_are_model_specific(self):
        self.assertEqual(set(PROFILES), {"kvv", "deepseek", "glm"})
        self.assertNotEqual(PROFILES["kvv"].answer_marker, PROFILES["deepseek"].answer_marker)

    def test_identity_requires_expected_model(self):
        ok, obs = validate_identity(200, {"model": "deepseek-v4-pro"}, ("deepseek-v4",))
        self.assertTrue(ok); self.assertEqual(obs["response_model"], "deepseek-v4-pro")
        self.assertFalse(validate_identity(200, {"model": "other"}, ("deepseek-v4",))[0])

    def test_usage_contract(self):
        self.assertTrue(validate_usage({"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}))
        self.assertFalse(validate_usage({"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 6}}))

    def test_json_object(self):
        self.assertEqual(json_object('{"answer":"OK"}') ["answer"], "OK")
        self.assertIsNone(json_object("not-json"))
