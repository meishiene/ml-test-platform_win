import unittest
from glm53.cases import CASES
from glm53.core import usage_ok, CaseResult, summarize_results

class GLM53Tests(unittest.TestCase):
    def test_case_groups_removed(self):
        self.assertNotIn("GLM_53_ALL", CASES)
        self.assertIn("GLM_53_CORE", CASES)
        self.assertIn("GLM_53_AGENT", CASES)
        self.assertIn("GLM_53_CACHE", CASES)
    def test_usage_contract(self):
        self.assertTrue(usage_ok({"usage": {"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}))
        self.assertFalse(usage_ok({"usage": {"prompt_tokens":1}}))
    def test_fingerprint_assessment_levels(self):
        results = [CaseResult(f"GLM-53-F0{i}", "probe", "glm-5.3", True) for i in range(1, 8)]
        result = summarize_results(results)
        self.assertEqual(result["fingerprint_assessment"]["level"], "满血能力证据充分")

if __name__ == "__main__": unittest.main()
