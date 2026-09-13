import unittest
from common.artifacts import SCHEMA_VERSION
from common.result_adapter import normalize_summary
class ResultAdapterTests(unittest.TestCase):
 def test_not_evaluable_blocks_admission(self):
  x=normalize_summary({'schema_version':'old','admission_passed':True,'results':[{'status':'not_evaluable'}]})
  self.assertFalse(x['admission_passed']); self.assertEqual(x['schema_version'],SCHEMA_VERSION)
 def test_legacy_pass_is_explicit(self):
  x=normalize_summary({'results':[{'passed':True,'status':'passed'}]})
  self.assertTrue(x['results'][0]['case_passed'])
