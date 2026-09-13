import json,tempfile,unittest
from pathlib import Path
from common.finalizer import finalize_result_dir
class FinalizerTests(unittest.TestCase):
 def test_warmup_only_load_cannot_pass(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td); (p/'summary.json').write_text(json.dumps({'admission_passed':True,'sent':0,'checks':{}})); (p/'requests.jsonl').write_text('{}\n')
   x=finalize_result_dir(p); self.assertFalse(x['admission_passed']); self.assertTrue((p/'validation.json').exists())
 def test_external_without_scores_cannot_pass(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td); (p/'summary.json').write_text(json.dumps({'admission_passed':True,'scored_tasks':0})); (p/'run.log').write_text('ran')
   self.assertFalse(finalize_result_dir(p)['admission_passed'])
