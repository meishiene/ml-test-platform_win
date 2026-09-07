import json
import tempfile
import unittest
from pathlib import Path
from web.aggregation import aggregate_batch,enrich_summary,SCHEMA_VERSION

class AggregationTests(unittest.TestCase):
    def test_enrich_and_aggregate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); child=root/'child';child.mkdir();summary=child/'summary.json'
            summary.write_text(json.dumps({'sent':10,'success_rate':1,'ttft_p90_seconds':1.25,'ttfc_p90_seconds':3.5,'admission_passed':True}),encoding='utf-8')
            enriched=enrich_summary(summary,record_id='r1',batch_id='b1',test_kind='load',case_name='T0',protocol='volcengine',base_url='https://x',model='m')
            self.assertEqual(enriched['schema_version'],SCHEMA_VERSION);self.assertEqual(enriched['record_id'],'r1')
            out=root/'batch';batch={'id':'b1','test_kind':'load','protocol':'volcengine','base_url':'https://x','model':'m'}
            childrow={'id':'r1','case_name':'T0','status':'completed','passed':1,'result_dir':str(child),'report_path':str(child/'report.md'),'summary_path':str(summary)}
            result=aggregate_batch(out,batch,[childrow])
            self.assertTrue(result['admission_passed']);self.assertEqual(result['child_record_ids'],['r1'])
            self.assertTrue((out/'aggregation_manifest.json').exists());report=(out/'report.md').read_text();self.assertIn('r1',report);self.assertIn('TTFT P90',report);self.assertIn('1.250',report)

if __name__=='__main__':unittest.main()
