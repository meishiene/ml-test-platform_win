import json, tempfile, unittest
from pathlib import Path
from common.artifacts import SCHEMA_VERSION, result_envelope
from common.test_framework import ExecutionRecord, validate_case
from common.validators import status, status_error

class CommonFrameworkTests(unittest.TestCase):
    def test_expected_4xx_is_not_transport_failure(self):
        r=ExecutionRecord('x','rid',{}, {}, 'now', 0, http_status=400,
                          response_body={'error': {'type':'invalid_request_error','message':'bad temperature'}})
        v=validate_case(r,[status(400), status_error(400,error_types=('invalid_request_error',),message_contains=('temperature',))])
        self.assertTrue(v.passed); self.assertEqual(v.status,'passed')
    def test_missing_evidence_fails(self):
        r=ExecutionRecord('x','rid',{}, {}, 'now', 0, http_status=200)
        v=validate_case(r,[status(200),status_error(200)])
        self.assertFalse(v.passed)
    def test_schema(self):
        x=result_envelope(record_id='r',suite='s',model='m',protocol='openai',records=[])
        self.assertEqual(x['schema_version'],SCHEMA_VERSION)
        self.assertEqual(json.loads(json.dumps(x))['suite'],'s')

class ValidatorContractTests(unittest.TestCase):
    def test_expected_error_requires_parseable_evidence(self):
        from common.validators import status_error
        r=ExecutionRecord('x','rid',{}, {}, 'now', 0, http_status=400, response_text='')
        self.assertFalse(validate_case(r,[status_error(400,error_types=('invalid',))]).passed)

    def test_percentile_reports_insufficient_samples(self):
        from common.validators import aggregate_percentile
        r=aggregate_percentile('ttft_p95',[10],p=.95,maximum=100,minimum_samples=2)
        self.assertFalse(r.passed)
