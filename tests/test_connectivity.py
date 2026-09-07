import unittest
from unittest.mock import patch
from volcperf.config import ApiConfig
from volcperf.connectivity import connectivity_endpoint, connectivity_payload, check_connectivity

class Response:
    def __init__(self,status=200,data=None,text=''):
        self.status_code=status;self._data=data if data is not None else {"ok":True};self.text=text
    def json(self): return self._data

class ConnectivityTests(unittest.TestCase):
    def test_openai_low_effort(self):
        cfg=ApiConfig('k','https://api.example/v1','m',protocol='volcengine')
        self.assertEqual(connectivity_endpoint(cfg),'https://api.example/v1/chat/completions')
        body=connectivity_payload(cfg);self.assertEqual(body['max_tokens'],32);self.assertEqual(body['reasoning_effort'],'low')

    def test_gemini_nonstream_endpoint(self):
        cfg=ApiConfig('k','https://google/v1beta','gemini/x',protocol='gemini')
        self.assertEqual(connectivity_endpoint(cfg),'https://google/v1beta/models/gemini%2Fx:generateContent')

    def test_success_does_not_require_visible_content(self):
        cfg=ApiConfig('k','https://api.example/v1','m',protocol='openai')
        fake=type('Client',(),{'__enter__':lambda s:s,'__exit__':lambda *a:None,'post':lambda *a,**k:Response(data={"choices":[{"message":{"content":""}}]})})
        with patch('volcperf.connectivity.httpx.Client',return_value=fake()): result=check_connectivity(cfg)
        self.assertTrue(result['passed'])

    def test_http_failure(self):
        cfg=ApiConfig('k','https://api.example/v1','m',protocol='openai')
        fake=type('Client',(),{'__enter__':lambda s:s,'__exit__':lambda *a:None,'post':lambda *a,**k:Response(401,text='bad key')})
        with patch('volcperf.connectivity.httpx.Client',return_value=fake()): result=check_connectivity(cfg)
        self.assertFalse(result['passed']);self.assertIn('401',result['error'])

if __name__=='__main__':unittest.main()
