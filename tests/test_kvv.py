import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from kvv.cases import CASES
from kvv.runner import DEFAULT_KVV_HOME, SUITE_PATHS, build_command, official_python, parse_junit, validate_runtime


class OfficialKvvTests(unittest.TestCase):
    def test_cases_map_to_official_suites(self):
        self.assertEqual(set(CASES), {"KVV_K3_FEATURES","KVV_PARAMS","KVV_TOOL_SCHEMA"})
        self.assertEqual(CASES["KVV_K3_FEATURES"].request_count,46)
        self.assertEqual(CASES["KVV_PARAMS"].request_count,18)
        self.assertEqual(CASES["KVV_TOOL_SCHEMA"].request_count,408)
        self.assertNotIn("KVV_K3_FINGERPRINT", CASES)

    def test_existing_official_runtime(self):
        python=official_python(DEFAULT_KVV_HOME)
        validate_runtime(DEFAULT_KVV_HOME,python)
        self.assertTrue(python.is_file())

    def test_command_directly_runs_official_pytest_without_key_in_argv(self):
        env={"VOLC_BASE_URL":"https://example.com/v1","VOLC_MODEL":"model-x","VOLC_API_KEY":"secret"}
        with patch.dict(os.environ,env,clear=False):
            cmd=build_command(DEFAULT_KVV_HOME,official_python(DEFAULT_KVV_HOME),"k3_features_core",Path("junit.xml"),Path("tool.json"))
        self.assertEqual(cmd[0],str(DEFAULT_KVV_HOME/".venv/bin/python"))
        self.assertEqual(cmd[1:3],["-m","pytest"])
        self.assertIn("tests/k3_features/test_response_format.py",cmd)
        self.assertNotIn("tests/k3_features/test_dynamic_tools.py",cmd)
        self.assertNotIn("tests/k3_features/test_thinking_effort.py",cmd)
        self.assertNotIn("secret",cmd)

    def test_tool_schema_command_uses_official_parallel_and_report_options(self):
        env={"VOLC_BASE_URL":"https://example.com/v1","VOLC_MODEL":"model-x","VOLC_API_KEY":"secret"}
        with patch.dict(os.environ,env,clear=False):
            cmd=build_command(DEFAULT_KVV_HOME,official_python(DEFAULT_KVV_HOME),"tool_call_json_schema",Path("junit.xml"),Path("tool.json"))
        self.assertIn("-n",cmd); self.assertIn("--selection",cmd); self.assertIn("--thinking",cmd)
        self.assertTrue(any(x.startswith("--tool-json-report=") for x in cmd))

    def test_parse_junit(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"junit.xml"
            root=ET.Element("testsuites")
            suite=ET.SubElement(root,"testsuite",tests="3",failures="1",errors="0",skipped="1",time="1.5")
            ET.SubElement(suite,"testcase",name="ok")
            bad=ET.SubElement(suite,"testcase",name="bad");ET.SubElement(bad,"failure",message="no")
            ET.SubElement(suite,"testcase",name="skip")
            ET.ElementTree(root).write(path)
            result=parse_junit(path,"params")
        self.assertEqual((result["total"],result["passed"],result["failed"],result["skipped"]),(3,1,1,1))

if __name__=="__main__":unittest.main()
