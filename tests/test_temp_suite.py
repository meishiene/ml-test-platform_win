import unittest
from collections import Counter

from common.temp_suite import find_case, load_manifest, manifest_cases, merge_cases
from web import server


class TempSuiteMergeTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest()
        self.atomic = manifest_cases(self.manifest)
        self.merged = merge_cases(self.atomic)

    def test_manifest_keeps_atomic_rows(self):
        # 清单本身仍是逐条原子案例：合并只发生在案例集与汇总层。
        self.assertEqual(len(self.atomic), 68)

    def test_same_category_and_batch_are_merged(self):
        self.assertEqual(len(self.merged), 24)
        counts = Counter(case.repetitions for case in self.merged)
        self.assertEqual(counts, Counter({1: 18, 10: 4, 7: 1, 3: 1}))

    def test_merged_ids_use_batch_name(self):
        ids = {case.id for case in self.merged}
        for expected in ("16K-10", "64K-10", "100K-10", "TPOT-10", "CACHE-HIT-10", "CACHE-TTL-3"):
            self.assertIn(expected, ids)
        # 没有 batch 的单条场景保持原子 id，且不与他条合并。
        self.assertIn("K3-CONN-001", ids)
        self.assertIn("K3-OBS-001", ids)

    def test_merged_case_expands_back_to_atomic_order(self):
        case = find_case(self.merged, "16K-10")
        self.assertIsNotNone(case)
        self.assertTrue(case.merged)
        self.assertEqual(case.category, "TTFT")
        self.assertEqual(list(case.case_ids), [f"K3-TTFT-{index:02d}" for index in range(1, 11)])
        # 每条原子请求都在清单里真实存在，合并不会吞掉请求。
        self.assertTrue(set(case.case_ids) <= {item["id"] for item in self.atomic})

    def test_merged_name_drops_repetition_suffix(self):
        case = find_case(self.merged, "TPOT-10")
        self.assertEqual(case.name, "流式生成 TPOT")

    def test_select_by_merged_or_atomic_id(self):
        by_merged = find_case(self.merged, "64K-10")
        by_atomic = find_case(self.merged, "K3-TTFT-12")
        self.assertIsNotNone(by_merged)
        self.assertIs(by_merged, by_atomic)
        self.assertIsNone(find_case(self.merged, "K3-NOT-EXIST"))
        self.assertIsNone(find_case(self.merged, None))

    def test_merge_is_pure(self):
        # 合并只读清单，不修改原始条目（执行器仍要拿到原始 request）。
        snapshot = [dict(item) for item in self.atomic]
        merge_cases(self.atomic)
        self.assertEqual(self.atomic, snapshot)

    def test_web_payload_reports_merged_count(self):
        payload = server.temporary_case_payload()
        self.assertEqual(payload["case_count"], 24)
        self.assertEqual(payload["atomic_case_count"], 68)
        self.assertEqual(len(payload["case_options"]), 24)
        options = {option["id"]: option for option in payload["case_options"]}
        self.assertEqual(options["16K-10"]["repetitions"], 10)
        self.assertTrue(options["16K-10"]["merged"])
        self.assertFalse(options["K3-CONN-001"]["merged"])


if __name__ == "__main__":
    unittest.main()
