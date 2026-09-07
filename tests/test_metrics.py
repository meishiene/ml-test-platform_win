import unittest
from volcperf.cases import CASES
from volcperf.metrics import percentile, summarize


def row(**changes):
    value = {"counted": True, "success": True, "prompt_tokens": 2048, "completion_tokens": 500,
             "visible_completion_tokens": 500, "finish_reason": "stop", "ttfc_seconds": 1,
             "first_sse_seconds": .1, "first_reasoning_seconds": .2, "ttft_seconds": .2, "e2e_seconds": 5,
             "tpot_ms_per_visible_token": 8, "tpot_token_basis": "visible", "send_delay_ms": None}
    value.update(changes); return value


class MetricTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([1,2,3,4], .5), 2.5)
        self.assertAlmostEqual(percentile([1,2,3,4], .9), 3.7)

    def test_t0_exact_success(self):
        self.assertTrue(summarize(CASES["T0"], [row() for _ in range(10)])["admission_passed"])

    def test_ttft_prefers_first_effective_token(self):
        rows=[row(ttft_seconds=None,first_reasoning_seconds=.4,ttfc_seconds=2.0),
              row(ttft_seconds=None,first_reasoning_seconds=None,ttfc_seconds=1.0)]
        result=summarize(CASES["B1"],rows)
        self.assertAlmostEqual(result["ttft_p50_seconds"],.7)
        self.assertAlmostEqual(result["ttft_p90_seconds"],.94)

    def test_b1_has_four_requests_per_size(self):
        rows=[]
        for dataset,tokens in (("P2K",2048),("P8K",8192),("P16K",16384),("P32K",32768)):
            rows += [row(dataset_id=f"{dataset}-{i:05d}", prompt_tokens=tokens) for i in range(4)]
        result=summarize(CASES["B1"], rows)
        self.assertTrue(result["admission_passed"])
        self.assertEqual({k:v["sent"] for k,v in result["baselines_by_dataset"].items()}, {"P2K":4,"P8K":4,"P16K":4,"P32K":4})

    def test_l1_all_requirements(self):
        self.assertTrue(summarize(CASES["L1"], [row() for _ in range(100)], baseline_p90=1)["admission_passed"])

    def test_failure_not_hidden(self):
        rows = [row() for _ in range(98)] + [row(success=False, error_type="timeout") for _ in range(2)]
        self.assertFalse(summarize(CASES["L1"], rows, baseline_p90=1)["admission_passed"])

    def test_short_outputs_excluded_from_tpot(self):
        result = summarize(CASES["L1"], [row(visible_completion_tokens=100) for _ in range(100)], baseline_p90=1)
        self.assertEqual(result["data_quality"]["tpot_valid_samples"], 0)
        self.assertFalse(result["checks"]["tpot_p90"]["passed"])

    def test_c1_three_windows(self):
        start, rows = 1_000_000.0, []
        for i in range(1890):
            rows.append(row(prompt_tokens=9500, completion_tokens=500, total_tokens=10000,
                            planned_epoch=start+i/10.5, phase="steady", send_delay_ms=1))
        result = summarize(CASES["C1"], rows)
        self.assertEqual(len(result["capacity_windows"]), 3)
        self.assertTrue(result["admission_passed"])

if __name__ == "__main__": unittest.main()
