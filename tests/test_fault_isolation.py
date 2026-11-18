from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from recall_harness import run_fault_isolation_checks, run_qualification


class AdapterBoundaryFaultInjectionTests(unittest.TestCase):
    def test_all_injected_failures_are_isolated_and_explicitly_non_live(self) -> None:
        report = run_fault_isolation_checks()

        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["live_outage_tests_run"])
        self.assertEqual(report["scenario_count"], 8)
        self.assertEqual(
            {item["injected_condition"] for item in report["scenarios"]},
            {
                "MathMesh unavailable",
                "System-Stability-Toolkit unavailable",
                "Deterministic RAG unavailable",
                "Zeta initialization fails",
                "Adapter returns malformed data",
                "Service times out",
                "Audit operation fails",
                "Numerical validation fails",
            },
        )
        self.assertTrue(all(item["failure_isolated"] for item in report["scenarios"]))
        self.assertTrue(all(item["execution_continued"] for item in report["scenarios"]))
        self.assertTrue(all(item["evidence_preserved"] for item in report["scenarios"]))
        self.assertTrue(
            all(
                item["evidence_scope"] == "LOCAL UNIT FAULT-INJECTION; NOT LIVE INTEGRATION"
                for item in report["scenarios"]
            )
        )


class FaultIsolationTests(unittest.TestCase):
    def test_unavailable_rag_isolated_from_real_local_probes(self) -> None:
        required = (
            "RECALL_ZETA_REPO",
            "RECALL_MATHMESH_REPO",
            "RECALL_STABILITY_REPO",
        )
        if any(not os.environ.get(name) for name in required):
            self.skipTest("Source checkouts required for real adapter probes are not configured")

        with patch.dict(os.environ, {"RECALL_RAG_BASE_URL": ""}):
            result = run_qualification(write_results=False)

        self.assertEqual(result["systems"]["deterministic_rag_governor"]["status"], "BLOCKED")
        self.assertEqual(result["systems"]["zeta_scaling_toolkit"]["status"], "PASS")
        self.assertEqual(result["systems"]["system_stability_toolkit"]["status"], "PASS")
        self.assertEqual(result["qualification_status"], "BLOCKED")
        self.assertTrue(result["fault_isolation"]["other_adapter_results_retained"])
        self.assertFalse(result["coordination"]["parallel_execution_claimed"])
        self.assertEqual(
            result["workload"]["actual_model_tokens_processed"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
