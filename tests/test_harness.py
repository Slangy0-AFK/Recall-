from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recall_harness import (
    EvidenceValidationError,
    _reference_model_cached,
    _run_adapter,
    render_reports,
    run_qualification,
    validate_evidence,
)


class ReportShapeTests(unittest.TestCase):
    def test_report_names_and_required_claim_boundaries(self) -> None:
        result = {
            "experiment_id": "test-experiment",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:00:01+00:00",
            "commit_identifiers": {
                "recall": {"commit": "recall-sha"},
                "connected_repositories": {
                    name: {"observed_revision": "source-sha", "expected_revision": "expected-sha"}
                    for name in (
                        "zeta-scaling-toolkit",
                        "mathmesh-governor",
                        "System-Stability-Toolkit",
                        "deterministic-rag-governor",
                    )
                },
            },
            "model_provenance": {
                "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
                "revision": "a10cc1512eabd3dde888204e902eca88bddb4951",
                "historical_workload_tokens": 100000,
                "historical_throughput_tokens_per_second": 10.7199,
                "historical_wall_seconds": 9636.7,
                "historical_final_token_position": 99999,
                "historical_checkpoints": "56/56 passed",
                "target_throughput_tokens_per_second": 21.4398,
                "tokenizer_provenance": {
                    "tokenizer_id": "NOT LOADED",
                    "note": "No tokenizer was instantiated.",
                },
            },
            "workload": {
                "description": "attention probe only",
                "actual_model_tokens_processed": 0,
            },
            "hardware_runtime": {
                "platform": "test-platform",
                "python": "3.14.2",
                "torch_runtime": "2.15.0",
                "node": "v24.22.2",
            },
            "qualification_status": "BLOCKED",
            "production_readiness": "NOT ASSESSED",
            "phases": {
                "1_dependency_environment_validation": "PARTIAL",
                "3_four_system_lightweight_integration": "BLOCKED",
            },
            "systems": {
                name: {"status": "BLOCKED", "details": {"reason": "not configured"}}
                for name in (
                    "zeta_scaling_toolkit",
                    "mathmesh_governor",
                    "system_stability_toolkit",
                    "deterministic_rag_governor",
                )
            },
            "performance": {
                "target_reached": None,
                "total_harness_elapsed_seconds": 1.0,
                "model_weights_cached": True,
                "reason": "no model runner",
            },
            "coordination": {
                "native_component_events_received": 0,
            },
            "fault_isolation": {"status": "PARTIAL"},
            "environment_validation": {
                "status": "PARTIAL",
                "installed_versions": {},
                "checks": {},
                "rag_governor_pinned_runtime": {"missing_or_mismatched": []},
            },
            "integrity": {"status": "NOT TESTED"},
        }
        documents = render_reports(result)
        self.assertEqual(
            set(documents),
            {
                "RECALL_TEST_RESULTS.md",
                "FOUR_SYSTEM_INTEGRATION_REPORT.md",
                "FAULT_ISOLATION_REPORT.md",
                "PERFORMANCE_COMPARISON.md",
                "ENVIRONMENT_AND_DEPENDENCY_REPORT.md",
                "RECALL_COMPLETION_REPORT.md",
            },
        )
        for document in documents.values():
            self.assertIn("test-experiment", document)
            self.assertIn("a10cc1512eabd3dde888204e902eca88bddb4951", document)
            self.assertIn("actual model tokens: 0", document)
            self.assertIn("Timestamp:", document)
            self.assertIn("Recall commit:", document)
            self.assertIn("Hardware/runtime:", document)
            self.assertIn("Test status:", document)
        self.assertIn(
            "- Pinned model weights cached locally: True",
            documents["PERFORMANCE_COMPARISON.md"],
        )

    def test_evidence_validator_rejects_mismatched_experiment_ids(self) -> None:
        result = {
            "schema_version": 1,
            "experiment_id": "expected-id",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:00:01+00:00",
            "systems": {
                name: {"status": "BLOCKED"}
                for name in (
                    "zeta_scaling_toolkit",
                    "mathmesh_governor",
                    "system_stability_toolkit",
                    "deterministic_rag_governor",
                )
            },
            "workload": {"actual_model_tokens_processed": 0},
            "events": [{"experiment_id": "different-id"}],
            "qualification_status": "BLOCKED",
            "commit_identifiers": {},
        }
        with self.assertRaisesRegex(EvidenceValidationError, "mismatched experiment_id"):
            validate_evidence(result)

    def test_native_events_are_counted_only_when_experiment_and_system_match(self) -> None:
        event = {
            "experiment_id": "shared-id",
            "system": "mathmesh_governor",
            "component": "audit",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "operation": "record",
            "status": "PASS",
        }
        events: list[dict[str, object]] = []
        result = _run_adapter(
            "mathmesh_governor",
            lambda: {"status": "PASS", "native_events": [event]},
            events,
            "shared-id",
        )
        self.assertEqual(result["native_component_events_received"], 1)
        self.assertTrue(events[0]["native_component_event"])

        invalid_events: list[dict[str, object]] = []
        invalid = _run_adapter(
            "mathmesh_governor",
            lambda: {
                "status": "PASS",
                "native_events": [{**event, "experiment_id": "wrong-id"}],
            },
            invalid_events,
            "shared-id",
        )
        self.assertEqual(invalid["status"], "FAIL")
        self.assertEqual(invalid["native_component_events_received"], 0)
        self.assertFalse(invalid_events[0]["native_component_event"])

        invalid_timestamp_events: list[dict[str, object]] = []
        invalid_timestamp = _run_adapter(
            "mathmesh_governor",
            lambda: {
                "status": "PASS",
                "native_events": [{**event, "timestamp": "not-a-timestamp"}],
            },
            invalid_timestamp_events,
            "shared-id",
        )
        self.assertEqual(invalid_timestamp["status"], "FAIL")
        self.assertEqual(invalid_timestamp["native_component_events_received"], 0)

    def test_evidence_validator_prevents_success_with_blocked_gate(self) -> None:
        result = {
            "schema_version": 1,
            "experiment_id": "expected-id",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:00:01+00:00",
            "systems": {
                name: {"status": "PASS"}
                for name in (
                    "zeta_scaling_toolkit",
                    "mathmesh_governor",
                    "system_stability_toolkit",
                    "deterministic_rag_governor",
                )
            },
            "workload": {"actual_model_tokens_processed": 0},
            "events": [{"experiment_id": "expected-id"}],
            "qualification_status": "PASS",
            "commit_identifiers": {},
        }
        result["systems"]["deterministic_rag_governor"]["status"] = "BLOCKED"
        with self.assertRaisesRegex(EvidenceValidationError, "cannot be PASS"):
            validate_evidence(result)

    def test_selected_system_run_has_complete_records_and_blocks_global_gate(self) -> None:
        empty_environment = {
            "RECALL_ZETA_REPO": "",
            "RECALL_MATHMESH_REPO": "",
            "RECALL_STABILITY_REPO": "",
            "RECALL_RAG_REPO": "",
            "RECALL_MATHMESH_BASE_URL": "",
            "RECALL_RAG_BASE_URL": "",
        }
        with patch.dict(os.environ, empty_environment):
            result = run_qualification(
                write_results=False,
                selected_systems={"mathmesh_governor"},
            )

        validate_evidence(result)
        self.assertEqual(result["systems"]["mathmesh_governor"]["operation"], "parity_check")
        self.assertEqual(result["systems"]["zeta_scaling_toolkit"]["status"], "NOT TESTED")
        self.assertEqual(result["qualification_status"], "BLOCKED")
        for system in result["systems"].values():
            self.assertEqual(system["experiment_id"], result["experiment_id"])
            self.assertIn("repository_commit", system)
            self.assertIn("elapsed_seconds", system)

    def test_model_index_without_weight_shards_is_not_cached(self) -> None:
        revision = "a10cc1512eabd3dde888204e902eca88bddb4951"
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = (
                Path(temporary)
                / "hub"
                / "models--HuggingFaceTB--SmolLM2-360M-Instruct"
                / "snapshots"
                / revision
            )
            snapshot.mkdir(parents=True)
            (snapshot / "pytorch_model.bin.index.json").write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"HF_HOME": temporary}):
                self.assertFalse(_reference_model_cached(revision))

            (snapshot / "pytorch_model-00001-of-00001.bin").write_bytes(b"weights")
            with patch.dict(os.environ, {"HF_HOME": temporary}):
                self.assertTrue(_reference_model_cached(revision))


if __name__ == "__main__":
    unittest.main()
