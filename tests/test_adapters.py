from __future__ import annotations

import os
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from adapters.common import AdapterBlocked
from adapters.mathmesh_adapter import MathMeshAdapter
from adapters.rag_adapter import DeterministicRAGAdapter
from adapters.stability_adapter import StabilityAdapter
from adapters.zeta_adapter import ZetaToolkitAdapter


def _configured_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise unittest.SkipTest(f"{name} is not configured")
    return Path(value)


class RealAdapterTests(unittest.TestCase):
    def test_zeta_published_attention_and_cache_apis(self) -> None:
        repo = _configured_path("RECALL_ZETA_REPO")
        result = ZetaToolkitAdapter(repo, "0.5.1").run_component_probe()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["attention_output_finite"])
        self.assertEqual(result["cache_positions_after_five_appends"], [0, 3, 4])
        self.assertEqual(result["actual_model_tokens_processed"], 0)

    def test_mathmesh_published_numeric_operation(self) -> None:
        repo = _configured_path("RECALL_MATHMESH_REPO")
        result = MathMeshAdapter(repo).test_numeric_operation()
        self.assertEqual(result["status"], "PARTIAL")
        self.assertTrue(result["same_parity_vector"]["aligned"])
        self.assertFalse(result["mixed_parity_vector"]["aligned"])
        self.assertIn("numeric round-trip", result["unsupported_operations"])

    def test_stability_toolkit_documented_json_cli(self) -> None:
        repo = _configured_path("RECALL_STABILITY_REPO")
        result = StabilityAdapter(repo).recommendation(32)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["applied"])
        self.assertEqual(set(result["recommendation"]), {"buffer_window", "parallel_tasks", "task_size"})

    def test_rag_does_not_claim_missing_audit_mode_as_pass(self) -> None:
        adapter = DeterministicRAGAdapter(base_url="")
        result = adapter.audit_only_probe()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["audit_mode"], "UNSUPPORTED")
        self.assertEqual(result["audit_events"], 0)
        self.assertEqual(result["retrieval_augmentation"], "NOT RUN")


class AdapterContractTests(unittest.TestCase):
    class Response:
        status = 200

        def __init__(self, body: dict[str, object]) -> None:
            self.body = json.dumps(body).encode()

        def read(self) -> bytes:
            return self.body

        def __enter__(self) -> "AdapterContractTests.Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def test_mathmesh_audit_write_uses_a_published_observational_event_type(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RECALL_MATHMESH_BASE_URL": "https://mathmesh.example",
                "RECALL_MATHMESH_ADMIN_TOKEN": "test-token",
                "RECALL_ALLOW_REMOTE_AUDIT_WRITES": "1",
            },
        ), patch(
            "adapters.mathmesh_adapter.urlopen",
            return_value=self.Response({"ok": True, "seq": 7}),
        ) as send:
            result = MathMeshAdapter(None).record_observation(
                "parity-check", "experiment-7", "checked"
            )

        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["experiment_id"], "experiment-7")
        self.assertEqual(payload["event_type"], "PIPELINE_COMPLETE")
        self.assertIn("parity-check", payload["details"])

    def test_rag_govern_operation_requires_explicit_opt_in(self) -> None:
        adapter = DeterministicRAGAdapter("https://rag.example")
        with patch.dict(os.environ, {"RECALL_ALLOW_RAG_GOVERN": ""}):
            with self.assertRaisesRegex(AdapterBlocked, "RECALL_ALLOW_RAG_GOVERN=1"):
                adapter.govern_query("test-agent", "test query", "experiment-8")

    def test_rag_govern_records_combined_operation_and_request_id(self) -> None:
        adapter = DeterministicRAGAdapter("https://rag.example")
        with patch.dict(os.environ, {"RECALL_ALLOW_RAG_GOVERN": "1"}), patch(
            "adapters.rag_adapter.urlopen",
            return_value=self.Response(
                {
                    "released": True,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "source_ids": ["doc-1"],
                    "answer": "response body is not retained",
                }
            ),
        ) as send:
            result = adapter.govern_query("test-agent", "test query", "experiment-9")

        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(result["operation"], "retrieval_and_generation")
        self.assertEqual(result["experiment_id"], "experiment-9")
        self.assertEqual(payload["request_id"], "experiment-9")
        self.assertEqual(payload["query"], "test query")
        self.assertEqual(result["source_identifiers"], ["doc-1"])
        self.assertEqual(result["response_metadata"]["timestamp"], "2026-01-01T00:00:00+00:00")
        self.assertNotIn("response", result)


if __name__ == "__main__":
    unittest.main()
