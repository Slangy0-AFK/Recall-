from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from adapters.mathmesh_adapter import MathMeshAdapter
from adapters.rag_adapter import DeterministicRAGAdapter
from adapters.stability_adapter import StabilityAdapter
from adapters.zeta_adapter import ZetaToolkitAdapter


ROOT = Path(__file__).resolve().parents[1]


class IntegrationPrerequisiteTests(unittest.TestCase):
    def test_public_adapter_lifecycle_and_integration_contracts_are_declared(self) -> None:
        adapters = (
            ZetaToolkitAdapter(None, "0.5.1"),
            MathMeshAdapter(None),
            StabilityAdapter(None),
            DeterministicRAGAdapter(base_url=""),
        )
        for adapter in adapters:
            for method_name in (
                "initialize",
                "capabilities",
                "health",
                "execute",
                "shutdown",
            ):
                self.assertTrue(callable(getattr(adapter, method_name)))

        capabilities = json.loads(
            (ROOT / "config" / "capabilities.json").read_text(encoding="utf-8")
        )
        integration = json.loads(
            (ROOT / "config" / "integrations.json").read_text(encoding="utf-8")
        )
        self.assertIn("NOT APPLICABLE", capabilities["status_vocabulary"])
        self.assertIn(
            "all four systems emit native events correlated to the shared experiment ID",
            integration["integration_gate_requirements"],
        )
        self.assertIn(
            "metadata-only audit endpoint",
            capabilities["systems"]["deterministic_rag_governor"][
                "unsupported_capabilities"
            ],
        )

    def test_metadata_only_rag_audit_is_blocked_without_a_supported_endpoint(self) -> None:
        adapter = DeterministicRAGAdapter(base_url="")
        result = adapter.audit_only_probe()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["failed_or_dropped_events"], "NOT TESTED")

    def test_optional_read_only_service_health_endpoints(self) -> None:
        mathmesh_url = os.environ.get("RECALL_MATHMESH_BASE_URL", "")
        rag_url = os.environ.get("RECALL_RAG_BASE_URL", "")
        if not mathmesh_url or not rag_url:
            self.skipTest("Read-only MathMesh and RAG health endpoints are not both configured")

        mathmesh = MathMeshAdapter(None).health(mathmesh_url)
        rag = DeterministicRAGAdapter(rag_url).audit_only_probe()
        self.assertEqual(mathmesh["status"], "PASS")
        self.assertEqual(rag["status"], "PARTIAL")
        self.assertEqual(rag["audit_mode"], "UNSUPPORTED")


if __name__ == "__main__":
    unittest.main()
