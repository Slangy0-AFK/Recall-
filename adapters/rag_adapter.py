from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from adapters.common import AdapterBlocked, utc_now


class DeterministicRAGAdapter:
    """Use published RAG health/govern endpoints with explicit write-side-effect gates."""

    def __init__(self, base_url: str | None = None) -> None:
        configured_url = (
            base_url if base_url is not None else os.environ.get("RECALL_RAG_BASE_URL", "")
        )
        self.base_url = configured_url.rstrip("/")

    def initialize(self) -> dict[str, Any]:
        if not self.base_url:
            raise AdapterBlocked("RAG service URL is not configured.")
        return {
            "status": "PASS",
            "timestamp": utc_now(),
            "base_url": self.base_url,
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "health": "GET /health",
            "readiness": "GET /ready",
            "retrieval_and_generation": "POST /govern (combined, state-changing)",
            "audit_chain_validation": "GET /health returns chain_valid",
            "audit_only_endpoint": "NOT EXPOSED",
            "retrieval_only_endpoint": "NOT EXPOSED",
            "generation_only_endpoint": "NOT EXPOSED",
        }

    def health(self) -> dict[str, Any]:
        self.initialize()
        request = Request(f"{self.base_url}/health", method="GET")
        try:
            with urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
                if response.status != 200 or body.get("chain_valid") is not True:
                    raise RuntimeError(f"RAG health/audit-chain check failed: {body!r}")
                return {
                    "status": "PASS",
                    "endpoint": "/health",
                    "http_status": response.status,
                    "timestamp": utc_now(),
                    "response": body,
                }
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"RAG health request failed: {error}") from error

    def readiness(self) -> dict[str, Any]:
        self.initialize()
        request = Request(f"{self.base_url}/ready", method="GET")
        try:
            with urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
                if response.status != 200 or body.get("ready") is not True:
                    raise RuntimeError(f"Unexpected RAG readiness response: {body!r}")
                return {
                    "status": "PASS",
                    "endpoint": "/ready",
                    "http_status": response.status,
                    "timestamp": utc_now(),
                    "response": body,
                }
        except HTTPError as error:
            if error.code == 503:
                return {
                    "status": "BLOCKED",
                    "endpoint": "/ready",
                    "http_status": error.code,
                    "timestamp": utc_now(),
                    "reason": error.read().decode("utf-8", errors="replace"),
                }
            raise RuntimeError(f"RAG readiness request failed: {error}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"RAG readiness request failed: {error}") from error

    def shutdown(self) -> None:
        return None

    def execute(
        self, operation: str, input_metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        if operation == "audit_capability_probe":
            if input_metadata:
                raise ValueError("The RAG capability probe does not accept input metadata.")
            return self.audit_only_probe()
        if operation == "govern":
            agent_id = input_metadata.get("agent_id")
            query = input_metadata.get("query")
            experiment_id = input_metadata.get("experiment_id")
            if not all(
                isinstance(value, str) for value in (agent_id, query, experiment_id)
            ):
                raise ValueError("The RAG govern operation requires string input fields.")
            return self.govern_query(agent_id, query, experiment_id)
        raise ValueError(f"Unsupported RAG adapter operation: {operation}")

    def audit_only_probe(self) -> dict[str, Any]:
        if not self.base_url:
            return self._blocked_probe(
                "No RAG service URL is configured. The repository has no documented "
                "metadata-only audit endpoint; /govern may retrieve and generate."
            )
        health = self.health()
        return {
            "status": "PARTIAL",
            "timestamp": utc_now(),
            "interface": "GET /health",
            "http_status": health["http_status"],
            "health_response": health["response"],
            "audit_mode": "UNSUPPORTED",
            "retrieval_augmentation": "NOT TESTED",
            "query_events": 0,
            "audit_events": 0,
            "event_timestamps": [],
            "evidence_references": [],
            "failed_or_dropped_events": "NOT TESTED",
            "notes": (
                "The repository documents a governed /govern request path that may "
                "retrieve and generate. It does not expose a metadata-only audit mode."
            ),
        }

    def govern_query(
        self, agent_id: str, query: str, experiment_id: str
    ) -> dict[str, Any]:
        if os.environ.get("RECALL_ALLOW_RAG_GOVERN") != "1":
            raise AdapterBlocked(
                "POST /govern can retrieve, generate, spend agent credit, and write audit "
                "events; set RECALL_ALLOW_RAG_GOVERN=1 to explicitly allow it."
            )
        if not self.base_url:
            raise AdapterBlocked("RAG service URL is not configured.")
        if not agent_id.strip() or not query.strip() or not experiment_id.strip():
            raise ValueError("agent_id, query, and experiment_id must be non-empty.")
        timestamp = utc_now()
        payload = {
            "agent_id": agent_id,
            "query": query,
            "request_id": experiment_id,
        }
        request = Request(
            f"{self.base_url}/govern",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
                if response.status != 200 or not isinstance(body, dict):
                    raise RuntimeError(f"RAG governed request returned invalid data: {body!r}")
                response_fields = (
                    "released",
                    "reason",
                    "request_id",
                    "timestamp",
                    "created_at",
                    "sources",
                    "source_ids",
                    "source_identifiers",
                )
                response_metadata = {
                    key: body[key] for key in response_fields if key in body
                }
                return {
                    "status": "PASS",
                    "timestamp": timestamp,
                    "experiment_id": experiment_id,
                    "operation": "retrieval_and_generation",
                    "http_status": response.status,
                    "response_fields": sorted(body),
                    "response_metadata": response_metadata,
                    "source_identifiers": response_metadata.get(
                        "source_identifiers",
                        response_metadata.get("source_ids", response_metadata.get("sources", "NOT EXPOSED")),
                    ),
                }
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"RAG governed request failed: {error}") from error

    @staticmethod
    def _blocked_probe(reason: str) -> dict[str, Any]:
        return {
            "status": "BLOCKED",
            "interface": "GET /health",
            "reason": reason,
            "audit_mode": "UNSUPPORTED",
            "retrieval_augmentation": "NOT RUN",
            "query_events": 0,
            "audit_events": 0,
            "event_timestamps": [],
            "evidence_references": [],
            "failed_or_dropped_events": "NOT TESTED",
        }

    @staticmethod
    def unsupported_audit_mode() -> None:
        raise AdapterBlocked(
            "The connected RAG repository has no documented metadata-only audit endpoint; "
            "calling /govern would invoke retrieval/generation."
        )
