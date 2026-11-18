from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from adapters.common import AdapterBlocked, repository_revision, utc_now


class MathMeshAdapter:
    """Use MathMesh's exported parity check and documented HTTP endpoints."""

    def __init__(self, repository: Path | None, node: str = "node") -> None:
        self.repository = repository
        self.node = node

    def initialize(self) -> dict[str, Any]:
        if self.repository is None:
            raise AdapterBlocked("Set RECALL_MATHMESH_REPO to a MathMesh checkout.")
        module = self.repository / "governor" / "shared" / "mathChecks.ts"
        if not module.is_file():
            raise FileNotFoundError(f"MathMesh numeric module not found: {module}")
        version = self._node_version()
        if version is None:
            raise AdapterBlocked(f"Node.js executable not found: {self.node}")
        numbers = tuple(int(part) for part in version.lstrip("v").split(".")[:2])
        if numbers < (22, 6):
            raise AdapterBlocked(
                f"MathMesh TypeScript module requires Node.js >=22.6; found {version}."
            )
        return {
            "status": "PASS",
            "timestamp": utc_now(),
            "node_version": version,
            "module_path": str(module.resolve()),
            "repository_commit": repository_revision(self.repository),
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "numeric_operation": "governor/shared/mathChecks.ts:evaluateVoteParity",
            "health_endpoint": "GET /api/health",
            "audit_endpoint": "POST /api/functions/recordAudit",
            "audit_event_types": [
                "GATE_PASS",
                "GATE_HALT",
                "CACHE_HIT",
                "CACHE_STORE",
                "OUTPUT_ATTRIBUTED",
                "PIPELINE_COMPLETE",
            ],
            "unsupported_operations": ["radix conversion", "numeric round-trip"],
        }

    def health(self, base_url: str | None = None) -> dict[str, Any]:
        url = (base_url or os.environ.get("RECALL_MATHMESH_BASE_URL", "")).rstrip("/")
        if not url:
            raise AdapterBlocked("MathMesh API URL is not configured.")
        timestamp = utc_now()
        request = Request(f"{url}/api/health", method="GET")
        try:
            with urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
                if response.status != 200 or body.get("status") != "ok":
                    raise RuntimeError(f"Unexpected MathMesh health response: {body!r}")
                return {
                    "status": "PASS",
                    "endpoint": "/api/health",
                    "http_status": response.status,
                    "timestamp": timestamp,
                    "response": body,
                }
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"MathMesh health request failed: {error}") from error

    def shutdown(self) -> None:
        return None

    def execute(
        self, operation: str, input_metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        if operation != "parity_check":
            raise ValueError(f"Unsupported MathMesh adapter operation: {operation}")
        if input_metadata:
            raise ValueError("The MathMesh parity check does not accept input metadata.")
        return self.test_numeric_operation()

    def test_numeric_operation(self) -> dict[str, Any]:
        if self.repository is None:
            raise AdapterBlocked("Set RECALL_MATHMESH_REPO to a MathMesh checkout.")
        initialized = self.initialize()
        module = self.repository / "governor" / "shared" / "mathChecks.ts"

        script = """
import { pathToFileURL } from 'node:url';
const { evaluateVoteParity } = await import(pathToFileURL(process.argv[1]).href);
const aligned = evaluateVoteParity([2, 4, 6]);
const mismatch = evaluateVoteParity([2, 3]);
process.stdout.write(JSON.stringify({ aligned, mismatch }));
"""
        result = subprocess.run(
            [self.node, "--experimental-strip-types", "--input-type=module", "-e", script, str(module)],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        observed = json.loads(result.stdout)
        if observed.get("aligned", {}).get("aligned") is not True:
            raise ArithmeticError("MathMesh parity operation rejected a same-parity vector")
        if observed.get("mismatch", {}).get("aligned") is not False:
            raise ArithmeticError("MathMesh parity operation accepted a mismatched vector")

        return {
            "status": "PARTIAL",
            "timestamp": utc_now(),
            "repository_commit": initialized["repository_commit"],
            "interface": "governor/shared/mathChecks.ts:evaluateVoteParity",
            "same_parity_vector": observed["aligned"],
            "mixed_parity_vector": observed["mismatch"],
            "unsupported_operations": self.capabilities()["unsupported_operations"],
            "audit_event_delivery": {
                "status": "BLOCKED",
                "attempted": 0,
                "delivered": 0,
                "failed": 0,
                "dropped": None,
                "reason": (
                    "No MathMesh deployment/admin token was configured. The supported "
                    "recordAudit endpoint requires an authenticated remote write."
                ),
            },
            "notes": (
                "This supported check validates parity; it does not implement "
                "base conversion or a round-trip operation."
            ),
        }

    def record_observation(
        self, event_type: str, experiment_id: str, details: str
    ) -> dict[str, Any]:
        """Write a supported observational event only after an explicit opt-in."""
        base_url = os.environ.get("RECALL_MATHMESH_BASE_URL", "").rstrip("/")
        token = os.environ.get("RECALL_MATHMESH_ADMIN_TOKEN")
        if os.environ.get("RECALL_ALLOW_REMOTE_AUDIT_WRITES") != "1":
            raise AdapterBlocked("Remote audit writes require RECALL_ALLOW_REMOTE_AUDIT_WRITES=1.")
        if not base_url or not token:
            raise AdapterBlocked(
                "Remote audit writes require RECALL_MATHMESH_BASE_URL and "
                "RECALL_MATHMESH_ADMIN_TOKEN."
            )
        if not event_type or len(event_type) > 128:
            raise ValueError("MathMesh audit action must contain 1 to 128 characters.")
        timestamp = utc_now()
        payload = {
            "event_type": "PIPELINE_COMPLETE",
            "session_nonce": experiment_id,
            "action": "recall_qualification",
            "details": json.dumps(
                {"operation": event_type, "details": details[:1800]},
                separators=(",", ":"),
            ),
        }
        request = Request(
            f"{base_url}/api/functions/recordAudit",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                body = json.loads(response.read().decode("utf-8"))
                if response.status != 200 or body.get("ok") is not True:
                    raise RuntimeError(f"MathMesh audit write was rejected: {body!r}")
                return {
                    "status": "PASS",
                    "timestamp": timestamp,
                    "experiment_id": experiment_id,
                    "http_status": response.status,
                    "response": body,
                }
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"MathMesh audit write failed: {error}") from error

    def _node_version(self) -> str | None:
        try:
            result = subprocess.run(
                [self.node, "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        version = result.stdout.strip()
        return version if re.fullmatch(r"v?\d+\.\d+\.\d+", version) else None
