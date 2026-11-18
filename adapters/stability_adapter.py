from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from adapters.common import AdapterBlocked, repository_revision, utc_now


class StabilityAdapter:
    """Invoke the documented System-Stability-Toolkit JSON CLI without applying advice."""

    def __init__(self, repository: Path | None) -> None:
        self.repository = repository

    def initialize(self) -> dict[str, Any]:
        if self.repository is None:
            raise AdapterBlocked(
                "Set RECALL_STABILITY_REPO to a System-Stability-Toolkit checkout."
            )
        script = self.repository / "crash_fix.py"
        if not script.is_file():
            raise FileNotFoundError(f"System-Stability-Toolkit CLI not found: {script}")
        return {
            "status": "PASS",
            "timestamp": utc_now(),
            "cli_path": str(script.resolve()),
            "repository_commit": repository_revision(self.repository),
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "advisory_cli": "crash_fix.py <scale> --json",
            "resource_measurements": "Recall via psutil",
            "recommendations_applied": False,
            "temperature_or_cooling_measurement": "NOT IMPLEMENTED",
        }

    def health(self) -> dict[str, Any]:
        initialized = self.initialize()
        result = subprocess.run(
            [sys.executable, initialized["cli_path"], "--help"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "status": "PASS",
            "timestamp": utc_now(),
            "return_code": result.returncode,
            "repository_commit": initialized["repository_commit"],
        }

    def shutdown(self) -> None:
        return None

    def execute(
        self, operation: str, input_metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        if operation != "advisory_recommendation":
            raise ValueError(f"Unsupported stability adapter operation: {operation}")
        workload_scale = input_metadata.get("workload_scale")
        if (
            isinstance(workload_scale, bool)
            or not isinstance(workload_scale, int)
            or workload_scale < 1
        ):
            raise ValueError("workload_scale must be a positive whole number.")
        return self.recommendation(workload_scale)

    def resource_snapshot(self) -> dict[str, Any]:
        import psutil

        memory = psutil.virtual_memory()
        process = psutil.Process()
        try:
            import resource
        except ImportError:
            peak_process_memory_mb = None
        else:
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_process_memory_mb = (
                peak_rss / (1024 * 1024)
                if sys.platform == "darwin"
                else peak_rss / 1024
            )
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp_epoch": time.time(),
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "memory_used_mb": memory.used / (1024 * 1024),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "disk_free_gb": psutil.disk_usage("/").free / (1024**3),
            "process_memory_mb": process.memory_info().rss / (1024 * 1024),
            "process_peak_memory_mb": peak_process_memory_mb,
            "measurement_source": "Recall via psutil/resource; not measured by the toolkit",
        }

    def recommendation(self, workload_scale: int) -> dict[str, Any]:
        initialized = self.initialize()
        script = Path(initialized["cli_path"])
        result = subprocess.run(
            [sys.executable, str(script), str(workload_scale), "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        recommendation = json.loads(result.stdout)
        required = {"buffer_window", "parallel_tasks", "task_size"}
        if not required.issubset(recommendation) or any(
            isinstance(recommendation[key], bool)
            or not isinstance(recommendation[key], int)
            or recommendation[key] < 1
            for key in required
        ):
            raise ValueError(f"Unexpected stability recommendation: {recommendation!r}")
        return {
            "status": "PASS",
            "timestamp": utc_now(),
            "repository_commit": initialized["repository_commit"],
            "interface": "crash_fix.py <workload> --json",
            "recommendation": recommendation,
            "applied": False,
            "actual_workload": {
                "recommendation_requests": 1,
                "recommendations_applied": 0,
            },
            "temperature_or_cooling_measurement": "NOT IMPLEMENTED",
            "notes": "Advice is recorded only; no workload or resource limits are changed.",
        }
