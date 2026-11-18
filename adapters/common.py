from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol


STATUSES = {
    "PASS",
    "FAIL",
    "PARTIAL",
    "BLOCKED",
    "NOT TESTED",
    "NOT APPLICABLE",
}


class AdapterBlocked(RuntimeError):
    """A supported adapter operation could not run because a prerequisite is absent."""


class AdapterLifecycle(Protocol):
    def initialize(self) -> dict[str, Any]: ...

    def capabilities(self) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]: ...

    def execute(
        self, operation: str, input_metadata: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repository_revision(path: Path | None) -> str | None:
    if path is None or not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def checked_status(status: str) -> str:
    if status not in STATUSES:
        raise ValueError(f"Unknown qualification status: {status}")
    return status
