from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapters.common import AdapterBlocked, STATUSES, repository_revision
from adapters.mathmesh_adapter import MathMeshAdapter
from adapters.rag_adapter import DeterministicRAGAdapter
from adapters.stability_adapter import StabilityAdapter
from adapters.zeta_adapter import ZetaToolkitAdapter


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "experiment.json"
STATUS_ORDER = {
    "FAIL": 6,
    "BLOCKED": 5,
    "PARTIAL": 4,
    "NOT TESTED": 3,
    "NOT APPLICABLE": 2,
    "PASS": 1,
}
SYSTEM_ORDER = (
    "zeta_scaling_toolkit",
    "mathmesh_governor",
    "system_stability_toolkit",
    "deterministic_rag_governor",
)
REPOSITORY_ORDER = (
    "zeta-scaling-toolkit",
    "mathmesh-governor",
    "System-Stability-Toolkit",
    "deterministic-rag-governor",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _node_at_least(major: int, minor: int) -> bool:
    version = _node_version()
    if not version:
        return False
    try:
        parts = version.lstrip("v").split(".")
        observed = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return False
    return observed >= (major, minor)


def _node_version() -> str | None:
    try:
        result = subprocess.run(
            ["node", "--version"], check=True, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _hardware_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "node": _node_version(),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        result["memory_total_mb"] = memory.total / (1024 * 1024)
        result["memory_available_mb_at_report"] = memory.available / (1024 * 1024)
        result["disk_free_gb_at_report"] = psutil.disk_usage("/").free / (1024**3)
    except ImportError:
        result["resource_probe"] = "BLOCKED: psutil is not installed"
    try:
        import torch

        result["torch_runtime"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        result["torch_runtime"] = None
        result["cuda_available"] = None
    return result


def _repo_metadata(config: dict[str, Any], paths: dict[str, Path | None]) -> dict[str, Any]:
    return {
        "recall": {
            "commit": _git_revision(ROOT),
            "branch": _git_branch(ROOT),
            "license": config["repository_inspection"]["recall_license"],
            "working_tree_at_phase1": config["repository_inspection"][
                "recall_initial_working_tree"
            ],
            "working_tree_at_run": _git_status(ROOT),
        },
        "connected_repositories": {
            name: {
                "url": details["url"],
                "default_branch": details["default_branch"],
                "license": details["license"],
                "expected_revision": details["revision"],
                "observed_revision": repository_revision(paths.get(name)),
                "checkout_path": str(paths[name]) if paths.get(name) else None,
            }
            for name, details in config["repositories"].items()
        },
    }


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_branch(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_status(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {type(error).__name__}"
    return result.stdout.strip() or "clean"


def validate_environment(
    config: dict[str, Any], paths: dict[str, Path | None]
) -> dict[str, Any]:
    versions = {
        "torch": _installed_version("torch"),
        "psutil": _installed_version("psutil"),
        "pytest": _installed_version("pytest"),
        "zeta-scaling-toolkit": _installed_version("zeta-scaling-toolkit"),
        "fastapi": _installed_version("fastapi"),
        "uvicorn": _installed_version("uvicorn"),
        "pydantic": _installed_version("pydantic"),
        "python-dotenv": _installed_version("python-dotenv"),
    }
    expected = config["dependencies"]
    zeta_path = paths["zeta-scaling-toolkit"]
    zeta_source_version: str | None = None
    zeta_source_revision = repository_revision(zeta_path)
    if zeta_path is not None:
        project_file = zeta_path / "pyproject.toml"
        if project_file.is_file():
            match = re.search(
                r'(?m)^version\s*=\s*["\']([^"\']+)["\']',
                project_file.read_text(encoding="utf-8"),
            )
            zeta_source_version = match.group(1) if match else None
    try:
        zeta_importable = importlib.util.find_spec("zeta_scaling_toolkit") is not None
    except (ImportError, ValueError):
        zeta_importable = False
    zeta_from_pinned_checkout = (
        zeta_source_version == expected["zeta_scaling_toolkit"]
        and zeta_source_revision == config["repositories"]["zeta-scaling-toolkit"]["revision"]
        and zeta_path is not None
        and (zeta_path / "zeta_scaling_toolkit" / "__init__.py").is_file()
    )
    checks = {
        "python_minimum": {
            "expected": expected["python_minimum"],
            "observed": platform.python_version(),
            "passed": sys.version_info >= (3, 10),
        },
        "torch": {
            "expected": expected["torch"],
            "observed": versions["torch"],
            "passed": versions["torch"] == expected["torch"],
        },
        "psutil": {
            "expected": expected["psutil"],
            "observed": versions["psutil"],
            "passed": versions["psutil"] == expected["psutil"],
        },
        "pytest": {
            "expected": expected["pytest"],
            "observed": versions["pytest"],
            "passed": versions["pytest"] == expected["pytest"],
        },
        "zeta-scaling-toolkit": {
            "expected": expected["zeta_scaling_toolkit"],
            "observed_distribution": versions["zeta-scaling-toolkit"],
            "observed_source_version": zeta_source_version,
            "observed_source_revision": zeta_source_revision,
            "module_importable_without_checkout": zeta_importable,
            "passed": zeta_from_pinned_checkout
            or (
                zeta_importable
                and versions["zeta-scaling-toolkit"] == expected["zeta_scaling_toolkit"]
            ),
        },
        "node": {
            "expected": ">=22.6 for direct TypeScript module loading",
            "observed": _node_version(),
            "passed": _node_at_least(22, 6),
        },
    }
    repository_checks: dict[str, dict[str, Any]] = {}
    for name, details in config["repositories"].items():
        path = paths.get(name)
        observed = repository_revision(path)
        repository_checks[name] = {
            "expected_revision": details["revision"],
            "observed_revision": observed,
            "checkout_path": str(path) if path else None,
            "status": (
                "BLOCKED"
                if observed is None
                else "PASS"
                if observed == details["revision"]
                else "FAIL"
            ),
        }
    rag_dependencies = {
        "fastapi": {"required": "0.115.6", "installed": versions["fastapi"]},
        "uvicorn": {"required": "0.34.0", "installed": versions["uvicorn"]},
        "pydantic": {"required": "2.10.6", "installed": versions["pydantic"]},
        "python-dotenv": {"required": "1.0.1", "installed": versions["python-dotenv"]},
    }
    rag_missing = [
        f"{name}=={details['required']} (installed: {details['installed'] or 'missing'})"
        for name, details in rag_dependencies.items()
        if details["installed"] != details["required"]
    ]
    core_checks_passed = all(check["passed"] for check in checks.values())
    repository_checks_passed = all(
        check["status"] == "PASS" for check in repository_checks.values()
    )
    return {
        "status": (
            "PASS"
            if core_checks_passed and repository_checks_passed and not rag_missing
            else "PARTIAL"
        ),
        "checks": checks,
        "repository_checks": repository_checks,
        "installed_versions": versions,
        "rag_governor_pinned_runtime": {
            "status": "BLOCKED" if rag_missing else "PASS",
            "missing_or_mismatched": rag_missing,
            "requirements_file": (
                "deterministic-rag-governor/requirements.txt; not installed by Recall"
            ),
        },
        "install_actions_taken": [],
        "notes": (
            "No packages were installed or upgraded. The RAG adapter's HTTP client uses "
            "the Python standard library, but the connected RAG server's pinned runtime "
            "is incomplete in this environment."
        ),
    }


def _run_adapter(
    name: str,
    operation: Any,
    events: list[dict[str, Any]],
    experiment_id: str,
    *,
    component: str = "adapter",
    operation_name: str = "execute",
    input_metadata: dict[str, Any] | None = None,
    repository_commit: str | None = None,
    software_version: str | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    monotonic_start = time.perf_counter()
    error_details: dict[str, str] | None = None
    observed_native_events: list[dict[str, Any]] = []
    try:
        result = operation()
        if not isinstance(result, dict):
            raise TypeError(
                f"{name} adapter returned {type(result).__name__}; expected a dictionary."
            )
        status = result.get("status", "PASS")
        if status not in STATUSES:
            raise ValueError(f"{name} adapter returned invalid status {status!r}")
        candidate_events = result.get("native_events", [])
        if not isinstance(candidate_events, list):
            raise TypeError(f"{name} adapter native_events must be a list.")
        required_event_fields = {
            "experiment_id",
            "system",
            "component",
            "timestamp",
            "operation",
            "status",
        }
        for candidate in candidate_events:
            if not isinstance(candidate, dict) or not required_event_fields.issubset(candidate):
                raise ValueError(f"{name} adapter returned a malformed native event.")
            if any(
                not isinstance(candidate[field], str) or not candidate[field].strip()
                for field in ("experiment_id", "system", "component", "timestamp", "operation")
            ):
                raise ValueError(f"{name} adapter native event has empty required fields.")
            try:
                event_timestamp = datetime.fromisoformat(
                    candidate["timestamp"].replace("Z", "+00:00")
                )
            except ValueError as error:
                raise ValueError(
                    f"{name} adapter native event timestamp is not ISO-8601."
                ) from error
            if event_timestamp.tzinfo is None or event_timestamp.utcoffset() is None:
                raise ValueError(f"{name} adapter native event timestamp must include a timezone.")
            if candidate["experiment_id"] != experiment_id or candidate["system"] != name:
                raise ValueError(
                    f"{name} adapter native event has a mismatched experiment or system."
                )
            if not isinstance(candidate["status"], str) or candidate["status"] not in STATUSES:
                raise ValueError(f"{name} adapter native event has an invalid status.")
            observed_native_events.append(dict(candidate))
        if observed_native_events:
            status = max(
                [status, *(event["status"] for event in observed_native_events)],
                key=STATUS_ORDER.__getitem__,
            )
        details = result
    except AdapterBlocked as error:
        status = "BLOCKED"
        details = {"reason": str(error)}
        error_details = {"type": type(error).__name__, "message": str(error)}
    except (FileNotFoundError, ModuleNotFoundError, ImportError) as error:
        status = "BLOCKED"
        details = {"reason": f"{type(error).__name__}: {error}"}
        error_details = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:
        status = "FAIL"
        message = f"{type(error).__name__}: {error}"
        details = {"error": message, "reason": message}
        error_details = {"type": type(error).__name__, "message": str(error)}

    finished_at = utc_now()
    duration = time.perf_counter() - monotonic_start
    for native_event in observed_native_events:
        events.append(
            {
                **native_event,
                "sequence": len(events) + 1,
                "source": "Native component",
                "event": "component_event",
                "native_component_event": True,
            }
        )
    events.append(
        {
            "experiment_id": experiment_id,
            "sequence": len(events) + 1,
            "timestamp": finished_at,
            "source": "Recall observation",
            "native_component_event": False,
            "system": name,
            "component": component,
            "operation": operation_name,
            "event": "adapter_completed",
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration,
            "input_metadata": input_metadata or {},
            "output_metadata": details,
            "actual_workload": details.get("actual_workload", {}),
            "resource_measurements": details.get("resource_measurements", {}),
            "error_details": error_details,
            "repository_commit": details.get("repository_commit", repository_commit),
            "software_version": details.get(
                "package_version", details.get("software_version", software_version)
            ),
        }
    )
    return {
        "experiment_id": experiment_id,
        "system": name,
        "component": component,
        "timestamp": finished_at,
        "operation": operation_name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": duration,
        "duration_seconds": duration,
        "input_metadata": input_metadata or {},
        "output_metadata": details,
        "actual_workload": details.get("actual_workload", {}),
        "resource_measurements": details.get("resource_measurements", {}),
        "error_details": error_details,
        "repository_commit": details.get("repository_commit", repository_commit),
        "software_version": details.get(
            "package_version", details.get("software_version", software_version)
        ),
        "native_component_events_received": len(observed_native_events),
        "details": details,
    }


def _combine_statuses(*statuses: str) -> str:
    return max(statuses, key=lambda status: STATUS_ORDER[status])


def run_fault_isolation_checks() -> dict[str, Any]:
    """Exercise local adapter-boundary error handling; never represent live outages."""

    def fail_with(error: Exception) -> None:
        raise error

    scenarios: list[tuple[str, str, Any, str]] = [
        (
            "MathMesh unavailable",
            "mathmesh_governor",
            lambda: fail_with(AdapterBlocked("MathMesh unavailable")),
            "BLOCKED",
        ),
        (
            "System-Stability-Toolkit unavailable",
            "system_stability_toolkit",
            lambda: fail_with(AdapterBlocked("Stability toolkit unavailable")),
            "BLOCKED",
        ),
        (
            "Deterministic RAG unavailable",
            "deterministic_rag_governor",
            lambda: fail_with(AdapterBlocked("RAG unavailable")),
            "BLOCKED",
        ),
        (
            "Zeta initialization fails",
            "zeta_scaling_toolkit",
            lambda: fail_with(RuntimeError("injected initialization failure")),
            "FAIL",
        ),
        (
            "Adapter returns malformed data",
            "adapter_contract",
            lambda: ["not", "a", "result"],
            "FAIL",
        ),
        (
            "Service times out",
            "service_boundary",
            lambda: fail_with(TimeoutError("injected timeout")),
            "FAIL",
        ),
        (
            "Audit operation fails",
            "mathmesh_audit",
            lambda: fail_with(RuntimeError("injected audit failure")),
            "FAIL",
        ),
        (
            "Numerical validation fails",
            "zeta_numerical_validation",
            lambda: fail_with(ArithmeticError("injected numerical failure")),
            "FAIL",
        ),
    ]
    results: list[dict[str, Any]] = []
    for index, (condition, affected, operation, expected_status) in enumerate(scenarios):
        experiment_id = f"fault-injection-{index + 1}"
        events: list[dict[str, Any]] = []
        affected_result = _run_adapter(
            affected,
            operation,
            events,
            experiment_id,
            component=affected,
            operation_name="injected_fault",
            input_metadata={"injected_condition": condition},
        )
        unaffected_result = _run_adapter(
            "independent_adapter_probe",
            lambda: {"status": "PASS", "probe": "local boundary control"},
            events,
            experiment_id,
            component="independent_adapter_probe",
            operation_name="control_probe",
        )
        isolated = (
            affected_result["status"] == expected_status
            and unaffected_result["status"] == "PASS"
            and len(events) == 2
        )
        results.append(
            {
                "injected_condition": condition,
                "affected_component": affected,
                "unaffected_components": ["independent_adapter_probe"],
                "affected_status": affected_result["status"],
                "execution_continued": unaffected_result["status"] == "PASS",
                "failure_isolated": isolated,
                "evidence_preserved": len(events) == 2
                and all(event["experiment_id"] == experiment_id for event in events),
                "final_status": "PASS" if isolated else "FAIL",
                "evidence_scope": "LOCAL UNIT FAULT-INJECTION; NOT LIVE INTEGRATION",
            }
        )
    return {
        "status": (
            "PASS"
            if all(result["final_status"] == "PASS" for result in results)
            else "FAIL"
        ),
        "live_outage_tests_run": False,
        "scenario_count": len(results),
        "scenarios": results,
        "notes": (
            "These checks validate Recall's local adapter-boundary isolation and evidence "
            "retention only. They do not prove that any external system continues during "
            "a live outage."
        ),
    }


def run_qualification(
    config_path: Path = CONFIG_PATH,
    write_results: bool = True,
    selected_systems: set[str] | None = None,
    output_root: Path = ROOT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    capability_configuration = json.loads(
        (ROOT / "config" / "capabilities.json").read_text(encoding="utf-8")
    )
    integration_configuration = json.loads(
        (ROOT / "config" / "integrations.json").read_text(encoding="utf-8")
    )
    experiment_id = f"{config['experiment_id_prefix']}-{uuid.uuid4()}"
    started_at = utc_now()
    start_clock = time.perf_counter()
    paths = {
        "zeta-scaling-toolkit": _env_path("RECALL_ZETA_REPO"),
        "mathmesh-governor": _env_path("RECALL_MATHMESH_REPO"),
        "System-Stability-Toolkit": _env_path("RECALL_STABILITY_REPO"),
        "deterministic-rag-governor": _env_path("RECALL_RAG_REPO"),
    }
    events: list[dict[str, Any]] = [
        {
            "experiment_id": experiment_id,
            "sequence": 1,
            "timestamp": started_at,
            "source": "Recall",
            "native_component_event": False,
            "system": "Recall",
            "event": "experiment_started",
        }
    ]
    environment = validate_environment(config, paths)

    zeta_adapter = ZetaToolkitAdapter(
        paths["zeta-scaling-toolkit"],
        config["dependencies"]["zeta_scaling_toolkit"],
        sequence_length=config["qualification_probe"]["zeta_attention_sequence_length"],
        head_count=config["qualification_probe"]["zeta_head_count"],
        head_dimension=config["qualification_probe"]["zeta_head_dimension"],
    )
    mathmesh_adapter = MathMeshAdapter(paths["mathmesh-governor"])
    stability_adapter = StabilityAdapter(paths["System-Stability-Toolkit"])
    rag_adapter = DeterministicRAGAdapter()

    resource_snapshots: dict[str, Any] = {}
    try:
        resource_snapshots["baseline"] = stability_adapter.resource_snapshot()
    except Exception as error:
        resource_snapshots["baseline"] = {
            "status": "BLOCKED",
            "reason": f"{type(error).__name__}: {error}",
        }

    def run_mathmesh_checks() -> dict[str, Any]:
        details = mathmesh_adapter.execute("parity_check", {})
        if os.environ.get("RECALL_ALLOW_REMOTE_AUDIT_WRITES") == "1":
            try:
                details["audit_event_delivery"] = mathmesh_adapter.record_observation(
                    "mathmesh_parity_check",
                    experiment_id,
                    json.dumps(
                        {
                            "same_parity": details.get("same_parity_vector"),
                            "mixed_parity": details.get("mixed_parity_vector"),
                        },
                        sort_keys=True,
                    ),
                )
            except AdapterBlocked as error:
                details["audit_event_delivery"] = {
                    "status": "BLOCKED",
                    "attempted": 0,
                    "delivered": 0,
                    "failed": 0,
                    "reason": str(error),
                }
            except Exception as error:
                details["audit_event_delivery"] = {
                    "status": "FAIL",
                    "attempted": 1,
                    "delivered": 0,
                    "failed": 1,
                    "reason": f"{type(error).__name__}: {error}",
                }
                details["status"] = "FAIL"
        details["actual_workload"] = {"parity_vectors_checked": 2}
        return details

    def run_rag_checks() -> dict[str, Any]:
        details = rag_adapter.execute("audit_capability_probe", {})
        details["actual_workload"] = {
            "health_probes": 1 if rag_adapter.base_url else 0,
            "retrieval_requests": 0,
            "generation_requests": 0,
        }
        if os.environ.get("RECALL_ALLOW_RAG_GOVERN") == "1":
            agent_id = os.environ.get("RECALL_RAG_AGENT_ID", "")
            query = os.environ.get("RECALL_RAG_QUERY", "")
            governed = rag_adapter.execute(
                "govern",
                {
                    "agent_id": agent_id,
                    "query": query,
                    "experiment_id": experiment_id,
                },
            )
            details["governed_request"] = governed
            details["request_metadata"] = {
                "experiment_id": experiment_id,
                "query_length": len(query),
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            }
            details["status"] = _combine_statuses(details["status"], governed["status"])
            details["actual_workload"] = {
                "request_count": 1,
                "operation": "retrieval_and_generation",
            }
        return details

    def rag_health_checks() -> dict[str, Any]:
        health = rag_adapter.health()
        readiness = rag_adapter.readiness()
        return {
            "status": _combine_statuses(health["status"], readiness["status"]),
            "health": health,
            "readiness": readiness,
        }

    configured_operations: dict[str, tuple[Any, Any, Any, str]] = {
        "zeta_scaling_toolkit": (
            zeta_adapter,
            lambda: zeta_adapter.execute(
                "component_probe",
                {
                    "sequence_length": config["qualification_probe"][
                        "zeta_attention_sequence_length"
                    ],
                    "head_count": config["qualification_probe"]["zeta_head_count"],
                    "head_dimension": config["qualification_probe"][
                        "zeta_head_dimension"
                    ],
                },
            ),
            {
                "sequence_length": config["qualification_probe"]["zeta_attention_sequence_length"],
                "head_count": config["qualification_probe"]["zeta_head_count"],
                "head_dimension": config["qualification_probe"]["zeta_head_dimension"],
            },
            "component_probe",
        ),
        "mathmesh_governor": (
            mathmesh_adapter,
            run_mathmesh_checks,
            {},
            "parity_check",
        ),
        "system_stability_toolkit": (
            stability_adapter,
            lambda: {
                **stability_adapter.execute(
                    "advisory_recommendation",
                    {
                        "workload_scale": config["qualification_probe"][
                            "stability_advisory_workload_scale"
                        ]
                    },
                ),
                "resource_measurements": resource_snapshots,
            },
            {
                "workload_scale": config["qualification_probe"][
                    "stability_advisory_workload_scale"
                ]
            },
            "advisory_recommendation",
        ),
        "deterministic_rag_governor": (
            rag_adapter,
            run_rag_checks,
            {},
            "audit_capability_probe",
        ),
    }
    systems: dict[str, dict[str, Any]] = {}
    for name, (adapter, operation, input_metadata, operation_name) in configured_operations.items():
        if selected_systems is not None and name not in selected_systems:
            not_tested = _run_adapter(
                name,
                lambda: {
                    "status": "NOT TESTED",
                    "reason": "Excluded by the --system selection.",
                },
                events,
                experiment_id,
                component="qualification",
                operation_name="not_selected",
            )
            lifecycle = {
                "initialization": {"status": "NOT TESTED"},
                "health_check": {"status": "NOT TESTED"},
                "shutdown": {"status": "NOT APPLICABLE"},
            }
            systems[name] = {
                **not_tested,
                "lifecycle": lifecycle,
                "capabilities": adapter.capabilities(),
            }
            continue

        source_name = {
            "zeta_scaling_toolkit": "zeta-scaling-toolkit",
            "mathmesh_governor": "mathmesh-governor",
            "system_stability_toolkit": "System-Stability-Toolkit",
            "deterministic_rag_governor": "deterministic-rag-governor",
        }[name]
        source_path = paths[source_name]
        initialization = _run_adapter(
            name,
            adapter.initialize,
            events,
            experiment_id,
            component="adapter",
            operation_name="initialize",
            repository_commit=repository_revision(source_path),
        )
        health_check = _run_adapter(
            name,
            rag_health_checks if name == "deterministic_rag_governor" else adapter.health,
            events,
            experiment_id,
            component="service",
            operation_name="health_check",
            repository_commit=repository_revision(source_path),
        )
        execution = _run_adapter(
            name,
            operation,
            events,
            experiment_id,
            component=name,
            operation_name=operation_name,
            input_metadata=input_metadata,
            repository_commit=repository_revision(source_path),
        )
        try:
            adapter.shutdown()
            shutdown = {"status": "PASS", "timestamp": utc_now()}
        except Exception as error:
            shutdown = {
                "status": "FAIL",
                "timestamp": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        execution["status"] = _combine_statuses(
            execution["status"], initialization["status"], health_check["status"], shutdown["status"]
        )
        execution["lifecycle"] = {
            "initialization": initialization,
            "health_check": health_check,
            "shutdown": shutdown,
        }
        execution["capabilities"] = adapter.capabilities()
        systems[name] = execution

    for phase in ("during", "post"):
        try:
            resource_snapshots[phase] = stability_adapter.resource_snapshot()
        except Exception as error:
            resource_snapshots[phase] = {
                "status": "BLOCKED",
                "reason": f"{type(error).__name__}: {error}",
            }
    systems["system_stability_toolkit"]["details"]["resource_measurements"] = resource_snapshots
    systems["system_stability_toolkit"]["resource_measurements"] = resource_snapshots
    events.append(
        {
            "experiment_id": experiment_id,
            "sequence": len(events) + 1,
            "timestamp": utc_now(),
            "source": "Recall",
            "native_component_event": False,
            "system": "Recall",
            "event": "experiment_completed",
        }
    )

    zeta_status = systems["zeta_scaling_toolkit"]["status"]
    mathmesh_status = systems["mathmesh_governor"]["status"]
    stability_status = systems["system_stability_toolkit"]["status"]
    rag_status = systems["deterministic_rag_governor"]["status"]
    native_systems = {
        event["system"] for event in events if event["native_component_event"]
    }
    native_events = sum(event["native_component_event"] for event in events)
    if (
        selected_systems is None
        and all(
            status == "PASS"
            for status in (zeta_status, mathmesh_status, stability_status, rag_status)
        )
        and environment["status"] == "PASS"
        and len(native_systems) == 4
    ):
        integration_status = "PASS"
        integration_reason = "All native adapters passed and emitted correlated events."
    elif selected_systems is not None:
        integration_status = "BLOCKED"
        integration_reason = (
            "A system-scoped run cannot qualify the complete four-system integration."
        )
    else:
        integration_status = "BLOCKED"
        integration_reason = (
            "Required live integration evidence is incomplete: adapter or environment "
            "checks include blocked/partial results, pinned revisions are unverified, "
            "and/or systems do not emit shared native events. "
            "Recall-local observations are not component telemetry."
        )

    ids_match = all(event["experiment_id"] == experiment_id for event in events)
    correlation_status = (
        "FAIL"
        if not ids_match
        else "PASS"
        if len(native_systems) == 4
        else "PARTIAL"
    )
    correlation = {
        "status": correlation_status,
        "experiment_id": experiment_id,
        "recall_event_count": len(events),
        "native_component_events_received": native_events,
        "native_systems_with_events": sorted(native_systems),
        "execution_order": [event["sequence"] for event in events],
        "execution_model": "serial",
        "parallel_execution_claimed": False,
        "notes": (
            "Recall's own observations carry the shared ID. Zeta, MathMesh parity, and "
            "System-Stability CLI have no shared experiment-event interface. RAG audit "
            "writes are opt-in and do not provide cross-system native telemetry."
        ),
    }

    numerical_status = "NOT TESTED"
    checkpoint_status = "NOT TESTED"
    if zeta_status == "PASS":
        numerical_status = (
            "PASS"
            if systems["zeta_scaling_toolkit"]["details"].get("attention_output_finite")
            else "FAIL"
        )
    integrity = {
        "status": "PARTIAL" if numerical_status == "PASS" else numerical_status,
        "numerical_finiteness": numerical_status,
        "cache_integrity": (
            "PASS" if zeta_status == "PASS" else "NOT TESTED"
        ),
        "checkpoint_integrity": checkpoint_status,
        "notes": "No model checkpoint was loaded, written, or verified.",
    }
    unit_fault_injection = run_fault_isolation_checks()
    fault_isolation = {
        "status": "PARTIAL",
        "observed_unavailable_components": [
            name
            for name in SYSTEM_ORDER
            if systems[name]["status"] in {"BLOCKED", "FAIL"}
        ],
        "other_adapter_results_retained": True,
        "simulated_failures_used_as_evidence": False,
        "unit_fault_injection": unit_fault_injection,
        "notes": (
            "Local unit fault-injection checks verify orchestration behavior only. "
            "There is no full live four-service outage test."
        ),
    }

    model_reference = config["model_reference"]
    model_cache = _reference_model_cached(model_reference["revision"])
    cache_note = (
        "Pinned model weights are present in the local Hugging Face cache."
        if model_cache
        else "Pinned model weights are not present in the local Hugging Face cache."
    )
    short_benchmark = {
        "status": "BLOCKED",
        "model_id": model_reference["model_id"],
        "model_revision": model_reference["revision"],
        "model_weights_cached": model_cache,
        "zeta_model_runner_available": False,
        "actual_tokens_processed": 0,
        "inference_seconds": None,
        "throughput_tokens_per_second": None,
        "target_tokens_per_second": model_reference["target_throughput_tokens_per_second"],
        "target_reached": None,
        "reason": (
            "The Zeta repository exposes attention/cache primitives, not a runner for "
            f"the pinned Hugging Face model. {cache_note} No model was loaded or run; "
            "the qualification gates do not permit inference."
        ),
        "total_harness_elapsed_seconds": None,
    }

    ended_at = utc_now()
    elapsed = time.perf_counter() - start_clock
    short_benchmark["total_harness_elapsed_seconds"] = elapsed
    result = {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "commit_identifiers": _repo_metadata(config, paths),
        "model_provenance": {
            **model_reference,
            "tokenizer_provenance": {
                "tokenizer_id": "NOT LOADED",
                "revision": "NOT TESTED",
                "note": "The benchmark's tokenizer was not instantiated in this run.",
            },
            "local_reference_evidence_verified": False,
            "context_clarification": (
                "100,000 contiguous processed tokens is a total workload, not "
                "100,000 tokens of simultaneous attention or retained context."
            ),
        },
        "workload": {
            "description": (
                "One Zeta attention/cache component probe with sequence_length=32; "
                "no model or tokenizer was run."
            ),
            "zeta_sequence_length": config["qualification_probe"]["zeta_attention_sequence_length"],
            "actual_model_tokens_processed": 0,
            "actual_workload": {
                "model_tokens_processed": 0,
                "attention_probe_sequence_length": config["qualification_probe"][
                    "zeta_attention_sequence_length"
                ],
                "model_inference_run": False,
            },
            "historical_workload_tokens": model_reference["historical_workload_tokens"],
        },
        "resources": resource_snapshots,
        "hardware_runtime": _hardware_snapshot(),
        "environment_validation": environment,
        "capability_configuration": capability_configuration,
        "integration_configuration": integration_configuration,
        "systems": systems,
        "events": events,
        "phases": {
            "1_dependency_environment_validation": environment["status"],
            "2_individual_adapter_tests": (
                "PASS"
                if all(value["status"] == "PASS" for value in systems.values())
                else "PARTIAL"
            ),
            "3_four_system_lightweight_integration": integration_status,
            "3_four_system_lightweight_integration_reason": integration_reason,
            "4_shared_experiment_id_telemetry_correlation": correlation_status,
            "5_numerical_and_checkpoint_integrity": integrity["status"],
            "6_component_outage_fault_isolation": fault_isolation["status"],
            "7_short_zeta_inference_benchmark": short_benchmark["status"],
            "8_long_benchmark": "NOT TESTED",
        },
        "coordination": correlation,
        "integrity": integrity,
        "fault_isolation": fault_isolation,
        "performance": short_benchmark,
        "qualification_status": integration_status,
        "run_scope": (
            "all systems"
            if selected_systems is None
            else "selected systems: " + ", ".join(sorted(selected_systems))
        ),
        "production_readiness": "NOT ASSESSED",
        "original_100k_evidence": {
            "run_repeated": False,
            "files_written_or_modified": False,
            "status": "PRESERVED; no original 100K evidence path was used as output",
        },
    }

    if write_results:
        write_reports(result, output_root)
    return result


def _reference_model_cached(revision: str) -> bool:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model = cache / "models--HuggingFaceTB--SmolLM2-360M-Instruct"
    snapshot = model / "snapshots" / revision
    if not snapshot.is_dir():
        return False
    weights = [
        *snapshot.glob("*.safetensors"),
        *(path for path in snapshot.glob("pytorch_model*.bin") if path.name.endswith(".bin")),
    ]
    return any(path.is_file() for path in weights)


def _summary_markdown(result: dict[str, Any]) -> str:
    systems = result["systems"]
    return "\n".join(
        [
            "# Recall Test Results",
            "",
            *_metadata_lines(result),
            "",
            f"**Qualification status:** {result['qualification_status']}  ",
            f"**{result['model_provenance']['target_throughput_tokens_per_second']} tokens/s target:** "
            f"NOT MEASURED (`target_reached={result['performance']['target_reached']}`)  ",
            f"**Production readiness:** {result['production_readiness']}",
            "",
            "## Phase results",
            "",
            "| Phase | Status |",
            "|---|---|",
            *[
                f"| {name.replace('_', ' ')} | {status} |"
                for name, status in result["phases"].items()
                if not name.endswith("_reason")
            ],
            "",
            "## Adapter results",
            "",
            *[
                f"- **{name}: {systems[name]['status']}** — "
                f"{systems[name]['details'].get('notes', systems[name]['details'].get('reason', 'See JSON details.'))}"
                for name in SYSTEM_ORDER
            ],
            "",
            "## Performance",
            "",
            f"- Actual model tokens: {result['workload']['actual_model_tokens_processed']}",
            "- Model inference elapsed time: not measured",
            "- Model workload throughput: not measured",
            f"- Total harness wall time: {result['performance']['total_harness_elapsed_seconds']:.6f} seconds",
            f"- Blocker: {result['performance']['reason']}",
            "",
            "The attention probe's sequence length is not a token count. The historical "
            "100,000-token run is a total workload and does not mean 100,000 tokens of "
            "simultaneous attention or context retention.",
            "",
        ]
    )


def _metadata_lines(result: dict[str, Any]) -> list[str]:
    commits = result["commit_identifiers"]["connected_repositories"]
    connected_commit_lines = []
    for name in REPOSITORY_ORDER:
        item = commits[name]
        observed = item["observed_revision"]
        expected = item["expected_revision"]
        connected_commit_lines.append(
            f"{name}=`{observed}`"
            if observed == expected
            else f"{name}=`{observed or 'UNAVAILABLE'}` (expected `{expected}`)"
        )
    return [
        f"- **Experiment ID:** `{result['experiment_id']}`",
        f"- **Timestamp:** {result['started_at']} to {result['ended_at']}",
        f"- **Recall commit:** `{result['commit_identifiers']['recall']['commit']}`",
        f"- **Connected commits:** " + "; ".join(connected_commit_lines),
        f"- **Model provenance:** `{result['model_provenance']['model_id']}` "
        f"@ `{result['model_provenance']['revision']}`",
        f"- **Tokenizer provenance:** "
        f"{result['model_provenance']['tokenizer_provenance']['tokenizer_id']} "
        f"({result['model_provenance']['tokenizer_provenance']['note']})",
        f"- **Workload / actual token count:** {result['workload']['description']} "
        f"(actual model tokens: {result['workload']['actual_model_tokens_processed']})",
        f"- **Hardware/runtime:** {result['hardware_runtime']['platform']}; "
        f"Python {result['hardware_runtime']['python']}; "
        f"torch {result['hardware_runtime'].get('torch_runtime')}; "
        f"Node {result['hardware_runtime'].get('node')}",
        f"- **Test status:** {result['qualification_status']}",
    ]


def _report_document(title: str, result: dict[str, Any], body: list[str]) -> str:
    return "\n".join([f"# {title}", "", *_metadata_lines(result), "", *body, ""])


def _snapshot_summary(snapshot: dict[str, Any]) -> str:
    if "memory_percent" not in snapshot:
        return "not available"
    peak_rss = snapshot.get("process_peak_memory_mb")
    peak_description = (
        f", process peak RSS {peak_rss:.1f} MB"
        if isinstance(peak_rss, (int, float))
        else ""
    )
    return (
        f"{snapshot['timestamp']}: memory {snapshot['memory_percent']:.1f}%, "
        f"CPU {snapshot['cpu_percent']:.1f}%, process RSS "
        f"{snapshot['process_memory_mb']:.1f} MB{peak_description}"
    )


def render_reports(result: dict[str, Any]) -> dict[str, str]:
    systems = result["systems"]
    zeta_details = systems["zeta_scaling_toolkit"]["details"]
    mathmesh_details = systems["mathmesh_governor"]["details"]
    mathmesh_delivery = mathmesh_details.get("audit_event_delivery", {})
    mathmesh_health_status = systems["mathmesh_governor"].get("lifecycle", {}).get(
        "health_check", {}
    ).get("status", "NOT TESTED")
    stability_measurements = systems["system_stability_toolkit"]["details"].get(
        "resource_measurements", {}
    )
    saved_resource_measurements = result.get("resources", stability_measurements)
    peak_memory_values = [
        snapshot["process_peak_memory_mb"]
        for snapshot in saved_resource_measurements.values()
        if isinstance(snapshot, dict)
        and isinstance(snapshot.get("process_peak_memory_mb"), (int, float))
    ]
    peak_memory_summary = (
        f"{max(peak_memory_values):.1f} MB (Recall process RSS; not model-only)"
        if peak_memory_values
        else "not measured"
    )
    rag_details = systems["deterministic_rag_governor"]["details"]
    rag_health_status = systems["deterministic_rag_governor"].get("lifecycle", {}).get(
        "health_check", {}
    ).get("status", "NOT TESTED")
    rag_workload = rag_details.get("actual_workload", {})
    return {
        "RECALL_TEST_RESULTS.md": _summary_markdown(result),
        "FOUR_SYSTEM_INTEGRATION_REPORT.md": _report_document(
            "Four-System Integration Report",
            result,
            [
                "## Integration map",
                "",
                "- **Zeta Scaling Toolkit:** `zeta_scaling_toolkit.NonCollapsingAttention` and "
                "`StreamingKVCache`; component probe only. No Hugging Face inference runner "
                "is present in the connected repository.",
                "- **MathMesh Governor:** `governor/shared/mathChecks.ts:evaluateVoteParity`; "
                f"parity result `{mathmesh_details.get('status', 'NOT TESTED')}`; live "
                f"`GET /api/health` status `{mathmesh_health_status}`. Audit endpoint writes "
                f"are opt-in; delivery status is `{mathmesh_delivery.get('status', 'NOT TESTED')}`.",
                "- **System-Stability-Toolkit:** documented `crash_fix.py <scale> --json`; "
                "recommendations are recorded but not applied. The toolkit does not monitor "
                "resources; snapshots here are Recall/psutil measurements.",
                f"- **Deterministic RAG Governor:** `GET /health`/`GET /ready` status "
                f"`{rag_health_status}`. The repo documents combined retrieval/generation "
                "through `/govern`, not a metadata-only audit mode. Retrieval requests: "
                f"{rag_workload.get('retrieval_requests', 0)}; generation requests: "
                f"{rag_workload.get('generation_requests', 0)}.",
                "",
                "## Results",
                "",
                f"- Zeta: {systems['zeta_scaling_toolkit']['status']}",
                f"- MathMesh: {systems['mathmesh_governor']['status']}",
                f"- System-Stability-Toolkit: {systems['system_stability_toolkit']['status']}",
                f"- Deterministic RAG Governor: {systems['deterministic_rag_governor']['status']}",
                f"- Four-system integration: {result['phases']['3_four_system_lightweight_integration']}",
                f"- Shared native telemetry events received: "
                f"{result['coordination']['native_component_events_received']}",
                "- Event order is serial; no parallel execution is inferred from timestamps.",
                f"- Zeta probe output shape: `{zeta_details.get('attention_shape', 'NOT TESTED')}`; "
                f"finite: `{zeta_details.get('attention_output_finite', 'NOT TESTED')}`; "
                f"cache positions: `{zeta_details.get('cache_positions_after_five_appends', 'NOT TESTED')}`; "
                f"model tokens: `{zeta_details.get('actual_model_tokens_processed', 0)}`.",
                f"- MathMesh numeric observation: `{mathmesh_details.get('same_parity_vector', 'NOT TESTED')}`; "
                f"unsupported operations: `{mathmesh_details.get('unsupported_operations', [])}`.",
                f"- MathMesh event delivery: {mathmesh_delivery.get('status', 'NOT TESTED')}; "
                f"attempted {mathmesh_delivery.get('attempted', 'NOT TESTED')}, "
                f"delivered {mathmesh_delivery.get('delivered', 'NOT TESTED')}, "
                f"failed {mathmesh_delivery.get('failed', 'NOT TESTED')}, "
                f"dropped {mathmesh_delivery.get('dropped', 'NOT TESTED')}.",
                f"- Stability snapshots (Recall/psutil): baseline "
                f"{_snapshot_summary(stability_measurements.get('baseline', {}))}; during "
                f"{_snapshot_summary(stability_measurements.get('during', {}))}; post "
                f"{_snapshot_summary(stability_measurements.get('post', {}))}.",
                f"- RAG queries/audit events: `{rag_details.get('query_events', 'NOT TESTED')}`/"
                f"`{rag_details.get('audit_events', 'NOT TESTED')}`; event timestamps and "
                f"evidence references: `{rag_details.get('event_timestamps', 'NOT TESTED')}` / "
                f"`{rag_details.get('evidence_references', 'NOT TESTED')}`; "
                f"audit-only mode: `{rag_details.get('audit_mode', 'UNSUPPORTED')}`.",
                "",
                "## Unsupported interfaces",
                "",
                "- Zeta model-token inference for the pinned Hugging Face model.",
                "- MathMesh radix conversion and numeric round-trip operations.",
                "- MathMesh live governance/audit delivery without a dedicated deployment and admin token.",
                "- A metadata-only RAG audit mode; `/govern` can retrieve and generate.",
            ],
        ),
        "FAULT_ISOLATION_REPORT.md": _report_document(
            "Fault Isolation Report",
            result,
            [
                f"**Fault-isolation status:** {result['fault_isolation']['status']}",
                "",
                "The harness isolates adapter outcomes and continues collecting evidence from "
                "available components when another adapter is blocked. No mock or simulated "
                "result is counted as a real integration.",
                "",
                "### Observed blockers",
                "",
                *[
                    f"- {name}: {systems[name]['details'].get('reason', systems[name]['status'])}"
                    for name in SYSTEM_ORDER
                    if systems[name]["status"] in {"BLOCKED", "FAIL"}
                ],
                f"- MathMesh audit delivery: {mathmesh_delivery.get('status', 'NOT TESTED')}; "
                f"{mathmesh_delivery.get('reason', 'see recorded API response or status')}.",
                "- RAG audit-only execution is unsupported; `/govern` combines retrieval, "
                "generation, spend, and audit side effects.",
                "",
                "### Limits",
                "",
                f"Local unit fault-injection checks: "
                f"{result['fault_isolation'].get('unit_fault_injection', {}).get('status', 'NOT TESTED')} "
                f"({result['fault_isolation'].get('unit_fault_injection', {}).get('scenario_count', 0)} "
                "scenarios). These validate Recall's adapter-boundary handling only; "
                "they are not live service outage evidence.",
                "",
                *[
                    f"- {scenario['injected_condition']}: {scenario['final_status']} "
                    f"(affected `{scenario['affected_component']}`; "
                    f"unaffected `{', '.join(scenario['unaffected_components'])}`; "
                    f"continued={scenario['execution_continued']}; "
                    f"isolated={scenario['failure_isolated']}; "
                    f"evidence_preserved={scenario['evidence_preserved']})"
                    for scenario in result["fault_isolation"]
                    .get("unit_fault_injection", {})
                    .get("scenarios", [])
                ],
                "",
                "A live outage test against all four services was not established by this run. "
                "Zeta model inference was also unavailable. "
                "This is not a production-readiness test.",
            ],
        ),
        "PERFORMANCE_COMPARISON.md": _report_document(
            "Performance Comparison",
            result,
            [
                "## Historical reference (provided by the user)",
                "",
                f"- Model/revision: `{result['model_provenance']['model_id']}` @ "
                f"`{result['model_provenance']['revision']}`",
                f"- Workload: {result['model_provenance']['historical_workload_tokens']:,} "
                "contiguous processed tokens; native context 8,192",
                f"- Reference throughput: "
                f"{result['model_provenance']['historical_throughput_tokens_per_second']} tokens/s",
                f"- Reference wall time: {result['model_provenance']['historical_wall_seconds']} s",
                f"- Reference final token position: "
                f"{result['model_provenance']['historical_final_token_position']}",
                f"- Checkpoints: {result['model_provenance']['historical_checkpoints']}",
                "- Reference evidence was not locally verified. A total processed-token "
                "count is not simultaneous attention/context retention.",
                "",
                "## Current result",
                "",
                "- Short inference: BLOCKED; no model inference was run.",
                "- Actual processed model tokens: 0",
                "- Measured inference time: none",
                f"- Peak process memory observed: {peak_memory_summary}",
                f"- Pinned model weights cached locally: "
                f"{result['performance']['model_weights_cached']}",
                "- Workload throughput formula: actual processed model tokens / measured "
                "inference time; not calculable for this probe.",
                "- Total end-to-end Recall harness elapsed time is recorded in "
                "`results/RECALL_TEST_RESULTS.json`; it is not model inference time.",
                f"- {result['model_provenance']['target_throughput_tokens_per_second']} "
                "tokens/s target: NOT MEASURED; achievement is unverified.",
                "- Comparability: not applicable. No inference run exists to compare for "
                "model, tokenizer, workload, hardware, precision, or measurement method.",
            ],
        ),
        "ENVIRONMENT_AND_DEPENDENCY_REPORT.md": _report_document(
            "Environment and Dependency Report",
            result,
            [
                f"- Environment status: {result['environment_validation']['status']}",
                f"- Python: {result['hardware_runtime']['python']}",
                f"- Installed versions: `{json.dumps(result['environment_validation']['installed_versions'], sort_keys=True)}`",
                f"- Runtime checks: `{json.dumps(result['environment_validation']['checks'], sort_keys=True)}`",
                f"- Connected repository revision checks: "
                f"`{json.dumps(result['environment_validation'].get('repository_checks', {}), sort_keys=True)}`",
                f"- RAG pinned packages missing/mismatched: "
                f"`{json.dumps(result['environment_validation']['rag_governor_pinned_runtime']['missing_or_mismatched'])}`",
                "- Recall runtime and test dependencies are pinned in `requirements.txt` and "
                "`requirements-dev.txt`; no packages were "
                "installed or upgraded during this run.",
                f"- MathMesh parity operation: {mathmesh_details.get('status', 'NOT TESTED')}; "
                f"service health: {mathmesh_health_status}. A database-backed service is not "
                "started by Recall.",
                "- System-Stability-Toolkit and MathMesh are consumed from separately cloned "
                "source checkouts. RAG health is accessed over HTTP; service result: "
                f"{rag_health_status}.",
                "- CUDA availability and system resource measurements are recorded in the JSON.",
                f"- Model weights for the pinned SmolLM2 revision are "
                f"{'cached' if result['performance']['model_weights_cached'] else 'not cached'}; "
                "no download or model load was attempted.",
            ],
        ),
        "RECALL_COMPLETION_REPORT.md": _report_document(
            "Recall Completion Report",
            result,
            [
                f"## Qualification conclusion",
                "",
                f"- Overall result: **{result['qualification_status']}**",
                f"- Production readiness: **{result['production_readiness']}**",
                f"- Real model inference run: **{'YES' if result['workload']['actual_model_tokens_processed'] else 'NO'}** "
                f"({result['workload']['actual_model_tokens_processed']} model tokens processed)",
                f"- Historical 100K experiment rerun: **"
                f"{result.get('original_100k_evidence', {}).get('run_repeated', 'NOT RECORDED')}**",
                f"- Original evidence status: "
                f"{result.get('original_100k_evidence', {}).get('status', 'NOT RECORDED')}",
                "",
                "## Adapter and integration outcomes",
                "",
                *[
                    f"- {name}: **{system['status']}**; operation "
                    f"`{system.get('operation', 'NOT TESTED')}`; commit "
                    f"`{system.get('repository_commit') or 'UNAVAILABLE'}`; software version "
                    f"`{system.get('software_version') or 'UNAVAILABLE'}`."
                    for name in SYSTEM_ORDER
                    for system in (result["systems"][name],)
                ],
                f"- Four-system integration gate: **{result['phases']['3_four_system_lightweight_integration']}** "
                f"— {result['phases'].get('3_four_system_lightweight_integration_reason', 'No separate reason recorded.')}",
                f"- Native component events received: "
                f"{result['coordination']['native_component_events_received']}; Recall-local "
                "observations are not presented as component telemetry.",
                f"- Fault boundary unit checks: "
                f"{result['fault_isolation'].get('unit_fault_injection', {}).get('status', 'NOT TESTED')} "
                f"({result['fault_isolation'].get('unit_fault_injection', {}).get('scenario_count', 0)} "
                "local injected scenarios; not live outage tests).",
                "",
                "## Delivered qualification components",
                "",
                "- `recall_harness.py`: safe run, per-system selection, evidence validation, report generation.",
                "- `adapters/`: versioned lifecycle, capability, health, operation and shutdown contracts.",
                "- `config/experiment.json`, `config/capabilities.json`, and `config/integrations.json`.",
                "- Adapter, harness, integration-contract and fault-isolation tests.",
                "- Evidence-backed JSON and Markdown qualification reports.",
                "",
                "## External blockers",
                "",
                "- Zeta's pinned repository exposes attention/cache primitives but no supported "
                "runner for the pinned Hugging Face model; inference remains BLOCKED.",
                "- MathMesh local parity is distinct from live governance/audit delivery; writes "
                "are opt-in and require an authenticated deployment.",
                "- RAG exposes health/readiness and a combined `/govern` operation, not a "
                "metadata-only audit endpoint. Governed requests remain disabled unless explicitly opted in.",
                "- A complete shared native experiment-event contract is not exposed by all four systems.",
                "",
                "Historical qualification documents are retained as originally recorded; their "
                "claims are not silently upgraded or substituted for the current structured evidence. "
                "In particular, the earlier FOUR_SYSTEM_4K_RESULTS.json COMPLETE claim is not "
                "treated as this run's live four-system qualification.",
            ],
        ),
    }


class EvidenceValidationError(ValueError):
    """Saved evidence is structurally invalid or internally inconsistent."""


def validate_evidence(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise EvidenceValidationError("Evidence root must be a JSON object.")
    errors: list[str] = []
    if result.get("schema_version") not in (1, 2):
        errors.append("schema_version must be 1 or 2")
    experiment_id = result.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        errors.append("experiment_id must be a non-empty string")
    for field in ("started_at", "ended_at"):
        value = result.get(field)
        if not isinstance(value, str):
            errors.append(f"{field} must be an ISO-8601 timestamp")
            continue
        try:
            parsed_timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"{field} is not a valid ISO-8601 timestamp")
        else:
            if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
                errors.append(f"{field} must include a timezone")

    systems = result.get("systems")
    if not isinstance(systems, dict):
        errors.append("systems must be an object")
        systems = {}
    for name in (
        "zeta_scaling_toolkit",
        "mathmesh_governor",
        "system_stability_toolkit",
        "deterministic_rag_governor",
    ):
        entry = systems.get(name)
        if not isinstance(entry, dict):
            errors.append(f"systems.{name} is missing or invalid")
            continue
        if not isinstance(entry.get("status"), str) or entry["status"] not in STATUSES:
            errors.append(f"systems.{name}.status is not a supported status")
        if result.get("schema_version") == 2:
            required_record_fields = {
                "experiment_id",
                "system",
                "component",
                "timestamp",
                "operation",
                "input_metadata",
                "output_metadata",
                "actual_workload",
                "elapsed_seconds",
                "resource_measurements",
                "error_details",
                "repository_commit",
                "software_version",
            }
            missing = required_record_fields.difference(entry)
            if missing:
                errors.append(
                    f"systems.{name} is missing structured fields: "
                    + ", ".join(sorted(missing))
                )
            elif entry.get("experiment_id") != experiment_id:
                errors.append(f"systems.{name} experiment_id does not match the result")

    workload = result.get("workload")
    if not isinstance(workload, dict):
        errors.append("workload must be an object")
    else:
        actual_tokens = workload.get("actual_model_tokens_processed")
        if isinstance(actual_tokens, bool) or not isinstance(actual_tokens, int) or actual_tokens < 0:
            errors.append("workload.actual_model_tokens_processed must be a non-negative integer")

    events = result.get("events")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
    else:
        for index, event in enumerate(events):
            if not isinstance(event, dict) or event.get("experiment_id") != experiment_id:
                errors.append(f"events[{index}] has a missing or mismatched experiment_id")
                continue
            if "status" in event and (
                not isinstance(event["status"], str) or event["status"] not in STATUSES
            ):
                errors.append(f"events[{index}].status is not a supported status")

    for field in ("model_provenance", "hardware_runtime", "performance", "phases", "coordination", "fault_isolation", "integrity"):
        if not isinstance(result.get(field), dict):
            errors.append(f"{field} must be an object")

    model_provenance = result.get("model_provenance")
    if isinstance(model_provenance, dict) and any(
        not isinstance(model_provenance.get(field), str) or not model_provenance[field]
        for field in ("model_id", "revision")
    ):
        errors.append("model_provenance must include model_id and exact revision")

    commit_identifiers = result.get("commit_identifiers")
    if isinstance(commit_identifiers, dict):
        if not isinstance(commit_identifiers.get("recall"), dict):
            errors.append("commit_identifiers.recall must be an object")
        if not isinstance(commit_identifiers.get("connected_repositories"), dict):
            errors.append("commit_identifiers.connected_repositories must be an object")

    coordination = result.get("coordination")
    native_event_count = (
        coordination.get("native_component_events_received")
        if isinstance(coordination, dict)
        else None
    )
    native_systems = (
        coordination.get("native_systems_with_events")
        if isinstance(coordination, dict)
        else None
    )

    qualification_status = result.get("qualification_status")
    if not isinstance(qualification_status, str) or qualification_status not in STATUSES:
        errors.append("qualification_status is not a supported status")
    elif qualification_status == "PASS" and any(
        system.get("status") != "PASS" for system in systems.values() if isinstance(system, dict)
    ):
        errors.append("qualification_status cannot be PASS when any system is not PASS")
    elif qualification_status == "PASS" and not isinstance(
        result.get("environment_validation"), dict
    ):
        errors.append("qualification_status cannot be PASS without environment evidence")
    elif qualification_status == "PASS" and result["environment_validation"].get(
        "status"
    ) != "PASS":
        errors.append("qualification_status cannot be PASS when environment gates are incomplete")
    elif qualification_status == "PASS" and not isinstance(result.get("coordination"), dict):
        errors.append("qualification_status cannot be PASS without coordination evidence")
    elif qualification_status == "PASS" and (
        isinstance(native_event_count, bool)
        or not isinstance(native_event_count, int)
        or native_event_count < 4
    ):
        errors.append(
            "qualification_status cannot be PASS without four native component event streams"
        )
    elif qualification_status == "PASS" and (
        coordination.get("status") != "PASS"
        or not isinstance(native_systems, list)
        or any(not isinstance(name, str) for name in native_systems)
        or set(native_systems)
        != {
            "zeta_scaling_toolkit",
            "mathmesh_governor",
            "system_stability_toolkit",
            "deterministic_rag_governor",
        }
    ):
        errors.append("qualification_status cannot be PASS without correlated events from all systems")
    if not isinstance(result.get("commit_identifiers"), dict):
        errors.append("commit_identifiers must be an object")

    if errors:
        raise EvidenceValidationError("Invalid evidence: " + "; ".join(errors))
    return result


def validate_evidence_file(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"Unable to read evidence {path}: {error}") from error
    return validate_evidence(result)


def regenerate_reports_from_evidence(
    evidence_path: Path,
    reports_dir: Path = ROOT / "reports",
    *,
    overwrite: bool = False,
) -> list[Path]:
    result = validate_evidence_file(evidence_path)
    reports = render_reports(result)
    reports_dir.mkdir(parents=True, exist_ok=True)
    destinations = [reports_dir / name for name in reports]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing reports; pass --overwrite-reports to replace "
            "generated Markdown reports from validated evidence: "
            + ", ".join(str(path) for path in existing)
        )
    temporary_paths: list[Path] = []
    try:
        for destination, content in zip(destinations, reports.values()):
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary_paths.append(temporary)
        for temporary, destination in zip(temporary_paths, destinations):
            temporary.replace(destination)
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()
    return destinations


def write_reports(result: dict[str, Any], output_root: Path = ROOT) -> None:
    validate_evidence(result)
    results_dir = output_root / "results"
    reports_dir = output_root / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        results_dir / "RECALL_TEST_RESULTS.json": json.dumps(result, indent=2, sort_keys=True) + "\n",
        **{
            reports_dir / name: body
            for name, body in render_reports(result).items()
        },
    }
    created: list[Path] = []
    try:
        for path, content in outputs.items():
            with path.open("x", encoding="utf-8") as file:
                file.write(content)
            created.append(path)
    except FileExistsError:
        for path in created:
            path.unlink()
        raise RuntimeError(
            "Refusing to overwrite an existing evidence file; choose a fresh checkout "
            "or preserve and review the existing evidence before rerunning."
        ) from None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Recall's non-destructive qualification probes.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Root directory for a new results/ and reports/ evidence set.",
    )
    parser.add_argument("--no-write", action="store_true", help="Run probes without creating evidence files.")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--system",
        choices=(
            "zeta_scaling_toolkit",
            "mathmesh_governor",
            "system_stability_toolkit",
            "deterministic_rag_governor",
        ),
        help="Run one system's safe adapter checks; the complete integration remains unqualified.",
    )
    actions.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only the safe integration-contract and optional read-only health tests.",
    )
    actions.add_argument(
        "--fault-isolation",
        action="store_true",
        help="Run local injected adapter-boundary checks (not live outage qualification).",
    )
    actions.add_argument(
        "--safe-suite",
        action="store_true",
        help="Run full unittest discovery and all safe adapter probes.",
    )
    actions.add_argument(
        "--from-evidence",
        type=Path,
        help="Regenerate Markdown reports from validated saved JSON without rerunning tests.",
    )
    actions.add_argument(
        "--validate-evidence",
        nargs="?",
        const=ROOT / "results" / "RECALL_TEST_RESULTS.json",
        type=Path,
        help="Validate saved JSON evidence without running experiments.",
    )
    parser.add_argument(
        "--overwrite-reports",
        action="store_true",
        help="Allow --from-evidence to replace existing Markdown reports; never replaces JSON evidence.",
    )
    args = parser.parse_args()
    if args.from_evidence:
        paths = regenerate_reports_from_evidence(
            args.from_evidence,
            args.output_dir / "reports",
            overwrite=args.overwrite_reports,
        )
        print(json.dumps({"validated_evidence": str(args.from_evidence), "reports": [
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            for path in paths
        ]}, indent=2))
        return 0
    if args.validate_evidence:
        result = validate_evidence_file(args.validate_evidence)
        print(json.dumps({
            "evidence": str(args.validate_evidence),
            "experiment_id": result["experiment_id"],
            "qualification_status": result["qualification_status"],
            "validation": "PASS",
        }, indent=2))
        return 0
    if args.integration_only:
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_integration")
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.fault_isolation:
        report = run_fault_isolation_checks()
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1

    if args.safe_suite:
        suite = unittest.defaultTestLoader.discover(str(ROOT))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            return 1
        if importlib.util.find_spec("pytest") is None:
            print(
                "The legacy DeepTrace pytest suite requires pytest; install the pinned "
                "development dependencies from requirements-dev.txt.",
                file=sys.stderr,
            )
            return 2
        legacy_tests = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(ROOT / "deeptrace" / "tests"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if legacy_tests.stdout:
            print(legacy_tests.stdout, end="")
        if legacy_tests.stderr:
            print(legacy_tests.stderr, end="", file=sys.stderr)
        if legacy_tests.returncode != 0:
            return legacy_tests.returncode
    selected_systems = {args.system} if args.system else None
    result = run_qualification(
        args.config,
        write_results=not args.no_write,
        selected_systems=selected_systems,
        output_root=args.output_dir,
    )
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "qualification_status": result["qualification_status"],
        "phases": result["phases"],
        "run_scope": result["run_scope"],
        "output_paths": [] if args.no_write else [
            str(args.output_dir / "results" / "RECALL_TEST_RESULTS.json"),
            *[
                str(args.output_dir / "reports" / name)
                for name in render_reports(result)
            ],
        ],
    }, indent=2))
    return 0 if result["qualification_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
