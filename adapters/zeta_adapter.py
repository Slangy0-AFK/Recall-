from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

from adapters.common import AdapterBlocked, repository_revision, utc_now


class ZetaRunnerError(RuntimeError):
    """A supported Zeta inference runner failed during execution or validation."""


class ZetaModelProvenance(TypedDict):
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    repository_commit: str
    software_version: str
    precision: str


class ZetaCacheMetadata(TypedDict):
    positions: list[int]
    next_position: int


class ZetaInferenceResult(TypedDict):
    status: Literal[
        "PASS", "FAIL", "PARTIAL", "BLOCKED", "NOT TESTED", "NOT APPLICABLE"
    ]
    provenance: ZetaModelProvenance
    input_tokens: int
    actual_tokens_processed: int
    elapsed_seconds: float
    throughput_tokens_per_second: float | None
    numerical_validation: dict[str, bool]
    cache_metadata: ZetaCacheMetadata
    output_metadata: dict[str, Any]


class ZetaInferenceRunner(Protocol):
    """Public contract required before Recall can run model inference qualification."""

    def initialize(
        self, *, model_id: str, revision: str, precision: str
    ) -> ZetaModelProvenance: ...

    def capabilities(self) -> Mapping[str, bool]: ...

    def health(self) -> Mapping[str, Any]: ...

    def infer(self, input_tokens: Sequence[int]) -> ZetaInferenceResult: ...

    def shutdown(self) -> None: ...


class ZetaToolkitAdapter:
    """Exercise Zeta's public attention/cache APIs without inventing a model runner."""

    def __init__(
        self,
        repository: Path | None,
        expected_version: str,
        sequence_length: int = 32,
        head_count: int = 2,
        head_dimension: int = 16,
    ) -> None:
        self.repository = repository
        self.expected_version = expected_version
        self.sequence_length = sequence_length
        self.head_count = head_count
        self.head_dimension = head_dimension
        self._module: Any | None = None

    def initialize(self) -> dict[str, Any]:
        if self._module is None:
            if self.repository is not None:
                package_path = self.repository / "zeta_scaling_toolkit"
                if not package_path.is_dir():
                    raise FileNotFoundError(f"Zeta package not found at {package_path}")
                repository_path = str(self.repository.resolve())
                if repository_path not in sys.path:
                    sys.path.insert(0, repository_path)
            try:
                self._module = importlib.import_module("zeta_scaling_toolkit")
            except ModuleNotFoundError as error:
                if error.name == "zeta_scaling_toolkit":
                    raise AdapterBlocked(
                        "The Zeta Scaling Toolkit package is not installed or configured."
                    ) from error
                raise

        zeta = self._module
        if zeta.__version__ != self.expected_version:
            raise RuntimeError(
                f"Expected Zeta {self.expected_version}, found {zeta.__version__}"
            )
        required_symbols = ("NonCollapsingAttention", "StreamingKVCache")
        missing_symbols = [name for name in required_symbols if not hasattr(zeta, name)]
        if missing_symbols:
            raise RuntimeError(
                f"Zeta package is missing public capabilities: {', '.join(missing_symbols)}"
            )
        return {
            "status": "PASS",
            "timestamp": utc_now(),
            "package_version": zeta.__version__,
            "package_path": str(Path(zeta.__file__).resolve()),
            "repository_commit": repository_revision(self.repository),
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "attention": "zeta_scaling_toolkit.NonCollapsingAttention",
            "kv_cache": "zeta_scaling_toolkit.StreamingKVCache",
            "model_runner": "NOT EXPOSED",
            "tokenizer": "NOT EXPOSED",
            "numerical_validation": "finite output and shape checks",
            "cache_metadata": "absolute positions and next_position",
        }

    def health(self) -> dict[str, Any]:
        initialized = self.initialize()
        return {"status": "PASS", "timestamp": utc_now(), "details": initialized}

    def shutdown(self) -> None:
        return None

    def execute(
        self, operation: str, input_metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        if operation != "component_probe":
            raise ValueError(f"Unsupported Zeta adapter operation: {operation}")
        expected_metadata = {
            "sequence_length": self.sequence_length,
            "head_count": self.head_count,
            "head_dimension": self.head_dimension,
        }
        if dict(input_metadata) != expected_metadata:
            raise ValueError(
                f"Zeta component probe metadata must match its configured input: "
                f"{expected_metadata!r}"
            )
        return self.run_component_probe()

    def run_component_probe(self) -> dict[str, Any]:
        initialized = self.initialize()
        zeta = self._module

        import torch

        generator = torch.Generator(device="cpu").manual_seed(20261118)
        shape = (1, self.head_count, self.sequence_length, self.head_dimension)
        query = torch.randn(shape, generator=generator, dtype=torch.float32)
        key = torch.randn(shape, generator=generator, dtype=torch.float32)
        value = torch.randn(shape, generator=generator, dtype=torch.float32)
        attention = zeta.NonCollapsingAttention(
            d_k=self.head_dimension, check_finite=True
        ).eval()

        started = time.perf_counter()
        with torch.inference_mode():
            output = attention(query, key, value)
        probe_seconds = time.perf_counter() - started

        cache = zeta.StreamingKVCache(window=2, sink_tokens=1)
        for position in range(5):
            token = torch.full(
                (1, self.head_count, 1, self.head_dimension),
                float(position),
                dtype=torch.float32,
            )
            _, _, positions = cache.append(token, token)

        finite = bool(torch.isfinite(output).all())
        expected_positions = [0, 3, 4]
        observed_positions = positions.tolist()
        if output.shape != value.shape or not finite:
            raise ArithmeticError("Zeta attention output has an invalid shape or non-finite values")
        if observed_positions != expected_positions or cache.next_position != 5:
            raise ArithmeticError(
                f"Zeta cache integrity mismatch: positions={observed_positions}, "
                f"next_position={cache.next_position}"
            )

        return {
            "status": "PASS",
            "timestamp": utc_now(),
            "repository_commit": initialized["repository_commit"],
            "package_version": zeta.__version__,
            "package_path": str(Path(zeta.__file__).resolve()),
            "capabilities": self.capabilities(),
            "attention_api": "zeta_scaling_toolkit.NonCollapsingAttention",
            "sequence_length": self.sequence_length,
            "attention_shape": list(output.shape),
            "actual_workload": {
                "attention_sequence_length": self.sequence_length,
                "input_tensor_shape": list(shape),
                "model_tokens_processed": 0,
            },
            "attention_output_finite": finite,
            "attention_probe_seconds": probe_seconds,
            "cache_api": "zeta_scaling_toolkit.StreamingKVCache",
            "cache_positions_after_five_appends": observed_positions,
            "cache_next_position": cache.next_position,
            "cache_state": "COMPONENT_CACHE_PROBE",
            "cache_metadata": {
                "positions": observed_positions,
                "next_position": cache.next_position,
            },
            "model_inference_interface": "UNSUPPORTED",
            "actual_model_tokens_processed": 0,
            "checkpoint_integrity": "NOT TESTED",
            "notes": (
                "The probe exercises toolkit attention/cache primitives only. "
                "The toolkit does not expose a Hugging Face model runner."
            ),
        }
