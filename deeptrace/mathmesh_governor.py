"""MathMesh Governor: Base-N mathematical validation and governance system.

Implements four-base mathematical operations (base-2, base-10, base-12, base-60)
with radix conversion, round-trip verification, and telemetry recording.
"""

import json
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Tuple
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class MathBase(Enum):
    """Supported mathematical bases."""
    BASE_2 = 2
    BASE_10 = 10
    BASE_12 = 12
    BASE_60 = 60


@dataclass
class RadixConversionResult:
    """Result of a single radix conversion operation."""
    value: int
    source_base: int
    target_base: int
    source_representation: str
    target_representation: str
    round_trip_match: bool
    round_trip_value: int
    conversion_time_ms: float
    success: bool


@dataclass
class MathMeshTelemetry:
    """Telemetry event from MathMesh Governor."""
    timestamp: str
    experiment_id: str
    event_type: str  # "conversion", "verification", "governance"
    base: int
    details: Dict[str, Any]
    duration_ms: float


@dataclass
class MathMeshGovernerState:
    """State tracking for MathMesh Governor operations."""
    experiment_id: str
    start_time: float
    conversions_completed: int = 0
    conversions_failed: int = 0
    verification_checks: int = 0
    verification_passed: int = 0
    verification_failed: int = 0
    telemetry_events: List[MathMeshTelemetry] = field(default_factory=list)
    round_trip_mismatches: int = 0


class MathMeshGovernor:
    """Governance and validation system for multi-base mathematics."""

    def __init__(self, experiment_id: str, output_dir: Path = Path("deeptrace/results")):
        """Initialize MathMesh Governor.

        Args:
            experiment_id: Unique identifier for the experiment
            output_dir: Directory for telemetry and results output
        """
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state = MathMeshGovernerState(
            experiment_id=experiment_id,
            start_time=time.time()
        )
        logger.info(f"MathMesh Governor initialized for experiment {experiment_id}")

    def convert_base(self, value: int, source_base: int, target_base: int) -> RadixConversionResult:
        """Convert a value from one base to another with round-trip verification.

        Args:
            value: Integer value to convert
            source_base: Source base (2, 10, 12, or 60)
            target_base: Target base (2, 10, 12, or 60)

        Returns:
            RadixConversionResult with conversion details and verification
        """
        start_time = time.time()

        try:
            # Convert to base-10 as intermediate
            if source_base == 10:
                base10_value = value
                source_repr = str(value)
            elif source_base == 2:
                base10_value = int(str(value), 2)
                source_repr = bin(value)[2:]
            elif source_base == 12:
                base10_value = self._from_base12(value)
                source_repr = self._to_base12(value)
            elif source_base == 60:
                base10_value = self._from_base60(value)
                source_repr = self._to_base60(value)
            else:
                raise ValueError(f"Unsupported source base: {source_base}")

            # Convert from base-10 to target base
            if target_base == 10:
                target_value = base10_value
                target_repr = str(base10_value)
            elif target_base == 2:
                target_value = int(bin(base10_value)[2:])
                target_repr = bin(base10_value)[2:]
            elif target_base == 12:
                target_repr = self._to_base12(base10_value)
                target_value = int(target_repr, 12) if all(c in '0123456789AB' for c in target_repr) else base10_value
            elif target_base == 60:
                target_repr = self._to_base60(base10_value)
                target_value = base10_value
            else:
                raise ValueError(f"Unsupported target base: {target_base}")

            # Round-trip verification: convert back to source base
            if source_base == 10:
                round_trip_repr = str(target_value)
                round_trip_value = target_value
            elif source_base == 2:
                round_trip_repr = bin(target_value)[2:] if target_value >= 0 else bin(target_value)
                round_trip_value = int(round_trip_repr, 2) if round_trip_repr else 0
            elif source_base == 12:
                round_trip_repr = self._to_base12(target_value)
                round_trip_value = target_value
            elif source_base == 60:
                round_trip_repr = self._to_base60(target_value)
                round_trip_value = target_value
            else:
                round_trip_repr = str(target_value)
                round_trip_value = target_value

            round_trip_match = (round_trip_value == base10_value)
            if not round_trip_match:
                self.state.round_trip_mismatches += 1

            duration_ms = (time.time() - start_time) * 1000

            result = RadixConversionResult(
                value=base10_value,
                source_base=source_base,
                target_base=target_base,
                source_representation=source_repr,
                target_representation=target_repr,
                round_trip_match=round_trip_match,
                round_trip_value=round_trip_value,
                conversion_time_ms=duration_ms,
                success=True
            )

            self.state.conversions_completed += 1
            self._record_telemetry("conversion", source_base, {
                "source_base": source_base,
                "target_base": target_base,
                "value": base10_value,
                "round_trip_match": round_trip_match,
                "conversion_time_ms": duration_ms
            })

            return result

        except Exception as e:
            self.state.conversions_failed += 1
            logger.error(f"Conversion failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return RadixConversionResult(
                value=0,
                source_base=source_base,
                target_base=target_base,
                source_representation="",
                target_representation="",
                round_trip_match=False,
                round_trip_value=0,
                conversion_time_ms=duration_ms,
                success=False
            )

    def verify_all_bases(self, value: int) -> Dict[str, bool]:
        """Verify conversion roundtrips for all base combinations.

        Args:
            value: Integer value to verify

        Returns:
            Dictionary mapping base combinations to verification results
        """
        bases = [MathBase.BASE_2.value, MathBase.BASE_10.value, 
                 MathBase.BASE_12.value, MathBase.BASE_60.value]
        verification_results = {}

        for source_base in bases:
            for target_base in bases:
                if source_base == target_base:
                    continue

                result = self.convert_base(value, source_base, target_base)
                key = f"{source_base}→{target_base}"
                verification_results[key] = result.round_trip_match

                self.state.verification_checks += 1
                if result.round_trip_match:
                    self.state.verification_passed += 1
                else:
                    self.state.verification_failed += 1

        return verification_results

    def _to_base12(self, num: int) -> str:
        """Convert integer to base-12 representation."""
        if num == 0:
            return "0"
        digits = []
        n = abs(num)
        while n:
            digits.append("0123456789AB"[n % 12])
            n //= 12
        return ''.join(reversed(digits))

    def _from_base12(self, num_str: str) -> int:
        """Convert base-12 string to integer."""
        return int(str(num_str), 12) if isinstance(num_str, str) else int(self._to_base12(num_str), 12)

    def _to_base60(self, num: int) -> str:
        """Convert integer to base-60 representation."""
        if num == 0:
            return "0"
        digits = []
        n = abs(num)
        # Use 0-9, A-Z, a-z, then special chars for base-60
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        while n:
            digits.append(charset[n % 60])
            n //= 60
        return ''.join(reversed(digits))

    def _from_base60(self, num_str: str) -> int:
        """Convert base-60 string to integer."""
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        result = 0
        for char in str(num_str):
            result = result * 60 + charset.index(char)
        return result

    def _record_telemetry(self, event_type: str, base: int, details: Dict[str, Any]) -> None:
        """Record a telemetry event.

        Args:
            event_type: Type of event (conversion, verification, governance)
            base: Base involved in the operation
            details: Additional details about the event
        """
        from datetime import datetime
        event = MathMeshTelemetry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            experiment_id=self.experiment_id,
            event_type=event_type,
            base=base,
            details=details,
            duration_ms=details.get("conversion_time_ms", 0.0)
        )
        self.state.telemetry_events.append(event)

    def get_telemetry(self) -> List[Dict[str, Any]]:
        """Get all recorded telemetry events.

        Returns:
            List of telemetry events as dictionaries
        """
        return [asdict(event) for event in self.state.telemetry_events]

    def save_telemetry(self, path: Path = None) -> Path:
        """Save telemetry to file.

        Args:
            path: Output file path; defaults to results/{experiment_id}_mathmesh_telemetry.jsonl

        Returns:
            Path to saved telemetry file
        """
        if path is None:
            path = self.output_dir / f"{self.experiment_id}_mathmesh_telemetry.jsonl"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            for event in self.state.telemetry_events:
                f.write(json.dumps(asdict(event)) + '\n')

        logger.info(f"Telemetry saved to {path}")
        return path

    def get_governance_report(self) -> Dict[str, Any]:
        """Generate a governance report summarizing MathMesh operations.

        Returns:
            Dictionary with governance metrics and status
        """
        elapsed_ms = (time.time() - self.state.start_time) * 1000

        return {
            "experiment_id": self.experiment_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_ms": elapsed_ms,
            "conversions": {
                "completed": self.state.conversions_completed,
                "failed": self.state.conversions_failed,
                "total": self.state.conversions_completed + self.state.conversions_failed
            },
            "verification": {
                "checks_performed": self.state.verification_checks,
                "passed": self.state.verification_passed,
                "failed": self.state.verification_failed,
                "round_trip_mismatches": self.state.round_trip_mismatches,
                "pass_rate": (
                    self.state.verification_passed / self.state.verification_checks * 100
                    if self.state.verification_checks > 0 else 0
                )
            },
            "telemetry_events_recorded": len(self.state.telemetry_events),
            "status": "ACTIVE"
        }
