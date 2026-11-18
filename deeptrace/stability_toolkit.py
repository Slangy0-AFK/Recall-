"""System-Stability Toolkit: Advisory resource recommendations and monitoring.

Provides resource monitoring, advisory recommendations for inference stability,
and telemetry collection without modifying core model configuration.
"""

import json
import time
import psutil
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ResourceLevel(Enum):
    """Resource availability levels."""
    CRITICAL = "critical"
    LOW = "low"
    ADEQUATE = "adequate"
    OPTIMAL = "optimal"


class StabilityRecommendation(Enum):
    """Advisory recommendations for system stability."""
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    REDUCE_SEQUENCE_LENGTH = "reduce_sequence_length"
    ENABLE_GRADIENT_CHECKPOINTING = "enable_gradient_checkpointing"
    USE_LOWER_PRECISION = "use_lower_precision"
    MONITOR_MEMORY = "monitor_memory"
    OPTIMAL = "system_optimal"


@dataclass
class ResourceSnapshot:
    """Point-in-time resource measurement."""
    timestamp: str
    memory_percent: float
    memory_available_mb: float
    memory_used_mb: float
    cpu_percent: float
    cpu_count: int
    disk_free_gb: float
    process_memory_mb: Optional[float] = None


@dataclass
class StabilityRecommendationRecord:
    """Record of an advisory recommendation."""
    timestamp: str
    experiment_id: str
    recommendation: str
    reasoning: str
    resource_snapshot: ResourceSnapshot
    implementation_advisory: str
    is_critical: bool


@dataclass
class StabilityToolkitState:
    """State tracking for System-Stability Toolkit."""
    experiment_id: str
    start_time: float
    resource_snapshots: List[ResourceSnapshot] = field(default_factory=list)
    recommendations: List[StabilityRecommendationRecord] = field(default_factory=list)
    critical_events: int = 0
    total_advisory_events: int = 0


class SystemStabilityToolkit:
    """Advisory monitoring and stability guidance system."""

    def __init__(
        self,
        experiment_id: str,
        output_dir: Path = Path("deeptrace/results"),
        enable_continuous_monitoring: bool = False
    ):
        """Initialize System-Stability Toolkit.

        Args:
            experiment_id: Unique identifier for the experiment
            output_dir: Directory for telemetry output
            enable_continuous_monitoring: If True, spawn monitoring thread
        """
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state = StabilityToolkitState(
            experiment_id=experiment_id,
            start_time=time.time()
        )

        self.enable_continuous_monitoring = enable_continuous_monitoring
        self.monitoring_active = False

        # Thresholds for advisory (not enforcement)
        self.memory_advisory_threshold_percent = 75.0
        self.cpu_advisory_threshold_percent = 80.0
        self.disk_advisory_threshold_gb = 2.0

        logger.info(f"System-Stability Toolkit initialized for experiment {experiment_id}")

    def snapshot_resources(self) -> ResourceSnapshot:
        """Take a point-in-time resource measurement.

        Returns:
            ResourceSnapshot with current system metrics
        """
        process = psutil.Process()
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage('/')

        snapshot = ResourceSnapshot(
            timestamp=datetime.utcnow().isoformat() + "Z",
            memory_percent=mem.percent,
            memory_available_mb=mem.available / (1024 * 1024),
            memory_used_mb=mem.used / (1024 * 1024),
            cpu_percent=cpu,
            cpu_count=psutil.cpu_count(),
            disk_free_gb=disk.free / (1024**3),
            process_memory_mb=process.memory_info().rss / (1024 * 1024)
        )

        self.state.resource_snapshots.append(snapshot)
        return snapshot

    def assess_stability(self, snapshot: Optional[ResourceSnapshot] = None) -> StabilityRecommendation:
        """Assess system stability and provide advisory recommendation.

        Args:
            snapshot: ResourceSnapshot to analyze; if None, takes new snapshot

        Returns:
            StabilityRecommendation advisory (not enforced)
        """
        if snapshot is None:
            snapshot = self.snapshot_resources()

        recommendation = StabilityRecommendation.OPTIMAL
        reasoning = "System resources are adequate for inference."
        is_critical = False
        advisory_text = "No special action recommended."

        # Advisory logic (not enforcement)
        if snapshot.memory_percent >= self.memory_advisory_threshold_percent:
            recommendation = StabilityRecommendation.MONITOR_MEMORY
            reasoning = f"Memory usage at {snapshot.memory_percent:.1f}%; consider monitoring."
            if snapshot.memory_percent >= 90:
                is_critical = True
                advisory_text = "ADVISORY: Monitor memory closely; consider batch size reduction if OOM risk increases."

        if snapshot.cpu_percent >= self.cpu_advisory_threshold_percent:
            recommendation = StabilityRecommendation.MONITOR_MEMORY  # Could use CPU differently, but group for simplicity
            reasoning = f"CPU usage at {snapshot.cpu_percent:.1f}%; system may be under load."

        if snapshot.disk_free_gb <= self.disk_advisory_threshold_gb:
            recommendation = StabilityRecommendation.REDUCE_BATCH_SIZE
            reasoning = f"Disk space low ({snapshot.disk_free_gb:.1f} GB); may limit logging/checkpointing."

        # Record recommendation (advisory only)
        record = StabilityRecommendationRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            experiment_id=self.experiment_id,
            recommendation=recommendation.value,
            reasoning=reasoning,
            resource_snapshot=snapshot,
            implementation_advisory=advisory_text,
            is_critical=is_critical
        )

        self.state.recommendations.append(record)
        self.state.total_advisory_events += 1
        if is_critical:
            self.state.critical_events += 1

        logger.info(f"Stability assessment: {recommendation.value} | {reasoning}")

        return recommendation

    def get_recommendations_for_phase(self, phase: str) -> Dict[str, Any]:
        """Get advisory recommendations for a specific inference phase.

        Args:
            phase: Description of the phase (e.g., "forward_pass", "generation")

        Returns:
            Dictionary with phase-specific advisory guidance
        """
        latest_snapshot = self.state.resource_snapshots[-1] if self.state.resource_snapshots else self.snapshot_resources()
        recommendation = self.assess_stability(latest_snapshot)

        advisory = {
            "phase": phase,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "current_recommendation": recommendation.value,
            "resource_advisory": {
                "memory_mb": latest_snapshot.memory_used_mb,
                "memory_percent": latest_snapshot.memory_percent,
                "cpu_percent": latest_snapshot.cpu_percent,
                "disk_free_gb": latest_snapshot.disk_free_gb
            },
            "implementation_guidance": self._get_implementation_guidance(recommendation),
            "note": "These are advisory only; core model configuration is not modified."
        }

        return advisory

    def _get_implementation_guidance(self, recommendation: StabilityRecommendation) -> str:
        """Get implementation guidance for a recommendation.

        Args:
            recommendation: The recommendation level

        Returns:
            Human-readable guidance text
        """
        guidance = {
            StabilityRecommendation.REDUCE_BATCH_SIZE: 
                "If using batching, reducing batch_size from N to N/2 may improve stability.",
            StabilityRecommendation.REDUCE_SEQUENCE_LENGTH: 
                "If sequence is longer than necessary, reducing input length may free resources.",
            StabilityRecommendation.ENABLE_GRADIENT_CHECKPOINTING: 
                "If training, enabling gradient_checkpointing=True may reduce peak memory.",
            StabilityRecommendation.USE_LOWER_PRECISION: 
                "Using lower precision (float16) instead of float32 may reduce memory; verify accuracy.",
            StabilityRecommendation.MONITOR_MEMORY: 
                "Monitor memory usage over time; set up alerts if approaching system limits.",
            StabilityRecommendation.OPTIMAL: 
                "System is operating within normal parameters."
        }
        return guidance.get(recommendation, "No specific guidance available.")

    def should_continue_inference(self) -> Tuple[bool, str]:
        """Advisory check: Should inference continue based on resource availability?

        Returns:
            Tuple of (should_continue: bool, reasoning: str)
        """
        snapshot = self.snapshot_resources()

        # This is advisory only - actual decision is external
        if snapshot.memory_percent > 95:
            return False, "ADVISORY: Memory critically high (>95%); recommend pausing to avoid OOM."

        if snapshot.disk_free_gb < 0.5:
            return False, "ADVISORY: Disk space critically low (<500 MB); logging may fail."

        return True, "ADVISORY: System appears stable to continue."

    def get_resource_telemetry(self) -> List[Dict[str, Any]]:
        """Get all resource snapshots as telemetry.

        Returns:
            List of resource snapshots as dictionaries
        """
        return [asdict(snapshot) for snapshot in self.state.resource_snapshots]

    def save_advisory_telemetry(self, path: Path = None) -> Path:
        """Save advisory recommendations and resource telemetry to file.

        Args:
            path: Output file path; defaults to results/{experiment_id}_stability_telemetry.jsonl

        Returns:
            Path to saved telemetry file
        """
        if path is None:
            path = self.output_dir / f"{self.experiment_id}_stability_telemetry.jsonl"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            # Write resource snapshots
            for snapshot in self.state.resource_snapshots:
                f.write(json.dumps({
                    "type": "resource_snapshot",
                    **asdict(snapshot)
                }) + '\n')

            # Write recommendations
            for rec in self.state.recommendations:
                rec_dict = asdict(rec)
                rec_dict["type"] = "advisory_recommendation"
                f.write(json.dumps(rec_dict) + '\n')

        logger.info(f"Stability telemetry saved to {path}")
        return path

    def get_stability_report(self) -> Dict[str, Any]:
        """Generate a comprehensive stability advisory report.

        Returns:
            Dictionary with stability metrics and recommendations
        """
        elapsed_ms = (time.time() - self.state.start_time) * 1000

        # Get current status
        latest_snapshot = self.state.resource_snapshots[-1] if self.state.resource_snapshots else self.snapshot_resources()
        should_continue, reason = self.should_continue_inference()

        # Compute averages
        avg_memory_percent = sum(s.memory_percent for s in self.state.resource_snapshots) / len(self.state.resource_snapshots) if self.state.resource_snapshots else 0
        avg_cpu_percent = sum(s.cpu_percent for s in self.state.resource_snapshots) / len(self.state.resource_snapshots) if self.state.resource_snapshots else 0

        return {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "elapsed_ms": elapsed_ms,
            "current_status": {
                "memory_percent": latest_snapshot.memory_percent,
                "memory_available_mb": latest_snapshot.memory_available_mb,
                "cpu_percent": latest_snapshot.cpu_percent,
                "disk_free_gb": latest_snapshot.disk_free_gb,
                "process_memory_mb": latest_snapshot.process_memory_mb
            },
            "averages": {
                "memory_percent": avg_memory_percent,
                "cpu_percent": avg_cpu_percent
            },
            "advisory": {
                "should_continue": should_continue,
                "reasoning": reason,
                "critical_events": self.state.critical_events,
                "total_advisories": self.state.total_advisory_events
            },
            "measurements": {
                "snapshots_taken": len(self.state.resource_snapshots),
                "recommendations_issued": len(self.state.recommendations)
            },
            "note": "All recommendations are advisory only; core model configuration is not automatically modified."
        }


# Type hint for tuple return
from typing import Tuple
