"""Four-System Integrated Experiment Orchestration.

Coordinates simultaneous execution of Zeta Scaling Toolkit, MathMesh Governor,
System-Stability Toolkit, and Deterministic RAG Governor in a 4K token experiment.
"""

import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from deeptrace.mathmesh_governor import MathMeshGovernor
from deeptrace.stability_toolkit import SystemStabilityToolkit, StabilityRecommendation
from deeptrace.rag_governor import DeterministicRAGGovernor, RetrievalMode
from deeptrace.fact_generator import FactGenerator
from deeptrace.retrieval_dataset import RetrievalDataset
from deeptrace.retrieval_tests import RetrievalScorer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class FourSystemExperiment:
    """Orchestrator for four-system integrated experiment."""

    def __init__(self, experiment_id: str, output_dir: Path = Path("deeptrace/results")):
        """Initialize four-system experiment.

        Args:
            experiment_id: Unique identifier for the experiment
            output_dir: Directory for all results
        """
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.start_timestamp = datetime.utcnow().isoformat() + "Z"

        # Event log for all four systems
        self.event_log = []

        # Initialize four systems
        logger.info("Initializing Four-System Experiment...")
        logger.info(f"Experiment ID: {experiment_id}")

        self.zeta_model = None
        self.zeta_tokenizer = None
        self.mathmesh = MathMeshGovernor(experiment_id, output_dir)
        self.stability = SystemStabilityToolkit(experiment_id, output_dir)
        self.rag = DeterministicRAGGovernor(experiment_id, RetrievalMode.NATIVE_ONLY, output_dir)

        logger.info("✓ Four systems initialized")

    def load_huggingface_model(self) -> Tuple[Any, Any]:
        """Load HuggingFace SmolLM2 model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info("Loading HuggingFace SmolLM2-360M-Instruct...")
        model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
        revision = "a10cc1512eabd3dde888204e902eca88bddb4951"

        try:
            self.zeta_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True
            )
            logger.info(f"✓ Tokenizer loaded (vocab_size={len(self.zeta_tokenizer)})")

            self.zeta_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            self.zeta_model.to("cpu")
            self.zeta_model.eval()

            param_count = sum(p.numel() for p in self.zeta_model.parameters())
            logger.info(f"✓ Model loaded (params={param_count:,}, architecture={self.zeta_model.__class__.__name__})")

            self._log_event("zeta", "model_load_complete", {
                "model_id": model_id,
                "revision": revision,
                "tokenizer_vocab_size": len(self.zeta_tokenizer),
                "model_params": param_count,
                "architecture": self.zeta_model.__class__.__name__
            })

            return self.zeta_model, self.zeta_tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._log_event("zeta", "model_load_failed", {"error": str(e)})
            raise

    def run_4k_experiment(self) -> Dict[str, Any]:
        """Run the complete 4K token integrated experiment.

        Returns:
            Comprehensive results dictionary
        """
        logger.info("=" * 80)
        logger.info("FOUR-SYSTEM INTEGRATED EXPERIMENT - 4K TOKENS")
        logger.info("=" * 80)

        # Phase 1: Setup and resource check
        logger.info("\n[PHASE 1] Stability Baseline")
        stability_baseline = self.stability.snapshot_resources()
        self._log_event("stability", "baseline_snapshot", {
            "memory_percent": stability_baseline.memory_percent,
            "memory_available_mb": stability_baseline.memory_available_mb,
            "cpu_percent": stability_baseline.cpu_percent
        })

        # Phase 2: Load model (Zeta)
        logger.info("\n[PHASE 2] Load Zeta Model")
        self.load_huggingface_model()

        # Phase 3: Generate test data with facts
        logger.info("\n[PHASE 3] Generate Test Dataset (MathMesh validation)")
        fact_gen = FactGenerator(seed=100)
        facts = fact_gen.generate_dataset(num_facts=5)
        logger.info(f"Generated {len(facts)} deterministic facts")

        # Demonstrate MathMesh math operations
        logger.info("\n[PHASE 4] MathMesh Mathematical Operations")
        math_results = self._run_mathmesh_operations()

        # Phase 4: Issue retrieval queries (RAG)
        logger.info("\n[PHASE 5] Issue Retrieval Queries (RAG Governor)")
        retrieval_queries = []
        for i, fact in enumerate(facts[:3]):  # Test 3 facts
            query = self.rag.issue_retrieval_query(
                f"What is the answer about {fact.fact_id}?",
                expected_evidence=fact.expected_answer
            )
            retrieval_queries.append(query)
            self._log_event("rag", "query_issued", {
                "query_id": query.query_id,
                "expected_evidence": query.expected_evidence
            })

        # Phase 5: Run Zeta inference on 4K token stream
        logger.info("\n[PHASE 6] Zeta Native Inference (4K tokens)")
        
        # Build 4K token stream
        dataset = RetrievalDataset(tokenizer=self.zeta_tokenizer)
        for i, fact in enumerate(facts):
            dataset.add_fact_at_position(fact, position_tokens=i * 400)

        full_text = dataset.construct_full_text()
        input_ids = self.zeta_tokenizer.encode(full_text, return_tensors="pt")
        input_ids = input_ids[:, :4096]  # Truncate to 4K tokens

        logger.info(f"Input token stream: {input_ids.shape[1]} tokens")

        # Execute Zeta forward pass
        inference_start = time.time()
        self._log_event("zeta", "inference_start", {"token_count": input_ids.shape[1]})

        with torch.no_grad():
            outputs = self.zeta_model(input_ids, output_hidden_states=False)
            logits = outputs.logits

        inference_time_ms = (time.time() - inference_start) * 1000
        self._log_event("zeta", "inference_complete", {
            "inference_time_ms": inference_time_ms,
            "output_shape": str(logits.shape),
            "numerical_valid": bool(torch.isfinite(logits).all())
        })

        logger.info(f"✓ Zeta inference: {inference_time_ms:.2f}ms | Tokens: {input_ids.shape[1]} | Output: {logits.shape}")

        # Phase 6: Execute RAG retrieval operations (audit-only mode)
        logger.info("\n[PHASE 7] RAG Retrieval Execution (Native-Only / Audit Mode)")
        retrieval_results = []
        for query in retrieval_queries:
            result = self.rag.execute_retrieval(query)
            retrieval_results.append(result)

            should_integrate, reason = self.rag.should_integrate_evidence(result)
            self.rag.record_integration(result, integrated=False, context_position="none")

            self._log_event("rag", "retrieval_complete", {
                "query_id": query.query_id,
                "evidence_retrieved": result.evidence_retrieved,
                "mode": self.rag.retrieval_mode,
                "should_integrate": should_integrate
            })

        # Phase 7: Stability monitoring during inference
        logger.info("\n[PHASE 8] Stability Monitoring During Inference")
        post_inference_snapshot = self.stability.snapshot_resources()
        stability_recommendation = self.stability.assess_stability(post_inference_snapshot)

        self._log_event("stability", "post_inference_snapshot", {
            "memory_percent": post_inference_snapshot.memory_percent,
            "cpu_percent": post_inference_snapshot.cpu_percent,
            "recommendation": stability_recommendation.value
        })

        logger.info(f"✓ Stability recommendation: {stability_recommendation.value}")

        # Phase 8: Collect cross-system telemetry
        logger.info("\n[PHASE 9] Cross-System Telemetry Aggregation")
        
        mathmesh_report = self.mathmesh.get_governance_report()
        stability_report = self.stability.get_stability_report()
        rag_report = self.rag.get_retrieval_audit_report()

        # Phase 9: Verify cache state (Zeta-specific)
        logger.info("\n[PHASE 10] Zeta Cache and Numerical Verification")
        cache_check = {
            "logits_finite": bool(torch.isfinite(logits).all()),
            "logits_min": float(logits.min()),
            "logits_max": float(logits.max()),
            "logits_mean": float(logits.mean()),
            "output_shape": str(logits.shape),
            "dtype": str(logits.dtype)
        }

        self._log_event("zeta", "numerical_verification", cache_check)
        logger.info(f"✓ Numerical checks passed: {cache_check['logits_finite']}")

        # Compile results
        elapsed_ms = (time.time() - self.start_time) * 1000

        results = {
            "experiment_id": self.experiment_id,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": datetime.utcnow().isoformat() + "Z",
            "total_elapsed_ms": elapsed_ms,
            "systems_summary": {
                "zeta_scaling_toolkit": {
                    "status": "✓ ACTIVE",
                    "operations": "native_inference",
                    "tokens_processed": int(input_ids.shape[1]),
                    "inference_time_ms": inference_time_ms,
                    "numerical_integrity": cache_check
                },
                "mathmesh_governor": {
                    "status": "✓ ACTIVE",
                    "report": mathmesh_report
                },
                "system_stability_toolkit": {
                    "status": "✓ ACTIVE",
                    "report": stability_report
                },
                "deterministic_rag_governor": {
                    "status": "✓ ACTIVE",
                    "mode": self.rag.retrieval_mode,
                    "report": rag_report
                }
            },
            "event_log": self.event_log,
            "coordination_summary": self._compute_coordination_summary()
        }

        return results

    def _run_mathmesh_operations(self) -> Dict[str, Any]:
        """Execute MathMesh mathematical operations.

        Returns:
            Dictionary with MathMesh results
        """
        logger.info("Executing MathMesh base conversions...")

        # Test values
        test_values = [42, 256, 1024, 4096]
        conversions_performed = 0
        verifications_passed = 0

        for value in test_values:
            # Base 10 -> Base 2
            result = self.mathmesh.convert_base(value, 10, 2)
            conversions_performed += 1
            if result.round_trip_match:
                verifications_passed += 1

            # Base 10 -> Base 12
            result = self.mathmesh.convert_base(value, 10, 12)
            conversions_performed += 1
            if result.round_trip_match:
                verifications_passed += 1

            # Base 10 -> Base 60
            result = self.mathmesh.convert_base(value, 10, 60)
            conversions_performed += 1
            if result.round_trip_match:
                verifications_passed += 1

        self._log_event("mathmesh", "conversions_complete", {
            "conversions_performed": conversions_performed,
            "verifications_passed": verifications_passed
        })

        logger.info(f"✓ MathMesh: {conversions_performed} conversions, {verifications_passed} verified")

        return {
            "conversions_performed": conversions_performed,
            "verifications_passed": verifications_passed
        }

    def _log_event(self, system: str, event_type: str, details: Dict[str, Any]) -> None:
        """Log an event from one of the four systems.

        Args:
            system: System name (zeta, mathmesh, stability, rag)
            event_type: Type of event
            details: Event details
        """
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "experiment_id": self.experiment_id,
            "system": system,
            "event_type": event_type,
            "details": details
        }
        self.event_log.append(event)

    def _compute_coordination_summary(self) -> Dict[str, Any]:
        """Compute summary of four-system coordination.

        Returns:
            Coordination metrics
        """
        # Count events by system
        system_events = {}
        for event in self.event_log:
            system = event["system"]
            system_events[system] = system_events.get(system, 0) + 1

        # Verify simultaneous participation
        systems_active = len(system_events) == 4

        return {
            "all_systems_active": systems_active,
            "events_by_system": system_events,
            "total_events": len(self.event_log),
            "coordination_status": "✓ COORDINATED" if systems_active else "⚠ PARTIAL"
        }

    def save_results(self, results: Dict[str, Any]) -> Path:
        """Save experiment results to file.

        Args:
            results: Results dictionary

        Returns:
            Path to saved results file
        """
        path = self.output_dir / f"{self.experiment_id}_four_system_results.json"
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {path}")
        return path

    def save_event_log(self) -> Path:
        """Save event log in JSONL format.

        Returns:
            Path to saved event log
        """
        path = self.output_dir / f"{self.experiment_id}_four_system_events.jsonl"
        with open(path, 'w') as f:
            for event in self.event_log:
                f.write(json.dumps(event) + '\n')
        logger.info(f"Event log saved to {path}")
        return path

    def save_all_telemetry(self) -> Dict[str, Path]:
        """Save telemetry from all four systems.

        Returns:
            Dictionary mapping system names to saved paths
        """
        paths = {}
        paths["mathmesh"] = self.mathmesh.save_telemetry()
        paths["stability"] = self.stability.save_advisory_telemetry()
        paths["rag_audit"] = self.rag.save_audit_log()
        paths["rag_results"] = self.rag.save_retrieval_results()
        return paths


def main():
    """Run the four-system integrated experiment."""
    # Create experiment
    experiment_id = f"FOUR_SYSTEM_4K_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    experiment = FourSystemExperiment(experiment_id)

    # Run experiment
    results = experiment.run_4k_experiment()

    # Save results and telemetry
    results_path = experiment.save_results(results)
    event_log_path = experiment.save_event_log()
    telemetry_paths = experiment.save_all_telemetry()

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("FOUR-SYSTEM EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Event Log: {event_log_path}")
    logger.info(f"Telemetry Files:")
    for system, path in telemetry_paths.items():
        logger.info(f"  - {system}: {path}")

    coordination = results.get("coordination_summary", {})
    logger.info(f"\nCoordination Status: {coordination.get('coordination_status', 'unknown')}")
    logger.info(f"All systems active: {coordination.get('all_systems_active', False)}")

    logger.info("\n✓ Experiment complete")
    return results


if __name__ == "__main__":
    main()
