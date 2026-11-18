#!/usr/bin/env python3
"""Quick start runner for DeepTrace smoke tests."""

import sys
import torch
from pathlib import Path

# Add parent to path so we can import deeptrace
sys.path.insert(0, str(Path(__file__).parent))

from deeptrace.experiment import DeepTraceConfig, DeepTraceExperiment


def main():
    """Run DeepTrace smoke tests."""
    print("\n" + "=" * 70)
    print("ZETA DEEPTRACE - NATIVE RETRIEVAL TEST")
    print("=" * 70)
    print()

    # Configure experiment
    config = DeepTraceConfig(
        num_facts=5,
        num_distractors=10,
        enable_positive_controls=True,
        enable_negative_controls=True,
        output_dir="deeptrace/results",
    )

    # Create experiment (without model for now - demonstration mode)
    print("[*] Initializing DeepTrace experiment...")
    experiment = DeepTraceExperiment(config=config, model=None, tokenizer=None)

    # Save configuration
    experiment.save_config()

    # Run smoke tests
    print("[*] Running smoke tests (1024 and 4096 tokens)...")
    print()
    experiment.run_smoke_tests()

    # Save summary
    print()
    experiment.save_summary()

    print()
    print("=" * 70)
    print("SMOKE TESTS COMPLETE")
    print("=" * 70)
    print()
    print("Output files in:", config.output_dir)
    print("  - config.json: Experiment configuration")
    print("  - ground_truth.json: Protected ground-truth facts (kept from model)")
    print("  - smoke_test_results.csv/.json: Test results")
    print("  - DEEPTRACE_SUMMARY.json: Final summary")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
