#!/usr/bin/env python3
"""Run native Zeta retrieval tests (1K and 4K tokens)."""

import logging
import json
from pathlib import Path
import torch
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from deeptrace.native_integration import NativeZetaIntegration
from deeptrace.zeta_inference import ZetaLanguageModel
from deeptrace.tokenizer import SimpleTokenizer


def main():
    """Run 1K and 4K native Zeta retrieval tests."""
    logger.info("=" * 80)
    logger.info("NATIVE ZETA DEEPDRACE INTEGRATION TEST")
    logger.info("=" * 80)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Create output directory
    output_dir = Path("deeptrace/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize integration
        logger.info("Initializing NativeZetaIntegration...")
        integration = NativeZetaIntegration(device=device, output_dir=output_dir)
        
        # Run 1K test
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: Running 1K-token test")
        logger.info("=" * 80)
        results_1k = integration.run_1k_test(seed=42)
        
        # Save 1K results
        results_1k_path = output_dir / "DEEPTRACE_1K_RESULTS.json"
        with open(results_1k_path, "w") as f:
            json.dump(results_1k, f, indent=2)
        logger.info(f"Saved 1K results to {results_1k_path}")
        
        # Run 4K test
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: Running 4K-token test")
        logger.info("=" * 80)
        results_4k = integration.run_4k_test(seed=43)
        
        # Save 4K results
        results_4k_path = output_dir / "DEEPTRACE_4K_RESULTS.json"
        with open(results_4k_path, "w") as f:
            json.dump(results_4k, f, indent=2)
        logger.info(f"Saved 4K results to {results_4k_path}")
        
        # Create summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        summary = {
            "1K_Test": {
                "num_facts": results_1k["num_facts"],
                "accuracy": results_1k["summary"]["overall_exact_match_accuracy"],
                "exact_matches": results_1k["summary"]["total_correct"],
                "total_queries": results_1k["summary"]["total_queries"],
            },
            "4K_Test": {
                "num_facts": results_4k["num_facts"],
                "accuracy": results_4k["summary"]["overall_exact_match_accuracy"],
                "exact_matches": results_4k["summary"]["total_correct"],
                "total_queries": results_4k["summary"]["total_queries"],
            },
        }
        
        logger.info(f"1K Test Accuracy: {summary['1K_Test']['accuracy']:.1f}%")
        logger.info(f"4K Test Accuracy: {summary['4K_Test']['accuracy']:.1f}%")
        
        # Save summary
        summary_path = output_dir / "DEEPTRACE_TEST_SUMMARY.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ NATIVE ZETA TESTS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
