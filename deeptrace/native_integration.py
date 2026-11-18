"""Native Zeta DeepTrace integration for real retrieval testing."""

import json
import torch
import time
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict
import logging

from deeptrace.fact_generator import FactGenerator, Fact
from deeptrace.retrieval_dataset import RetrievalDataset
from deeptrace.retrieval_tests import RetrievalResult, RetrievalScorer
from deeptrace.zeta_inference import ZetaLanguageModel
from deeptrace.tokenizer import SimpleTokenizer


logger = logging.getLogger(__name__)


class NativeZetaIntegration:
    """Execute real retrieval tests through Zeta native math."""

    def __init__(
        self,
        model: Optional[ZetaLanguageModel] = None,
        tokenizer: Optional[SimpleTokenizer] = None,
        device: str = "cpu",
        output_dir: Path = Path("deeptrace/results"),
    ):
        """Initialize native Zeta integration.

        Args:
            model: ZetaLanguageModel instance; if None, creates a new one
            tokenizer: SimpleTokenizer instance; if None, creates a new one
            device: torch device
            output_dir: Directory for results
        """
        self.device = torch.device(device)

        # Create or use provided model
        if model is None:
            logger.info("Creating ZetaLanguageModel...")
            self.model = ZetaLanguageModel(
                vocab_size=10000,
                d_model=256,
                num_layers=2,
                num_heads=8,
                d_ff=512,
                max_seq_len=8192,
                tau0=10.0,
                tau_max=50.0,
            )
        else:
            self.model = model

        self.model = self.model.to(self.device)
        self.model.eval()

        # Create or use provided tokenizer
        if tokenizer is None:
            logger.info("Creating SimpleTokenizer...")
            self.tokenizer = SimpleTokenizer(vocab_size=10000)
        else:
            self.tokenizer = tokenizer

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_1k_test(self, seed: int = 42) -> dict:
        """Run 1024-token retrieval test with native Zeta inference.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Results dictionary with all measurements
        """
        logger.info("=" * 80)
        logger.info("NATIVE ZETA 1K RETRIEVAL TEST")
        logger.info("=" * 80)

        # Generate facts
        logger.info("Generating facts...")
        fact_gen = FactGenerator(seed=seed)
        facts = fact_gen.generate_dataset(num_facts=5)
        logger.info(f"Generated {len(facts)} facts")

        # Build token stream
        logger.info("Building 1024-token stream...")
        dataset = RetrievalDataset(tokenizer=self.tokenizer)
        for i, fact in enumerate(facts):
            pos = i * 100
            dataset.add_fact_at_position(fact, position_tokens=pos)
        logger.info(f"Facts at positions: {list(range(0, 500, 100))}")

        # Create full token stream
        context_text = dataset.construct_full_text()
        token_ids = self.tokenizer.encode(context_text, return_tensors="pt").to(self.device)
        logger.info(f"Full token stream: {token_ids.shape[1]} tokens")

        # Truncate to 1K
        if token_ids.shape[1] > 1024:
            token_ids = token_ids[:, :1024]
            logger.info(f"Truncated to 1024 tokens")

        # Generate positions
        positions = torch.arange(token_ids.shape[1], dtype=torch.int64, device=self.device)

        # Forward pass through Zeta model
        logger.info("Running forward pass through native Zeta attention...")
        start_time = time.time()

        with torch.no_grad():
            embeddings = self.model.forward(token_ids, positions=positions)
        forward_pass_ms = (time.time() - start_time) * 1000

        logger.info(f"Forward pass completed in {forward_pass_ms:.2f}ms")
        logger.info(f"Output shape: {embeddings.shape}")

        # Verify numerical integrity
        if not torch.isfinite(embeddings).all():
            raise ValueError("Model output contains NaN or inf!")

        logger.info("✅ All embeddings are finite")

        # Test retrieval for each fact
        logger.info("Testing retrieval for each fact...")
        results = []
        scorer = RetrievalScorer()

        for fact in facts:
            # Create query
            query_text = f"What is {fact.expected_answer}?"
            query_ids = self.tokenizer.encode(query_text, return_tensors="pt").to(self.device)

            # Find fact position in tokens
            fact_text_in_stream = f"Fact: {fact.expected_answer}"
            fact_pos = context_text.find(fact_text_in_stream)
            if fact_pos == -1:
                fact_pos = fact.insertion_position  # Use actual insertion position

            query_pos = token_ids.shape[1] + 10  # Approximately after context

            # Generate answer
            logger.info(f"  Generating answer for fact {fact.fact_id}...")
            full_input = torch.cat([token_ids, query_ids], dim=1)

            with torch.no_grad():
                answer_logits = self.model.forward(full_input)

            # Extract answer (simple greedy decoding)
            answer_tokens = answer_logits.argmax(dim=-1)
            answer_text = self.tokenizer.decode(answer_tokens[0, -5:].tolist())

            # Exact match check
            is_match = answer_text.lower().strip() == fact.expected_answer.lower().strip()

            # Create result
            result = RetrievalResult(
                test_id=f"1K-{fact.fact_id}",
                distance_band=1024,
                fact_id=fact.fact_id,
                original_token_position=fact_pos,
                query_token_position=query_pos,
                expected_answer=fact.expected_answer,
                actual_answer=answer_text,
                is_exact_match=is_match,
                cache_status="retained",  # At 1K, usually retained
                retrieval_latency_ms=forward_pass_ms / len(facts),
                control_result="test",
                random_seed=seed,
                model_revision="Zeta-Native-0.1.0",
            )

            results.append(result)
            scorer.add_result(result)

            logger.info(
                f"    {fact.fact_id}: Expected='{fact.expected_answer}', "
                f"Got='{answer_text}', Match={is_match}"
            )

        # Generate report
        logger.info("Generating report...")
        report = scorer.generate_report()

        # Save results
        results_data = {
            "test_name": "1K_tokens",
            "seed": seed,
            "num_facts": len(facts),
            "results": [asdict(r) for r in results],
            "summary": report,
            "metadata": {
                "model": "ZetaLanguageModel",
                "forward_pass_ms": forward_pass_ms,
                "token_count": token_ids.shape[1],
                "device": str(self.device),
            },
        }

        logger.info("=" * 80)
        return results_data

    def run_4k_test(self, seed: int = 43) -> dict:
        """Run 4096-token retrieval test with native Zeta inference.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Results dictionary with all measurements
        """
        logger.info("=" * 80)
        logger.info("NATIVE ZETA 4K RETRIEVAL TEST")
        logger.info("=" * 80)

        # Generate facts
        logger.info("Generating facts...")
        fact_gen = FactGenerator(seed=seed)
        facts = fact_gen.generate_dataset(num_facts=5)
        logger.info(f"Generated {len(facts)} facts")

        # Build token stream
        logger.info("Building 4096-token stream...")
        dataset = RetrievalDataset(tokenizer=self.tokenizer)
        for i, fact in enumerate(facts):
            pos = i * 500  # Spread them out more
            dataset.add_fact_at_position(fact, position_tokens=pos)
        logger.info(f"Facts at positions: {list(range(0, 2500, 500))}")

        # Create full token stream
        context_text = dataset.construct_full_text()
        token_ids = self.tokenizer.encode(context_text, return_tensors="pt").to(self.device)
        logger.info(f"Full token stream: {token_ids.shape[1]} tokens")

        # Truncate to 4K
        if token_ids.shape[1] > 4096:
            token_ids = token_ids[:, :4096]
            logger.info(f"Truncated to 4096 tokens")

        # Generate positions
        positions = torch.arange(token_ids.shape[1], dtype=torch.int64, device=self.device)

        # Forward pass through Zeta model
        logger.info("Running forward pass through native Zeta attention...")
        start_time = time.time()

        with torch.no_grad():
            embeddings = self.model.forward(token_ids, positions=positions)
        forward_pass_ms = (time.time() - start_time) * 1000

        logger.info(f"Forward pass completed in {forward_pass_ms:.2f}ms")
        logger.info(f"Output shape: {embeddings.shape}")

        # Verify numerical integrity
        if not torch.isfinite(embeddings).all():
            raise ValueError("Model output contains NaN or inf!")

        logger.info("✅ All embeddings are finite")

        # Test retrieval for each fact
        logger.info("Testing retrieval for each fact...")
        results = []
        scorer = RetrievalScorer()

        for fact in facts:
            # Create query
            query_text = f"What is {fact.expected_answer}?"
            query_ids = self.tokenizer.encode(query_text, return_tensors="pt").to(self.device)

            # Find fact position in tokens
            fact_text_in_stream = f"Fact: {fact.expected_answer}"
            fact_pos = context_text.find(fact_text_in_stream)
            if fact_pos == -1:
                fact_pos = fact.insertion_position  # Use actual insertion position

            query_pos = token_ids.shape[1] + 10

            # Generate answer
            logger.info(f"  Generating answer for fact {fact.fact_id}...")
            full_input = torch.cat([token_ids, query_ids], dim=1)

            with torch.no_grad():
                answer_logits = self.model.forward(full_input)

            # Extract answer
            answer_tokens = answer_logits.argmax(dim=-1)
            answer_text = self.tokenizer.decode(answer_tokens[0, -5:].tolist())

            # Exact match check
            is_match = answer_text.lower().strip() == fact.expected_answer.lower().strip()

            # At 4K, facts are likely outside cache window
            # StreamingKVCache(window=1024, sink_tokens=4) would evict facts
            distance = query_pos - fact_pos
            cache_status = "evicted" if distance > 1024 else "retained"

            # Create result
            result = RetrievalResult(
                test_id=f"4K-{fact.fact_id}",
                distance_band=4096,
                fact_id=fact.fact_id,
                original_token_position=fact_pos,
                query_token_position=query_pos,
                expected_answer=fact.expected_answer,
                actual_answer=answer_text,
                is_exact_match=is_match,
                cache_status=cache_status,
                retrieval_latency_ms=forward_pass_ms / len(facts),
                control_result="test",
                random_seed=seed,
                model_revision="Zeta-Native-0.1.0",
            )

            results.append(result)
            scorer.add_result(result)

            logger.info(
                f"    {fact.fact_id}: Expected='{fact.expected_answer}', "
                f"Got='{answer_text}', Match={is_match}, Cache={cache_status}"
            )

        # Generate report
        logger.info("Generating report...")
        report = scorer.generate_report()

        # Save results
        results_data = {
            "test_name": "4K_tokens",
            "seed": seed,
            "num_facts": len(facts),
            "results": [asdict(r) for r in results],
            "summary": report,
            "metadata": {
                "model": "ZetaLanguageModel",
                "forward_pass_ms": forward_pass_ms,
                "token_count": token_ids.shape[1],
                "device": str(self.device),
            },
        }

        logger.info("=" * 80)
        return results_data
