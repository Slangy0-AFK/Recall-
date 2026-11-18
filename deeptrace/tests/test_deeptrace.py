"""Unit tests for DeepTrace components."""

import pytest
import torch
import json
from pathlib import Path
import tempfile

from deeptrace.fact_generator import FactGenerator, Fact, generate_distractors
from deeptrace.retrieval_dataset import RetrievalDataset, TokenPosition
from deeptrace.retrieval_tests import RetrievalResult, RetrievalScorer


class TestFactGenerator:
    """Test fact generation."""

    def test_generator_initialization(self):
        """Test that generator initializes with seed."""
        gen = FactGenerator(seed=42)
        assert gen.seed == 42
        assert gen.fact_counter == 0

    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        gen1 = FactGenerator(seed=42)
        gen2 = FactGenerator(seed=42)

        fact1 = gen1.generate_fact("equipment")
        fact2 = gen2.generate_fact("equipment")

        assert fact1.expected_answer == fact2.expected_answer
        assert fact1.fact_id == fact2.fact_id

    def test_unique_fact_ids(self):
        """Test that fact IDs are unique."""
        gen = FactGenerator(seed=42)
        fact_ids = set()

        for _ in range(10):
            fact = gen.generate_fact("equipment")
            assert fact.fact_id not in fact_ids
            fact_ids.add(fact.fact_id)

    def test_equipment_fact_format(self):
        """Test equipment fact format."""
        gen = FactGenerator(seed=42)
        fact = gen.generate_fact("equipment")

        assert fact.fact_id.startswith("ZD-")
        assert "equipment code" in fact.content
        assert fact.expected_answer.startswith("QX-")
        assert len(fact.expected_answer) == 7  # "QX-XXXX"

    def test_location_fact_format(self):
        """Test location fact format."""
        gen = FactGenerator(seed=42)
        fact = gen.generate_fact("location")

        assert fact.expected_answer.startswith("LM-")
        assert "location code" in fact.content

    def test_project_fact_format(self):
        """Test project fact format."""
        gen = FactGenerator(seed=42)
        fact = gen.generate_fact("project")

        assert fact.expected_answer.startswith("PR-")
        assert "project code" in fact.content

    def test_numeric_fact_format(self):
        """Test numeric fact format."""
        gen = FactGenerator(seed=42)
        fact = gen.generate_fact("numeric")

        assert fact.expected_answer.isdigit()
        assert len(fact.expected_answer) == 5  # 5-digit number

    def test_invalid_fact_type(self):
        """Test that invalid fact type raises error."""
        gen = FactGenerator()
        with pytest.raises(ValueError):
            gen.generate_fact("invalid_type")

    def test_dataset_generation(self):
        """Test generating a full dataset."""
        gen = FactGenerator(seed=42)
        facts = gen.generate_dataset(num_facts=5)

        assert len(facts) == 5
        assert all(isinstance(f, Fact) for f in facts)
        assert len(set(f.fact_id for f in facts)) == 5  # All unique

    def test_ground_truth_save_and_load(self):
        """Test saving and loading ground truth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = FactGenerator(seed=42)
            facts = gen.generate_dataset(num_facts=3)

            path = f"{tmpdir}/ground_truth.json"
            gen.save_ground_truth(facts, path)

            # Verify file was created
            assert Path(path).exists()

            # Load and verify content
            with open(path) as f:
                data = json.load(f)

            assert "facts" in data
            assert len(data["facts"]) == 3
            assert data["facts"][0]["fact_id"] == facts[0].fact_id
            assert data["facts"][0]["expected_answer"] == facts[0].expected_answer


class TestDistractors:
    """Test distractor generation."""

    def test_distractor_generation(self):
        """Test generating distractors."""
        distractors = generate_distractors(num_distractors=5, seed=42)

        assert len(distractors) == 5
        assert all(isinstance(d, str) for d in distractors)
        assert all(len(d) > 0 for d in distractors)

    def test_distractors_deterministic(self):
        """Test distractors are deterministic with same seed."""
        d1 = generate_distractors(num_distractors=5, seed=42)
        d2 = generate_distractors(num_distractors=5, seed=42)

        assert d1 == d2

    def test_distractors_different_seed(self):
        """Test distractors are different with different seed."""
        d1 = generate_distractors(num_distractors=5, seed=42)
        d2 = generate_distractors(num_distractors=5, seed=43)

        assert d1 != d2


class TestRetrievalDataset:
    """Test retrieval dataset creation."""

    def test_dataset_initialization(self):
        """Test dataset initializes correctly."""
        dataset = RetrievalDataset()

        assert len(dataset) == 0
        assert dataset.tokens == []
        assert dataset.fact_positions == {}

    def test_add_fact_to_dataset(self):
        """Test adding a fact to dataset."""
        dataset = RetrievalDataset()
        fact = Fact(
            fact_id="ZD-0001",
            content="The code is ABC-123.",
            expected_answer="ABC-123",
        )

        pos = dataset.add_fact_at_position(fact)

        assert pos == 0  # First fact at position 0
        assert "ZD-0001" in dataset.fact_positions
        assert len(dataset) > 0

    def test_add_multiple_facts(self):
        """Test adding multiple facts preserves positions."""
        dataset = RetrievalDataset()

        fact1 = Fact(
            fact_id="ZD-0001",
            content="First fact is 123.",
            expected_answer="123",
        )
        fact2 = Fact(
            fact_id="ZD-0002",
            content="Second fact is 456.",
            expected_answer="456",
        )

        pos1 = dataset.add_fact_at_position(fact1)
        pos2 = dataset.add_fact_at_position(fact2)

        assert pos1 == 0
        assert pos2 > pos1  # Second fact is after first
        assert dataset.fact_positions["ZD-0001"] == pos1
        assert dataset.fact_positions["ZD-0002"] == pos2

    def test_tokens_as_tensor(self):
        """Test converting tokens to tensor."""
        dataset = RetrievalDataset()
        fact = Fact(
            fact_id="ZD-0001",
            content="Test fact",
            expected_answer="Test",
        )

        dataset.add_fact_at_position(fact)
        tensor = dataset.get_tokens_as_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert len(tensor) == len(dataset)

    def test_token_position_tracking(self):
        """Test that token positions are tracked correctly."""
        dataset = RetrievalDataset()
        fact = Fact(
            fact_id="ZD-0001",
            content="Test",
            expected_answer="Test",
        )

        dataset.add_fact_at_position(fact)

        # Check that positions are tracked
        assert len(dataset.token_positions) == len(dataset.tokens)
        assert all(isinstance(p, TokenPosition) for p in dataset.token_positions)


class TestRetrievalResult:
    """Test retrieval result tracking."""

    def test_result_initialization(self):
        """Test creating a retrieval result."""
        result = RetrievalResult(
            test_id="A-001",
            distance_band=1024,
            fact_id="ZD-0001",
            original_token_position=0,
            query_token_position=1024,
            expected_answer="ABC-123",
            actual_answer="ABC-123",
            is_exact_match=True,
            cache_status="retained",
            retrieval_latency_ms=42.5,
            control_result="pass",
            random_seed=42,
            model_revision="0.1.0",
        )

        assert result.test_id == "A-001"
        assert result.is_exact_match is True
        assert result.expected_answer == "ABC-123"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = RetrievalResult(
            test_id="A-001",
            distance_band=1024,
            fact_id="ZD-0001",
            original_token_position=0,
            query_token_position=1024,
            expected_answer="ABC-123",
            actual_answer="ABC-123",
            is_exact_match=True,
            cache_status="retained",
            retrieval_latency_ms=42.5,
            control_result="pass",
            random_seed=42,
            model_revision="0.1.0",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["test_id"] == "A-001"
        assert result_dict["is_exact_match"] is True


class TestRetrievalScorer:
    """Test scoring and reporting."""

    def test_scorer_initialization(self):
        """Test scorer initializes correctly."""
        scorer = RetrievalScorer()

        assert len(scorer.results) == 0
        assert scorer.get_overall_accuracy() == 0.0

    def test_add_single_result(self):
        """Test adding a single result."""
        scorer = RetrievalScorer()
        result = RetrievalResult(
            test_id="A-001",
            distance_band=1024,
            fact_id="ZD-0001",
            original_token_position=0,
            query_token_position=1024,
            expected_answer="ABC",
            actual_answer="ABC",
            is_exact_match=True,
            cache_status="retained",
            retrieval_latency_ms=42.5,
            control_result="pass",
            random_seed=42,
            model_revision="0.1.0",
        )

        scorer.add_result(result)

        assert len(scorer.results) == 1
        assert scorer.get_overall_accuracy() == 1.0

    def test_multiple_results_accuracy(self):
        """Test accuracy with multiple results."""
        scorer = RetrievalScorer()

        for i in range(10):
            is_match = i < 7  # First 7 match, last 3 don't
            result = RetrievalResult(
                test_id=f"A-{i:03d}",
                distance_band=1024,
                fact_id=f"ZD-{i:04d}",
                original_token_position=0,
                query_token_position=1024,
                expected_answer=f"Answer{i}",
                actual_answer=f"Answer{i}" if is_match else f"Wrong{i}",
                is_exact_match=is_match,
                cache_status="retained",
                retrieval_latency_ms=42.5,
                control_result="pass",
                random_seed=42,
                model_revision="0.1.0",
            )
            scorer.add_result(result)

        accuracy = scorer.get_overall_accuracy()
        assert 0.69 < accuracy < 0.71  # Should be ~0.70

    def test_accuracy_by_distance(self):
        """Test accuracy broken down by distance."""
        scorer = RetrievalScorer()

        # Add results for two distances
        for distance in [1024, 4096]:
            for i in range(5):
                is_match = i < 3  # 3/5 correct per distance
                result = RetrievalResult(
                    test_id=f"A-{distance}-{i}",
                    distance_band=distance,
                    fact_id=f"ZD-{i:04d}",
                    original_token_position=0,
                    query_token_position=distance,
                    expected_answer=f"Answer{i}",
                    actual_answer=f"Answer{i}" if is_match else f"Wrong{i}",
                    is_exact_match=is_match,
                    cache_status="retained",
                    retrieval_latency_ms=42.5,
                    control_result="pass",
                    random_seed=42,
                    model_revision="0.1.0",
                )
                scorer.add_result(result)

        accuracy_by_dist = scorer.get_accuracy_by_distance()

        assert 1024 in accuracy_by_dist
        assert 4096 in accuracy_by_dist
        assert accuracy_by_dist[1024] == 0.6
        assert accuracy_by_dist[4096] == 0.6

    def test_control_pass_rate(self):
        """Test control pass rate calculation."""
        scorer = RetrievalScorer()

        for i in range(5):
            control_result = "pass" if i < 3 else "fail"
            result = RetrievalResult(
                test_id=f"A-{i}",
                distance_band=1024,
                fact_id=f"ZD-{i:04d}",
                original_token_position=0,
                query_token_position=1024,
                expected_answer=f"Answer{i}",
                actual_answer=f"Answer{i}",
                is_exact_match=True,
                cache_status="retained",
                retrieval_latency_ms=42.5,
                control_result=control_result,
                random_seed=42,
                model_revision="0.1.0",
            )
            scorer.add_result(result)

        pass_rate = scorer.get_control_pass_rate()
        assert 0.59 < pass_rate < 0.61  # Should be ~0.60

    def test_generate_report(self):
        """Test generating a comprehensive report."""
        scorer = RetrievalScorer()

        for i in range(3):
            result = RetrievalResult(
                test_id=f"A-{i}",
                distance_band=1024,
                fact_id=f"ZD-{i:04d}",
                original_token_position=0,
                query_token_position=1024,
                expected_answer=f"Answer{i}",
                actual_answer=f"Answer{i}",
                is_exact_match=True,
                cache_status="retained",
                retrieval_latency_ms=42.5,
                control_result="pass",
                random_seed=42,
                model_revision="0.1.0",
            )
            scorer.add_result(result)

        report = scorer.generate_report()

        assert "overall_exact_match_accuracy" in report
        assert "accuracy_by_distance" in report
        assert "control_pass_rate" in report
        assert "total_queries" in report
        assert report["total_queries"] == 3
        assert report["total_correct"] == 3

    def test_save_results(self):
        """Test saving results to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer = RetrievalScorer()

            for i in range(2):
                result = RetrievalResult(
                    test_id=f"A-{i}",
                    distance_band=1024,
                    fact_id=f"ZD-{i:04d}",
                    original_token_position=0,
                    query_token_position=1024,
                    expected_answer=f"Answer{i}",
                    actual_answer=f"Answer{i}",
                    is_exact_match=True,
                    cache_status="retained",
                    retrieval_latency_ms=42.5,
                    control_result="pass",
                    random_seed=42,
                    model_revision="0.1.0",
                )
                scorer.add_result(result)

            path = f"{tmpdir}/results.csv"
            scorer.save_results(path)

            # Check files were created
            assert Path(path).exists()
            assert Path(str(path).replace(".csv", ".json")).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
