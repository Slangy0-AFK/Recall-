import csv
import json
from pathlib import Path
import re
import tempfile
import unittest

from deeptrace import (
    build_negative_case,
    build_needle_case,
    evaluate,
    generate_needle_cases,
    write_reports,
)


class WordTokenizer:
    def encode(self, text):
        return re.findall(r"\S+", text)


class FakeNativeModel:
    def __init__(self, tokenizer, answers):
        self.tokenizer = tokenizer
        self.answers = answers
        self.context = ""
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def reset(self):
        self.context = ""

    def prefill(self, text):
        self.context = text

    def generate(self, prompt):
        return self.answers[(self.context, prompt)]

    def cache_metadata(self):
        return {"positions": list(range(len(self.tokenizer.encode(self.context))))}

    def metadata(self):
        return {"model_id": "test-native-adapter"}


class DeepTraceTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_needle_positions_are_measured_by_tokenizer(self):
        case = build_needle_case(
            self.tokenizer,
            seed=14,
            target_distance=80,
            insertion_position="middle",
        )
        self.assertEqual(case.query_position, len(self.tokenizer.encode(case.context)))
        self.assertEqual(case.distance_tokens, case.query_position - case.fact_position)
        self.assertGreaterEqual(case.distance_tokens, case.target_distance)
        self.assertTrue(case.expected in case.context)

    def test_case_generation_is_reproducible_and_includes_negative_controls(self):
        kwargs = dict(
            distances=[32, 64],
            seed=7,
            cases_per_condition=1,
            positions=("early",),
        )
        first = generate_needle_cases(self.tokenizer, **kwargs)
        second = generate_needle_cases(self.tokenizer, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(
            [case.category for case in first],
            ["positive-control", "needle", "needle"] * 2,
        )

    def test_negative_case_does_not_contain_queried_fact(self):
        case = build_negative_case(self.tokenizer, seed=2, target_distance=40)
        queried_key = re.search(r"project (PROJECT-\d+)", case.question).group(1)
        self.assertNotIn(queried_key, case.context)
        self.assertIsNone(case.expected)

    def test_native_evaluation_tracks_accuracy_latency_and_cache(self):
        cases = generate_needle_cases(
            self.tokenizer,
            distances=[24],
            seed=3,
            positions=("early",),
        )
        answers = {
            (case.context, case.question): (
                "UNKNOWN" if case.control_type == "negative" else case.expected
            )
            for case in cases
        }
        model = FakeNativeModel(self.tokenizer, answers)
        result = evaluate(
            model,
            self.tokenizer,
            cases,
            seed=3,
            tokenizer_metadata={"tokenizer_id": "test-word-tokenizer"},
        )
        summary = result.summary()
        self.assertEqual(result.configuration, "native_zeta")
        self.assertEqual(model.seed, 3)
        self.assertEqual(summary["exact_match_accuracy"], 1.0)
        self.assertEqual(summary["false_positive_count"], 0)
        needle = next(question for question in result.questions if question.category == "needle")
        self.assertTrue(needle.fact_retained)
        self.assertTrue(needle.control_passed)
        self.assertEqual(needle.validity, "valid")
        self.assertGreaterEqual(needle.latency_ms, 0)

    def test_failed_short_control_marks_long_retrieval_inconclusive(self):
        cases = generate_needle_cases(
            self.tokenizer,
            distances=[24],
            seed=3,
            positions=("early",),
        )
        answers = {
            (case.context, case.question): (
                "WRONG"
                if case.category == "positive-control"
                else "UNKNOWN"
                if case.control_type == "negative"
                else case.expected
            )
            for case in cases
        }
        result = evaluate(FakeNativeModel(self.tokenizer, answers), self.tokenizer, cases)
        needle = next(question for question in result.questions if question.category == "needle")
        self.assertEqual(needle.validity, "inconclusive")
        self.assertFalse(needle.control_passed)
        self.assertEqual(result.summary()["valid_retrieval_count"], 0)

    def test_extended_experiment_requires_explicit_approval(self):
        cases = generate_needle_cases(
            self.tokenizer,
            distances=[16_385],
            seed=3,
            positions=("early",),
            include_negative_controls=False,
        )
        with self.assertRaisesRegex(ValueError, "allow_extended=True"):
            evaluate(FakeNativeModel(self.tokenizer, {}), self.tokenizer, cases)

    def test_reports_write_json_jsonl_csv_and_markdown(self):
        cases = generate_needle_cases(
            self.tokenizer,
            distances=[16],
            seed=1,
            positions=("early",),
        )
        answers = {
            (case.context, case.question): (
                "UNKNOWN" if case.control_type == "negative" else case.expected
            )
            for case in cases
        }
        result = evaluate(FakeNativeModel(self.tokenizer, answers), self.tokenizer, cases)
        with tempfile.TemporaryDirectory() as directory:
            write_reports(result, directory)
            output = Path(directory)
            expected_files = {"results.json", "questions.jsonl", "summary.csv", "report.md"}
            self.assertEqual({path.name for path in output.iterdir()}, expected_files)
            payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["configuration"], "native_zeta")
            self.assertEqual(len((output / "questions.jsonl").read_text().splitlines()), 3)
            with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            self.assertIn("External retrieval: disabled", (output / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
