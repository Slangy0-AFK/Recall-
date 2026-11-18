from __future__ import annotations

import csv
import json
from pathlib import Path

from .core import ExperimentResult


def write_reports(result: ExperimentResult, output_dir: str | Path) -> None:
    """Write JSON, JSONL, CSV, and Markdown artifacts for one experiment."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = result.to_dict()
    (output / "results.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "questions.jsonl").open("w", encoding="utf-8") as stream:
        for question in data["questions"]:
            stream.write(json.dumps(question, sort_keys=True) + "\n")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_id",
                "category",
                "control_type",
                "control_group",
                "control_passed",
                "validity",
                "target_distance",
                "distance_tokens",
                "fact_position",
                "query_position",
                "insertion_position",
                "expected",
                "answer",
                "correct",
                "false_positive",
                "latency_ms",
                "fact_retained",
                "fact_evicted",
                "cache_metadata",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                **question,
                "cache_metadata": json.dumps(question["cache_metadata"], sort_keys=True),
            }
            for question in data["questions"]
        )
    summary = result.summary()
    lines = [
        "# Zeta DeepTrace experiment",
        "",
        f"- Configuration: `{result.configuration}`",
        f"- Dataset: `{result.dataset_id}`",
        f"- Seed: `{result.seed}`",
        f"- Started: `{result.started_at}`",
        f"- Completed: `{result.completed_at}`",
        "- External retrieval: disabled",
        "",
        "## Results",
        "",
        f"- Questions: {summary['question_count']}",
        f"- Valid retrieval questions: {summary['valid_retrieval_count']}",
        f"- Inconclusive retrieval questions: {summary['inconclusive_retrieval_count']}",
        f"- Exact-match accuracy: {summary['exact_match_accuracy']:.3f}",
        f"- Positive-control accuracy: {summary['positive_control_accuracy']}",
        f"- Mean retrieval latency: {summary['mean_retrieval_latency_ms']:.3f} ms",
        f"- False positives: {summary['false_positive_count']}",
        "",
        "## Accuracy by token distance",
        "",
        "| Distance (tokens) | Accuracy | Correct / total |",
        "| ---: | ---: | ---: |",
    ]
    for distance, metric in sorted(
        summary["accuracy_by_distance"].items(), key=lambda item: int(item[0])
    ):
        lines.append(
            f"| {distance} | {metric['accuracy']:.3f} | "
            f"{metric['correct_count']} / {metric['question_count']} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Only the supplied native model adapter is measured. Cache retention is reported "
            "only when the adapter exposes position metadata. These results do not establish "
            "retention beyond the active cache or evaluate external-memory recovery. Long-distance "
            "retrieval results are marked inconclusive when their short-prompt positive control fails.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
