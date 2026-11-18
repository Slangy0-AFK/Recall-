from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import random
import string
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence


class Tokenizer(Protocol):
    """The tokenizer used by the model backend; counts are never character estimates."""

    def encode(self, text: str) -> Sequence[int]: ...


class NativeModel(Protocol):
    """A thin adapter over the existing model execution and cache implementation."""

    def set_seed(self, seed: int) -> None: ...

    def reset(self) -> None: ...

    def prefill(self, text: str) -> None: ...

    def generate(self, prompt: str) -> str: ...

    def cache_metadata(self) -> Mapping[str, Any]: ...

    def metadata(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class NeedleCase:
    case_id: str
    category: str
    control_type: str
    control_group: str | None
    context: str
    question: str
    expected: str | None
    fact_position: int | None
    query_position: int
    distance_tokens: int | None
    target_distance: int
    insertion_position: str


@dataclass(frozen=True)
class QuestionResult:
    case_id: str
    category: str
    control_type: str
    control_group: str | None
    control_passed: bool | None
    validity: str
    target_distance: int
    distance_tokens: int | None
    fact_position: int | None
    query_position: int
    insertion_position: str
    expected: str | None
    answer: str
    correct: bool
    false_positive: bool
    latency_ms: float
    cache_metadata: Mapping[str, Any]
    fact_retained: bool | None
    fact_evicted: bool | None


@dataclass(frozen=True)
class ExperimentResult:
    configuration: str
    seed: int
    dataset_id: str
    started_at: str
    completed_at: str
    model_metadata: Mapping[str, Any]
    tokenizer_metadata: Mapping[str, Any]
    questions: tuple[QuestionResult, ...]

    def summary(self) -> dict[str, Any]:
        total = len(self.questions)
        correct = sum(result.correct for result in self.questions)
        retrievals = [result for result in self.questions if result.category != "positive-control"]
        valid_retrievals = [result for result in retrievals if result.validity == "valid"]
        positive_controls = [
            result for result in self.questions if result.category == "positive-control"
        ]
        negative = [result for result in self.questions if result.control_type == "negative"]
        by_distance: dict[str, dict[str, int | float]] = {}
        by_position: dict[str, dict[str, int | float]] = {}
        for result in valid_retrievals:
            if result.distance_tokens is not None and result.control_type != "negative":
                _add_metric(by_distance, str(result.distance_tokens), result.correct)
            if result.control_type != "negative":
                _add_metric(by_position, result.insertion_position, result.correct)
        false_positives = sum(result.false_positive for result in negative)
        return {
            "question_count": total,
            "correct_count": correct,
            "retrieval_question_count": len(retrievals),
            "valid_retrieval_count": len(valid_retrievals),
            "inconclusive_retrieval_count": len(retrievals) - len(valid_retrievals),
            "raw_retrieval_accuracy": (
                sum(result.correct for result in retrievals) / len(retrievals)
                if retrievals
                else 0.0
            ),
            "exact_match_accuracy": (
                sum(result.correct for result in valid_retrievals) / len(valid_retrievals)
                if valid_retrievals
                else 0.0
            ),
            "positive_control_count": len(positive_controls),
            "positive_control_accuracy": (
                sum(result.correct for result in positive_controls) / len(positive_controls)
                if positive_controls
                else None
            ),
            "negative_control_count": len(negative),
            "negative_control_accuracy": (
                sum(result.correct for result in negative) / len(negative) if negative else None
            ),
            "false_positive_count": false_positives,
            "false_positive_rate": (
                false_positives / len(negative)
                if negative
                else None
            ),
            "accuracy_by_distance": by_distance,
            "accuracy_by_insertion_position": by_position,
            "mean_retrieval_latency_ms": (
                sum(result.latency_ms for result in retrievals) / len(retrievals)
                if retrievals
                else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "seed": self.seed,
            "dataset_id": self.dataset_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "model_metadata": dict(self.model_metadata),
            "tokenizer_metadata": dict(self.tokenizer_metadata),
            "summary": self.summary(),
            "questions": [asdict(question) for question in self.questions],
        }


def build_needle_case(
    tokenizer: Tokenizer,
    *,
    seed: int,
    target_distance: int,
    insertion_position: str = "early",
    case_id: str = "needle-0",
    control_group: str | None = None,
) -> NeedleCase:
    """Create one deterministic synthetic retrieval prompt at a tokenizer-measured distance."""
    if target_distance < 1:
        raise ValueError("target_distance must be a positive token count")
    if insertion_position not in {"early", "middle", "late"}:
        raise ValueError("insertion_position must be early, middle, or late")

    rng = random.Random(f"{seed}:{control_group or case_id}")
    key = f"PROJECT-{rng.randrange(10**8):08d}"
    value = "".join(rng.choices(string.ascii_uppercase + string.digits, k=10))
    fact = f"The unique access code for project {key} is {value}."
    prefix_targets = {"early": 0, "middle": target_distance, "late": 3 * target_distance}
    prefix = _pad_to_tokens(tokenizer, "", prefix_targets[insertion_position])
    separator = "\n" if prefix else ""
    fact_position = len(tokenizer.encode(prefix + separator)) if prefix else 0
    context = prefix + separator + fact
    context = _pad_to_tokens(tokenizer, context, fact_position + target_distance)
    query_position = len(tokenizer.encode(context))
    return NeedleCase(
        case_id=case_id,
        category="needle",
        control_type="positive",
        control_group=control_group,
        context=context,
        question=f"What is the unique access code for project {key}? Answer with only the code.",
        expected=value,
        fact_position=fact_position,
        query_position=query_position,
        distance_tokens=query_position - fact_position,
        target_distance=target_distance,
        insertion_position=insertion_position,
    )


def build_positive_control_case(
    tokenizer: Tokenizer,
    *,
    seed: int,
    case_id: str = "positive-control-0",
    control_group: str | None = None,
) -> NeedleCase:
    """Create a short-prompt known-answer control for a retrieval condition."""
    rng = random.Random(f"{seed}:{control_group or case_id}")
    key = f"PROJECT-{rng.randrange(10**8):08d}"
    value = "".join(rng.choices(string.ascii_uppercase + string.digits, k=10))
    fact = f"The unique access code for project {key} is {value}."
    return NeedleCase(
        case_id=case_id,
        category="positive-control",
        control_type="positive",
        control_group=control_group,
        context=fact,
        question=f"What is the unique access code for project {key}? Answer with only the code.",
        expected=value,
        fact_position=0,
        query_position=len(tokenizer.encode(fact)),
        distance_tokens=len(tokenizer.encode(fact)),
        target_distance=0,
        insertion_position="short-control",
    )


def build_negative_case(
    tokenizer: Tokenizer,
    *,
    seed: int,
    target_distance: int,
    case_id: str = "negative-0",
    control_group: str | None = None,
) -> NeedleCase:
    """Create a negative control whose queried key is deliberately absent."""
    if target_distance < 1:
        raise ValueError("target_distance must be a positive token count")
    rng = random.Random(f"{seed}:{case_id}:{target_distance}:negative")
    present_key = f"PROJECT-{rng.randrange(10**8):08d}"
    absent_key = f"PROJECT-{rng.randrange(10**8):08d}"
    while absent_key == present_key:
        absent_key = f"PROJECT-{rng.randrange(10**8):08d}"
    context = f"The unique access code for project {present_key} is NOTRETRIEVAL."
    context = _pad_to_tokens(tokenizer, context, target_distance)
    return NeedleCase(
        case_id=case_id,
        category="needle",
        control_type="negative",
        control_group=control_group,
        context=context,
        question=(
            f"What is the unique access code for project {absent_key}? "
            "If it was not provided, answer UNKNOWN."
        ),
        expected=None,
        fact_position=None,
        query_position=len(tokenizer.encode(context)),
        distance_tokens=None,
        target_distance=target_distance,
        insertion_position="negative-control",
    )


def generate_needle_cases(
    tokenizer: Tokenizer,
    *,
    distances: Sequence[int],
    seed: int = 0,
    cases_per_condition: int = 1,
    positions: Sequence[str] = ("early", "middle", "late"),
    include_negative_controls: bool = True,
) -> tuple[NeedleCase, ...]:
    """Generate reproducible needle cases and optional absent-fact controls."""
    if cases_per_condition < 1:
        raise ValueError("cases_per_condition must be positive")
    cases: list[NeedleCase] = []
    index = 0
    for distance in distances:
        if distance < 1:
            raise ValueError("distances must contain positive token counts")
        for replicate in range(cases_per_condition):
            control_group = f"{distance}-{replicate}"
            cases.append(
                build_positive_control_case(
                    tokenizer,
                    seed=seed + replicate,
                    case_id=f"positive-control-{index}",
                    control_group=control_group,
                )
            )
            index += 1
            for position in positions:
                case_id = f"needle-{index}"
                cases.append(
                    build_needle_case(
                        tokenizer,
                        seed=seed + replicate,
                        target_distance=distance,
                        insertion_position=position,
                        case_id=case_id,
                        control_group=control_group,
                    )
                )
                index += 1
                if include_negative_controls:
                    cases.append(
                        build_negative_case(
                            tokenizer,
                            seed=seed + replicate,
                            target_distance=distance,
                            case_id=f"negative-{index}",
                            control_group=control_group,
                        )
                    )
                    index += 1
    return tuple(cases)


def evaluate(
    model: NativeModel,
    tokenizer: Tokenizer,
    cases: Sequence[NeedleCase],
    *,
    seed: int = 0,
    dataset_id: str = "synthetic-needles",
    tokenizer_metadata: Mapping[str, Any] | None = None,
    allow_extended: bool = False,
) -> ExperimentResult:
    """Run cases through the supplied native model adapter without external retrieval."""
    if not allow_extended and any(case.target_distance > 16_384 for case in cases):
        raise ValueError(
            "Distances above 16,384 tokens require explicit allow_extended=True approval"
        )
    model.set_seed(seed)
    started_at = _utc_now()
    results: list[QuestionResult] = []
    for case in cases:
        actual_query_position = len(tokenizer.encode(case.context))
        if actual_query_position != case.query_position:
            raise ValueError(
                f"Tokenizer position mismatch for {case.case_id}: "
                f"recorded {case.query_position}, measured {actual_query_position}"
            )
        if (
            case.fact_position is not None
            and case.distance_tokens != actual_query_position - case.fact_position
        ):
            raise ValueError(f"Inconsistent fact/query positions for {case.case_id}")
        model.reset()
        model.prefill(case.context)
        cache = dict(model.cache_metadata())
        retained = _retention_status(cache, case.fact_position)
        started = perf_counter()
        answer = str(model.generate(case.question))
        latency_ms = (perf_counter() - started) * 1000
        correct = _is_correct(answer, case.expected)
        results.append(
            QuestionResult(
                case_id=case.case_id,
                category=case.category,
                control_type=case.control_type,
                control_group=case.control_group,
                control_passed=None,
                validity="valid",
                target_distance=case.target_distance,
                distance_tokens=case.distance_tokens,
                fact_position=case.fact_position,
                query_position=case.query_position,
                insertion_position=case.insertion_position,
                expected=case.expected,
                answer=answer,
                correct=correct,
                false_positive=case.expected is None and not _is_unknown(answer),
                latency_ms=latency_ms,
                cache_metadata=cache,
                fact_retained=retained,
                fact_evicted=None if retained is None else not retained,
            )
        )
    controls = {
        result.control_group: result.correct
        for result in results
        if result.category == "positive-control" and result.control_group is not None
    }
    results = [
        replace(
            result,
            control_passed=(
                result.correct
                if result.category == "positive-control"
                else controls.get(result.control_group)
            ),
            validity=(
                "inconclusive"
                if result.category != "positive-control"
                and result.control_group is not None
                and not controls.get(result.control_group, False)
                else "valid"
            ),
        )
        for result in results
    ]
    return ExperimentResult(
        configuration="native_zeta",
        seed=seed,
        dataset_id=dataset_id,
        started_at=started_at,
        completed_at=_utc_now(),
        model_metadata=dict(model.metadata()),
        tokenizer_metadata=dict(tokenizer_metadata or {}),
        questions=tuple(results),
    )


def _pad_to_tokens(tokenizer: Tokenizer, text: str, target: int) -> str:
    filler = " The archive records a routine status update."
    unit_tokens = max(1, len(tokenizer.encode(filler)))
    for _ in range(4):
        deficit = target - len(tokenizer.encode(text))
        if deficit <= 0:
            return text
        text += filler * max(1, deficit // unit_tokens)
    while len(tokenizer.encode(text)) < target:
        text += filler
    return text


def _is_unknown(answer: str) -> bool:
    normalized = " ".join(answer.casefold().split()).strip(" .,!?:;")
    return normalized in {
        "unknown",
        "not provided",
        "not introduced",
        "cannot determine",
        "i don't know",
        "i do not know",
    }


def _is_correct(answer: str, expected: str | None) -> bool:
    if expected is None:
        return _is_unknown(answer)
    normalized = lambda value: " ".join(value.casefold().split()).strip(" .,!?:;")
    return normalized(answer) == normalized(expected)


def _retention_status(cache: Mapping[str, Any], fact_position: int | None) -> bool | None:
    if fact_position is None:
        return None
    positions = cache.get("positions")
    if isinstance(positions, Sequence) and not isinstance(positions, (str, bytes)):
        return fact_position in positions
    start = cache.get("start_position")
    end = cache.get("end_position")
    if isinstance(start, int) and isinstance(end, int):
        return start <= fact_position <= end
    return None


def _add_metric(
    metrics: dict[str, dict[str, int | float]], key: str, correct: bool
) -> None:
    metric = metrics.setdefault(key, {"correct_count": 0, "question_count": 0, "accuracy": 0.0})
    metric["question_count"] += 1
    metric["correct_count"] += int(correct)
    metric["accuracy"] = metric["correct_count"] / metric["question_count"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
