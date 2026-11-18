"""Deterministic synthetic information stream generator for retrieval tests."""

import json
import random
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class Fact:
    """A single fact to be inserted into a retrieval test stream."""

    fact_id: str
    content: str
    expected_answer: str
    insertion_position: int = -1  # Will be filled during generation
    token_distance: Optional[int] = None  # Distance from insertion to query


class FactGenerator:
    """Generate deterministic synthetic facts with unique identifiers and expected answers."""

    def __init__(self, seed: int = 42):
        """Initialize the generator with a reproducible random seed."""
        self.seed = seed
        self.rng = random.Random(seed)
        self.fact_counter = 0

    def generate_fact(self, fact_type: str = "equipment") -> Fact:
        """Generate a single unique fact with identifier and expected answer."""
        self.fact_counter += 1

        if fact_type == "equipment":
            prefix = "QX"
            answer = f"{prefix}-{self.rng.randint(1000, 9999)}"
            content = f"The assigned equipment code is {answer}."
        elif fact_type == "location":
            prefix = "LM"
            answer = f"{prefix}-{self.rng.randint(1000, 9999)}"
            content = f"The assigned location code is {answer}."
        elif fact_type == "project":
            prefix = "PR"
            answer = f"{prefix}-{self.rng.randint(1000, 9999)}"
            content = f"The assigned project code is {answer}."
        elif fact_type == "numeric":
            answer = str(self.rng.randint(10000, 99999))
            content = f"The numeric identifier is {answer}."
        else:
            raise ValueError(f"Unknown fact_type: {fact_type}")

        fact_id = f"ZD-{self.fact_counter:04d}"
        return Fact(
            fact_id=fact_id,
            content=content,
            expected_answer=answer,
        )

    def generate_dataset(
        self, num_facts: int = 5, fact_types: Optional[list[str]] = None
    ) -> list[Fact]:
        """Generate a deterministic dataset of facts."""
        if fact_types is None:
            fact_types = ["equipment", "location", "project", "numeric"]

        facts = []
        for i in range(num_facts):
            fact_type = fact_types[i % len(fact_types)]
            facts.append(self.generate_fact(fact_type))

        return facts

    def save_ground_truth(self, facts: list[Fact], path: str) -> None:
        """Save protected ground-truth dataset (kept away from model during execution)."""
        ground_truth = {
            "facts": [
                {
                    "fact_id": f.fact_id,
                    "content": f.content,
                    "expected_answer": f.expected_answer,
                }
                for f in facts
            ]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(ground_truth, f, indent=2)


def generate_distractors(
    num_distractors: int = 10, seed: int = 42
) -> list[str]:
    """Generate realistic distractor information unrelated to facts."""
    rng = random.Random(seed)
    distractors = []

    topics = [
        "weather conditions", "traffic updates", "news headlines",
        "sports scores", "stock prices", "temperature readings",
        "distance measurements", "time values", "count values"
    ]

    for i in range(num_distractors):
        topic = topics[i % len(topics)]
        value = rng.randint(1, 999)
        unit = rng.choice(["meters", "minutes", "degrees", "points", "units"])
        distractor = f"The {topic} reading is {value} {unit}."
        distractors.append(distractor)

    return distractors
