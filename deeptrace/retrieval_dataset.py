"""Build deterministic synthetic information streams with facts at known positions."""

import torch
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import json

from .fact_generator import Fact, generate_distractors


@dataclass
class TokenPosition:
    """Track absolute token position in a stream."""

    absolute_position: int
    fact_id: Optional[str] = None
    is_fact: bool = False


class RetrievalDataset:
    """Build a token stream with facts at known positions."""

    def __init__(self, tokenizer=None):
        """Initialize with optional tokenizer for accurate position tracking."""
        self.tokenizer = tokenizer
        self.tokens: list[int] = []
        self.text_parts: list[str] = []  # Track original text for reconstruction
        self.token_positions: list[TokenPosition] = []
        self.fact_positions: dict[str, int] = {}  # fact_id -> absolute_position
        self.queries: list[dict] = []

    def add_fact_at_position(self, fact: Fact, context_before: list[str] = None, position_tokens: int = None) -> int:
        """
        Add a fact to the stream at a specific position.

        Args:
            fact: The fact to add
            context_before: Optional context text before the fact
            position_tokens: If provided, pad with filler to reach this position

        Returns the absolute token position of the fact content start.
        """
        if context_before is None:
            context_before = []

        # Pad to position if requested
        if position_tokens is not None and len(self.tokens) < position_tokens:
            padding_needed = position_tokens - len(self.tokens)
            padding_text = " ".join([f"filler{i}" for i in range(padding_needed)])
            self._add_text(padding_text, is_fact=False)

        # Add context before the fact
        for context_text in context_before:
            self._add_text(context_text, is_fact=False)

        # Record position before adding fact
        fact_start_pos = len(self.tokens)
        self.fact_positions[fact.fact_id] = fact_start_pos

        # Add the fact content
        self._add_text(fact.content, is_fact=True, fact_id=fact.fact_id)

        return fact_start_pos

    def _add_text(self, text: str, is_fact: bool = False, fact_id: Optional[str] = None) -> None:
        """Add text to the stream, tracking positions."""
        self.text_parts.append(text)
        
        if self.tokenizer:
            # Use actual tokenizer for accurate position tracking
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # Fallback: estimate tokens as words
            token_ids = self._estimate_tokens(text)

        for token_id in token_ids:
            self.tokens.append(token_id)
            self.token_positions.append(
                TokenPosition(
                    absolute_position=len(self.tokens) - 1,
                    fact_id=fact_id if is_fact else None,
                    is_fact=is_fact,
                )
            )

    @staticmethod
    def _estimate_tokens(text: str) -> list[int]:
        """Estimate tokens by word splitting (used when tokenizer unavailable)."""
        words = text.split()
        # Return hashes as token IDs for determinism
        return [hash(word) % 50000 for word in words]

    def add_distractors(self, num_distractors: int = 5, seed: int = 42) -> None:
        """Add unrelated distractor information."""
        distractors = generate_distractors(num_distractors, seed)
        for distractor in distractors:
            self._add_text(distractor, is_fact=False)

    def create_retrieval_question(
        self, fact: Fact, distance_tokens: int
    ) -> dict:
        """
        Create a retrieval question at a specific distance from a fact.

        Args:
            fact: The fact to query about
            distance_tokens: Tokens to process between fact insertion and question

        Returns:
            Query dict with position info and expected answer
        """
        if fact.fact_id not in self.fact_positions:
            raise ValueError(f"Fact {fact.fact_id} not in this dataset")

        fact_pos = self.fact_positions[fact.fact_id]
        query_pos = fact_pos + distance_tokens
        query_pos = min(query_pos, len(self.tokens) - 1)

        question = f"What is the {fact.content.split()[-3]}?"  # Extract identifier type

        return {
            "query_id": f"{fact.fact_id}-query-{distance_tokens}",
            "fact_id": fact.fact_id,
            "question": question,
            "fact_insertion_position": fact_pos,
            "query_position": query_pos,
            "distance_tokens": distance_tokens,
            "expected_answer": fact.expected_answer,
        }

    def get_tokens_as_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return the token stream as a 1D tensor."""
        if device is None:
            device = torch.device("cpu")
        return torch.tensor(self.tokens, dtype=torch.long, device=device)

    def save_manifest(self, path: str) -> None:
        """Save the dataset manifest with position information."""
        manifest = {
            "total_tokens": len(self.tokens),
            "fact_positions": self.fact_positions,
            "queries": self.queries,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def __len__(self) -> int:
        """Return total number of tokens in the stream."""
        return len(self.tokens)

    def construct_full_text(self) -> str:
        """Reconstruct the full text from added parts."""
        return " ".join(self.text_parts)

