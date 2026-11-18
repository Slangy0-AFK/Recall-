"""Minimal tokenizer for retrieval tests."""

import torch
from typing import List, Union, Optional


class SimpleTokenizer:
    """Ultra-simple word-based tokenizer for deterministic testing."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 0

        # Reserve special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.bos_token_id = 3
        self.next_id = 4

    def _add_word(self, word: str) -> int:
        """Add word to vocabulary and return its ID."""
        word_lower = word.lower()
        if word_lower not in self.word_to_id:
            if self.next_id >= self.vocab_size:
                return self.unk_token_id
            self.word_to_id[word_lower] = self.next_id
            self.id_to_word[self.next_id] = word_lower
            self.next_id += 1
        return self.word_to_id[word_lower]

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False,
    ) -> Union[List[int], torch.Tensor]:
        """Tokenize text into token IDs."""
        # Simple whitespace + punctuation tokenization
        import re

        words = re.findall(r"\b\w+\b|[.!?]", text.lower())
        token_ids = [self._add_word(word) for word in words]

        if return_tensors == "pt":
            return torch.tensor([token_ids], dtype=torch.long)
        elif return_tensors == "np":
            import numpy as np

            return np.array([token_ids], dtype=np.int64)
        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Convert token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            # Flatten if needed
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]

        words = []
        for token_id in token_ids:
            if isinstance(token_id, (list, tuple)):
                token_id = token_id[0]

            if skip_special_tokens and token_id < 4:
                continue

            word = self.id_to_word.get(token_id, f"<unk_{token_id}>")
            words.append(word)

        return " ".join(words)

    def __len__(self) -> int:
        """Vocabulary size."""
        return self.next_id

    def get_vocab(self) -> dict:
        """Return word to ID mapping."""
        return self.word_to_id.copy()
