"""Deterministic tokenization and vocabulary construction."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

_TOKEN = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphabetic word tokens."""
    return _TOKEN.findall(text.lower())


class Vocabulary:
    """A bidirectional word/index map built from a token frequency threshold."""

    def __init__(self, tokens: Iterable[str], *, min_count: int = 1) -> None:
        counts = Counter(tokens)
        kept = sorted(
            (word for word, count in counts.items() if count >= min_count),
            key=lambda word: (-counts[word], word),
        )
        self._index: dict[str, int] = {word: index for index, word in enumerate(kept)}
        self._words: tuple[str, ...] = tuple(kept)
        self._counts: dict[str, int] = {word: counts[word] for word in kept}

    def __len__(self) -> int:
        return len(self._words)

    def __contains__(self, word: str) -> bool:
        return word in self._index

    def index(self, word: str) -> int:
        return self._index[word]

    def word(self, index: int) -> str:
        return self._words[index]

    @property
    def words(self) -> tuple[str, ...]:
        return self._words

    def count(self, word: str) -> int:
        return self._counts.get(word, 0)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        """Map known tokens to indices, dropping out-of-vocabulary tokens."""
        return [self._index[token] for token in tokens if token in self._index]
