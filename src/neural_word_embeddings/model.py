"""Skip-gram word embeddings with negative sampling, trained from scratch.

The skip-gram objective predicts context words from a center word. Rather than a
full softmax over the vocabulary (expensive), negative sampling trains a binary
logistic objective: push the center word's vector toward each true context word
and away from a few randomly sampled "negative" words. The gradient of that
objective is derived and applied by hand, so the whole training loop is explicit.

Two embedding matrices are learned (center and context); the center matrix is the
word representation used for similarity and analogy.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from neural_word_embeddings.linalg import Vector, add_scaled, cosine, dot, sigmoid
from neural_word_embeddings.vocabulary import Vocabulary


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for skip-gram training."""

    dimension: int = 32
    window: int = 2
    negatives: int = 5
    epochs: int = 5
    learning_rate: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be at least 1")
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.negatives < 1:
            raise ValueError("negatives must be at least 1")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")


class SkipGramModel:
    """Skip-gram with negative sampling over a fixed vocabulary."""

    def __init__(self, vocabulary: Vocabulary, config: TrainingConfig) -> None:
        self._vocabulary = vocabulary
        self._config = config
        self._rng = random.Random(config.seed)
        size = len(vocabulary)
        scale = 0.5 / config.dimension
        self._center: list[Vector] = [
            [(self._rng.random() - 0.5) * scale for _ in range(config.dimension)]
            for _ in range(size)
        ]
        self._context: list[Vector] = [[0.0 for _ in range(config.dimension)] for _ in range(size)]
        self._negative_table = self._build_negative_table(vocabulary)

    def _build_negative_table(self, vocabulary: Vocabulary) -> list[int]:
        # Unigram distribution raised to the 3/4 power, the standard word2vec choice.
        weights = [vocabulary.count(word) ** 0.75 for word in vocabulary.words]
        total = sum(weights)
        table: list[int] = []
        table_size = 1000
        cumulative = 0.0
        index = 0
        for slot in range(table_size):
            target = (slot + 0.5) / table_size
            while index < len(weights) - 1 and cumulative + weights[index] / total < target:
                cumulative += weights[index] / total
                index += 1
            table.append(index)
        return table

    def _sample_negative(self, avoid: int) -> int:
        for _ in range(16):
            candidate = self._negative_table[self._rng.randrange(len(self._negative_table))]
            if candidate != avoid:
                return candidate
        return avoid

    def _update_pair(self, center_index: int, context_index: int, label: float) -> float:
        center = self._center[center_index]
        context = self._context[context_index]
        score = dot(center, context)
        prediction = sigmoid(score)
        gradient = self._config.learning_rate * (label - prediction)
        # Apply the symmetric gradient update to both vectors.
        center_copy = list(center)
        add_scaled(center, context, gradient)
        add_scaled(context, center_copy, gradient)
        probability = prediction if label == 1.0 else 1.0 - prediction
        return -math.log(max(probability, 1e-12))

    def train(self, corpus: Sequence[Sequence[int]]) -> list[float]:
        """Train on encoded sentences, returning the mean loss per epoch."""
        losses: list[float] = []
        for _ in range(self._config.epochs):
            total_loss = 0.0
            pairs = 0
            for sentence in corpus:
                for position, center_index in enumerate(sentence):
                    window = self._rng.randint(1, self._config.window)
                    start = max(0, position - window)
                    end = min(len(sentence), position + window + 1)
                    for context_position in range(start, end):
                        if context_position == position:
                            continue
                        context_index = sentence[context_position]
                        total_loss += self._update_pair(center_index, context_index, 1.0)
                        pairs += 1
                        for _ in range(self._config.negatives):
                            negative = self._sample_negative(context_index)
                            total_loss += self._update_pair(center_index, negative, 0.0)
                            pairs += 1
            losses.append(total_loss / max(pairs, 1))
        return losses

    def vector(self, word: str) -> Vector:
        """Return the learned center vector for a word."""
        return list(self._center[self._vocabulary.index(word)])

    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity between two words' vectors."""
        return cosine(self.vector(left), self.vector(right))

    def most_similar(self, word: str, *, k: int = 5) -> list[tuple[str, float]]:
        """Return the ``k`` nearest words by cosine similarity."""
        target = self.vector(word)
        scored = [
            (other, cosine(target, self._center[self._vocabulary.index(other)]))
            for other in self._vocabulary.words
            if other != word
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def analogy(self, a: str, b: str, c: str, *, k: int = 1) -> list[tuple[str, float]]:
        """Solve ``a is to b as c is to ?`` via vector arithmetic (b - a + c)."""
        target = [
            self.vector(b)[i] - self.vector(a)[i] + self.vector(c)[i]
            for i in range(self._config.dimension)
        ]
        exclude = {a, b, c}
        scored = [
            (word, cosine(target, self._center[self._vocabulary.index(word)]))
            for word in self._vocabulary.words
            if word not in exclude
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
