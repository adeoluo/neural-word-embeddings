"""Evaluate embeddings by category-retrieval accuracy.

For each word, the model's top-1 nearest neighbor should belong to the same
semantic category. Accuracy is the fraction of words whose nearest neighbor is
in-category, compared against a random baseline. A higher-than-baseline score
shows the embeddings captured semantic structure.
"""

from __future__ import annotations

from dataclasses import dataclass

from neural_word_embeddings.data import CATEGORIES, category_of
from neural_word_embeddings.model import SkipGramModel
from neural_word_embeddings.vocabulary import Vocabulary


@dataclass(frozen=True)
class RetrievalReport:
    """Category-retrieval accuracy for the model and a random baseline."""

    total: int
    correct: int
    random_baseline: float

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def format(self) -> str:
        return (
            f"category retrieval accuracy: {self.accuracy:.3f} "
            f"({self.correct}/{self.total}), random baseline {self.random_baseline:.3f}"
        )


def _random_baseline(vocabulary: Vocabulary) -> float:
    # Average probability that a random other word shares the category.
    total = 0.0
    counted = 0
    for word in vocabulary.words:
        category = category_of(word)
        if category is None:
            continue
        same = sum(1 for other in CATEGORIES[category] if other in vocabulary and other != word)
        others = len(vocabulary) - 1
        if others > 0:
            total += same / others
            counted += 1
    return total / counted if counted else 0.0


def evaluate_retrieval(model: SkipGramModel, vocabulary: Vocabulary) -> RetrievalReport:
    """Score top-1 category-retrieval accuracy for the model."""
    total = 0
    correct = 0
    for word in vocabulary.words:
        category = category_of(word)
        if category is None:
            continue
        total += 1
        neighbors = model.most_similar(word, k=1)
        if neighbors and category_of(neighbors[0][0]) == category:
            correct += 1
    return RetrievalReport(
        total=total, correct=correct, random_baseline=_random_baseline(vocabulary)
    )
