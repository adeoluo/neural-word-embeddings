"""A bundled, semantically structured corpus for demonstration and evaluation.

Words are organized into semantic categories (royalty, countries, capitals,
animals, nature). The generated sentences co-locate words from the same category,
so after training a word's nearest neighbor should belong to its own category.
This supports a standard intrinsic evaluation - category retrieval - that a
from-scratch model can genuinely pass, unlike full linear analogies which need a
large natural corpus.
"""

from __future__ import annotations

import random

CATEGORIES: dict[str, tuple[str, ...]] = {
    "royalty": ("king", "queen", "prince", "princess", "throne", "palace", "crown"),
    "countries": ("france", "japan", "italy", "spain", "germany", "nation", "border"),
    "capitals": ("paris", "tokyo", "rome", "madrid", "berlin", "city", "street"),
    "animals": ("cat", "dog", "bird", "fish", "pet", "paw", "fur"),
    "nature": ("tree", "river", "mountain", "forest", "ocean", "cloud", "rain"),
}


def category_of(word: str) -> str | None:
    """Return the category a word belongs to, or None."""
    for name, members in CATEGORIES.items():
        if word in members:
            return name
    return None


def corpus(*, seed: int = 0) -> list[str]:
    """Generate sentences that co-locate words from the same category."""
    rng = random.Random(seed)
    sentences: list[str] = []
    for members in CATEGORIES.values():
        pool = list(members)
        for _ in range(60):
            length = rng.randint(3, 5)
            words = rng.sample(pool, k=min(length, len(pool)))
            sentences.append(" ".join(words))
    rng.shuffle(sentences)
    return sentences
