"""Minimal dependency-free vector operations for the embedding model."""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = list[float]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, returning 0.0 for a zero vector.

    Each vector is scaled by its largest absolute component before the dot
    product and norms are formed. Cosine similarity is invariant to positive
    per-vector scaling, so this changes nothing mathematically while preventing
    ``norm`` or the dot product from overflowing to ``inf`` (which previously
    produced ``nan``) on vectors with very large magnitudes.
    """
    scale_a = max((abs(x) for x in a), default=0.0)
    scale_b = max((abs(y) for y in b), default=0.0)
    if scale_a == 0.0 or scale_b == 0.0:
        return 0.0
    sa = [x / scale_a for x in a]
    sb = [y / scale_b for y in b]
    denominator = norm(sa) * norm(sb)
    if denominator == 0.0:
        return 0.0
    return dot(sa, sb) / denominator


def add_scaled(target: Vector, source: Sequence[float], scale: float) -> None:
    """In-place ``target += scale * source``."""
    for index in range(len(target)):
        target[index] += scale * source[index]


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exponent = math.exp(x)
    return exponent / (1.0 + exponent)
