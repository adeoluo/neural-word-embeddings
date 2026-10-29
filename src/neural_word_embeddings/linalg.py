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
    """Cosine similarity, returning 0.0 for a zero vector."""
    denominator = norm(a) * norm(b)
    if denominator == 0.0:
        return 0.0
    return dot(a, b) / denominator


def add_scaled(target: Vector, source: Sequence[float], scale: float) -> None:
    """In-place ``target += scale * source``."""
    for index in range(len(target)):
        target[index] += scale * source[index]


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exponent = math.exp(x)
    return exponent / (1.0 + exponent)
