"""Minimal from-scratch vector operations for the embedding model."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

import numpy as np
import numpy.typing as npt

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


class CosineIndex:
    """Cosine nearest-neighbor search over a fixed set of row vectors.

    Each row is scaled by its largest absolute component and then normalized once, as
    :func:`cosine` does per call, so a query costs one matrix-vector product instead of a
    Python loop over every row. Scores match :func:`cosine` to floating-point rounding.
    """

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self._unit = _unit_rows(np.asarray(rows, dtype=np.float64).reshape(len(rows), -1))

    def __len__(self) -> int:
        return int(self._unit.shape[0])

    def scores(self, query: Sequence[float]) -> npt.NDArray[np.float64]:
        """Cosine similarity of ``query`` to every row (0.0 against zero vectors)."""
        unit = _unit_rows(np.asarray(query, dtype=np.float64).reshape(1, -1))[0]
        result: npt.NDArray[np.float64] = self._unit @ unit
        return result

    def nearest(
        self, query: Sequence[float], k: int, *, exclude: Collection[int] = ()
    ) -> list[tuple[int, float]]:
        """The ``k`` rows most similar to ``query``, best first; ties keep row order."""
        scores = self.scores(query)
        candidates = np.ones(len(scores), dtype=bool)
        candidates[list(exclude)] = False
        rows = np.flatnonzero(candidates)
        values = scores[rows]
        if k <= 0 or len(rows) == 0:
            return []
        if len(rows) > k:
            kth = np.partition(values, len(values) - k)[len(values) - k]
            above = np.flatnonzero(values > kth)
            tied = np.flatnonzero(values == kth)[: k - len(above)]
            chosen = np.sort(np.concatenate([above, tied]))
            rows, values = rows[chosen], values[chosen]
        order = np.lexsort((rows, -values))
        return [(int(rows[i]), float(values[i])) for i in order]


def _unit_rows(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    scale = np.abs(matrix).max(axis=1, keepdims=True) if matrix.size else matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(scale > 0.0, matrix / scale, 0.0)
        norms = np.sqrt((scaled * scaled).sum(axis=1, keepdims=True))
        return np.where(norms > 0.0, scaled / norms, 0.0)
