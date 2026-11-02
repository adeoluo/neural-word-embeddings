"""A deterministic scale benchmark for nearest-neighbor query throughput.

The core serving operation of a word-embedding model is a brute-force cosine
nearest-neighbor search over the embedding matrix (``SkipGramModel.most_similar``
scores every other vector through ``linalg.CosineIndex``, one matrix-vector product
over rows normalized once). This benchmark exercises that exact kernel over
progressively larger synthetic embedding matrices, up to 100,000 vectors. For each
index size it issues a fixed batch of planted queries (a stored vector plus tiny
noise) and reports wall-clock seconds, query throughput in queries per second,
peak memory in MiB measured with the standard-library ``tracemalloc`` module,
and recall@1 of the planted nearest neighbor as the correctness metric.

Timing is measured without tracemalloc (which otherwise slows every allocation);
peak memory is traced separately around the matrix plus a single query scan,
which is faithful because queries are scored one at a time. The vectors are
synthetic. Index construction (normalizing the rows) is timed separately from
the queries. The search is exact, not approximate. Deterministic and offline; run:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import random
import time
import tracemalloc
from dataclasses import dataclass

from neural_word_embeddings.linalg import CosineIndex

WORKLOAD_SIZES: tuple[int, ...] = (1000, 2000, 4000, 8000, 100_000)
DIMENSION = 32
QUERIES = 200
NOISE = 0.01
_BYTES_PER_MIB = 1024 * 1024

Matrix = list[list[float]]
Query = tuple[list[float], int]


@dataclass(frozen=True)
class ScaleRow:
    """One measured workload: size, timing, throughput, memory, and quality."""

    size: int
    build_seconds: float
    seconds: float
    throughput: float
    peak_mib: float
    recall_at_1: float


def _build_index(size: int, rng: random.Random) -> Matrix:
    return [[rng.gauss(0.0, 1.0) for _ in range(DIMENSION)] for _ in range(size)]


def _build_queries(index: Matrix, rng: random.Random) -> list[Query]:
    queries: list[Query] = []
    for _ in range(QUERIES):
        target = rng.randrange(len(index))
        vector = [value + rng.gauss(0.0, NOISE) for value in index[target]]
        queries.append((vector, target))
    return queries


def _nearest(query: list[float], index: CosineIndex) -> int:
    return index.nearest(query, 1)[0][0]


def _peak_mib(size: int) -> float:
    """Peak memory to hold an index of ``size`` vectors and run one query scan."""

    rng = random.Random(0)
    tracemalloc.start()
    tracemalloc.reset_peak()
    index = _build_index(size, rng)
    _nearest(index[0], CosineIndex(index))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / _BYTES_PER_MIB


def run_scale(sizes: tuple[int, ...] = WORKLOAD_SIZES) -> list[ScaleRow]:
    """Measure brute-force nearest-neighbor scaling across index sizes."""

    rows: list[ScaleRow] = []
    for size in sizes:
        rng = random.Random(size)
        index = _build_index(size, rng)
        queries = _build_queries(index, rng)
        start = time.perf_counter()
        search = CosineIndex(index)
        build = time.perf_counter() - start
        start = time.perf_counter()
        hits = sum(1 for vector, target in queries if _nearest(vector, search) == target)
        elapsed = time.perf_counter() - start
        rows.append(
            ScaleRow(
                size=size,
                build_seconds=build,
                seconds=elapsed,
                throughput=len(queries) / elapsed if elapsed else 0.0,
                peak_mib=_peak_mib(size),
                recall_at_1=hits / len(queries),
            )
        )
    return rows


def main() -> None:
    print("Synthetic embedding matrices; exact brute-force cosine search.")
    print(
        f"{'index':>8}  {'build s':>8}  {'seconds':>9}  {'query/sec':>10}  "
        f"{'peak MiB':>9}  {'recall@1':>9}"
    )
    for row in run_scale():
        print(
            f"{row.size:>8d}  {row.build_seconds:>8.3f}  {row.seconds:>9.3f}  "
            f"{row.throughput:>10.2f}  "
            f"{row.peak_mib:>9.2f}  {row.recall_at_1:>9.3f}"
        )


if __name__ == "__main__":
    main()
