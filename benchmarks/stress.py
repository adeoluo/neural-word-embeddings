"""A production-style stress benchmark for the embedding service.

Trains once, then drives the similar handler under sustained, concurrent load and
reports throughput with p50, p95, and p99 latency. Deterministic and offline;
run with:

    uv run python benchmarks/stress.py
"""

from __future__ import annotations

import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from neural_word_embeddings.service import EmbeddingService, handle


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    service = EmbeddingService.trained(epochs=15)
    words = list(service.vocabulary.words)
    rng = random.Random(0)
    requests = [{"word": rng.choice(words), "k": 5} for _ in range(3000)]

    def one(body: dict[str, object]) -> float:
        start = time.perf_counter()
        handle("POST", "/similar", body, service)
        return (time.perf_counter() - start) * 1000

    print(f"Vocabulary: {len(words)} words | Requests: {len(requests)}")
    for workers in (1, 4, 8):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            latencies = list(pool.map(one, requests))
        wall = time.perf_counter() - start
        print(
            f"workers={workers:<2}  "
            f"throughput={len(requests) / wall:>7.0f} req/s  "
            f"p50={statistics.median(latencies):>6.3f} ms  "
            f"p95={_percentile(latencies, 0.95):>6.3f} ms  "
            f"p99={_percentile(latencies, 0.99):>6.3f} ms"
        )


if __name__ == "__main__":
    main()
