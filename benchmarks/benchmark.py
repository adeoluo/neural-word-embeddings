"""A deterministic micro-benchmark for training and query throughput.

Measures training time over the bundled corpus and query throughput for
nearest-neighbor and analogy lookups, so the README can quote concrete numbers.
Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from neural_word_embeddings.data import corpus
from neural_word_embeddings.evaluation import evaluate_retrieval
from neural_word_embeddings.model import SkipGramModel, TrainingConfig
from neural_word_embeddings.vocabulary import Vocabulary, tokenize


def main() -> None:
    sentences = corpus()
    tokens = [token for sentence in sentences for token in tokenize(sentence)]
    vocab = Vocabulary(tokens)
    encoded = [vocab.encode(tokenize(sentence)) for sentence in sentences]

    start = time.perf_counter()
    model = SkipGramModel(vocab, TrainingConfig(dimension=32, epochs=15, seed=0))
    model.train(encoded)
    train_ms = (time.perf_counter() - start) * 1000

    report = evaluate_retrieval(model, vocab)

    start = time.perf_counter()
    for word in vocab.words:
        model.most_similar(word, k=5)
    query_s = time.perf_counter() - start

    print(f"Vocabulary: {len(vocab)} words | Sentences: {len(sentences)}")
    print(f"Training (15 epochs): {train_ms:.0f} ms")
    print(
        f"Category-retrieval accuracy: {report.accuracy:.3f} (random {report.random_baseline:.3f})"
    )
    print(f"Nearest-neighbor queries: {len(vocab) / query_s:.0f} queries/sec")


if __name__ == "__main__":
    main()
