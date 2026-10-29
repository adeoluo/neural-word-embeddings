"""Tests for tokenization, the vocabulary, and the skip-gram model."""

from __future__ import annotations

import pytest

from neural_word_embeddings import SkipGramModel, TrainingConfig, Vocabulary, cosine, tokenize
from neural_word_embeddings.linalg import sigmoid


def test_tokenize() -> None:
    assert tokenize("The cat, the dog!") == ["the", "cat", "the", "dog"]


def test_tokenize_preserves_unicode_digits_and_underscores() -> None:
    # Regression: an ASCII-only [a-z]+ tokenizer silently dropped accented text,
    # non-Latin scripts, digits, and underscored identifiers.
    assert tokenize("café") == ["café"]
    assert tokenize("word_embedding") == ["word_embedding"]
    assert tokenize("word2vec") == ["word2vec"]
    assert tokenize("東京") == ["東京"]
    assert tokenize("covid19 Rocket_Launch") == ["covid19", "rocket_launch"]


def test_vocabulary_frequency_order_and_min_count() -> None:
    vocab = Vocabulary(["a", "a", "a", "b", "b", "c"], min_count=2)
    assert "c" not in vocab
    assert vocab.words[0] == "a"  # most frequent first
    assert vocab.count("a") == 3
    assert vocab.encode(["a", "c", "b"]) == [vocab.index("a"), vocab.index("b")]


def test_sigmoid_bounds() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < sigmoid(-50.0) < 1e-10
    assert 1.0 - sigmoid(50.0) < 1e-10


def _toy_corpus() -> tuple[Vocabulary, list[list[int]]]:
    # Two tight semantic clusters that should become similar after training.
    sentences = [
        "king queen royal palace throne",
        "queen king royal throne palace",
        "cat dog pet animal fur",
        "dog cat animal pet fur",
    ] * 30
    tokens = [token for sentence in sentences for token in tokenize(sentence)]
    vocab = Vocabulary(tokens)
    corpus = [vocab.encode(tokenize(sentence)) for sentence in sentences]
    return vocab, corpus


def test_training_reduces_loss() -> None:
    vocab, corpus = _toy_corpus()
    model = SkipGramModel(vocab, TrainingConfig(dimension=16, epochs=6, seed=0))
    losses = model.train(corpus)
    assert losses[-1] < losses[0]


def test_similar_words_cluster() -> None:
    vocab, corpus = _toy_corpus()
    model = SkipGramModel(vocab, TrainingConfig(dimension=24, epochs=10, seed=1))
    model.train(corpus)
    # Words that co-occur should be more similar than words that do not.
    assert model.similarity("king", "queen") > model.similarity("king", "dog")
    assert model.similarity("cat", "dog") > model.similarity("cat", "palace")


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(dimension=0)
    with pytest.raises(ValueError):
        TrainingConfig(negatives=0)


def test_cosine_identity() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_is_stable_for_large_magnitudes() -> None:
    # Regression: norm(a) * norm(b) overflowed to inf for very large vectors,
    # bypassing the zero guard and returning nan.
    assert cosine([1e200, 1e200], [1e200, 1e200]) == pytest.approx(1.0)
    assert cosine([1e200, 0.0], [0.0, 1e200]) == pytest.approx(0.0)
    assert cosine([1e-200, 2e-200], [2e-200, 4e-200]) == pytest.approx(1.0)
