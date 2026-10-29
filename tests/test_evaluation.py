"""Tests for the bundled corpus, category-retrieval evaluation, and the CLI."""

from __future__ import annotations

from neural_word_embeddings.cli import main
from neural_word_embeddings.data import CATEGORIES, category_of, corpus
from neural_word_embeddings.evaluation import evaluate_retrieval
from neural_word_embeddings.model import SkipGramModel, TrainingConfig
from neural_word_embeddings.vocabulary import Vocabulary, tokenize


def _trained() -> tuple[SkipGramModel, Vocabulary]:
    sentences = corpus()
    tokens = [token for sentence in sentences for token in tokenize(sentence)]
    vocab = Vocabulary(tokens)
    model = SkipGramModel(vocab, TrainingConfig(dimension=32, epochs=15, seed=0))
    model.train([vocab.encode(tokenize(sentence)) for sentence in sentences])
    return model, vocab


def test_category_of() -> None:
    assert category_of("king") == "royalty"
    assert category_of("tokyo") == "capitals"
    assert category_of("notaword") is None


def test_corpus_is_deterministic() -> None:
    assert corpus(seed=0) == corpus(seed=0)
    assert corpus(seed=0) != corpus(seed=1)


def test_retrieval_beats_random_baseline() -> None:
    model, vocab = _trained()
    report = evaluate_retrieval(model, vocab)
    assert report.accuracy > report.random_baseline
    assert report.accuracy >= 0.9  # the model recovers the categories


def test_every_category_word_retrieves_in_category() -> None:
    model, _ = _trained()
    for members in CATEGORIES.values():
        word = members[0]
        neighbor = model.most_similar(word, k=1)[0][0]
        assert category_of(neighbor) == category_of(word)


def test_cli_evaluate(capsys) -> None:
    main(["--epochs", "10", "evaluate"])
    out = capsys.readouterr().out
    assert "category retrieval accuracy" in out


def test_cli_similar(capsys) -> None:
    main(["--epochs", "10", "similar", "king"])
    out = capsys.readouterr().out
    assert "Nearest words" in out


def test_cli_analogy(capsys) -> None:
    main(["--epochs", "10", "analogy", "king", "queen", "prince"])
    out = capsys.readouterr().out
    assert "is to" in out
