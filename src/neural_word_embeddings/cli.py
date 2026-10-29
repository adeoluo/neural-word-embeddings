"""Command line interface for the word-embedding model."""

from __future__ import annotations

import argparse

from neural_word_embeddings.data import corpus
from neural_word_embeddings.evaluation import evaluate_retrieval
from neural_word_embeddings.model import SkipGramModel, TrainingConfig
from neural_word_embeddings.vocabulary import Vocabulary, tokenize


def _train(epochs: int, dimension: int, seed: int) -> tuple[SkipGramModel, Vocabulary]:
    sentences = corpus()
    tokens = [token for sentence in sentences for token in tokenize(sentence)]
    vocab = Vocabulary(tokens)
    model = SkipGramModel(vocab, TrainingConfig(dimension=dimension, epochs=epochs, seed=seed))
    model.train([vocab.encode(tokenize(sentence)) for sentence in sentences])
    return model, vocab


def _cmd_evaluate(args: argparse.Namespace) -> None:
    model, vocab = _train(args.epochs, args.dimension, args.seed)
    print(evaluate_retrieval(model, vocab).format())


def _cmd_similar(args: argparse.Namespace) -> None:
    model, vocab = _train(args.epochs, args.dimension, args.seed)
    if args.word not in vocab:
        raise SystemExit(f"'{args.word}' is not in the vocabulary")
    print(f"Nearest words to '{args.word}':")
    for word, score in model.most_similar(args.word, k=args.k):
        print(f"  {word:<12} {score:.3f}")


def _cmd_analogy(args: argparse.Namespace) -> None:
    model, vocab = _train(args.epochs, args.dimension, args.seed)
    for word in (args.a, args.b, args.c):
        if word not in vocab:
            raise SystemExit(f"'{word}' is not in the vocabulary")
    print(f"{args.a} is to {args.b} as {args.c} is to:")
    for word, score in model.analogy(args.a, args.b, args.c, k=args.k):
        print(f"  {word:<12} {score:.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="neural-word-embeddings", description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("evaluate", help="category-retrieval accuracy on the bundled corpus")

    similar = sub.add_parser("similar", help="nearest words to a query")
    similar.add_argument("word")
    similar.add_argument("--k", type=int, default=5)

    analogy = sub.add_parser("analogy", help="solve a : b :: c : ? by vector arithmetic")
    analogy.add_argument("a")
    analogy.add_argument("b")
    analogy.add_argument("c")
    analogy.add_argument("--k", type=int, default=3)

    handlers = {"evaluate": _cmd_evaluate, "similar": _cmd_similar, "analogy": _cmd_analogy}
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
