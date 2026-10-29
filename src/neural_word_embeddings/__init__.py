"""Neural Word Embeddings - skip-gram with negative sampling from scratch."""

from __future__ import annotations

from neural_word_embeddings.data import CATEGORIES, category_of, corpus
from neural_word_embeddings.evaluation import RetrievalReport, evaluate_retrieval
from neural_word_embeddings.linalg import cosine
from neural_word_embeddings.model import SkipGramModel, TrainingConfig
from neural_word_embeddings.vocabulary import Vocabulary, tokenize

__version__ = "0.1.0"

__all__ = [
    "CATEGORIES",
    "RetrievalReport",
    "SkipGramModel",
    "TrainingConfig",
    "Vocabulary",
    "__version__",
    "category_of",
    "corpus",
    "cosine",
    "evaluate_retrieval",
    "tokenize",
]
