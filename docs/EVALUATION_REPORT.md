# Evaluation Report - Neural Word Embeddings

> Reproduce with `python -m neural_word_embeddings evaluate`. Deterministic and offline.

## Setup

- **Corpus:** a synthetic corpus with five semantic categories whose sentences co-locate
  same-category words.
- **Metric:** category-retrieval accuracy - for each word, is its top-1 nearest neighbor (by
  cosine) in the same category?
- **Baseline:** random guessing, the average probability a random other word shares the
  category.

## Results

| Metric | Value |
|---|---:|
| category-retrieval accuracy | 1.000 |
| random baseline | 0.176 |

## Interpretation

After training, every word's nearest neighbor lies in its own semantic category, giving a
perfect retrieval score against a random baseline of 0.176. This is the standard intrinsic
evaluation of word embeddings, and the from-scratch skip-gram model passes it completely,
confirming that negative-sampling training with hand-derived gradients learns meaningful
geometry. Full linear analogies are a harder test that needs a large natural corpus; the
nearest-neighbor result is the honest, reproducible claim at this scale.

## Reproducibility

Training and sampling are seeded; the numbers regenerate exactly.
