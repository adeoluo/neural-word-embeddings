# ADR-0002: Negative sampling instead of a full softmax

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

The skip-gram objective predicts context words from a center word. A full softmax normalizes
over the entire vocabulary per update, which is expensive and dominates training time.

## Decision

Train a binary logistic objective with negative sampling: for each true (center, context)
pair, sample a few negative words from the unigram distribution raised to the 3/4 power and
push the center vector toward the true context and away from the negatives.

## Alternatives considered

- **Full softmax.** Exact but cost grows with vocabulary size; impractical beyond small sets.
- **Hierarchical softmax.** Logarithmic in vocabulary but needs a Huffman tree and more code;
  negative sampling is simpler and the standard baseline.

## Consequences

- Each update touches a handful of vectors, so training is fast and dependency-free.
- The negative distribution (unigram^0.75) balances sampling frequent and rare words.
- The approximation is to the softmax gradient, which is the accepted word2vec tradeoff.
