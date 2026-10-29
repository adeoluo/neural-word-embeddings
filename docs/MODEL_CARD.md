# System Card - Neural Word Embeddings

## Overview

- **What it is:** a skip-gram word-embedding model with negative sampling, trained from
  scratch, with similarity, nearest-neighbor, and analogy queries.
- **What it is not:** a contextual language model; each word has one static vector.
- **Version:** 0.1.0.

## Intended use

- **Primary use cases:** learning how word embeddings are trained; computing static word
  vectors for small corpora.
- **Primary users:** engineers and students studying representation learning.

## Out-of-scope / prohibited use

- Not for web-scale corpora (pure-Python training).
- Out-of-vocabulary words are unsupported.

## Performance

See the README: 1.000 category-retrieval accuracy against a 0.176 random baseline on the
bundled corpus.

## Risks and limitations

- Static embeddings cannot disambiguate word senses.
- Full linear analogies need a larger natural corpus.

## Safety mitigations

- Deterministic under a fixed seed; offline; bounded query parameters in the service.

## Environmental / cost footprint

Negligible; all in-memory, trained in well under a second on the bundled corpus.
