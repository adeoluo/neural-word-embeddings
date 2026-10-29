# Scaling and Design Tradeoffs - Neural Word Embeddings

How the reference design grows, and the tradeoffs taken.

## Reference architecture (today)

A single process trains skip-gram embeddings over an in-memory corpus and serves similarity
and analogy queries over a standard-library HTTP service. Deterministic for a fixed seed.

## Scaling path

| Stage | Bottleneck | Change |
|---|---|---|
| One process | Query CPU, global interpreter lock | One replica per core |
| Large vocabulary | Linear nearest-neighbor scan | Add an approximate-nearest-neighbor index |
| Large corpus | Pure-Python training | Train offline; load precomputed vectors |
| Out-of-vocabulary words | Closed vocabulary | Add subword units |
| Many replicas | Repeated training at startup | Train once; ship vectors as an artifact |

## Substantial tradeoffs

- **Negative sampling vs full softmax.** Negative sampling makes training tractable and is the
  standard word2vec objective, at the cost of approximating the softmax gradient. Full softmax
  is exact but scales with vocabulary size.
- **Hand-derived gradients vs autodiff.** Hand gradients keep the project dependency-free and
  transparent but are specific to this objective; autodiff generalizes at the cost of a heavy
  dependency.
- **Linear scan vs an index.** A full cosine scan is exact and simple for a small vocabulary;
  an approximate index trades a little recall for sublinear query time at scale.
- **Static vs contextual embeddings.** Static vectors are cheap and interpretable but assign
  one vector per word; contextual models disambiguate senses at far greater cost.

## What would change under real load

Train offline and ship the vectors as an artifact, serve from process-per-core replicas, and
add an approximate-nearest-neighbor index for large vocabularies. The training math and the
query interface are unchanged.
