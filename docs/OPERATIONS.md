# Operations Guide - Neural Word Embeddings

The operational contract of the embedding service.

## Running the service

```bash
uv run python -m neural_word_embeddings serve --host 0.0.0.0 --port 8000
```

A standard-library HTTP server with **zero runtime dependencies**. The embeddings are trained
once at startup and then serve stateless queries.

## Endpoints

| Method | Path | Body | Success |
|---|---|---|---|
| GET | `/health` | none | status and vocabulary size (liveness) |
| GET | `/ready` | none | readiness once the embeddings are trained |
| GET | `/metrics` | none | per-route request, error, and latency metrics (Prometheus text) |
| POST | `/similar` | `{"word", "k"?}` | nearest words by cosine |
| POST | `/analogy` | `{"a", "b", "c", "k"?}` | vector-arithmetic analogy |

Errors return JSON `{"error": ...}` with a 4xx status; unknown words and out-of-range `k` are
rejected.

## Service level objectives

Measured by `benchmarks/stress.py` on a developer laptop:

| Objective | Target | Observed (single worker) |
|---|---|---|
| Similar latency p50 | < 1 ms | about 0.13 ms |
| Similar latency p99 | < 1 ms | about 0.23 ms |
| Throughput | > 5000 queries/second/core | about 7000/second |
| Startup training | < 2 s | about 0.8 s |

## Operational constraints

- **Training is a startup cost; queries are cheap.** The embeddings are trained once (well
  under a second on the bundled corpus); each query is a cosine ranking over the vocabulary.
- **Query latency is linear in vocabulary size** (a full nearest-neighbor scan); for large
  vocabularies add an approximate-nearest-neighbor index behind the query interface.
- **The query path is CPU bound and holds the global interpreter lock.** Throughput is flat
  across threads; scale with processes.
- **Requests are bounded**: `k` is capped and words must be in the vocabulary.
- **No persistence, no authentication.** Front the service with a gateway.

## Failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| Slow startup | Large corpus or many epochs | Train offline and load vectors |
| `400` on query | Unknown word or bad `k` | Inspect the `error` field |
| Rising p99 under load | Concurrency above one worker per core | Add replicas; cap concurrency |

## Capacity planning

A single core serves thousands of queries per second. For large vocabularies, replace the
linear scan with an approximate-nearest-neighbor index.
