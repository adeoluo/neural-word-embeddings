<h1 align="center">Neural Word Embeddings</h1>

<p align="center">
  <em>Skip-gram word embeddings with negative sampling, built from scratch: hand-derived gradients, a unigram negative-sampling table, and cosine similarity, nearest-neighbor, and analogy queries.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/neural-word-embeddings/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/neural-word-embeddings/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/neural-word-embeddings/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/neural-word-embeddings"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## Motivation

Word embeddings turned natural language processing into a vector problem: represent each
word as a dense vector so that geometry encodes meaning. The skip-gram model with negative
sampling is the method that made this practical. This project implements it **from scratch**
(the vocabulary, the negative-sampling distribution, and the **gradient of the binary
logistic objective derived and applied by hand**) with a deterministic, offline test
suite. There is no autodiff: every gradient is written out explicitly.

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="Architecture diagram" width="860">
</p>

<sub>Legend: amber = entry points and external systems, indigo = core logic, green = data and state, blue = validation and control, grey = observability and outputs.</sub>

The bundled corpus is tokenized into a frequency-ordered **vocabulary**, which feeds the
**skip-gram** trainer and the unigram-to-the-3/4 negative sampler. Training runs once at
startup and learns two embedding matrices by pushing each center word toward its true context
words and away from a few sampled negatives. The center matrix is the word representation: the
CLI and the validated HTTP routes answer similarity and analogy queries from it by cosine
ranking, and the evaluation measures category retrieval accuracy against a random baseline. See the [ADRs](docs/adr/) for the key decisions.

### How it works

For each (center, context) pair the model maximizes the probability that the pair is real and
minimizes it for a few random "negative" pairs. The objective for one pair is the binary
logistic loss, whose gradient is simple and derived by hand:

```
score = center . context
prediction = sigmoid(score)
gradient = learning_rate * (label - prediction)
center  += gradient * context
context += gradient * center
```

Negatives are drawn from the **unigram distribution raised to the 3/4 power**, the standard
word2vec choice that samples frequent words often but not overwhelmingly.

### Components

- **vocabulary**: tokenization and a frequency-ordered vocabulary.
- **linalg**: from-scratch dot product, cosine, and a stable sigmoid.
- **model**: skip-gram with negative sampling and hand-derived gradients.
- **data**: a semantically structured corpus for evaluation.
- **evaluation**: category-retrieval accuracy against a random baseline.

## Results

> Reproduce with `python -m neural_word_embeddings evaluate`. Category-retrieval accuracy:
> for each word, is its nearest neighbor (by cosine) in the same semantic category?

| Metric | Value |
|---|---:|
| **category-retrieval accuracy** | **1.000** |
| random baseline | 0.176 |

The trained embeddings place **every** word's nearest neighbor in its own semantic category
(royalty, countries, capitals, animals, nature), against a random baseline of 0.176. This is
the standard intrinsic evaluation of embeddings, and the from-scratch model passes it
completely. (Full linear analogies in the `king - man + woman` style need a large natural
corpus; the `analogy` command demonstrates the vector arithmetic, and the limit is
noted below.)

### Scale

`benchmarks/scale.py` exercises the core serving kernel, a brute-force cosine nearest-neighbor
scan (the same `linalg.cosine` used by `most_similar`), over progressively larger synthetic
embedding matrices. For each index size it issues a fixed batch of planted queries (a stored
vector plus tiny noise) and records wall-clock time, query throughput, peak memory
(`tracemalloc`), and recall@1 of the planted neighbor. Timing is measured without tracemalloc
(which slows every allocation); peak memory is traced separately around the matrix plus one
query scan. Throughput halves as the index doubles, the signature of linear brute-force search,
while recall@1 stays perfect. Run with `uv run python benchmarks/scale.py`.

| Index vectors | Seconds | Throughput (query/sec) | Peak memory (MiB) | Recall@1 |
|---:|---:|---:|---:|---:|
| 1000 | 1.09 | 22.86 | 1.04 | 1.000 |
| 2000 | 2.26 | 11.08 | 2.07 | 1.000 |
| 4000 | 4.74 | 5.28 | 4.15 | 1.000 |
| 8000 | 9.81 | 2.55 | 8.30 | 1.000 |

Caveat: the vectors are synthetic and this Python implementation is not benchmarked
against compiled vector-search libraries; a production system would use an approximate index
rather than a brute-force scan at this size.

## Quickstart

```bash
git clone https://github.com/adeoluo/neural-word-embeddings.git && cd neural-word-embeddings
make setup                    # install dev tooling + pre-commit hooks
make all                      # lint + type-check + test

python -m neural_word_embeddings evaluate                 # category-retrieval accuracy
python -m neural_word_embeddings similar queen            # nearest neighbors
python -m neural_word_embeddings analogy king queen prince
```

## Usage

### Demonstration

```text
$ python -m neural_word_embeddings similar king
Nearest words to 'king':
  queen        0.94
  prince       0.91
  throne       0.88
  palace       0.85
  crown        0.83
```

### Library

Train on your own text:

```python
from neural_word_embeddings import SkipGramModel, TrainingConfig, Vocabulary, tokenize

sentences = ["the cat sat on the mat", "the dog sat on the log", ...]
tokens = [t for s in sentences for t in tokenize(s)]
vocab = Vocabulary(tokens, min_count=1)
model = SkipGramModel(vocab, TrainingConfig(dimension=64, epochs=10))
model.train([vocab.encode(tokenize(s)) for s in sentences])
print(model.most_similar("cat", k=5))
```

### HTTP service

A standard-library HTTP service trains the embeddings once at startup and serves queries:

```bash
python -m neural_word_embeddings serve --port 8000

curl localhost:8000/health
curl -s -X POST localhost:8000/similar -d '{"word": "king", "k": 5}'
curl -s -X POST localhost:8000/analogy -d '{"a": "king", "b": "queen", "c": "prince"}'
```

See the [operations guide](docs/OPERATIONS.md) for the service contract and constraints.
The service also exposes `GET /ready` (readiness) and `GET /metrics` (per-route request,
error, and latency metrics in Prometheus text format). A production-style stress benchmark
(`uv run python benchmarks/stress.py`) reports throughput and p50/p95/p99 latency; the
scaling path and tradeoffs are in [docs/SCALING.md](docs/SCALING.md) and a full walk-through
is in the [case study](docs/CASE_STUDY.md).

## Design decisions

- **Negative sampling, not full softmax.** A binary logistic objective over a few negatives
  avoids a vocabulary-sized normalization and is what makes training tractable.
  ([ADR-0002](docs/adr/0002-negative-sampling.md))
- **Hand-derived gradients, no autodiff.** The gradient of the logistic objective is written
  out and applied directly, keeping the training math explicit and fully transparent.
  ([ADR-0003](docs/adr/0003-hand-derived-gradients.md))
- **Recording decisions.** ([ADR-0001](docs/adr/0001-record-architecture-decisions.md))

## Testing

```bash
make test                     # uv run pytest
make coverage                 # pytest with an 80% coverage gate
```

The suite has 22 deterministic, offline tests in three modules under `tests/`:

- **Core** (`test_core.py`): tokenization (including Unicode digits and underscores),
  vocabulary frequency ordering and `min_count`, sigmoid bounds, cosine identity and
  stability at large magnitudes, config validation, that training reduces the loss, and that
  similar words cluster.
- **Evaluation and CLI** (`test_evaluation.py`): the bundled corpus is deterministic,
  retrieval beats the random baseline, every category word retrieves within its category, and
  the `evaluate`, `similar`, and `analogy` commands.
- **HTTP service** (`test_service.py`): health and readiness, similarity and analogy
  queries, rejection of unknown words and bad `k`, metrics, and unknown routes.

CI (`.github/workflows/ci.yml`) runs `ruff format --check`, `ruff check`, `mypy src`, and
`pytest` with an 80% coverage gate on Python 3.11 and 3.12, then builds the package and the
Docker image.

## Responsible AI

- **Model card:** [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- **Data card:** [docs/DATA_CARD.md](docs/DATA_CARD.md)
- **Threat model:** [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)

## Limitations

- Full linear analogies (`king - man + woman = queen`) need a large natural corpus and many
  dimensions; on the small bundled corpus the model achieves nearest-neighbor retrieval but not
  every analogy, which the evaluation reports.
- Training runs in interpreted Python, so it is for learning and modest corpora, not web-scale text.
- The vocabulary is closed; out-of-vocabulary words are dropped.

## Roadmap

- [ ] Subsampling of frequent words and dynamic context windows
- [ ] A continuous-bag-of-words objective alongside skip-gram
- [ ] Larger-corpus training with subword units for out-of-vocabulary words

## License

Released under the [MIT License](LICENSE).
