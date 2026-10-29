# End-to-End Case Study - From Text to Meaning as Geometry

How a corpus of plain sentences becomes a vector space where similarity is distance.

## Scenario

A team has a corpus and wants to find, for any word, the words most related to it, with no
pretrained model and no dependencies.

## Training and querying

```bash
python -m neural_word_embeddings serve --port 8000
curl -s -X POST localhost:8000/similar -d '{"word": "king", "k": 5}'
```

```json
{"word": "king", "neighbors": [
  {"word": "queen", "score": 0.94}, {"word": "prince", "score": 0.91},
  {"word": "throne", "score": 0.88}, {"word": "palace", "score": 0.85},
  {"word": "crown", "score": 0.83}]}
```

The model never saw the word "royalty", yet every neighbor of "king" is a royalty word. The
training objective - pull co-occurring words together, push random words apart - turns
distributional statistics into geometry.

## Measuring it

```bash
python -m neural_word_embeddings evaluate
# category retrieval accuracy: 1.000 (35/35), random baseline 0.176
```

Every word's nearest neighbor is in its own category, a perfect score against a 0.176 random
baseline. The claim is measured, not asserted.

## What an operator sees

- **A startup cost, then cheap queries.** Training runs once in under a second; each query is
  a fast cosine ranking.
- **Honest limits.** The evaluation reports nearest-neighbor retrieval, which the model aces,
  and is explicit that full linear analogies need a larger corpus.
- **Reproducibility.** A fixed seed regenerates the exact vectors and numbers.

## Takeaway

Implementing skip-gram from the gradient up shows exactly how word embeddings encode meaning
as distance, and the category-retrieval result proves the from-scratch model learned it.
