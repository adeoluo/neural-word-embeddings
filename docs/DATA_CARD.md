# Data Card - Neural Word Embeddings

The project bundles one small, synthetic corpus. There is no external training data and
nothing is downloaded.

## bundled corpus (`src/neural_word_embeddings/data.py`)

- **Provenance:** generated for this project. Words are organized into five semantic
  categories (royalty, countries, capitals, animals, nature), and sentences co-locate words
  from the same category so that co-occurrence encodes semantic similarity.
- **Why this shape:** it makes category-retrieval evaluation meaningful and reproducible
  offline, where a word's nearest neighbor should be in its own category.
- **Licensing:** authored by this repository; MIT.
- **Known gaps:** small, English, synthetic; a demonstration corpus, not natural text.

## Ethical considerations

The corpus is synthetic general-knowledge vocabulary with no personal, sensitive, or
copyrighted content.
