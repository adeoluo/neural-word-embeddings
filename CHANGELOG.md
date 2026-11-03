# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.1.0] 2026-11-03

### Features

- Add vocabulary, vector operations, and the skip-gram model
- Add structured corpus, category-retrieval evaluation, and CLI
- Add HTTP service with observability, benchmarks, and documentation

### Bug fixes

- Keep Unicode tokens and stabilize cosine for large vectors (#11)

### Performance

- Vectorized exact cosine search for most_similar and analogy (#3)

### Documentation

- Use plain, factual wording (#9)
- Use plain wording (replace a line-wrapped phrase missed earlier) (#10)
- Standard README structure and language framing (#13)
- Vertical layered architecture diagram
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Recorded demos of more commands, with their vhs tapes (#1)
- Recorded demos of the HTTP service and more commands (#2)
- Describe the supply chain accurately (#4)

### Build and CI

- Run CodeQL on main and on a weekly schedule only

### Maintenance

- Scaffold neural-word-embeddings project
- Bump continuous integration action versions and dev dependencies
- Remove Dependabot configuration
