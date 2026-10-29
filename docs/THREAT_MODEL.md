# Threat Model - Neural Word Embeddings

## Assets

- Integrity of the trained vectors and the similarity and analogy results.

## Trust boundaries

- **Query words and parameters are untrusted input to the service.** They are validated
  against the vocabulary and numeric bounds; there is no path from text to execution.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Unbounded results | Huge `k` | Large responses | `k` bounded to 25 |
| Unknown-word queries | Out-of-vocabulary words | Errors | Validated against the vocabulary with a 400 |
| Irreproducible training | Unseeded initialization | Flaky vectors | Deterministic, seeded training |
| Supply-chain compromise | Malicious dependency | Code execution | Zero runtime dependencies; Dependabot and CodeQL in continuous integration |

## Residual risk

Embeddings reflect their training corpus; on natural text they can encode social biases. The
bundled corpus is synthetic and neutral, and that risk is inherent to the method, not this
implementation.
