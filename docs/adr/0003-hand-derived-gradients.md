# ADR-0003: Hand-derived gradients, no autodiff

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

Training needs gradients of the objective. An automatic-differentiation library would provide
them but add a heavy dependency and hide the mechanics the project exists to show.

## Decision

Derive the gradient of the binary logistic objective by hand and apply it directly: the
update is ``gradient = learning_rate * (label - sigmoid(score))`` applied symmetrically to the
center and context vectors.

## Alternatives considered

- **An autodiff framework (PyTorch, JAX).** Convenient and scalable, but a large dependency
  and opaque for a from-scratch reference.
- **Numerical gradients.** Dependency-free but slow and imprecise; the analytic gradient is
  simple here.

## Consequences

- The project has zero runtime dependencies and the full training math is visible.
- The implementation is specific to this objective; a new loss needs a new derivation.
- Pure-Python updates limit scale, which is documented.
