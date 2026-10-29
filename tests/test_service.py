"""Tests for the dependency-free HTTP embedding service (pure handler layer)."""

from __future__ import annotations

from neural_word_embeddings.service import EmbeddingService, handle

_SERVICE = EmbeddingService.trained(epochs=10)


def test_health_and_ready() -> None:
    status, payload = handle("GET", "/health", {}, _SERVICE)
    assert status == 200 and payload["vocabulary_size"] > 0
    status, payload = handle("GET", "/ready", {}, _SERVICE)
    assert status == 200 and payload["status"] == "ready"


def test_similar() -> None:
    status, payload = handle("POST", "/similar", {"word": "king", "k": 3}, _SERVICE)
    assert status == 200
    assert len(payload["neighbors"]) == 3


def test_analogy() -> None:
    status, payload = handle(
        "POST", "/analogy", {"a": "king", "b": "queen", "c": "prince"}, _SERVICE
    )
    assert status == 200
    assert payload["answers"]


def test_similar_rejects_unknown_word() -> None:
    status, payload = handle("POST", "/similar", {"word": "zzz"}, _SERVICE)
    assert status == 400
    assert "word" in payload["error"]


def test_rejects_bad_k() -> None:
    status, _ = handle("POST", "/similar", {"word": "king", "k": 0}, _SERVICE)
    assert status == 400


def test_metrics_and_unknown_route() -> None:
    service = EmbeddingService.trained(epochs=8)
    handle("POST", "/similar", {"word": "cat"}, service)
    text = service.metrics.render_prometheus()
    assert "service_requests_total" in text
    assert 'route="POST /similar"' in text
    status, payload = handle("GET", "/nope", {}, service)
    assert status == 404
    assert "error" in payload
