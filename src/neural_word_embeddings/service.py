"""A dependency-free HTTP service exposing the trained word embeddings.

The service trains the embeddings once at startup over the bundled corpus and
then answers similarity, nearest-neighbor, and analogy queries. A pure handler is
separated from the socket binding for testing.

Endpoints
---------
``GET  /health``    liveness and vocabulary size.
``GET  /ready``     readiness once the embeddings are trained.
``GET  /metrics``   per-route request, error, and latency metrics (Prometheus text).
``POST /similar``   body ``{"word": str, "k"?: int}``; nearest words by cosine.
``POST /analogy``   body ``{"a", "b", "c", "k"?}``; vector-arithmetic analogy.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from neural_word_embeddings.data import corpus
from neural_word_embeddings.model import SkipGramModel, TrainingConfig
from neural_word_embeddings.observability import ServiceMetrics
from neural_word_embeddings.vocabulary import Vocabulary, tokenize

JsonObject = dict[str, Any]
_MAX_K = 25


@dataclass(frozen=True)
class EmbeddingService:
    """Holds the trained model and answers embedding queries."""

    model: SkipGramModel
    vocabulary: Vocabulary
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def trained(cls, *, epochs: int = 15, dimension: int = 32, seed: int = 0) -> EmbeddingService:
        sentences = corpus()
        tokens = [token for sentence in sentences for token in tokenize(sentence)]
        vocab = Vocabulary(tokens)
        model = SkipGramModel(vocab, TrainingConfig(dimension=dimension, epochs=epochs, seed=seed))
        model.train([vocab.encode(tokenize(sentence)) for sentence in sentences])
        return cls(model=model, vocabulary=vocab)

    def similar(self, word: str, k: int) -> JsonObject:
        neighbors = self.model.most_similar(word, k=k)
        return {
            "word": word,
            "neighbors": [{"word": w, "score": round(s, 6)} for w, s in neighbors],
        }

    def analogy(self, a: str, b: str, c: str, k: int) -> JsonObject:
        results = self.model.analogy(a, b, c, k=k)
        return {
            "a": a,
            "b": b,
            "c": c,
            "answers": [{"word": w, "score": round(s, 6)} for w, s in results],
        }


def _validate_k(body: Mapping[str, Any]) -> int:
    k = body.get("k", 5)
    if not isinstance(k, int) or not 1 <= k <= _MAX_K:
        raise ValueError(f"'k' must be an integer in [1, {_MAX_K}]")
    return k


def _route(
    method: str, path: str, body: Mapping[str, Any], service: EmbeddingService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "vocabulary_size": len(service.vocabulary)}
    if method == "POST" and path == "/similar":
        word = body.get("word")
        if not isinstance(word, str) or word not in service.vocabulary:
            return 400, {"error": "'word' must be a known vocabulary word"}
        try:
            k = _validate_k(body)
        except ValueError as error:
            return 400, {"error": str(error)}
        return 200, service.similar(word, k)
    if method == "POST" and path == "/analogy":
        words = [body.get("a"), body.get("b"), body.get("c")]
        if any(not isinstance(w, str) or w not in service.vocabulary for w in words):
            return 400, {"error": "'a', 'b', and 'c' must be known vocabulary words"}
        try:
            k = _validate_k(body)
        except ValueError as error:
            return 400, {"error": str(error)}
        return 200, service.analogy(str(words[0]), str(words[1]), str(words[2]), k)
    return 404, {"error": f"no route for {method} {path}"}


def handle(
    method: str, path: str, body: Mapping[str, Any], service: EmbeddingService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready", "vocabulary_size": len(service.vocabulary)}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: EmbeddingService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": EmbeddingService.trained()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"neural-word-embeddings service listening on http://{host}:{port}")
    server.serve_forever()
