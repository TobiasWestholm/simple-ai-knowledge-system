from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from ai_ks.config import Settings
from ai_ks.embeddings import RemoteBgeM3Embedder
from ai_ks.errors import DependencyUnavailableError
from ai_ks.main import app
from ai_ks.observability import (
    LocalObservability,
    TimingCollector,
    reset_active_collector,
    set_active_collector,
)
from ai_ks.query import QueryRequest, QueryResponse, QueryService, get_query_service
from ai_ks.retrieval import CitationRecord, HybridRetriever, HybridSearchResult, RetrievedChunk


@dataclass
class FakeRetriever(HybridRetriever):
    def __init__(self) -> None:
        pass

    def search(
        self,
        query: str,
        limit: int | None = None,
        candidate_limit: int | None = None,
        semantic_weight: float | None = None,
        lexical_weight: float | None = None,
        rrf_k: int | None = None,
    ) -> HybridSearchResult:
        return HybridSearchResult(
            query=query.strip(),
            hits=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    title="FastAPI Notes",
                    source_uri="knowledge/fastapi.md",
                    chunk_index=0,
                    text="FastAPI is a modern Python web framework for APIs.",
                    fused_score=0.42,
                    semantic_rank=1,
                    lexical_rank=2,
                    metadata={"kind": "note"},
                )
            ],
            citations=[
                CitationRecord(
                    citation_id=1,
                    chunk_id="chunk-1",
                    title="FastAPI Notes",
                    source_uri="knowledge/fastapi.md",
                    chunk_index=0,
                    excerpt="FastAPI is a modern Python web framework for APIs.",
                )
            ],
            diagnostics={
                "limit": limit or 5,
                "candidate_limit": candidate_limit or 8,
                "semantic_weight": semantic_weight or 0.7,
                "lexical_weight": lexical_weight or 0.3,
                "rrf_k": rrf_k or 60,
                "semantic_hits": 1,
                "lexical_hits": 1,
                "timings_ms": {
                    "embedding_request": 11.0,
                    "qdrant_search": 4.5,
                    "bm25_search": 0.9,
                    "fusion": 0.1,
                },
            },
        )


def test_query_service_returns_answer_hits_and_diagnostics() -> None:
    settings = Settings()
    service = QueryService(
        settings=settings,
        retriever=FakeRetriever(),
        answer_chain=RunnableLambda(
            lambda _: "FastAPI is a modern Python web framework for APIs [1]."
        ),
    )

    response = service.run(
        QueryRequest(
            query="What is FastAPI?",
            limit=4,
            candidate_limit=7,
            semantic_weight=0.6,
            lexical_weight=0.4,
            rrf_k=50,
        )
    )

    assert response.query == "What is FastAPI?"
    assert response.answer == "FastAPI is a modern Python web framework for APIs [1]."
    assert response.citations[0].citation_id == 1
    assert response.hits[0].chunk_id == "chunk-1"
    assert response.diagnostics["semantic_hits"] == 1
    assert response.diagnostics["lexical_hits"] == 1
    assert "timings_ms" in response.diagnostics
    assert "retriever_search" in response.diagnostics["timings_ms"]
    assert "answer_chain_invoke" in response.diagnostics["timings_ms"]
    assert response.diagnostics["timings_ms"]["embedding_request"] == 11.0
    assert response.diagnostics["timings_ms"]["qdrant_search"] == 4.5
    assert response.diagnostics["timings_ms"]["bm25_search"] == 0.9
    assert response.diagnostics["timings_ms"]["fusion"] == 0.1
    assert response.diagnostics["runtime"]["embedding"]["url"] == settings.embedding_url


def test_query_service_records_logs_and_telemetry(tmp_path: Path) -> None:
    settings = Settings(
        sqlite_path=tmp_path / "logs" / "telemetry.db",
        log_jsonl_path=tmp_path / "logs" / "requests.jsonl",
    )
    service = QueryService(
        settings=settings,
        retriever=FakeRetriever(),
        answer_chain=RunnableLambda(
            lambda _: "FastAPI is a modern Python web framework for APIs [1]."
        ),
        observability=LocalObservability(
            sqlite_path=settings.sqlite_path,
            log_jsonl_path=settings.log_jsonl_path,
        ),
    )

    response = service.run(QueryRequest(query="What is FastAPI?"))

    log_lines = settings.log_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    log_payload = json.loads(log_lines[-1])
    assert log_payload["route"] == "/query"
    assert log_payload["status"] == "success"
    assert "timings_ms" in log_payload["diagnostics"]
    assert log_payload["runtime"]["embedding"]["url"] == settings.embedding_url

    connection = sqlite3.connect(settings.sqlite_path)
    try:
        request_row = connection.execute(
            "SELECT route, status FROM request_runs WHERE request_id = ?",
            (response.request_id,),
        ).fetchone()
        span_rows = connection.execute(
            "SELECT name FROM timing_spans WHERE request_id = ? ORDER BY sequence",
            (response.request_id,),
        ).fetchall()
    finally:
        connection.close()

    assert request_row == ("/query", "success")
    assert [row[0] for row in span_rows] == [
        "retriever.search",
        "answer_chain.invoke",
    ]


def test_query_service_wraps_llm_errors() -> None:
    class BrokenAnswerChain:
        def invoke(self, input: Any, config: Any = None) -> str:
            raise RuntimeError("host unavailable")

    service = QueryService(
        settings=Settings(),
        retriever=FakeRetriever(),
        answer_chain=BrokenAnswerChain(),
    )

    try:
        service.run(QueryRequest(query="What is FastAPI?"))
    except DependencyUnavailableError as exc:
        assert exc.service == "llm"
        assert "host unavailable" in str(exc)
    else:
        raise AssertionError("Expected DependencyUnavailableError for broken answer chain")


def test_query_endpoint_uses_dependency_override() -> None:
    class FakeQueryService:
        def run(self, request: QueryRequest) -> QueryResponse:
            return QueryResponse(
                request_id="req-2",
                query=request.query,
                answer="Direct query answer [1].",
                hits=[],
                citations=[],
                diagnostics={"semantic_hits": 0, "lexical_hits": 0},
            )

    app.dependency_overrides[get_query_service] = lambda: FakeQueryService()
    client = TestClient(app)

    response = client.post("/query", json={"query": "Explain FastAPI"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "Explain FastAPI"
    assert payload["answer"] == "Direct query answer [1]."

    app.dependency_overrides.clear()


def test_query_endpoint_rejects_empty_query() -> None:
    client = TestClient(app)

    response = client.post("/query", json={"query": "   "})

    assert response.status_code == 422


def test_query_endpoint_rejects_overlong_query() -> None:
    client = TestClient(app)

    response = client.post("/query", json={"query": "x" * 2001})

    assert response.status_code == 422


def test_query_endpoint_accepts_boundary_length_query() -> None:
    class FakeQueryService:
        def run(self, request: QueryRequest) -> QueryResponse:
            return QueryResponse(
                request_id="req-boundary",
                query=request.query,
                answer="ok",
                hits=[],
                citations=[],
                diagnostics={},
            )

    app.dependency_overrides[get_query_service] = lambda: FakeQueryService()
    client = TestClient(app)

    response = client.post("/query", json={"query": "x" * 2000})

    assert response.status_code == 200
    app.dependency_overrides.clear()


def test_query_endpoint_maps_dependency_unavailable_to_503() -> None:
    class FakeQueryService:
        def run(self, request: QueryRequest) -> QueryResponse:
            raise DependencyUnavailableError("embedding", "Embedding service unavailable")

    app.dependency_overrides[get_query_service] = lambda: FakeQueryService()
    client = TestClient(app)

    response = client.post("/query", json={"query": "What is FastAPI?"})

    assert response.status_code == 503
    assert "Embedding service unavailable" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_hybrid_retriever_defaults_to_remote_embedder() -> None:
    retriever = HybridRetriever(settings=Settings())

    assert isinstance(retriever.embedder, RemoteBgeM3Embedder)


def test_hybrid_retriever_records_retrieval_subspans(tmp_path: Path) -> None:
    settings = Settings(
        index_dir=tmp_path / "index",
        sqlite_path=tmp_path / "logs" / "telemetry.db",
        log_jsonl_path=tmp_path / "logs" / "requests.jsonl",
    )
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    (settings.index_dir / "bm25_documents.json").write_text(
        json.dumps(
            [
                {
                    "id": "chunk-1",
                    "title": "FastAPI Notes",
                    "source_uri": "knowledge/fastapi.md",
                    "chunk_index": 0,
                    "text": "FastAPI is a modern Python web framework for APIs.",
                    "tokens": ["fastapi", "modern", "python", "web", "framework"],
                    "metadata": {},
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeDetailedEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

        def embed_with_details(self, texts: list[str]) -> Any:
            class Result:
                model_id = "BAAI/bge-m3"
                device = "mps:0"
                embeddings = [[0.1, 0.2, 0.3] for _ in texts]
                diagnostics = {"timings_ms": {"embed_texts": 7.5}}

            return Result()

    class FakePoint:
        def __init__(self) -> None:
            self.id = "chunk-1"
            self.score = 0.9
            self.payload = {
                "title": "FastAPI Notes",
                "source_uri": "knowledge/fastapi.md",
                "chunk_index": 0,
                "text": "FastAPI is a modern Python web framework for APIs.",
                "metadata": {},
            }

    class FakeQueryResponse:
        points = [FakePoint()]

    class FakeQdrantClient:
        def query_points(self, **_: Any) -> FakeQueryResponse:
            return FakeQueryResponse()

    retriever = HybridRetriever(
        settings=settings,
        embedder=FakeDetailedEmbedder(),
        qdrant_client=FakeQdrantClient(),  # type: ignore[arg-type]
    )

    collector = TimingCollector(route="/query", request_id="req-retrieval")
    token = set_active_collector(collector)
    try:
        result = retriever.search("What is FastAPI?")
    finally:
        reset_active_collector(token)

    assert result.diagnostics["embedding_service"]["device"] == "mps:0"
    assert result.diagnostics["timings_ms"]["embedding_service_inference"] == 7.5
    assert [span.name for span in collector.spans] == [
        "embedding_service.request",
        "qdrant.search",
        "bm25.search",
        "fusion.rank",
    ]


def test_hybrid_retriever_wraps_qdrant_errors(tmp_path: Path) -> None:
    settings = Settings(index_dir=tmp_path / "index")
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    (settings.index_dir / "bm25_documents.json").write_text(
        json.dumps(
            [
                {
                    "id": "chunk-1",
                    "title": "FastAPI Notes",
                    "source_uri": "knowledge/fastapi.md",
                    "chunk_index": 0,
                    "text": "FastAPI is a modern Python web framework for APIs.",
                    "tokens": ["fastapi"],
                    "metadata": {},
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeEmbedder:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    class BrokenQdrantClient:
        def query_points(self, **_: Any) -> Any:
            raise RuntimeError("connection refused")

    retriever = HybridRetriever(
        settings=settings,
        embedder=FakeEmbedder(),
        qdrant_client=BrokenQdrantClient(),  # type: ignore[arg-type]
    )

    try:
        retriever.search("What is FastAPI?")
    except DependencyUnavailableError as exc:
        assert exc.service == "qdrant"
        assert "connection refused" in str(exc)
    else:
        raise AssertionError("Expected DependencyUnavailableError for broken Qdrant")
