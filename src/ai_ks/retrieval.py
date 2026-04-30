from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from ai_ks.config import Settings
from ai_ks.embeddings import EmbeddingServiceResult, RemoteBgeM3Embedder
from ai_ks.errors import DependencyUnavailableError
from ai_ks.ingestion import build_qdrant_client
from ai_ks.observability import get_active_collector


@dataclass(frozen=True)
class CitationRecord:
    citation_id: int
    chunk_id: str
    title: str
    source_uri: str
    chunk_index: int
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "source_uri": self.source_uri,
            "chunk_index": self.chunk_index,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    title: str
    source_uri: str
    chunk_index: int
    text: str
    fused_score: float
    semantic_rank: int | None
    lexical_rank: int | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "source_uri": self.source_uri,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "fused_score": self.fused_score,
            "semantic_rank": self.semantic_rank,
            "lexical_rank": self.lexical_rank,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class HybridSearchResult:
    query: str
    hits: list[RetrievedChunk]
    citations: list[CitationRecord]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "citations": [citation.to_dict() for citation in self.citations],
            "diagnostics": self.diagnostics,
        }


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def weighted_reciprocal_rank_fusion(
    semantic_ranks: dict[str, int],
    lexical_ranks: dict[str, int],
    semantic_weight: float,
    lexical_weight: float,
    rrf_k: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}

    for chunk_id, rank in semantic_ranks.items():
        scores[chunk_id] = scores.get(chunk_id, 0.0) + semantic_weight / (rrf_k + rank)

    for chunk_id, rank in lexical_ranks.items():
        scores[chunk_id] = scores.get(chunk_id, 0.0) + lexical_weight / (rrf_k + rank)

    return scores


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: TextEmbedder | None = None,
        qdrant_client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or RemoteBgeM3Embedder(
            base_url=settings.embedding_url,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
        self.qdrant_client = qdrant_client or build_qdrant_client(settings.qdrant_url)
        self._bm25_documents: list[dict[str, Any]] | None = None
        self._bm25_index: BM25Okapi | None = None

    def search(
        self,
        query: str,
        limit: int | None = None,
        candidate_limit: int | None = None,
        semantic_weight: float | None = None,
        lexical_weight: float | None = None,
        rrf_k: int | None = None,
    ) -> HybridSearchResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty.")

        limit = limit or self.settings.retrieval_limit
        candidate_limit = candidate_limit or self.settings.retrieval_candidate_limit
        semantic_weight = semantic_weight or self.settings.retrieval_semantic_weight
        lexical_weight = lexical_weight or self.settings.retrieval_lexical_weight
        rrf_k = rrf_k or self.settings.retrieval_rrf_k

        bm25_documents = self._load_bm25_documents()
        bm25_by_id = {document["id"]: document for document in bm25_documents}

        semantic_hits, semantic_diagnostics = self._dense_search(
            normalized_query,
            candidate_limit,
        )
        lexical_hits, bm25_duration_ms = self._lexical_search(
            normalized_query,
            candidate_limit,
            bm25_documents,
        )

        semantic_ranks = {
            chunk_id: rank
            for rank, chunk_id in enumerate((hit["id"] for hit in semantic_hits), start=1)
        }
        lexical_ranks = {
            chunk_id: rank
            for rank, chunk_id in enumerate((hit["id"] for hit in lexical_hits), start=1)
        }
        fusion_started = perf_counter()
        fused_scores = weighted_reciprocal_rank_fusion(
            semantic_ranks=semantic_ranks,
            lexical_ranks=lexical_ranks,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            rrf_k=rrf_k,
        )
        fusion_duration_ms = round((perf_counter() - fusion_started) * 1000, 3)
        collector = get_active_collector()
        if collector is not None:
            collector.record(
                kind="operation",
                name="fusion.rank",
                duration_ms=fusion_duration_ms,
                metadata={
                    "semantic_hits": len(semantic_hits),
                    "lexical_hits": len(lexical_hits),
                },
            )

        ordered_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )[:limit]

        semantic_payloads = {hit["id"]: hit for hit in semantic_hits}
        result_hits: list[RetrievedChunk] = []
        citations: list[CitationRecord] = []

        for index, chunk_id in enumerate(ordered_ids, start=1):
            artifact_document = bm25_by_id.get(chunk_id, {})
            semantic_payload = semantic_payloads.get(chunk_id, {})
            payload = artifact_document or semantic_payload
            hit = RetrievedChunk(
                chunk_id=chunk_id,
                title=str(payload.get("title", "")),
                source_uri=str(payload.get("source_uri", "")),
                chunk_index=int(payload.get("chunk_index", 0)),
                text=str(payload.get("text", "")),
                fused_score=fused_scores[chunk_id],
                semantic_rank=semantic_ranks.get(chunk_id),
                lexical_rank=lexical_ranks.get(chunk_id),
                metadata=dict(payload.get("metadata", {})),
            )
            result_hits.append(hit)
            citations.append(
                CitationRecord(
                    citation_id=index,
                    chunk_id=hit.chunk_id,
                    title=hit.title,
                    source_uri=hit.source_uri,
                    chunk_index=hit.chunk_index,
                    excerpt=hit.text[:280],
                )
            )

        return HybridSearchResult(
            query=normalized_query,
            hits=result_hits,
            citations=citations,
            diagnostics={
                "limit": limit,
                "candidate_limit": candidate_limit,
                "semantic_weight": semantic_weight,
                "lexical_weight": lexical_weight,
                "rrf_k": rrf_k,
                "semantic_hits": len(semantic_hits),
                "lexical_hits": len(lexical_hits),
                "timings_ms": {
                    "embedding_request": semantic_diagnostics["embedding_request_ms"],
                    "embedding_service_inference": semantic_diagnostics[
                        "embedding_service_inference_ms"
                    ],
                    "qdrant_search": semantic_diagnostics["qdrant_search_ms"],
                    "bm25_search": bm25_duration_ms,
                    "fusion": fusion_duration_ms,
                },
                "embedding_service": {
                    "model_id": semantic_diagnostics["embedding_model_id"],
                    "device": semantic_diagnostics["embedding_device"],
                },
            },
        )

    def _dense_search(
        self,
        query: str,
        candidate_limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        embed_started = perf_counter()
        embedding_result = self._embed_query(query)
        query_vector = embedding_result.embeddings[0]
        embedding_request_ms = round((perf_counter() - embed_started) * 1000, 3)
        collector = get_active_collector()
        if collector is not None:
            collector.record(
                kind="operation",
                name="embedding_service.request",
                duration_ms=embedding_request_ms,
                metadata={
                    "model_id": embedding_result.model_id,
                    "device": embedding_result.device,
                },
            )

        qdrant_started = perf_counter()
        try:
            response = self.qdrant_client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=query_vector,
                limit=candidate_limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise DependencyUnavailableError(
                "qdrant",
                f"Qdrant search failed against {self.settings.qdrant_url}: {exc}",
            ) from exc
        qdrant_search_ms = round((perf_counter() - qdrant_started) * 1000, 3)
        if collector is not None:
            collector.record(
                kind="operation",
                name="qdrant.search",
                duration_ms=qdrant_search_ms,
                metadata={"candidate_limit": candidate_limit},
            )

        hits: list[dict[str, Any]] = []
        for point in response.points:
            payload = dict(point.payload or {})
            payload["id"] = str(point.id)
            payload["score"] = float(point.score)
            hits.append(payload)
        return hits, {
            "embedding_request_ms": embedding_request_ms,
            "embedding_service_inference_ms": float(
                embedding_result.diagnostics.get("timings_ms", {}).get("embed_texts", 0.0)
            ),
            "embedding_model_id": embedding_result.model_id,
            "embedding_device": embedding_result.device,
            "qdrant_search_ms": qdrant_search_ms,
        }

    def _lexical_search(
        self,
        query: str,
        candidate_limit: int,
        bm25_documents: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], float]:
        started = perf_counter()
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return [], round((perf_counter() - started) * 1000, 3)

        bm25_index = self._load_bm25_index()
        scores = bm25_index.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        hits: list[dict[str, Any]] = []
        for index in ranked_indices:
            if scores[index] <= 0:
                continue
            document = dict(bm25_documents[index])
            document["bm25_score"] = float(scores[index])
            hits.append(document)
            if len(hits) >= candidate_limit:
                break
        duration_ms = round((perf_counter() - started) * 1000, 3)
        collector = get_active_collector()
        if collector is not None:
            collector.record(
                kind="operation",
                name="bm25.search",
                duration_ms=duration_ms,
                metadata={"candidate_limit": candidate_limit},
            )
        return hits, duration_ms

    def _embed_query(self, query: str) -> EmbeddingServiceResult:
        if isinstance(self.embedder, RemoteBgeM3Embedder):
            return self.embedder.embed_with_details([query])
        embed_with_details = getattr(self.embedder, "embed_with_details", None)
        if callable(embed_with_details):
            return embed_with_details([query])
        return EmbeddingServiceResult(
            model_id=self.settings.embed_model_id,
            device="unknown",
            embeddings=self.embedder.embed_texts([query]),
            diagnostics={},
        )

    def _load_bm25_documents(self) -> list[dict[str, Any]]:
        if self._bm25_documents is not None:
            return self._bm25_documents

        artifact_path = self.settings.index_dir / "bm25_documents.json"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"BM25 artifact not found at {artifact_path}. Run ingest before querying."
            )

        self._bm25_documents = json.loads(artifact_path.read_text(encoding="utf-8"))
        return self._bm25_documents

    def _load_bm25_index(self) -> BM25Okapi:
        if self._bm25_index is not None:
            return self._bm25_index

        documents = self._load_bm25_documents()
        self._bm25_index = BM25Okapi(
            [list(document.get("tokens", [])) for document in documents]
        )
        return self._bm25_index
