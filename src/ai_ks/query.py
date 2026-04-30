from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

from ai_ks.config import DEFAULT_MAX_QUERY_CHARS, Settings, get_settings
from ai_ks.errors import DependencyUnavailableError
from ai_ks.observability import (
    LocalObservability,
    ObservabilitySink,
    TimingCollector,
    build_runtime_context,
    elapsed_ms,
    get_active_collector,
    langsmith_request_context,
    reset_active_collector,
    set_active_collector,
    time_call,
)
from ai_ks.retrieval import HybridRetriever
from ai_ks.validation import normalize_user_text


class CitationResponse(BaseModel):
    citation_id: int
    chunk_id: str
    title: str
    source_uri: str
    chunk_index: int
    excerpt: str


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    title: str
    source_uri: str
    chunk_index: int
    text: str
    fused_score: float
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(max_length=DEFAULT_MAX_QUERY_CHARS)
    limit: int | None = Field(default=None, ge=1, le=10)
    candidate_limit: int | None = Field(default=None, ge=1, le=20)
    semantic_weight: float | None = Field(default=None, ge=0)
    lexical_weight: float | None = Field(default=None, ge=0)
    rrf_k: int | None = Field(default=None, ge=1)
    answer_style: str = Field(default="a concise grounded answer")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return normalize_user_text(
            value,
            field_name="Query",
            max_chars=DEFAULT_MAX_QUERY_CHARS,
        )


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    hits: list[RetrievedChunkResponse]
    citations: list[CitationResponse]
    diagnostics: dict[str, Any]


class QueryService:
    SYSTEM_PROMPT = (
        "You answer questions using only the retrieved context from the local knowledge base. "
        "If the context is insufficient, say that clearly. Use bracket citations like [1] or [2] "
        "that match the supplied citations and do not invent sources."
    )

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever | None = None,
        answer_model: Any | None = None,
        answer_chain: Any | None = None,
        observability: ObservabilitySink | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever or HybridRetriever(settings)
        self.answer_model = answer_model or self._build_model()
        self.answer_chain = answer_chain or self._build_answer_chain()
        self.observability = observability or LocalObservability(
            sqlite_path=settings.sqlite_path,
            log_jsonl_path=settings.log_jsonl_path,
        )

    def run(self, request: QueryRequest) -> QueryResponse:
        request.query = normalize_user_text(
            request.query,
            field_name="Query",
            max_chars=self.settings.max_query_chars,
        )
        request_id = str(uuid4())
        collector = TimingCollector(route="/query", request_id=request_id)
        token = set_active_collector(collector)
        started = perf_counter()
        response: QueryResponse | None = None
        error: str | None = None
        try:
            with langsmith_request_context(
                self.settings,
                route="/query",
                request_id=request_id,
            ):
                result, retriever_duration_ms = time_call(
                    kind="operation",
                    name="retriever.search",
                    call=lambda: self.retriever.search(
                        query=request.query,
                        limit=request.limit,
                        candidate_limit=request.candidate_limit,
                        semantic_weight=request.semantic_weight,
                        lexical_weight=request.lexical_weight,
                        rrf_k=request.rrf_k,
                    ),
                )
                citations = [
                    CitationResponse.model_validate(citation.to_dict())
                    for citation in result.citations
                ]
                context = self._build_context(result.hits, citations)
                try:
                    answer, answer_chain_duration_ms = time_call(
                        kind="operation",
                        name="answer_chain.invoke",
                        call=lambda: self.answer_chain.invoke(
                            {
                                "query": result.query,
                                "answer_style": request.answer_style,
                                "context": context,
                            },
                            config=self._langchain_metadata_config(),
                        ),
                    )
                except Exception as exc:
                    raise DependencyUnavailableError(
                        "llm",
                        f"LLM service unavailable during answer generation: {exc}",
                    ) from exc
            collector_diagnostics = collector.build_diagnostics(elapsed_ms(started))
            result_timings = dict(result.diagnostics.get("timings_ms", {}))
            collector_timings = dict(collector_diagnostics.get("timings_ms", {}))
            diagnostics = {
                **result.diagnostics,
                **collector_diagnostics,
                "timings_ms": {
                    **result_timings,
                    **collector_timings,
                },
                "runtime": build_runtime_context(self.settings, route="/query"),
            }
            diagnostics["timings_ms"]["retriever_search"] = round(retriever_duration_ms, 3)
            diagnostics["timings_ms"]["answer_chain_invoke"] = round(
                answer_chain_duration_ms, 3
            )
            response = QueryResponse(
                request_id=request_id,
                query=result.query,
                answer=str(answer).strip(),
                hits=[
                    RetrievedChunkResponse.model_validate(hit.to_dict())
                    for hit in result.hits
                ],
                citations=citations,
                diagnostics=diagnostics,
            )
            return response
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            diagnostics = collector.build_diagnostics(elapsed_ms(started))
            diagnostics["runtime"] = build_runtime_context(self.settings, route="/query")
            self.observability.record_request(
                route="/query",
                request_id=request_id,
                status="error" if error else "success",
                diagnostics=response.diagnostics if response else diagnostics,
                final_query=response.query if response else request.query,
                answer=response.answer if response else None,
                error=error,
            )
            reset_active_collector(token)

    def _build_model(self) -> ChatOllama:
        return ChatOllama(
            model=self.settings.llm_runtime_model,
            base_url=self.settings.ollama_url,
            temperature=0,
        )

    def _build_answer_chain(self) -> Any:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                (
                    "human",
                    "Question: {query}\n\n"
                    "Write {answer_style}.\n\n"
                    "Retrieved context:\n{context}",
                ),
            ]
        )
        return prompt | self.answer_model | StrOutputParser()

    @staticmethod
    def _build_context(
        hits: list[Any],
        citations: list[CitationResponse],
    ) -> str:
        if not hits:
            return "No retrieved context was found."

        lines: list[str] = []
        for hit, citation in zip(hits, citations, strict=False):
            lines.append(
                f"[{citation.citation_id}] {citation.title} "
                f"({citation.source_uri}, chunk {citation.chunk_index})"
            )
            lines.append(hit.text)
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _langchain_metadata_config() -> dict[str, Any]:
        collector = get_active_collector()
        if collector is None:
            return {}
        return {
            "metadata": {
                "request_id": collector.request_id,
                "route": collector.route,
                "chain_name": "answer_chain",
            }
        }


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    return QueryService(settings=get_settings())
