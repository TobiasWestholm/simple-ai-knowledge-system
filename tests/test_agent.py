from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from ai_ks.agent import (
    AgentRequest,
    AgentResponse,
    LangChainAgentService,
    get_agent_service,
)
from ai_ks.config import Settings
from ai_ks.errors import DependencyUnavailableError
from ai_ks.main import app
from ai_ks.observability import (
    LocalObservability,
    TimingCollector,
    reset_active_collector,
    set_active_collector,
)
from ai_ks.retrieval import (
    HybridRetriever,
    HybridSearchResult,
    weighted_reciprocal_rank_fusion,
)


@dataclass
class FakeUtilityModel:
    response: str

    def __call__(self, input: Any) -> str:
        return self.response


@dataclass
class FakeAgentGraph:
    result: dict[str, Any]

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.result


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
            query=query,
            hits=[],
            citations=[],
            diagnostics={
                "limit": limit or 5,
                "candidate_limit": candidate_limit or 8,
                "semantic_weight": semantic_weight or 0.7,
                "lexical_weight": lexical_weight or 0.3,
                "rrf_k": rrf_k or 60,
                "semantic_hits": 0,
                "lexical_hits": 0,
            },
        )


def test_weighted_reciprocal_rank_fusion_combines_semantic_and_lexical_scores() -> None:
    scores = weighted_reciprocal_rank_fusion(
        semantic_ranks={"a": 1, "b": 2},
        lexical_ranks={"b": 1, "c": 2},
        semantic_weight=0.7,
        lexical_weight=0.3,
        rrf_k=60,
    )

    assert scores["b"] > scores["a"] > scores["c"]


def test_langchain_tools_return_json_payloads() -> None:
    service = LangChainAgentService(
        settings=Settings(),
        retriever=FakeRetriever(),
        utility_model=RunnableLambda(FakeUtilityModel(response="rewritten question")),
        agent_graph=FakeAgentGraph(result={"messages": []}),
        observability=FakeObservability(),
    )

    tools_by_name = {tool.name: tool for tool in service.tools}
    rewrite_output = tools_by_name["rewrite_query"].invoke({"query": "What is FastAPI?"})
    summary_output = tools_by_name["summarize_context"].invoke(
        {"text": "FastAPI is a Python framework.", "style": "a short summary"}
    )
    rag_output = tools_by_name["rag_search"].invoke({"query": "What is FastAPI?", "limit": 5})

    assert rewrite_output == "rewritten question"
    assert summary_output == "rewritten question"
    assert rag_output == "No matching chunks found in the local knowledge base."

    collector = TimingCollector(route="/agent", request_id="req-tools")
    token = set_active_collector(collector)
    try:
        tools_by_name["rewrite_query"].invoke({"query": "What is FastAPI?"})
        tools_by_name["summarize_context"].invoke(
            {"text": "FastAPI is a Python framework.", "style": "a short summary"}
        )
        tools_by_name["rag_search"].invoke({"query": "What is FastAPI?", "limit": 5})
    finally:
        reset_active_collector(token)

    assert [span.name for span in collector.spans] == [
        "rewrite_chain.invoke",
        "rewrite_query",
        "summary_chain.invoke",
        "summarize_context",
        "retriever.search",
        "rag_search",
    ]


def test_agent_service_maps_langchain_messages_into_api_response() -> None:
    agent_graph = FakeAgentGraph(
        result={
            "messages": [
                HumanMessage(content="What is FastAPI?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "rewrite_query",
                            "args": {"query": "What is FastAPI?"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="fastapi overview",
                    artifact={"rewritten_query": "fastapi overview"},
                    tool_call_id="call-1",
                    name="rewrite_query",
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "rag_search",
                            "args": {"query": "fastapi overview", "limit": 5},
                            "id": "call-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="Search query: fastapi overview\n\n[1] FastAPI Notes",
                    artifact={
                        "query": "fastapi overview",
                        "hits": [],
                        "citations": [
                            {
                                "citation_id": 1,
                                "chunk_id": "chunk-1",
                                "title": "FastAPI Notes",
                                "source_uri": "knowledge/fastapi.md",
                                "chunk_index": 0,
                                "excerpt": "FastAPI is a modern Python web framework.",
                            }
                        ],
                        "diagnostics": {"semantic_hits": 1, "lexical_hits": 1},
                    },
                    tool_call_id="call-2",
                    name="rag_search",
                ),
                AIMessage(content="FastAPI is a modern Python web framework [1]."),
            ]
        }
    )
    service = LangChainAgentService(
        settings=Settings(),
        retriever=FakeRetriever(),
        utility_model=RunnableLambda(FakeUtilityModel(response="unused")),
        agent_graph=agent_graph,
        observability=FakeObservability(),
    )

    response = service.run(AgentRequest(message="What is FastAPI?"))

    assert response.answer == "FastAPI is a modern Python web framework [1]."
    assert response.final_query == "fastapi overview"
    assert [record.name for record in response.tool_calls] == ["rewrite_query", "rag_search"]
    assert response.citations[0].citation_id == 1
    assert "timings_ms" in response.diagnostics
    assert "agent_graph_invoke" in response.diagnostics["timings_ms"]


def test_agent_endpoint_uses_dependency_override() -> None:
    class FakeAgentService:
        def run(self, request: AgentRequest) -> AgentResponse:
            return AgentResponse(
                request_id="req-1",
                answer=f"Echo: {request.message}",
                tool_calls=[],
                citations=[],
                final_query=request.message,
            )

    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    client = TestClient(app)

    response = client.post("/agent", json={"message": "Summarize FastAPI"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Echo: Summarize FastAPI"
    assert payload["final_query"] == "Summarize FastAPI"

    app.dependency_overrides.clear()


def test_agent_endpoint_rejects_removed_retrieval_limit_field() -> None:
    client = TestClient(app)

    response = client.post(
        "/agent",
        json={
            "message": "Summarize FastAPI",
            "retrieval_limit": 3,
        },
    )

    assert response.status_code == 422


def test_agent_endpoint_rejects_empty_message() -> None:
    client = TestClient(app)

    response = client.post("/agent", json={"message": "   "})

    assert response.status_code == 422


def test_agent_endpoint_rejects_overlong_message() -> None:
    client = TestClient(app)

    response = client.post("/agent", json={"message": "x" * 2001})

    assert response.status_code == 422


def test_agent_endpoint_accepts_boundary_length_message() -> None:
    class FakeAgentService:
        def run(self, request: AgentRequest) -> AgentResponse:
            return AgentResponse(
                request_id="req-boundary",
                answer="ok",
                tool_calls=[],
                citations=[],
                final_query=request.message,
                diagnostics={},
            )

    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    client = TestClient(app)

    response = client.post("/agent", json={"message": "x" * 2000})

    assert response.status_code == 200
    app.dependency_overrides.clear()


def test_agent_endpoint_maps_dependency_unavailable_to_503() -> None:
    class FakeAgentService:
        def run(self, request: AgentRequest) -> AgentResponse:
            raise DependencyUnavailableError(
                "llm",
                "LLM service unavailable during agent execution",
            )

    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    client = TestClient(app)

    response = client.post("/agent", json={"message": "What is FastAPI?"})

    assert response.status_code == 503
    assert "LLM service unavailable" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_agent_service_records_logs_and_telemetry(tmp_path: Path) -> None:
    settings = Settings(
        sqlite_path=tmp_path / "logs" / "telemetry.db",
        log_jsonl_path=tmp_path / "logs" / "requests.jsonl",
    )
    service = LangChainAgentService(
        settings=settings,
        retriever=FakeRetriever(),
        utility_model=RunnableLambda(FakeUtilityModel(response="unused")),
        agent_graph=FakeAgentGraph(result={"messages": [AIMessage(content="Done.")]}),
        observability=LocalObservability(
            sqlite_path=settings.sqlite_path,
            log_jsonl_path=settings.log_jsonl_path,
        ),
    )

    response = service.run(AgentRequest(message="What is FastAPI?"))

    assert response.answer == "Done."
    log_lines = settings.log_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    log_payload = json.loads(log_lines[-1])
    assert log_payload["route"] == "/agent"
    assert log_payload["status"] == "success"
    assert "timings_ms" in log_payload["diagnostics"]

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

    assert request_row == ("/agent", "success")
    assert [row[0] for row in span_rows] == ["agent_graph.invoke"]


def test_agent_service_wraps_llm_errors() -> None:
    class BrokenAgentGraph:
        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("connection refused")

    service = LangChainAgentService(
        settings=Settings(),
        retriever=FakeRetriever(),
        utility_model=RunnableLambda(FakeUtilityModel(response="unused")),
        agent_graph=BrokenAgentGraph(),
        observability=FakeObservability(),
    )

    try:
        service.run(AgentRequest(message="What is FastAPI?"))
    except DependencyUnavailableError as exc:
        assert exc.service == "llm"
        assert "connection refused" in str(exc)
    else:
        raise AssertionError("Expected DependencyUnavailableError for broken LLM graph")


class FakeObservability:
    def record_request(
        self,
        *,
        route: str,
        request_id: str,
        status: str,
        diagnostics: dict[str, Any],
        final_query: str | None,
        answer: str | None,
        error: str | None,
    ) -> None:
        return None
