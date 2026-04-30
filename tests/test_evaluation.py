from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ai_ks.agent import AgentResponse, CitationResponse, ToolCallRecord
from ai_ks.config import Settings
from ai_ks.evaluation import (
    EvaluateRequest,
    EvaluateResponse,
    EvaluationService,
    ToolBehaviorSummary,
    get_evaluation_service,
    load_tool_behavior_cases,
)
from ai_ks.main import app


class FakeAgentService:
    def run(self, request: Any) -> AgentResponse:
        message = request.message
        if "FastAPI" in message:
            return AgentResponse(
                request_id="req-rag",
                answer="FastAPI is a web framework [1].",
                tool_calls=[
                    ToolCallRecord(
                        name="rag_search",
                        arguments={"query": message},
                        output={
                            "query": message,
                            "citations": [
                                CitationResponse(
                                    citation_id=1,
                                    chunk_id="chunk-1",
                                    title="FastAPI Notes",
                                    source_uri="knowledge/fastapi.md",
                                    chunk_index=0,
                                    excerpt="FastAPI is a modern Python web framework.",
                                ).model_dump()
                            ],
                            "diagnostics": {"timings_ms": {"tool_execution": 18.5}},
                        },
                        status="success",
                        duration_ms=18.5,
                    )
                ],
                citations=[],
                final_query=message,
                diagnostics={
                    "timings_ms": {
                        "total_request": 30.0,
                        "agent_graph_invoke": 29.0,
                        "retriever_search": 18.0,
                    }
                },
            )
        return AgentResponse(
            request_id="req-no-tool",
            answer="ready",
            tool_calls=[],
            citations=[],
            final_query=None,
            diagnostics={
                "timings_ms": {
                    "total_request": 4.0,
                    "agent_graph_invoke": 3.5,
                }
            },
        )


def test_load_tool_behavior_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "tool_behavior_cases.json"
    path.write_text(
        json.dumps(
            [
                {"id": "dup", "category": "x", "prompt": "a"},
                {"id": "dup", "category": "y", "prompt": "b"},
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_tool_behavior_cases(path)
    except ValueError as exc:
        assert "Duplicate tool behavior case ids" in str(exc)
    else:
        raise AssertionError("Expected duplicate id validation to fail")


def test_tool_behavior_dataset_contains_twenty_plus_cases() -> None:
    cases = load_tool_behavior_cases(
        Path("data/evals/tool_behavior_cases.json").resolve()
    )

    assert len(cases) >= 20
    assert len({case.id for case in cases}) == len(cases)


def test_evaluation_service_scores_tool_behavior_failure_and_timing(tmp_path: Path) -> None:
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "tool_behavior_cases.json").write_text(
        json.dumps(
            [
                {
                    "id": "rag-case",
                    "category": "rag_lookup",
                    "prompt": "Using the local knowledge base, what is FastAPI?",
                    "required_tools": ["rag_search"],
                    "forbidden_tools": ["rewrite_query"],
                    "max_tool_calls": 1,
                },
                {
                    "id": "no-tool-case",
                    "category": "no_tool",
                    "prompt": "Reply with exactly the word ready.",
                    "forbidden_tools": ["rag_search", "rewrite_query", "summarize_context"],
                    "max_tool_calls": 0,
                },
            ]
        ),
        encoding="utf-8",
    )
    settings = Settings(eval_dir=eval_dir)
    service = EvaluationService(
        settings=settings,
        agent_service_factory=lambda: FakeAgentService(),
    )

    result = service.evaluate(EvaluateRequest())

    assert result.tool_behavior is not None
    assert result.failure is not None
    assert result.timing is not None
    assert result.tool_behavior.passed_cases == 2
    assert result.failure.failed_cases == 0
    assert "total_request" in result.timing.request_metrics
    assert "rag_search" in result.timing.tool_metrics


def test_tool_behavior_rewrite_reference_query_is_checked() -> None:
    class RewriteOnlyAgentService:
        def run(self, request: Any) -> AgentResponse:
            return AgentResponse(
                request_id="req-rewrite",
                answer="done",
                tool_calls=[
                    ToolCallRecord(
                        name="rewrite_query",
                        arguments={"query": "tls lung cancer immune structures why matter"},
                        output={
                            "rewritten_query": "tls lung cancer immune structures why matter",
                            "diagnostics": {"timings_ms": {"tool_execution": 5.0}},
                        },
                        status="success",
                        duration_ms=5.0,
                    ),
                    ToolCallRecord(
                        name="rag_search",
                        arguments={"query": "tls lung cancer immune structures why matter"},
                        output={"diagnostics": {"timings_ms": {"tool_execution": 8.0}}},
                        status="success",
                        duration_ms=8.0,
                    ),
                ],
                citations=[],
                final_query="tls lung cancer immune structures why matter",
                diagnostics={"timings_ms": {"total_request": 20.0}},
            )

    settings = Settings()
    service = EvaluationService(
        settings=settings,
        agent_service_factory=lambda: RewriteOnlyAgentService(),
    )

    result = service.evaluate(
        EvaluateRequest(suites=["tool_behavior"], case_ids=["rewrite-rag-tls"])
    )

    assert result.tool_behavior is not None
    assert result.tool_behavior.failed_cases == 1
    assert "did not materially change the query" in result.tool_behavior.cases[0].reasons[0]


def test_tool_behavior_allows_rewrite_plus_two_rag_searches_when_case_requests_it() -> None:
    class TwoSearchAgentService:
        def run(self, request: Any) -> AgentResponse:
            return AgentResponse(
                request_id="req-two-searches",
                answer=(
                    "I could not find relevant information about FastAPI "
                    "in the local knowledge base."
                ),
                tool_calls=[
                    ToolCallRecord(
                        name="rewrite_query",
                        arguments={"query": "FastAPI"},
                        output={
                            "rewritten_query": "python web framework fastapi",
                            "diagnostics": {"timings_ms": {"tool_execution": 4.0}},
                        },
                        status="success",
                        duration_ms=4.0,
                    ),
                    ToolCallRecord(
                        name="rag_search",
                        arguments={"query": "FastAPI Docker"},
                        output={"diagnostics": {"timings_ms": {"tool_execution": 6.0}}},
                        status="success",
                        duration_ms=6.0,
                    ),
                    ToolCallRecord(
                        name="rag_search",
                        arguments={"query": "web framework container platform"},
                        output={"diagnostics": {"timings_ms": {"tool_execution": 7.0}}},
                        status="success",
                        duration_ms=7.0,
                    ),
                ],
                citations=[],
                final_query="web framework container platform",
                diagnostics={"timings_ms": {"total_request": 18.0}},
            )

    settings = Settings()
    service = EvaluationService(
        settings=settings,
        agent_service_factory=lambda: TwoSearchAgentService(),
    )

    result = service.evaluate(
        EvaluateRequest(
            suites=["tool_behavior"],
            case_ids=["out-of-corpus-fastapi-two-searches"],
        )
    )

    assert result.tool_behavior is not None
    assert result.tool_behavior.passed_cases == 1
    assert result.tool_behavior.cases[0].called_tools == [
        "rewrite_query",
        "rag_search",
        "rag_search",
    ]


def test_evaluation_endpoint_uses_dependency_override() -> None:
    class FakeEvaluationService:
        def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
            return EvaluateResponse(
                evaluation_id="eval-1",
                suites=request.suites,
                tool_behavior=ToolBehaviorSummary(
                    total_cases=1,
                    passed_cases=1,
                    failed_cases=0,
                    cases=[],
                ),
            )

    app.dependency_overrides[get_evaluation_service] = lambda: FakeEvaluationService()
    client = TestClient(app)

    response = client.post("/evaluate", json={"suites": ["tool_behavior"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["evaluation_id"] == "eval-1"
    assert payload["tool_behavior"]["passed_cases"] == 1

    app.dependency_overrides.clear()
