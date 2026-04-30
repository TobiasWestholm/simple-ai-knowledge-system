from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from ai_ks.agent import AgentRequest, AgentResponse, LangChainAgentService
from ai_ks.config import DEFAULT_MAX_QUERY_CHARS, Settings, get_settings
from ai_ks.query import QueryRequest

EvaluationSuite = Literal["tool_behavior", "failure", "timing"]


def _default_suites() -> list[EvaluationSuite]:
    return ["tool_behavior", "failure", "timing"]


class EvaluateRequest(BaseModel):
    suites: list[EvaluationSuite] = Field(default_factory=_default_suites)
    case_ids: list[str] = Field(default_factory=list)


class TimingStats(BaseModel):
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float


class ToolBehaviorCase(BaseModel):
    id: str
    category: str
    prompt: str
    source_documents: list[str] = Field(default_factory=list)
    source_excerpt: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    ordered_tools: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = None
    allow_repeated_tools: bool = False
    rewrite_must_change_query: bool = False
    rewrite_reference_query: str | None = None
    notes: str = ""


class ToolBehaviorCaseResult(BaseModel):
    id: str
    category: str
    prompt: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    called_tools: list[str] = Field(default_factory=list)
    final_query: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ToolBehaviorSummary(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    cases: list[ToolBehaviorCaseResult]


class FailureCase(BaseModel):
    id: str
    target: Literal["agent", "query"]
    payload: dict[str, Any]
    expect_valid: bool
    expected_error_contains: str | None = None


class FailureCaseResult(BaseModel):
    id: str
    target: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    accepted: bool


class FailureSummary(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    cases: list[FailureCaseResult]


class TimingCaseSample(BaseModel):
    id: str
    called_tools: list[str]
    timings_ms: dict[str, float]


class TimingSummary(BaseModel):
    total_cases: int
    successful_cases: int
    failed_cases: int
    request_metrics: dict[str, TimingStats]
    tool_metrics: dict[str, TimingStats]
    cases: list[TimingCaseSample]


class EvaluateResponse(BaseModel):
    evaluation_id: str
    suites: list[EvaluationSuite]
    tool_behavior: ToolBehaviorSummary | None = None
    failure: FailureSummary | None = None
    timing: TimingSummary | None = None


class _ToolRunResult(BaseModel):
    case: ToolBehaviorCase
    response: AgentResponse | None = None
    error: str | None = None


class EvaluationService:
    def __init__(
        self,
        settings: Settings,
        agent_service_factory: Callable[[], LangChainAgentService] | None = None,
    ) -> None:
        self.settings = settings
        self.agent_service_factory = agent_service_factory or (
            lambda: LangChainAgentService(settings=settings)
        )

    def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        suites = request.suites or ["tool_behavior", "failure", "timing"]
        evaluation_id = str(uuid4())

        tool_runs: list[_ToolRunResult] = []
        if "tool_behavior" in suites or "timing" in suites:
            cases = self._select_tool_behavior_cases(request.case_ids)
            agent_service = self.agent_service_factory()
            tool_runs = [self._run_tool_behavior_case(agent_service, case) for case in cases]

        return EvaluateResponse(
            evaluation_id=evaluation_id,
            suites=suites,
            tool_behavior=self._build_tool_behavior_summary(tool_runs)
            if "tool_behavior" in suites
            else None,
            failure=self._build_failure_summary() if "failure" in suites else None,
            timing=self._build_timing_summary(tool_runs) if "timing" in suites else None,
        )

    def _select_tool_behavior_cases(self, case_ids: list[str]) -> list[ToolBehaviorCase]:
        cases = load_tool_behavior_cases(self.settings.eval_dir / "tool_behavior_cases.json")
        if not case_ids:
            return cases
        by_id = {case.id: case for case in cases}
        missing = [case_id for case_id in case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"Unknown evaluation case ids: {', '.join(sorted(missing))}")
        return [by_id[case_id] for case_id in case_ids]

    def _run_tool_behavior_case(
        self,
        agent_service: LangChainAgentService,
        case: ToolBehaviorCase,
    ) -> _ToolRunResult:
        try:
            response = agent_service.run(AgentRequest(message=case.prompt))
            return _ToolRunResult(case=case, response=response)
        except Exception as exc:
            return _ToolRunResult(case=case, error=str(exc))

    def _build_tool_behavior_summary(
        self,
        tool_runs: list[_ToolRunResult],
    ) -> ToolBehaviorSummary:
        results = [self._evaluate_tool_behavior(run) for run in tool_runs]
        passed_cases = sum(1 for result in results if result.passed)
        return ToolBehaviorSummary(
            total_cases=len(results),
            passed_cases=passed_cases,
            failed_cases=len(results) - passed_cases,
            cases=results,
        )

    def _evaluate_tool_behavior(self, run: _ToolRunResult) -> ToolBehaviorCaseResult:
        case = run.case
        if run.response is None:
            return ToolBehaviorCaseResult(
                id=case.id,
                category=case.category,
                prompt=case.prompt,
                passed=False,
                reasons=[f"Agent execution failed: {run.error or 'unknown error'}"],
            )

        response = run.response
        called_tools = [record.name for record in response.tool_calls if record.status == "success"]
        reasons: list[str] = []

        missing_tools = [tool for tool in case.required_tools if tool not in called_tools]
        if missing_tools:
            reasons.append(f"Missing required tools: {', '.join(missing_tools)}.")

        forbidden_tools = [tool for tool in case.forbidden_tools if tool in called_tools]
        if forbidden_tools:
            reasons.append(f"Called forbidden tools: {', '.join(forbidden_tools)}.")

        if case.max_tool_calls is not None and len(called_tools) > case.max_tool_calls:
            reasons.append(
                f"Used {len(called_tools)} tool calls, above the limit of {case.max_tool_calls}."
            )

        if not case.allow_repeated_tools and len(set(called_tools)) != len(called_tools):
            reasons.append("Repeated a tool call in a case that forbids repetition.")

        if case.ordered_tools and not _is_subsequence(case.ordered_tools, called_tools):
            reasons.append(
                "Required tool order was not respected: "
                f"{' -> '.join(case.ordered_tools)}."
            )

        if case.rewrite_must_change_query:
            rewrite_outputs = [
                record.output.get("rewritten_query", "")
                for record in response.tool_calls
                if record.name == "rewrite_query" and record.status == "success"
            ]
            if not rewrite_outputs:
                reasons.append("Expected rewrite_query to produce a rewritten query.")
            else:
                rewrite_reference = case.rewrite_reference_query or case.prompt
                if _normalize_case_text(rewrite_outputs[-1]) == _normalize_case_text(
                    rewrite_reference
                ):
                    reasons.append("rewrite_query did not materially change the query.")

        return ToolBehaviorCaseResult(
            id=case.id,
            category=case.category,
            prompt=case.prompt,
            passed=not reasons,
            reasons=reasons,
            called_tools=called_tools,
            final_query=run.response.final_query,
            timings_ms=_numeric_timings(run.response.diagnostics.get("timings_ms", {})),
        )

    def _build_failure_summary(self) -> FailureSummary:
        results = [self._evaluate_failure_case(case) for case in _failure_cases()]
        passed_cases = sum(1 for result in results if result.passed)
        return FailureSummary(
            total_cases=len(results),
            passed_cases=passed_cases,
            failed_cases=len(results) - passed_cases,
            cases=results,
        )

    def _evaluate_failure_case(self, case: FailureCase) -> FailureCaseResult:
        model_cls = AgentRequest if case.target == "agent" else QueryRequest
        reasons: list[str] = []
        try:
            model_cls.model_validate(case.payload)
            accepted = True
        except ValidationError as exc:
            accepted = False
            error_message = str(exc)
            if case.expect_valid:
                reasons.append(f"Unexpected validation error: {error_message}")
            elif case.expected_error_contains and case.expected_error_contains not in error_message:
                reasons.append(
                    "Validation error did not mention the expected contract: "
                    f"{case.expected_error_contains!r}."
                )
        else:
            if not case.expect_valid:
                reasons.append("Payload was accepted but should have been rejected.")

        return FailureCaseResult(
            id=case.id,
            target=case.target,
            passed=not reasons,
            reasons=reasons,
            accepted=accepted,
        )

    def _build_timing_summary(self, tool_runs: list[_ToolRunResult]) -> TimingSummary:
        successful_runs = [run for run in tool_runs if run.response is not None]
        request_samples: dict[str, list[float]] = {}
        tool_samples: dict[str, list[float]] = {}
        case_samples: list[TimingCaseSample] = []

        for run in successful_runs:
            assert run.response is not None
            timings_ms = _numeric_timings(run.response.diagnostics.get("timings_ms", {}))
            for name, duration in timings_ms.items():
                request_samples.setdefault(name, []).append(duration)
            for tool_call in run.response.tool_calls:
                if tool_call.duration_ms is not None and tool_call.status == "success":
                    tool_samples.setdefault(tool_call.name, []).append(tool_call.duration_ms)
            case_samples.append(
                TimingCaseSample(
                    id=run.case.id,
                    called_tools=[
                        record.name
                        for record in run.response.tool_calls
                        if record.status == "success"
                    ],
                    timings_ms=timings_ms,
                )
            )

        return TimingSummary(
            total_cases=len(tool_runs),
            successful_cases=len(successful_runs),
            failed_cases=len(tool_runs) - len(successful_runs),
            request_metrics={
                name: _build_timing_stats(values)
                for name, values in sorted(request_samples.items())
            },
            tool_metrics={
                name: _build_timing_stats(values)
                for name, values in sorted(tool_samples.items())
            },
            cases=case_samples,
        )


def _failure_cases() -> list[FailureCase]:
    valid_boundary_query = "x" * DEFAULT_MAX_QUERY_CHARS
    invalid_query = "x" * (DEFAULT_MAX_QUERY_CHARS + 1)
    return [
        FailureCase(
            id="agent-empty",
            target="agent",
            payload={"message": ""},
            expect_valid=False,
            expected_error_contains="must not be empty or whitespace",
        ),
        FailureCase(
            id="agent-whitespace",
            target="agent",
            payload={"message": "   "},
            expect_valid=False,
            expected_error_contains="must not be empty or whitespace",
        ),
        FailureCase(
            id="agent-over-limit",
            target="agent",
            payload={"message": invalid_query},
            expect_valid=False,
            expected_error_contains=f"at most {DEFAULT_MAX_QUERY_CHARS}",
        ),
        FailureCase(
            id="agent-at-limit",
            target="agent",
            payload={"message": valid_boundary_query},
            expect_valid=True,
        ),
        FailureCase(
            id="query-empty",
            target="query",
            payload={"query": ""},
            expect_valid=False,
            expected_error_contains="must not be empty or whitespace",
        ),
        FailureCase(
            id="query-whitespace",
            target="query",
            payload={"query": "  \n  "},
            expect_valid=False,
            expected_error_contains="must not be empty or whitespace",
        ),
        FailureCase(
            id="query-over-limit",
            target="query",
            payload={"query": invalid_query},
            expect_valid=False,
            expected_error_contains=f"at most {DEFAULT_MAX_QUERY_CHARS}",
        ),
        FailureCase(
            id="query-at-limit",
            target="query",
            payload={"query": valid_boundary_query},
            expect_valid=True,
        ),
    ]


@lru_cache(maxsize=4)
def load_tool_behavior_cases(path: Path) -> list[ToolBehaviorCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Tool behavior cases at {path} must be a JSON list.")

    cases = [ToolBehaviorCase.model_validate(item) for item in payload]
    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate tool behavior case ids: {', '.join(duplicates)}")
    return cases


def _normalize_case_text(value: str) -> str:
    return " ".join(value.lower().split())


def _is_subsequence(required: list[str], actual: list[str]) -> bool:
    if not required:
        return True
    index = 0
    for tool_name in actual:
        if tool_name == required[index]:
            index += 1
            if index == len(required):
                return True
    return False


def _numeric_timings(payload: dict[str, Any]) -> dict[str, float]:
    return {
        name: round(float(value), 3)
        for name, value in payload.items()
        if isinstance(value, (int, float))
    }


def _build_timing_stats(values: list[float]) -> TimingStats:
    ordered = sorted(float(value) for value in values)
    return TimingStats(
        count=len(ordered),
        min_ms=round(ordered[0], 3),
        max_ms=round(ordered[-1], 3),
        mean_ms=round(mean(ordered), 3),
        median_ms=round(median(ordered), 3),
    )


@lru_cache(maxsize=1)
def get_evaluation_service() -> EvaluationService:
    return EvaluationService(settings=get_settings())
