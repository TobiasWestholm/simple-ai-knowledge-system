from __future__ import annotations

import json
import os
import sqlite3
from contextlib import nullcontext
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol

try:
    from langsmith import run_helpers as _langsmith_run_helpers
except ImportError:  # pragma: no cover - dependency is available in runtime env
    tracing_context: Any = None
else:
    tracing_context = _langsmith_run_helpers.tracing_context

if TYPE_CHECKING:
    from ai_ks.config import Settings


@dataclass
class TimingSpan:
    sequence: int
    kind: str
    name: str
    duration_ms: float
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_ms"] = round(self.duration_ms, 3)
        return payload


class TimingCollector:
    def __init__(self, route: str, request_id: str) -> None:
        self.route = route
        self.request_id = request_id
        self.spans: list[TimingSpan] = []
        self._sequence = 0

    def record(
        self,
        *,
        kind: str,
        name: str,
        duration_ms: float,
        status: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._sequence += 1
        self.spans.append(
            TimingSpan(
                sequence=self._sequence,
                kind=kind,
                name=name,
                duration_ms=duration_ms,
                status=status,
                metadata=metadata or {},
            )
        )

    def build_diagnostics(self, total_duration_ms: float) -> dict[str, Any]:
        timings_ms = {"total_request": round(total_duration_ms, 3)}
        for span in self.spans:
            if span.kind == "agent" and span.name == "agent_graph.invoke":
                timings_ms["agent_graph_invoke"] = round(span.duration_ms, 3)
            elif span.kind == "operation":
                normalized_name = span.name.replace(".", "_")
                timings_ms[normalized_name] = round(span.duration_ms, 3)
        return {
            "timings_ms": timings_ms,
            "spans": [span.to_dict() for span in self.spans],
        }


_ACTIVE_COLLECTOR: ContextVar[TimingCollector | None] = ContextVar(
    "ai_ks_active_timing_collector",
    default=None,
)


def set_active_collector(collector: TimingCollector | None) -> Token[TimingCollector | None]:
    return _ACTIVE_COLLECTOR.set(collector)


def reset_active_collector(token: Token[TimingCollector | None]) -> None:
    _ACTIVE_COLLECTOR.reset(token)


def get_active_collector() -> TimingCollector | None:
    return _ACTIVE_COLLECTOR.get()


def time_call(
    *,
    kind: str,
    name: str,
    call: Any,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, float]:
    started = perf_counter()
    status = "success"
    try:
        result = call()
        return result, elapsed_ms(started)
    except Exception:
        status = "error"
        raise
    finally:
        collector = get_active_collector()
        if collector is not None:
            collector.record(
                kind=kind,
                name=name,
                duration_ms=elapsed_ms(started),
                status=status,
                metadata=metadata,
            )


def elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000


class ObservabilitySink(Protocol):
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
        ...


class LocalObservability:
    def __init__(self, sqlite_path: Path, log_jsonl_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.log_jsonl_path = log_jsonl_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

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
        created_at = datetime.now(UTC).isoformat()
        total_duration_ms = (
            diagnostics.get("timings_ms", {}).get("total_request")
            if isinstance(diagnostics, dict)
            else None
        )
        agent_invoke_ms = (
            diagnostics.get("timings_ms", {}).get("agent_graph_invoke")
            if isinstance(diagnostics, dict)
            else None
        )
        spans = diagnostics.get("spans", []) if isinstance(diagnostics, dict) else []

        log_record = {
            "created_at": created_at,
            "route": route,
            "request_id": request_id,
            "status": status,
            "total_duration_ms": total_duration_ms,
            "agent_graph_invoke_ms": agent_invoke_ms,
            "final_query": final_query,
            "error": error,
            "runtime": diagnostics.get("runtime") if isinstance(diagnostics, dict) else None,
            "diagnostics": diagnostics,
        }
        with self.log_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_record, ensure_ascii=True) + "\n")

        connection = sqlite3.connect(self.sqlite_path)
        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO request_runs (
                    request_id,
                    route,
                    status,
                    total_duration_ms,
                    agent_invoke_ms,
                    final_query,
                    answer_preview,
                    error,
                    diagnostics_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    route,
                    status,
                    total_duration_ms,
                    agent_invoke_ms,
                    final_query,
                    (answer or "")[:280],
                    error,
                    json.dumps(diagnostics, ensure_ascii=True, sort_keys=True),
                    created_at,
                ),
            )
            for span in spans:
                connection.execute(
                    """
                    INSERT INTO timing_spans (
                        request_id,
                        route,
                        sequence,
                        kind,
                        name,
                        duration_ms,
                        status,
                        metadata_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        route,
                        int(span.get("sequence", 0)),
                        str(span.get("kind", "")),
                        str(span.get("name", "")),
                        float(span.get("duration_ms", 0.0)),
                        str(span.get("status", "success")),
                        json.dumps(span.get("metadata", {}), ensure_ascii=True, sort_keys=True),
                        created_at,
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    def _ensure_tables(self) -> None:
        connection = sqlite3.connect(self.sqlite_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_runs (
                    request_id TEXT PRIMARY KEY,
                    route TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_duration_ms REAL,
                    agent_invoke_ms REAL,
                    final_query TEXT,
                    answer_preview TEXT,
                    error TEXT,
                    diagnostics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS timing_spans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    route TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()


def build_runtime_context(
    settings: Settings,
    *,
    route: str,
    embedding_device: str | None = None,
) -> dict[str, Any]:
    return {
        "route": route,
        "llm": {
            "url": settings.ollama_url,
            "model": settings.llm_runtime_model,
        },
        "embedding": {
            "url": settings.embedding_url,
            "model": settings.embed_model_id,
            "device": embedding_device or settings.embedding_device,
        },
        "qdrant": {
            "url": settings.qdrant_url,
            "collection": settings.qdrant_collection,
        },
        "langsmith": {
            "enabled": settings.langsmith_tracing and bool(settings.langsmith_api_key),
            "project": settings.langsmith_project,
        },
    }


def apply_langsmith_environment(settings: Settings) -> None:
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


def langsmith_request_context(
    settings: Settings,
    *,
    route: str,
    request_id: str,
):
    enabled = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    if not enabled or tracing_context is None:
        return nullcontext()
    apply_langsmith_environment(settings)
    return tracing_context(
        project_name=settings.langsmith_project,
        tags=["ai-ks", route.lstrip("/")],
        metadata={
            "request_id": request_id,
            "route": route,
            "llm_model": settings.llm_runtime_model,
            "embedding_model": settings.embed_model_id,
        },
        enabled=True,
    )
