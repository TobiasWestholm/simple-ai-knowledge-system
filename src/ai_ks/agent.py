from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any, Literal, Protocol
from uuid import uuid4

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=DEFAULT_MAX_QUERY_CHARS)
    conversation: list[ConversationTurn] = Field(default_factory=list)
    max_steps: int | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return normalize_user_text(
            value,
            field_name="Message",
            max_chars=DEFAULT_MAX_QUERY_CHARS,
        )


class CitationResponse(BaseModel):
    citation_id: int
    chunk_id: str
    title: str
    source_uri: str
    chunk_index: int
    excerpt: str


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    status: Literal["success", "error"]
    duration_ms: float | None = None


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    tool_calls: list[ToolCallRecord]
    citations: list[CitationResponse]
    final_query: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class RagSearchInput(BaseModel):
    query: str = Field(description="User question or rewritten search query to search for.")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of hits to return.")


class RagSearchOutput(BaseModel):
    query: str
    hits: list[dict[str, Any]]
    citations: list[CitationResponse]
    diagnostics: dict[str, Any]


class SummarizeContextInput(BaseModel):
    text: str = Field(description="The text to summarize.")
    style: str = Field(
        default="a concise answer-focused summary",
        description="How the summary should be written.",
    )


class SummarizeContextOutput(BaseModel):
    summary: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class RewriteQueryInput(BaseModel):
    query: str = Field(description="The original user query to rewrite for retrieval.")


class RewriteQueryOutput(BaseModel):
    rewritten_query: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class AgentGraphLike(Protocol):
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        ...


class LangChainAgentService:
    SYSTEM_PROMPT = (
        "You are a local knowledge assistant. Use the provided tools to answer questions about "
        "the indexed knowledge base. Rewrite ambiguous retrieval requests before searching. "
        "When a question needs evidence from the knowledge base, use rag_search. When the user "
        "asks for a summary, use summarize_context. After using rag_search, answer with bracket "
        "citations like [1] or [2] that match the returned citations. Prefer concise, grounded "
        "answers and avoid unsupported claims."
    )

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever | None = None,
        utility_model: Any | None = None,
        agent_graph: AgentGraphLike | None = None,
        observability: ObservabilitySink | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever or HybridRetriever(settings)
        self.utility_model = utility_model or self._build_model()
        self.rewrite_chain = self._build_rewrite_chain()
        self.summary_chain = self._build_summary_chain()
        self.tools = self._build_tools()
        self.agent_graph = agent_graph or self._build_agent_graph()
        self.observability = observability or LocalObservability(
            sqlite_path=settings.sqlite_path,
            log_jsonl_path=settings.log_jsonl_path,
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        request.message = normalize_user_text(
            request.message,
            field_name="Message",
            max_chars=self.settings.max_query_chars,
        )
        request_id = str(uuid4())
        collector = TimingCollector(route="/agent", request_id=request_id)
        token = set_active_collector(collector)
        started = perf_counter()
        config = {
            "recursion_limit": max(25, (request.max_steps or self.settings.agent_max_steps) * 4),
        }
        response: AgentResponse | None = None
        error: str | None = None
        try:
            try:
                with langsmith_request_context(
                    self.settings,
                    route="/agent",
                    request_id=request_id,
                ):
                    result, _ = time_call(
                        kind="agent",
                        name="agent_graph.invoke",
                        call=lambda: self.agent_graph.invoke(
                            {"messages": self._build_messages(request)},
                            config={
                                **config,
                                "metadata": {
                                    "request_id": request_id,
                                    "route": "/agent",
                                },
                            },
                        ),
                    )
            except Exception as exc:
                raise DependencyUnavailableError(
                    "llm",
                    f"LLM service unavailable during agent execution: {exc}",
                ) from exc
            diagnostics = collector.build_diagnostics(elapsed_ms(started))
            diagnostics["runtime"] = build_runtime_context(self.settings, route="/agent")
            response = self._build_response(
                request_id=request_id,
                result=result,
                diagnostics=diagnostics,
            )
            return response
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            diagnostics = collector.build_diagnostics(elapsed_ms(started))
            diagnostics["runtime"] = build_runtime_context(self.settings, route="/agent")
            self.observability.record_request(
                route="/agent",
                request_id=request_id,
                status="error" if error else "success",
                diagnostics=response.diagnostics if response else diagnostics,
                final_query=response.final_query if response else None,
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

    def _build_agent_graph(self) -> AgentGraphLike:
        return create_agent(
            model=self._build_model(),
            tools=self.tools,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def _build_rewrite_chain(self) -> Any:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the user query for document retrieval. Preserve the user's intent. "
                    "Return only the rewritten query.",
                ),
                ("human", "{query}"),
            ]
        )
        return prompt | self.utility_model | StrOutputParser()

    def _build_summary_chain(self) -> Any:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Summarize the provided text. Keep the summary factual, concise, and useful "
                    "for answering the user's question.",
                ),
                ("human", "Write {style}.\n\nText to summarize:\n{text}"),
            ]
        )
        return prompt | self.utility_model | StrOutputParser()

    def _build_tools(self) -> list[BaseTool]:
        @tool(args_schema=RagSearchInput, response_format="content_and_artifact")
        def rag_search(query: str, limit: int = 5) -> tuple[str, dict[str, Any]]:
            """Search the local knowledge base with hybrid retrieval and citations."""
            started = perf_counter()
            output = self._rag_search(RagSearchInput(query=query, limit=limit))
            duration_ms = elapsed_ms(started)
            artifact = output.model_dump()
            timings_ms = dict(artifact.get("diagnostics", {}).get("timings_ms", {}))
            timings_ms["tool_execution"] = round(duration_ms, 3)
            artifact.setdefault("diagnostics", {})["timings_ms"] = timings_ms
            collector = get_active_collector()
            if collector is not None:
                collector.record(
                    kind="tool",
                    name="rag_search",
                    duration_ms=duration_ms,
                    metadata={"limit": limit},
                )
            return self._rag_search_content(output), artifact

        @tool(args_schema=SummarizeContextInput, response_format="content_and_artifact")
        def summarize_context(
            text: str,
            style: str = "a concise answer-focused summary",
        ) -> tuple[str, dict[str, Any]]:
            """Summarize supplied text into a concise answer-focused summary."""
            started = perf_counter()
            output = self._summarize_context(
                SummarizeContextInput(text=text, style=style)
            )
            duration_ms = elapsed_ms(started)
            artifact = output.model_dump()
            timings_ms = dict(artifact.get("diagnostics", {}).get("timings_ms", {}))
            timings_ms["tool_execution"] = round(duration_ms, 3)
            artifact.setdefault("diagnostics", {})["timings_ms"] = timings_ms
            collector = get_active_collector()
            if collector is not None:
                collector.record(
                    kind="tool",
                    name="summarize_context",
                    duration_ms=duration_ms,
                )
            return output.summary, artifact

        @tool(args_schema=RewriteQueryInput, response_format="content_and_artifact")
        def rewrite_query(query: str) -> tuple[str, dict[str, Any]]:
            """Rewrite a user query into a retrieval-friendly search query."""
            started = perf_counter()
            output = self._rewrite_query(RewriteQueryInput(query=query))
            duration_ms = elapsed_ms(started)
            artifact = output.model_dump()
            timings_ms = dict(artifact.get("diagnostics", {}).get("timings_ms", {}))
            timings_ms["tool_execution"] = round(duration_ms, 3)
            artifact.setdefault("diagnostics", {})["timings_ms"] = timings_ms
            collector = get_active_collector()
            if collector is not None:
                collector.record(
                    kind="tool",
                    name="rewrite_query",
                    duration_ms=duration_ms,
                )
            return output.rewritten_query, artifact

        return [rag_search, summarize_context, rewrite_query]

    def _build_messages(self, request: AgentRequest) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        for turn in request.conversation:
            if turn.role == "assistant":
                messages.append(AIMessage(content=turn.content))
            else:
                messages.append(HumanMessage(content=turn.content))
        messages.append(HumanMessage(content=request.message))
        return messages

    def _rag_search(self, tool_input: RagSearchInput) -> RagSearchOutput:
        result, duration_ms = time_call(
            kind="operation",
            name="retriever.search",
            call=lambda: self.retriever.search(
                query=tool_input.query,
                limit=tool_input.limit,
            ),
            metadata={"tool_name": "rag_search"},
        )
        return RagSearchOutput(
            query=result.query,
            hits=[hit.to_dict() for hit in result.hits],
            citations=[
                CitationResponse.model_validate(citation.to_dict())
                for citation in result.citations
            ],
            diagnostics={
                **result.diagnostics,
                "timings_ms": {
                    **result.diagnostics.get("timings_ms", {}),
                    "retriever_search": round(duration_ms, 3),
                },
            },
        )

    def _summarize_context(self, tool_input: SummarizeContextInput) -> SummarizeContextOutput:
        try:
            summary, duration_ms = time_call(
                kind="operation",
                name="summary_chain.invoke",
                call=lambda: self.summary_chain.invoke(
                    {
                        "style": tool_input.style,
                        "text": tool_input.text,
                    },
                    config=self._langchain_metadata_config("summarize_context"),
                ),
                metadata={"tool_name": "summarize_context"},
            )
        except Exception as exc:
            raise DependencyUnavailableError(
                "llm",
                f"LLM service unavailable during summarize_context: {exc}",
            ) from exc
        return SummarizeContextOutput(
            summary=str(summary).strip(),
            diagnostics={"timings_ms": {"summary_chain_invoke": round(duration_ms, 3)}},
        )

    def _rewrite_query(self, tool_input: RewriteQueryInput) -> RewriteQueryOutput:
        try:
            rewritten_query, duration_ms = time_call(
                kind="operation",
                name="rewrite_chain.invoke",
                call=lambda: self.rewrite_chain.invoke(
                    {"query": tool_input.query},
                    config=self._langchain_metadata_config("rewrite_query"),
                ),
                metadata={"tool_name": "rewrite_query"},
            )
        except Exception as exc:
            raise DependencyUnavailableError(
                "llm",
                f"LLM service unavailable during rewrite_query: {exc}",
            ) from exc
        return RewriteQueryOutput(
            rewritten_query=str(rewritten_query).strip(),
            diagnostics={"timings_ms": {"rewrite_chain_invoke": round(duration_ms, 3)}},
        )

    def _build_response(
        self,
        request_id: str,
        result: Any,
        diagnostics: dict[str, Any],
    ) -> AgentResponse:
        raw_messages = result.get("messages", []) if isinstance(result, dict) else []
        messages = [message for message in raw_messages if isinstance(message, BaseMessage)]

        tool_call_arguments: dict[str, tuple[str, dict[str, Any]]] = {}
        tool_calls: list[ToolCallRecord] = []
        citations: list[CitationResponse] = []
        final_query: str | None = None

        for message in messages:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    tool_call_arguments[str(tool_call.get("id", ""))] = (
                        str(tool_call.get("name", "")),
                        dict(tool_call.get("args", {})),
                    )
            elif isinstance(message, ToolMessage):
                tool_name, arguments = tool_call_arguments.get(
                    message.tool_call_id,
                    (message.name or "", {}),
                )
                payload = self._tool_payload(message)
                tool_calls.append(
                    ToolCallRecord(
                        name=tool_name,
                        arguments=arguments,
                        output=payload,
                        status="error" if message.status == "error" else "success",
                        duration_ms=self._tool_duration(payload),
                    )
                )
                if tool_name == "rag_search" and message.status != "error":
                    citations = [
                        CitationResponse.model_validate(citation)
                        for citation in payload.get("citations", [])
                    ]
                    final_query = str(payload.get("query", final_query or ""))
                elif tool_name == "rewrite_query" and message.status != "error":
                    final_query = str(payload.get("rewritten_query", final_query or ""))

        answer = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                answer = self._message_text(message)
                break

        return AgentResponse(
            request_id=request_id,
            answer=answer,
            tool_calls=tool_calls,
            citations=citations,
            final_query=final_query,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _tool_payload(message: ToolMessage) -> dict[str, Any]:
        if isinstance(message.artifact, dict):
            return message.artifact
        return LangChainAgentService._parse_tool_payload(message.content)

    @staticmethod
    def _parse_tool_payload(content: Any) -> dict[str, Any]:
        if isinstance(content, str):
            try:
                from json import JSONDecodeError, loads

                payload = loads(content)
                if isinstance(payload, dict):
                    return payload
            except JSONDecodeError:
                return {"content": content}

        if isinstance(content, list):
            joined = []
            for item in content:
                if isinstance(item, str):
                    joined.append(item)
                elif isinstance(item, dict) and "text" in item:
                    joined.append(str(item["text"]))
            combined = "\n".join(part for part in joined if part)
            return LangChainAgentService._parse_tool_payload(combined)

        return {"content": str(content)}

    @staticmethod
    def _message_text(message: AIMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "\n".join(part for part in parts if part).strip()
        return str(content)

    @staticmethod
    def _tool_duration(payload: dict[str, Any]) -> float | None:
        diagnostics = payload.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            return None
        timings_ms = diagnostics.get("timings_ms", {})
        if not isinstance(timings_ms, dict):
            return None
        duration_ms = timings_ms.get("tool_execution")
        if isinstance(duration_ms, (int, float)):
            return float(duration_ms)
        return None

    @staticmethod
    def _rag_search_content(output: RagSearchOutput) -> str:
        if not output.hits:
            return "No matching chunks found in the local knowledge base."

        lines = [f"Search query: {output.query}"]
        for citation, hit in zip(output.citations, output.hits, strict=False):
            lines.append(
                f"[{citation.citation_id}] {citation.title} "
                f"({citation.source_uri}, chunk {citation.chunk_index})"
            )
            lines.append(hit["text"])
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _langchain_metadata_config(tool_name: str) -> dict[str, Any]:
        collector = get_active_collector()
        if collector is None:
            return {}
        return {
            "metadata": {
                "request_id": collector.request_id,
                "route": collector.route,
                "tool_name": tool_name,
            }
        }


@lru_cache(maxsize=1)
def get_agent_service() -> LangChainAgentService:
    return LangChainAgentService(settings=get_settings())
