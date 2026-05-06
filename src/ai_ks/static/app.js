const shell = document.querySelector(".app-shell");
const apiRoute = shell?.dataset.apiRoute ?? "/agent";
const maxQueryChars = Number(shell?.dataset.maxQueryChars ?? "2000");

const state = {
  conversation: [],
  threadId: createThreadId(),
  pendingMessage: null,
  pendingAssistantText: "",
  latestResponse: null,
};

const transcriptEl = document.getElementById("transcript");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");
const sendButtonEl = document.getElementById("send-button");
const resetButtonEl = document.getElementById("reset-button");
const charCountEl = document.getElementById("char-count");
const errorBannerEl = document.getElementById("error-banner");
const requestMetaEl = document.getElementById("request-meta");
const toolTimelineEl = document.getElementById("tool-timeline");
const timingsTableEl = document.getElementById("timings-table");
const citationsListEl = document.getElementById("citations-list");
const statusBandEl = document.getElementById("status-band");

function setError(message) {
  if (!message) {
    errorBannerEl.classList.add("hidden");
    errorBannerEl.textContent = "";
    return;
  }
  errorBannerEl.textContent = message;
  errorBannerEl.classList.remove("hidden");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
}

function renderMarkdown(value) {
  const source = String(value ?? "").replace(/\r\n/g, "\n").trim();
  if (!source) {
    return "";
  }

  const blocks = source.split(/\n\s*\n/);
  return blocks
    .map((block) => {
      const lines = block.split("\n");
      if (lines.every((line) => /^\s*[*-]\s+/.test(line))) {
        const items = lines
          .map((line) => line.replace(/^\s*[*-]\s+/, "").trim())
          .filter(Boolean)
          .map((line) => `<li>${renderInlineMarkdown(line)}</li>`)
          .join("");
        return `<ul>${items}</ul>`;
      }

      const heading = lines.length === 1 ? lines[0].match(/^(#{1,3})\s+(.*)$/) : null;
      if (heading) {
        const level = Math.min(heading[1].length, 3);
        return `<h${level}>${renderInlineMarkdown(heading[2].trim())}</h${level}>`;
      }

      const content = lines
        .map((line) => renderInlineMarkdown(line.trimEnd()))
        .join("<br>");
      return `<p>${content}</p>`;
    })
    .join("");
}

function renderMessageBody(message) {
  if (message.role === "assistant") {
    return `<div class="message-body markdown">${renderMarkdown(message.content)}</div>`;
  }
  return `<div class="message-body plain">${escapeHtml(message.content)}</div>`;
}

function formatDuration(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(1)} ms`;
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "number") {
    return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value);
}

function summarizeToolOutput(toolCall) {
  const output = toolCall.output ?? {};
  if (toolCall.name === "rag_search") {
    const hitCount = Array.isArray(output.hits) ? output.hits.length : 0;
    const citationCount = Array.isArray(output.citations) ? output.citations.length : 0;
    return [
      ["Query", output.query ?? "-"],
      ["Hits", String(hitCount)],
      ["Citations", String(citationCount)],
    ];
  }
  if (toolCall.name === "rewrite_query") {
    return [["Rewritten query", output.rewritten_query ?? "-"]];
  }
  if (toolCall.name === "summarize_context") {
    return [["Summary", output.summary ?? "-"]];
  }
  return Object.entries(output).slice(0, 3);
}

function renderTranscript() {
  const messages = [...state.conversation];
  if (state.pendingMessage) {
    messages.push({ role: "user", content: state.pendingMessage, pending: true });
    messages.push({
      role: "assistant",
      content: state.pendingAssistantText || "Working through the agent loop...",
      pending: true,
    });
  }

  if (messages.length === 0) {
    transcriptEl.innerHTML = `
      <section class="empty-state">
        <h3>Ask the knowledge base a question</h3>
        <p>
          The response panel will show the answer, citations, tool calls, and timings
          from the same agent path the API uses.
        </p>
      </section>
    `;
    return;
  }

  transcriptEl.innerHTML = messages
    .map((message, index) => {
      const roleLabel = message.role === "user" ? "You" : "Agent";
      const metaLabel = message.pending ? "In progress" : `Turn ${index + 1}`;
      return `
        <article class="message ${escapeHtml(message.role)}">
          <div class="message-header">
            <strong>${roleLabel}</strong>
            <span>${metaLabel}</span>
          </div>
          ${renderMessageBody(message)}
        </article>
      `;
    })
    .join("");
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function renderRequestMeta() {
  if (!state.latestResponse) {
    requestMetaEl.innerHTML = `
      <div><dt>Request ID</dt><dd>Waiting for first run</dd></div>
      <div><dt>Final query</dt><dd>-</dd></div>
    `;
    return;
  }

  requestMetaEl.innerHTML = `
    <div>
      <dt>Request ID</dt>
      <dd>${escapeHtml(state.latestResponse.request_id ?? "-")}</dd>
    </div>
    <div>
      <dt>Final query</dt>
      <dd>${escapeHtml(state.latestResponse.final_query ?? "-")}</dd>
    </div>
  `;
}

function renderTimeline() {
  const toolCalls = state.latestResponse?.tool_calls ?? [];
  if (toolCalls.length === 0) {
    toolTimelineEl.innerHTML = `<li class="placeholder-row">No tools used yet.</li>`;
    return;
  }

  toolTimelineEl.innerHTML = toolCalls
    .map((toolCall) => {
      const outputRows = summarizeToolOutput(toolCall)
        .map(
          ([label, value]) => `
            <div>
              <span>${escapeHtml(label)}</span>
              <strong>${escapeHtml(formatValue(value))}</strong>
            </div>
          `
        )
        .join("");

      return `
        <li class="timeline-row">
          <div class="row-title">
            <strong>${escapeHtml(toolCall.name)}</strong>
            <span class="status-tag ${escapeHtml(toolCall.status)}">
              ${escapeHtml(toolCall.status)} · ${escapeHtml(formatDuration(toolCall.duration_ms))}
            </span>
          </div>
          <div class="mini-grid">
            <div>
              <span>Arguments</span>
              <strong>${escapeHtml(JSON.stringify(toolCall.arguments ?? {}))}</strong>
            </div>
            ${outputRows}
          </div>
        </li>
      `;
    })
    .join("");
}

function renderTimings() {
  const timings = state.latestResponse?.diagnostics?.timings_ms ?? {};
  const entries = Object.entries(timings).sort((left, right) => right[1] - left[1]);
  if (entries.length === 0) {
    timingsTableEl.innerHTML = `<div class="placeholder-row">No timings yet.</div>`;
    return;
  }

  timingsTableEl.innerHTML = entries
    .map(
      ([name, value]) => `
        <div class="kv-row">
          <span>${escapeHtml(name)}</span>
          <strong>${escapeHtml(formatDuration(value))}</strong>
        </div>
      `
    )
    .join("");
}

function renderCitations() {
  const citations = state.latestResponse?.citations ?? [];
  if (citations.length === 0) {
    citationsListEl.innerHTML = `<div class="placeholder-row">No citations yet.</div>`;
    return;
  }

  citationsListEl.innerHTML = citations
    .map(
      (citation) => `
        <article class="citation-row">
          <div class="row-title">
            <strong>[${escapeHtml(citation.citation_id)}] ${escapeHtml(citation.title)}</strong>
            <span>${escapeHtml(citation.source_uri)}</span>
          </div>
          <p>${escapeHtml(citation.excerpt)}</p>
        </article>
      `
    )
    .join("");
}

function renderCharCount() {
  const length = inputEl.value.length;
  charCountEl.textContent = `${length} / ${maxQueryChars}`;
}

function setPending(isPending) {
  sendButtonEl.disabled = isPending;
  inputEl.disabled = isPending;
  resetButtonEl.disabled = isPending;
}

async function fetchHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error(`Health check failed with ${response.status}`);
    }
    const payload = await response.json();
    const services = payload.services ?? {};
    const labels = [
      ["API", "ok"],
      ["LLM", services.llm],
      ["Embedding", services.embedding],
      ["Qdrant", services.qdrant],
    ];
    statusBandEl.innerHTML = labels
      .map(([label, status]) => {
        const normalized = status === "available" || status === "reachable" ? "ok" : "error";
        return `
          <span class="status-pill ${escapeHtml(normalized)}">
            ${escapeHtml(label)}: ${escapeHtml(status)}
          </span>
        `;
      })
      .join("");
  } catch (error) {
    statusBandEl.innerHTML = `<span class="status-pill error">Health unavailable</span>`;
  }
}

function resetUi() {
  state.conversation = [];
  state.threadId = createThreadId();
  state.pendingMessage = null;
  state.pendingAssistantText = "";
  state.latestResponse = null;
  setError("");
  renderTranscript();
  renderRequestMeta();
  renderTimeline();
  renderTimings();
  renderCitations();
}

function createThreadId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `thread-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function decodeSseEvent(block) {
  const lines = block.split("\n");
  let eventName = "message";
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      let value = line.slice(5);
      if (value.startsWith(" ")) {
        value = value.slice(1);
      }
      dataLines.push(value);
    }
  }
  return { event: eventName, data: dataLines.join("\n") };
}

async function readStreamedAgentResponse(response) {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Streaming response body was not available.");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      if (!part.trim()) {
        continue;
      }
      const payload = decodeSseEvent(part);
      if (payload.event === "token") {
        state.pendingAssistantText += payload.data;
        renderTranscript();
      } else if (payload.event === "response") {
        finalResponse = JSON.parse(payload.data);
      } else if (payload.event === "error") {
        throw new Error(payload.data || "The request failed.");
      }
    }

    if (done) {
      break;
    }
  }

  if (buffer.trim()) {
    const payload = decodeSseEvent(buffer);
    if (payload.event === "response") {
      finalResponse = JSON.parse(payload.data);
    } else if (payload.event === "error") {
      throw new Error(payload.data || "The request failed.");
    }
  }

  if (!finalResponse) {
    throw new Error("The stream completed without a final response payload.");
  }
  return finalResponse;
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) {
    setError("Message is required.");
    return;
  }
  if (message.length > maxQueryChars) {
    setError(`Message must be at most ${maxQueryChars} characters.`);
    return;
  }

  setError("");
  state.pendingMessage = message;
  state.pendingAssistantText = "";
  renderTranscript();
  setPending(true);

  try {
    const response = await fetch(apiRoute, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        thread_id: state.threadId,
      }),
    });

    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail ?? "The request failed.");
    }
    const payload = await readStreamedAgentResponse(response);

    state.conversation.push({ role: "user", content: message });
    state.conversation.push({
      role: "assistant",
      content: payload.answer || state.pendingAssistantText || "",
    });
    state.latestResponse = payload;
    state.threadId = payload.thread_id || state.threadId;
    state.pendingMessage = null;
    state.pendingAssistantText = "";
    inputEl.value = "";
    renderCharCount();
    renderTranscript();
    renderRequestMeta();
    renderTimeline();
    renderTimings();
    renderCitations();
    fetchHealth();
  } catch (error) {
    state.pendingMessage = null;
    state.pendingAssistantText = "";
    renderTranscript();
    setError(error instanceof Error ? error.message : "The request failed.");
  } finally {
    setPending(false);
    inputEl.focus();
  }
});

resetButtonEl.addEventListener("click", () => {
  resetUi();
  inputEl.value = "";
  renderCharCount();
  inputEl.focus();
});

inputEl.addEventListener("input", renderCharCount);

renderCharCount();
renderTranscript();
renderRequestMeta();
renderTimeline();
renderTimings();
renderCitations();
fetchHealth();
