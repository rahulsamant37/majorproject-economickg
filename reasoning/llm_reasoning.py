"""
LLM reasoning over knowledge-graph context.

Takes graph data (causal chains + neighbours from Neo4j) and the original
user query, then produces a structured, explainable answer via a LangChain
chain.  Falls back to a template-based answer when no LLM is configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Result model ────────────────────────────────────────────────────────────


@dataclass
class ReasoningResult:
    """Structured output from the reasoning step."""
    answer: str
    causal_chain: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


# ── System prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert macroeconomic analyst for India. You are given a causal
knowledge graph extracted from real economic data. Using the graph
relationships provided, explain the answer to the user's question step by
step. Identify the cause-effect chain explicitly.

If the graph does not contain enough information, say so clearly instead of
guessing.

Some edges may be marked as PREDICTED_LINK; treat them as hypotheses and
clearly label any inferences that rely on them.

When presenting your answer:
1. Start with a concise summary.
2. List the causal chain as numbered steps (e.g. "1. RBI raised repo rate →").
3. Cite the supporting evidence from the graph data.
"""


# ── Graph context formatter ─────────────────────────────────────────────────

def _format_graph_context(
    graph_data: list[dict[str, Any]],
    neighbor_data: list[dict[str, Any]],
    predicted_links: list[dict[str, Any]],
) -> str:
    """Format graph query results into a textual context block for the LLM."""
    lines: list[str] = []

    if graph_data:
        lines.append("=== Causal Chain (from Knowledge Graph) ===")
        for edge in graph_data:
            src = edge.get("from", "?")
            rel = edge.get("relation", "?")
            tgt = edge.get("to", "?")
            sentence = edge.get("sentence", "")
            lines.append(f"  {src} --[{rel}]--> {tgt}")
            if sentence:
                lines.append(f"    Evidence: {sentence}")

    if neighbor_data:
        lines.append("\n=== Neighbouring Entities ===")
        for nb in neighbor_data:
            name = nb.get("neighbor", "?")
            rel = nb.get("relation", "?")
            direction = nb.get("direction", "?")
            lines.append(f"  {name} ({rel}, {direction})")

    if predicted_links:
        lines.append("\n=== Predicted Links (Hypotheses) ===")
        for edge in predicted_links:
            src = edge.get("from", "?")
            tgt = edge.get("to", "?")
            score = edge.get("score", 0)
            lines.append(f"  {src} --[PREDICTED_LINK]--> {tgt} (score {score})")

    if not lines:
        lines.append("(No graph data available for this query.)")

    return "\n".join(lines)


def _extract_causal_chain(graph_data: list[dict[str, Any]]) -> list[str]:
    """Build a human-readable causal chain list from graph edges."""
    chain: list[str] = []
    for edge in graph_data:
        src = edge.get("from", "?")
        rel = edge.get("relation", "?").replace("_", " ").lower()
        tgt = edge.get("to", "?")
        chain.append(f"{src} {rel} {tgt}")
    return chain


def _extract_sources(graph_data: list[dict[str, Any]]) -> list[str]:
    """Collect unique supporting sentences from graph edges."""
    sources: list[str] = []
    seen: set[str] = set()
    for edge in graph_data:
        sentence = edge.get("sentence", "")
        if sentence and sentence not in seen:
            seen.add(sentence)
            sources.append(sentence)
    return sources


# ── LLM-based reasoning ────────────────────────────────────────────────────

def _reason_with_llm(
    query: str,
    graph_context: str,
    graph_data: list[dict[str, Any]],
) -> ReasoningResult:
    """Use LangChain to generate an insight from graph context."""
    settings = get_settings()

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=settings.openai_api_key,
        )
    elif settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=settings.groq_api_key,
        )
    else:
        return _reason_without_llm(query, graph_data)

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Graph Context:\n{graph_context}\n\n"
                f"User Question: {query}\n\n"
                "Provide a step-by-step causal explanation."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content.strip()

        return ReasoningResult(
            answer=answer,
            causal_chain=_extract_causal_chain(graph_data),
            sources=_extract_sources(graph_data),
        )

    except Exception as exc:
        logger.error("LLM reasoning failed: %s — using template fallback.", exc)
        return _reason_without_llm(query, graph_data)


# ── Template-based fallback ─────────────────────────────────────────────────

def _reason_without_llm(
    query: str,
    graph_data: list[dict[str, Any]],
) -> ReasoningResult:
    """Generate a structured answer without an LLM using the graph data."""
    causal_chain = _extract_causal_chain(graph_data)
    sources = _extract_sources(graph_data)

    if not causal_chain:
        answer = (
            f"Regarding your question: \"{query}\"\n\n"
            "The knowledge graph does not contain enough information to "
            "provide a causal explanation. Try ingesting more data or "
            "refining your query."
        )
    else:
        chain_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(causal_chain))
        answer = (
            f"Based on the economic knowledge graph, here is the causal "
            f"chain related to your question:\n\n{chain_text}\n\n"
            f"This chain was derived from {len(sources)} supporting "
            f"evidence snippet(s) in the graph."
        )

    return ReasoningResult(
        answer=answer,
        causal_chain=causal_chain,
        sources=sources,
    )


# ── Public API ──────────────────────────────────────────────────────────────

def reason_over_graph(
    query: str,
    graph_data: list[dict[str, Any]],
    neighbor_data: list[dict[str, Any]] | None = None,
    predicted_links: list[dict[str, Any]] | None = None,
) -> ReasoningResult:
    """
    Produce an explainable answer by reasoning over graph context.

    Uses LLM when available, otherwise constructs a template-based
    response from the causal chain.
    """
    neighbor_data = neighbor_data or []
    predicted_links = predicted_links or []
    graph_context = _format_graph_context(graph_data, neighbor_data, predicted_links)
    logger.debug("Graph context for reasoning:\n%s", graph_context)

    settings = get_settings()
    if settings.has_llm:
        return _reason_with_llm(query, graph_context, graph_data)
    return _reason_without_llm(query, graph_data)
