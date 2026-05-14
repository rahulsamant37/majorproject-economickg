"""
LangGraph-based macro-economic analysis agent.

Implements a StateGraph with four nodes:
    1. retrieve_graph   — fetch causal chains from Neo4j
    2. retrieve_neighbors — fetch neighbouring entities for context
    3. reason           — run LLM reasoning over graph data
    4. respond          — assemble the final QueryResponse

Entity extraction (SpaCy) on the incoming query drives which graph
lookups are performed.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from graph.neo4j_client import Neo4jClient
from graph.graph_retriever import GraphRetriever
from nlp.entity_extractor import extract_entities
from reasoning.llm_reasoning import reason_over_graph
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Agent state ─────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """State dictionary threaded through the LangGraph nodes."""
    query: str
    entities: list[str]
    graph_data: list[dict[str, Any]]
    neighbor_data: list[dict[str, Any]]
    predicted_links: list[dict[str, Any]]
    use_predictions: bool
    answer: str
    causal_chain: list[str]
    sources: list[str]


# ── Node functions ──────────────────────────────────────────────────────────

def _extract_query_entities(state: AgentState) -> AgentState:
    """Pre-processing: extract key entities from the user query for
    targeted graph lookup."""
    query = state["query"]
    ents = extract_entities(query)
    entity_names = list({e.text for e in ents})

    # If SpaCy did not find anything, try splitting the query for
    # common macro keywords as a fallback.
    if not entity_names:
        _KEYWORDS = [
            "RBI", "NIFTY", "Sensex", "GDP", "CPI", "Inflation",
            "Repo Rate", "Bond Yield", "FII", "DII", "SEBI", "Rupee",
        ]
        for kw in _KEYWORDS:
            if kw.lower() in query.lower():
                entity_names.append(kw)

    logger.info("Query entities: %s", entity_names)
    state["entities"] = entity_names
    return state


def _retrieve_graph(state: AgentState) -> AgentState:
    """Node 1: retrieve causal chains from Neo4j for each entity."""
    client = Neo4jClient()
    retriever = GraphRetriever(client)

    all_chains: list[dict[str, Any]] = []
    for entity in state.get("entities", []):
        chains = retriever.get_causal_chain(entity)
        all_chains.extend(chains)

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for edge in all_chains:
        key = f"{edge.get('from')}|{edge.get('relation')}|{edge.get('to')}"
        if key not in seen:
            seen.add(key)
            unique.append(edge)

    state["graph_data"] = unique
    logger.info("Retrieved %d causal edges.", len(unique))
    return state


def _retrieve_neighbors(state: AgentState) -> AgentState:
    """Node 2: retrieve neighbour context for each entity."""
    client = Neo4jClient()
    retriever = GraphRetriever(client)

    all_neighbors: list[dict[str, Any]] = []
    for entity in state.get("entities", []):
        neighbors = retriever.get_neighbors(entity)
        all_neighbors.extend(neighbors)

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for nb in all_neighbors:
        key = f"{nb.get('neighbor')}|{nb.get('relation')}|{nb.get('direction')}"
        if key not in seen:
            seen.add(key)
            unique.append(nb)

    state["neighbor_data"] = unique
    logger.info("Retrieved %d neighbor edges.", len(unique))
    return state


def _predict_links(state: AgentState) -> AgentState:
    """Node 2b: predict missing links to reduce sparsity."""
    if not state.get("use_predictions"):
        state["predicted_links"] = []
        return state

    client = Neo4jClient()
    retriever = GraphRetriever(client)

    predictions: list[dict[str, Any]] = []
    for entity in state.get("entities", []):
        predictions.extend(retriever.predict_links(entity, limit=5))

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for edge in predictions:
        key = f"{edge.get('from')}|{edge.get('relation')}|{edge.get('to')}"
        if key not in seen:
            seen.add(key)
            unique.append(edge)

    state["predicted_links"] = unique
    logger.info("Predicted %d candidate edges.", len(unique))
    return state


def _reason(state: AgentState) -> AgentState:
    """Node 3: run LLM reasoning over the collected graph data."""
    result = reason_over_graph(
        query=state["query"],
        graph_data=state.get("graph_data", []),
        neighbor_data=state.get("neighbor_data", []),
        predicted_links=state.get("predicted_links", []),
    )
    state["answer"] = result.answer
    state["causal_chain"] = result.causal_chain
    state["sources"] = result.sources
    return state


def _respond(state: AgentState) -> AgentState:
    """Node 4: final assembly — nothing extra needed, the state already
    holds everything the API route will use."""
    logger.info("Agent completed. Answer length: %d chars.", len(state.get("answer", "")))
    return state


# ── Graph construction ──────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("extract_entities", _extract_query_entities)
    graph.add_node("retrieve_graph", _retrieve_graph)
    graph.add_node("retrieve_neighbors", _retrieve_neighbors)
    graph.add_node("predict_links", _predict_links)
    graph.add_node("reason", _reason)
    graph.add_node("respond", _respond)

    graph.set_entry_point("extract_entities")
    graph.add_edge("extract_entities", "retrieve_graph")
    graph.add_edge("retrieve_graph", "retrieve_neighbors")
    graph.add_edge("retrieve_neighbors", "predict_links")
    graph.add_edge("predict_links", "reason")
    graph.add_edge("reason", "respond")
    graph.add_edge("respond", END)

    return graph


# Compile once at module level
_compiled_graph = _build_graph().compile()


# ── Public API ──────────────────────────────────────────────────────────────

def run_macro_agent(
    query: str,
    *,
    use_predictions: bool = False,
    return_debug: bool = False,
) -> dict[str, Any]:
    """
    Run the macro analysis agent on a natural-language query.

    Returns a dict with keys: query, answer, causal_chain, sources.
    """
    initial_state: AgentState = {
        "query": query,
        "entities": [],
        "graph_data": [],
        "neighbor_data": [],
        "predicted_links": [],
        "use_predictions": use_predictions,
        "answer": "",
        "causal_chain": [],
        "sources": [],
    }

    final_state = _compiled_graph.invoke(initial_state)

    result = {
        "question": query,
        "insight": final_state.get("answer", ""),
        "causal_chain": final_state.get("causal_chain", []),
        "sources": final_state.get("sources", []),
    }

    if return_debug:
        result["predicted_links"] = final_state.get("predicted_links", [])
        result["entities"] = final_state.get("entities", [])
        result["graph_data_count"] = len(final_state.get("graph_data", []))

    return result
