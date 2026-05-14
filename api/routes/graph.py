"""
Graph inspection endpoints.

GET /api/v1/graph/summary        — node/relationship counts + top entities
GET /api/v1/graph/chain/{entity} — causal chain from a given entity
GET /api/v1/graph/search?q=      — keyword entity search
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from graph.neo4j_client import Neo4jClient
from graph.graph_retriever import GraphRetriever
from api.models.response import GraphSummaryResponse
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])


@router.get("/summary", response_model=GraphSummaryResponse)
async def graph_summary() -> GraphSummaryResponse:
    """Return high-level knowledge-graph statistics."""
    client = Neo4jClient()
    retriever = GraphRetriever(client)
    summary = retriever.get_graph_summary()

    return GraphSummaryResponse(
        node_count=summary.get("node_count", 0),
        relationship_count=summary.get("relationship_count", 0),
        top_connected_nodes=summary.get("top_connected_nodes", []),
    )


@router.get("/chain/{entity}")
async def causal_chain(entity: str) -> dict[str, Any]:
    """Return the causal chain originating from *entity*."""
    client = Neo4jClient()
    retriever = GraphRetriever(client)
    chain = retriever.get_causal_chain(entity)

    return {
        "entity": entity,
        "chain_length": len(chain),
        "chain": chain,
    }


@router.get("/search")
async def search_entities(
    q: str = Query(..., min_length=1, description="Keyword to search for"),
) -> dict[str, Any]:
    """Case-insensitive keyword search across all graph entities."""
    client = Neo4jClient()
    retriever = GraphRetriever(client)
    results = retriever.search_entities(q)

    return {
        "query": q,
        "count": len(results),
        "entities": results,
    }


@router.get("/visualize")
async def visualize_graph(
    limit: int = Query(default=100, ge=10, le=500, description="Max nodes/edges"),
) -> dict[str, Any]:
    """Return nodes and edges formatted for D3.js force-directed graph."""
    client = Neo4jClient()
    retriever = GraphRetriever(client)
    return retriever.get_visualization_data(limit=limit)

