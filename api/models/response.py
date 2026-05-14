"""
Pydantic response models for the API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    """Response from the macro analysis agent."""
    question: str
    insight: str
    causal_chain: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    """Summary of an ingestion run."""
    status: str
    documents_processed: int = 0
    entities_found: int = 0
    relationships_created: int = 0


class GraphSummaryResponse(BaseModel):
    """High-level graph statistics."""
    node_count: int = 0
    relationship_count: int = 0
    top_connected_nodes: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Application health check response."""
    status: str
    neo4j_connected: bool
    graph_summary: dict = Field(default_factory=dict)
