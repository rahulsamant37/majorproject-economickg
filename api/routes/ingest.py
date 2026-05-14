"""
POST /api/v1/ingest — ingest data from World Bank, NewsAPI, or raw text.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models.request import IngestRequest
from api.models.response import IngestResponse
from ingestion.world_bank import Document, fetch_world_bank_indicators
from ingestion.news_fetcher import fetch_india_news
from graph.neo4j_client import Neo4jClient
from graph.graph_builder import GraphBuilder
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_data(request: IngestRequest) -> IngestResponse:
    """
    Ingest macroeconomic data and populate the knowledge graph.

    Supported sources:
    - ``worldbank``: fetch indicators from the World Bank API
    - ``news``: fetch financial news via NewsAPI
    - ``text``: process raw text provided in ``content``
    """
    logger.info("Ingest request: source=%s", request.source)

    try:
        # ── Fetch documents ─────────────────────────────────────────────
        documents: list[Document] = []

        if request.source == "worldbank":
            documents = fetch_world_bank_indicators()

        elif request.source == "news":
            documents = fetch_india_news()

        elif request.source == "text":
            if not request.content:
                raise HTTPException(
                    status_code=400,
                    detail="'content' field is required when source='text'.",
                )
            documents = [
                Document(
                    page_content=request.content,
                    metadata={"source": "user_text"},
                )
            ]

        if not documents:
            return IngestResponse(
                status="warning",
                documents_processed=0,
                entities_found=0,
                relationships_created=0,
            )

        # ── Build graph ─────────────────────────────────────────────────
        client = Neo4jClient()
        builder = GraphBuilder(client)
        totals = builder.build_from_documents(documents)

        return IngestResponse(
            status="success",
            documents_processed=totals["documents_processed"],
            entities_found=totals["entities_found"],
            relationships_created=totals["relationships_created"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        )
