"""
POST /api/v1/query — run the macro analysis agent on a natural-language question.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models.request import QueryRequest
from api.models.response import QueryResponse
from agent.macro_agent import run_macro_agent
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Query"])


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_graph(request: QueryRequest) -> QueryResponse:
    """
    Accept a natural-language question, run the LangGraph macro agent,
    and return an explainable, causal-chain-driven insight.
    """
    logger.info("Query received: %s", request.question)

    try:
        result = run_macro_agent(request.question)

        return QueryResponse(
            question=result["question"],
            insight=result["insight"],
            causal_chain=result.get("causal_chain", []),
            sources=result.get("sources", []),
        )

    except Exception as exc:
        logger.error("Query processing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        )
