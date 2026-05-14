"""
Pydantic request models for the API.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Natural-language query to the macro analysis agent."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["Why did NIFTY fall?"],
        description="The economic question to analyse.",
    )


class IngestRequest(BaseModel):
    """Request to ingest data from a specific source."""
    source: Literal["worldbank", "news", "text"] = Field(
        ...,
        description="Data source to ingest from.",
    )
    content: Optional[str] = Field(
        default=None,
        description=(
            "Raw text to ingest when source='text'. "
            "Ignored for 'worldbank' and 'news' sources."
        ),
    )
