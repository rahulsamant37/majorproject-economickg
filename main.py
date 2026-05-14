"""
AI-Powered Economic Knowledge Graph for India Macro Analysis
=============================================================

FastAPI application entrypoint.

Startup lifecycle:
    1. Connect to Neo4j
    2. Load SpaCy model
    3. Seed demo data if the graph is empty

Shutdown lifecycle:
    1. Close Neo4j driver
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from graph.neo4j_client import Neo4jClient
from graph.graph_builder import GraphBuilder
from graph.graph_retriever import GraphRetriever
from ingestion.world_bank import Document
from nlp.entity_extractor import load_spacy_model
from api.models.response import HealthResponse
from api.routes import query as query_router
from api.routes import ingest as ingest_router
from api.routes import graph as graph_router
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Demo seed documents ─────────────────────────────────────────────────────

DEMO_DOCUMENTS: list[Document] = [
    Document(
        page_content=(
            "The RBI increased the repo rate by 50 basis points to control "
            "rising inflation. Higher rates make borrowing expensive, slowing "
            "economic activity."
        ),
        metadata={"source": "demo_seed", "doc_id": 1},
    ),
    Document(
        page_content=(
            "Following the RBI rate hike, bond yields surged as fixed income "
            "became more attractive relative to equities."
        ),
        metadata={"source": "demo_seed", "doc_id": 2},
    ),
    Document(
        page_content=(
            "Rising bond yields triggered FII outflows from Indian equity "
            "markets. Foreign investors pulled out over $2 billion in a "
            "single week."
        ),
        metadata={"source": "demo_seed", "doc_id": 3},
    ),
    Document(
        page_content=(
            "The combined effect of FII outflows and reduced liquidity caused "
            "NIFTY to fall by 2.3% in a single session."
        ),
        metadata={"source": "demo_seed", "doc_id": 4},
    ),
]


def _seed_demo_data(client: Neo4jClient) -> None:
    """Seed the graph with demo documents if it is currently empty."""
    retriever = GraphRetriever(client)
    summary = retriever.get_graph_summary()

    if summary.get("node_count", 0) > 0:
        logger.info(
            "Graph already contains %d nodes — skipping demo seed.",
            summary["node_count"],
        )
        return

    logger.info("Graph is empty — seeding %d demo documents …", len(DEMO_DOCUMENTS))
    builder = GraphBuilder(client)
    totals = builder.build_from_documents(DEMO_DOCUMENTS)
    logger.info(
        "Demo seed complete: %d docs, %d entities, %d relationships.",
        totals["documents_processed"],
        totals["entities_found"],
        totals["relationships_created"],
    )


# ── Lifespan context manager ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup / shutdown lifecycle."""
    settings = get_settings()

    # ── Startup ─────────────────────────────────────────────────────────
    logger.info("Starting AI Economic Knowledge Graph …")
    logger.info("LLM provider: %s", settings.llm_provider)

    # 1. Neo4j connection
    client = Neo4jClient()
    client.connect()

    # 2. SpaCy model
    load_spacy_model()

    # 3. Demo seed
    if settings.demo_seed:
        _seed_demo_data(client)

    yield

    # ── Shutdown ────────────────────────────────────────────────────────
    logger.info("Shutting down …")
    client.close()


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Economic Knowledge Graph",
    description=(
        "Ingest macroeconomic data, build a causal knowledge graph in Neo4j, "
        "and query it with natural language for explainable insights."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(query_router.router)
app.include_router(ingest_router.router)
app.include_router(graph_router.router)


# ── Health check ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Return Neo4j connectivity status and a graph summary snapshot."""
    client = Neo4jClient()
    connected = client.ping()

    summary: dict = {}
    if connected:
        retriever = GraphRetriever(client)
        summary = retriever.get_graph_summary()

    return HealthResponse(
        status="healthy" if connected else "degraded",
        neo4j_connected=connected,
        graph_summary=summary,
    )


# ── Dashboard ───────────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard() -> HTMLResponse:
    """Serve the interactive knowledge graph dashboard."""
    html_path = _STATIC_DIR / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── Root ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root() -> dict:
    """Redirect-friendly root endpoint."""
    return {
        "app": "AI Economic Knowledge Graph",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "dashboard": "/dashboard",
    }
