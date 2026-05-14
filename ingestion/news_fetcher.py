"""
Fetch India-related financial/macroeconomic news via NewsAPI.

If NEWS_API_KEY is missing or the API is unreachable the module returns
hardcoded sample snippets so the pipeline can still run.
"""

from __future__ import annotations

import requests
from dataclasses import dataclass, field
from typing import Any

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Re-use the same Document model from world_bank to keep a single type
from ingestion.world_bank import Document  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────────────

NEWS_API_URL = "https://newsapi.org/v2/everything"

KEYWORDS: list[str] = [
    "RBI",
    "NIFTY",
    "India GDP",
    "repo rate",
    "inflation India",
]

# ── Hardcoded fallback news ─────────────────────────────────────────────────

_FALLBACK_NEWS: list[dict[str, str]] = [
    {
        "title": "RBI holds repo rate steady at 6.5% amid easing inflation",
        "content": (
            "The Reserve Bank of India kept the benchmark repo rate unchanged "
            "at 6.5% for the eighth consecutive meeting. Governor Shaktikanta "
            "Das noted that inflation has moderated to 4.8% but remains above "
            "the 4% target. Markets responded positively with NIFTY gaining "
            "0.6% on the day."
        ),
        "source": "Economic Times",
        "published_at": "2024-08-08",
    },
    {
        "title": "FII outflows hit $3 billion as bond yields rise globally",
        "content": (
            "Foreign institutional investors pulled out over $3 billion from "
            "Indian equities in October as rising US Treasury yields made "
            "emerging-market assets less attractive. The sell-off weighed on "
            "NIFTY and Sensex, both declining more than 3% during the month."
        ),
        "source": "Mint",
        "published_at": "2024-10-31",
    },
    {
        "title": "India GDP growth beats expectations at 8.2% for FY24",
        "content": (
            "India's real GDP expanded 8.2% in 2023-24, surpassing the "
            "advance estimate of 7.6%. The growth was driven by strong "
            "manufacturing activity and public capital expenditure. "
            "Economists expect the momentum to slow slightly in FY25 as "
            "global demand weakens."
        ),
        "source": "Business Standard",
        "published_at": "2024-05-31",
    },
    {
        "title": "SEBI tightens F&O regulations to curb retail speculation",
        "content": (
            "The Securities and Exchange Board of India announced stricter "
            "margin requirements and position limits for futures and options "
            "trading. The move aims to reduce excessive retail participation "
            "in derivatives, which has surged in recent years. NIFTY options "
            "volumes fell 12% following the announcement."
        ),
        "source": "LiveMint",
        "published_at": "2024-09-15",
    },
    {
        "title": "Rupee weakens to 84.5 against the dollar on FII selling",
        "content": (
            "The Indian rupee depreciated to 84.5 per US dollar, driven by "
            "persistent foreign investor outflows and a strengthening dollar. "
            "The RBI intervened in the forex market to prevent sharp "
            "volatility. A weaker rupee raises import costs and puts upward "
            "pressure on inflation."
        ),
        "source": "Reuters",
        "published_at": "2024-11-10",
    },
]


def _fallback_documents() -> list[Document]:
    """Return hardcoded sample news documents."""
    logger.warning("NewsAPI unavailable or key missing — using sample news data.")
    return [
        Document(
            page_content=f"{item['title']}. {item['content']}",
            metadata={
                "source": "newsapi",
                "news_source": item["source"],
                "title": item["title"],
                "published_at": item["published_at"],
            },
        )
        for item in _FALLBACK_NEWS
    ]


# ── Public API ──────────────────────────────────────────────────────────────

def fetch_india_news(max_articles: int = 20) -> list[Document]:
    """
    Fetch recent India macro/financial news via NewsAPI.

    Returns List[Document] with metadata including title, source, date.
    Falls back to hardcoded sample snippets if the key is missing or
    the API is unreachable.
    """
    settings = get_settings()

    if not settings.has_news_api:
        return _fallback_documents()

    query = " OR ".join(KEYWORDS)
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "apiKey": settings.news_api_key,
    }

    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", [])
        if not articles:
            return _fallback_documents()

        documents: list[Document] = []
        for article in articles:
            title = article.get("title", "")
            content = article.get("content") or article.get("description") or ""
            source_name = article.get("source", {}).get("name", "Unknown")
            published = article.get("publishedAt", "")

            if not content:
                continue

            documents.append(
                Document(
                    page_content=f"{title}. {content}",
                    metadata={
                        "source": "newsapi",
                        "news_source": source_name,
                        "title": title,
                        "published_at": published,
                        "url": article.get("url", ""),
                    },
                )
            )

        logger.info("Fetched %d news articles from NewsAPI.", len(documents))
        return documents if documents else _fallback_documents()

    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.error("NewsAPI request failed: %s", exc)
        return _fallback_documents()
