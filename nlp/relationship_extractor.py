"""
LLM-based relationship extraction using LangChain.

Given a text passage the module prompts an LLM to extract
(subject, relation, object) triples with a constrained set of
relation types.  Entity names are normalised during extraction
(e.g. "central bank" → "RBI").

When no LLM key is configured the module falls back to a
deterministic rule-based extractor so the pipeline still works.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Relationship data class ─────────────────────────────────────────────────


@dataclass
class Relationship:
    """A single extracted causal/economic relationship."""
    source: str
    relation: str
    target: str
    sentence: str


ALLOWED_RELATIONS = [
    "CAUSES",
    "IMPACTS",
    "LEADS_TO",
    "CORRELATES_WITH",
    "AFFECTS",
]

# ── Entity normalisation map ────────────────────────────────────────────────

_NORMALISATION: dict[str, str] = {
    "rbi": "RBI",
    "central bank": "RBI",
    "reserve bank": "RBI",
    "reserve bank of india": "RBI",
    "rbi rate hike": "Repo Rate",
    "benchmark rate": "Repo Rate",
    "repo rate": "Repo Rate",
    "policy rate": "Repo Rate",
    "interest rate": "Repo Rate",
    "rate hike": "Repo Rate",
    "higher rates": "Repo Rate",
    "real interest rate": "Real Interest Rate",
    "real interest rates": "Real Interest Rate",
    "consumer price index": "CPI",
    "cpi": "CPI",
    "inflation": "Inflation",
    "rising inflation": "Inflation",
    "cpi inflation": "Inflation",
    "bond yield": "Bond Yield",
    "bond yields": "Bond Yield",
    "rising bond yields": "Bond Yield",
    "treasury yields": "Bond Yield",
    "10-year bond": "Bond Yield",
    "fixed income": "Bond Yield",
    "nifty": "NIFTY",
    "nifty 50": "NIFTY",
    "nifty fall": "NIFTY",
    "indian equity markets": "NIFTY",
    "equity markets": "NIFTY",
    "sensex": "Sensex",
    "bse sensex": "Sensex",
    "foreign institutional investors": "FII",
    "fii": "FII",
    "fii outflows": "FII",
    "fii outflow": "FII",
    "fii selling": "FII",
    "foreign investors": "FII",
    "domestic institutional investors": "DII",
    "dii": "DII",
    "sebi": "SEBI",
    "gdp": "GDP",
    "gdp growth": "GDP",
    "gdp growth (annual %)": "GDP",
    "economic activity": "GDP",
    "economic growth": "GDP",
    "rupee": "Rupee",
    "inr": "Rupee",
    "indian rupee": "Rupee",
    "liquidity": "Liquidity",
    "reduced liquidity": "Liquidity",
    "monetary policy": "Monetary Policy",
    "inflation rate": "Inflation",
    "borrowing": "Borrowing Cost",
    "borrowing cost": "Borrowing Cost",
    "borrowing costs": "Borrowing Cost",
    "real interest rate": "Real Interest Rate",
    "real interest rate (%)": "Real Interest Rate",
    "real interest rates": "Real Interest Rate",
    "cpi inflation": "Inflation",
    "unemployment": "Unemployment",
    "unemployment rate": "Unemployment",
    "exports": "Exports",
    "imports": "Imports",
    "trade balance": "Trade Balance",
    "current account": "Current Account",
    "current account balance": "Current Account",
    "fiscal balance": "Fiscal Balance",
    "government debt": "Government Debt",
    "public debt": "Government Debt",
    "fdi": "FDI",
    "foreign direct investment": "FDI",
    "savings": "Savings",
    "gross savings": "Savings",
    "credit growth": "Credit Growth",
    "private sector credit": "Credit Growth",
    "domestic credit": "Credit Growth",
    "fx reserves": "FX Reserves",
    "foreign exchange reserves": "FX Reserves",
    "exchange rate": "Exchange Rate",
    "official exchange rate": "Exchange Rate",
    "trade": "Trade",
    "energy use": "Energy Use",
    "electricity access": "Electricity Access",
}


def _normalise(entity: str) -> str:
    """Normalise an entity name using the lookup table."""
    return _NORMALISATION.get(entity.lower().strip(), entity.strip())


# ── LLM-based extraction ───────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert economist. From the given text extract causal economic
relationships as JSON.  Return a JSON array of objects, each with keys:
"source", "relation", "target", "sentence".

Rules:
1. relation MUST be one of: CAUSES, IMPACTS, LEADS_TO, CORRELATES_WITH, AFFECTS.
2. Normalise entity names to standard forms:
   - "central bank" / "reserve bank" → "RBI"
   - "benchmark rate" / "policy rate" → "Repo Rate"
   - "consumer price index" → "CPI"
   - "nifty 50" → "NIFTY"
   - "foreign institutional investors" → "FII"
3. Each sentence should be the original text that supports the relationship.
4. Return ONLY valid JSON, no markdown fencing, no commentary.
"""


def _extract_with_llm(text: str) -> list[Relationship]:
    """Use LangChain + LLM to extract relationships."""
    settings = get_settings()

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.openai_api_key,
        )
    elif settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=settings.groq_api_key,
        )
    else:
        logger.warning("No LLM configured — falling back to rule-based extraction.")
        return _extract_rules(text)

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Extract relationships from:\n\n{text}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        triples: list[dict[str, str]] = json.loads(raw)

        relationships: list[Relationship] = []
        for triple in triples:
            rel_type = triple.get("relation", "").upper()
            if rel_type not in ALLOWED_RELATIONS:
                continue
            relationships.append(
                Relationship(
                    source=_normalise(triple.get("source", "")),
                    relation=rel_type,
                    target=_normalise(triple.get("target", "")),
                    sentence=triple.get("sentence", text[:200]),
                )
            )

        logger.info("LLM extracted %d relationships.", len(relationships))
        return relationships

    except Exception as exc:
        logger.error("LLM extraction failed (%s) — falling back to rules.", exc)
        return _extract_rules(text)


# ── Rule-based fallback ────────────────────────────────────────────────────

# Simple regex-based heuristics that look for causal language patterns
_CAUSAL_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)(\b\w[\w\s]{1,30})\s+(?:caused|causes|cause)\s+(\b\w[\w\s]{1,30})", "CAUSES"),
    (r"(?i)(\b\w[\w\s]{1,30})\s+(?:impacts?|impacted|impacting)\s+(\b\w[\w\s]{1,30})", "IMPACTS"),
    (r"(?i)(\b\w[\w\s]{1,30})\s+(?:leads?\s+to|led\s+to)\s+(\b\w[\w\s]{1,30})", "LEADS_TO"),
    (r"(?i)(\b\w[\w\s]{1,30})\s+(?:affects?|affected|affecting)\s+(\b\w[\w\s]{1,30})", "AFFECTS"),
    (r"(?i)(\b\w[\w\s]{1,30})\s+(?:correlates?\s+with)\s+(\b\w[\w\s]{1,30})", "CORRELATES_WITH"),
]

# Additional keyword-proximity heuristics for economic text
_KEYWORD_PAIRS: list[tuple[str, str, str]] = [
    ("repo rate", "inflation", "CAUSES"),
    ("rbi", "repo rate", "CAUSES"),
    ("rate hike", "bond yield", "LEADS_TO"),
    ("bond yield", "fii", "IMPACTS"),
    ("fii outflow", "nifty", "CAUSES"),
    ("fii", "nifty", "IMPACTS"),
    ("inflation", "gdp", "AFFECTS"),
    ("rupee", "inflation", "CORRELATES_WITH"),
]


def _extract_rules(text: str) -> list[Relationship]:
    """Deterministic rule-based relationship extraction."""
    relationships: list[Relationship] = []
    text_lower = text.lower()

    # 1. Regex causal patterns
    for pattern, rel_type in _CAUSAL_PATTERNS:
        for match in re.finditer(pattern, text):
            src = _normalise(match.group(1).strip())
            tgt = _normalise(match.group(2).strip())
            if src and tgt and src != tgt:
                relationships.append(
                    Relationship(
                        source=src,
                        relation=rel_type,
                        target=tgt,
                        sentence=text[max(0, match.start() - 20): match.end() + 20].strip(),
                    )
                )

    # 2. Keyword-proximity heuristics
    for kw1, kw2, rel_type in _KEYWORD_PAIRS:
        if kw1 in text_lower and kw2 in text_lower:
            src = _normalise(kw1)
            tgt = _normalise(kw2)
            # Avoid adding duplicate relationships
            if not any(r.source == src and r.target == tgt for r in relationships):
                relationships.append(
                    Relationship(
                        source=src,
                        relation=rel_type,
                        target=tgt,
                        sentence=text[:200],
                    )
                )

    logger.info("Rule-based extractor found %d relationships.", len(relationships))
    return relationships


# ── Public API ──────────────────────────────────────────────────────────────

def extract_relationships(text: str) -> list[Relationship]:
    """
    Extract economic relationships from *text*.

    Uses LLM when available, otherwise falls back to rule-based
    extraction.  All entity names are normalised.
    """
    settings = get_settings()
    if settings.has_llm:
        return _extract_with_llm(text)
    return _extract_rules(text)
