"""
Entity extraction using SpaCy with a custom EntityRuler for Indian
macroeconomic domain terms.

The module lazily loads the SpaCy model on first use and adds pattern-based
entity rules for terms that the base model may not recognise
(e.g. RBI, NIFTY, Sensex, repo rate, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import spacy
from spacy.language import Language

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Entity data class ───────────────────────────────────────────────────────


@dataclass
class Entity:
    """A single extracted entity."""
    text: str
    label: str
    start: int
    end: int


# ── Custom domain patterns ──────────────────────────────────────────────────

_ECONOMIC_PATTERNS: list[dict] = [
    # Institutions / organisations
    {"label": "ORG", "pattern": "RBI"},
    {"label": "ORG", "pattern": "Reserve Bank of India"},
    {"label": "ORG", "pattern": "SEBI"},
    {"label": "ORG", "pattern": "Securities and Exchange Board of India"},
    {"label": "ORG", "pattern": "FII"},
    {"label": "ORG", "pattern": "DII"},
    {"label": "ORG", "pattern": "Foreign Institutional Investors"},
    {"label": "ORG", "pattern": "Domestic Institutional Investors"},

    # Market indices
    {"label": "MARKET", "pattern": "NIFTY"},
    {"label": "MARKET", "pattern": "Nifty 50"},
    {"label": "MARKET", "pattern": "NIFTY 50"},
    {"label": "MARKET", "pattern": "Sensex"},
    {"label": "MARKET", "pattern": "BSE Sensex"},

    # Macro indicators — multi-token patterns
    {"label": "INDICATOR", "pattern": [{"LOWER": "gdp"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "gdp"}, {"LOWER": "growth"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "cpi"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "inflation"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "consumer"}, {"LOWER": "price"}, {"LOWER": "index"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "unemployment"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "unemployment"}, {"LOWER": "rate"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "government"}, {"LOWER": "debt"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "public"}, {"LOWER": "debt"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "fiscal"}, {"LOWER": "balance"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "current"}, {"LOWER": "account"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "trade"}, {"LOWER": "balance"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "credit"}, {"LOWER": "growth"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "fx"}, {"LOWER": "reserves"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "foreign"}, {"LOWER": "exchange"}, {"LOWER": "reserves"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "energy"}, {"LOWER": "use"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "electricity"}, {"LOWER": "access"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "exports"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "imports"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "fdi"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "savings"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "investment"}]},
    {"label": "INDICATOR", "pattern": [{"LOWER": "employment"}]},

    # Policy / rates
    {"label": "POLICY", "pattern": [{"LOWER": "repo"}, {"LOWER": "rate"}]},
    {"label": "POLICY", "pattern": [{"LOWER": "reverse"}, {"LOWER": "repo"}, {"LOWER": "rate"}]},
    {"label": "POLICY", "pattern": [{"LOWER": "benchmark"}, {"LOWER": "rate"}]},
    {"label": "POLICY", "pattern": [{"LOWER": "interest"}, {"LOWER": "rate"}]},
    {"label": "POLICY", "pattern": [{"LOWER": "real"}, {"LOWER": "interest"}, {"LOWER": "rate"}]},
    {"label": "POLICY", "pattern": [{"LOWER": "monetary"}, {"LOWER": "policy"}]},

    # Financial instruments
    {"label": "INSTRUMENT", "pattern": [{"LOWER": "bond"}, {"LOWER": "yield"}]},
    {"label": "INSTRUMENT", "pattern": [{"LOWER": "bond"}, {"LOWER": "yields"}]},
    {"label": "INSTRUMENT", "pattern": [{"LOWER": "bond"}, {"LOWER": "yields"}, {"LOWER": "spreads"}]},
    {"label": "INSTRUMENT", "pattern": [{"LOWER": "treasury"}, {"LOWER": "yields"}]},
    {"label": "INSTRUMENT", "pattern": [{"LOWER": "10"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "year"}, {"LOWER": "bond"}]},

    # Currencies
    {"label": "CURRENCY", "pattern": [{"LOWER": "rupee"}]},
    {"label": "CURRENCY", "pattern": [{"LOWER": "inr"}]},
]

# ── Module-level SpaCy model cache ──────────────────────────────────────────

_nlp: Optional[Language] = None


def load_spacy_model() -> Language:
    """Load and configure the SpaCy model (singleton)."""
    global _nlp
    if _nlp is not None:
        return _nlp

    logger.info("Loading SpaCy model en_core_web_sm …")
    _nlp = spacy.load("en_core_web_sm")

    # Add entity ruler *before* the NER component so patterns take precedence
    if "entity_ruler" not in _nlp.pipe_names:
        ruler = _nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(_ECONOMIC_PATTERNS)
        logger.info("Added %d entity-ruler patterns.", len(_ECONOMIC_PATTERNS))

    return _nlp


# ── Public API ──────────────────────────────────────────────────────────────

def extract_entities(text: str) -> list[Entity]:
    """
    Extract named entities from *text* using SpaCy + custom domain rules.

    Returns a deduplicated list of ``Entity`` objects sorted by appearance.
    """
    nlp = load_spacy_model()
    doc = nlp(text)

    seen: set[str] = set()
    entities: list[Entity] = []

    for ent in doc.ents:
        key = f"{ent.text.lower()}|{ent.label_}"
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            )
        )

    logger.debug("Extracted %d entities from text (%d chars).", len(entities), len(text))
    return entities
