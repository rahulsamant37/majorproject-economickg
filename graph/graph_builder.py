"""
Graph builder — orchestrates entity extraction, relationship extraction,
and writes the results to Neo4j.

All writes use MERGE (not CREATE) to prevent duplicate nodes/edges.
"""

from __future__ import annotations

from typing import Any

from ingestion.world_bank import Document
from nlp.entity_extractor import Entity, extract_entities
from nlp.relationship_extractor import Relationship, extract_relationships
from graph.neo4j_client import Neo4jClient
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Label mapping ───────────────────────────────────────────────────────────

# Map entity labels from SpaCy / EntityRuler to Neo4j node labels.
_LABEL_MAP: dict[str, str] = {
    "ORG": "EconomicEntity",
    "MARKET": "Market",
    "INDICATOR": "MacroIndicator",
    "POLICY": "Policy",
    "INSTRUMENT": "Market",
    "CURRENCY": "MacroIndicator",
    "GPE": "EconomicEntity",
    "PERSON": "EconomicEntity",
    "MONEY": "MacroIndicator",
    "PERCENT": "MacroIndicator",
}

# ── World Bank indicator relationship hints ────────────────────────────────

_WORLDBANK_RELATION_HINTS: dict[str, list[tuple[str, str, str, str]]] = {
    "NY.GDP.MKTP.KD.ZG": [
        (
            "GDP",
            "IMPACTS",
            "NIFTY",
            "GDP growth typically supports equity market performance such as NIFTY.",
        ),
        (
            "GDP",
            "AFFECTS",
            "Economic Activity",
            "GDP growth reflects changes in overall economic activity.",
        ),
        (
            "GDP",
            "CORRELATES_WITH",
            "Employment",
            "GDP growth is often correlated with employment conditions.",
        ),
    ],
    "FP.CPI.TOTL.ZG": [
        (
            "Inflation",
            "LEADS_TO",
            "Repo Rate",
            "Higher CPI inflation often leads to tighter monetary policy via repo rate changes.",
        ),
        (
            "Inflation",
            "AFFECTS",
            "Borrowing Cost",
            "Inflation can influence borrowing costs across the economy.",
        ),
        (
            "Inflation",
            "AFFECTS",
            "GDP",
            "Inflation can reduce real purchasing power and affect GDP growth.",
        ),
        (
            "Inflation",
            "LEADS_TO",
            "Bond Yield",
            "Rising inflation expectations can push up bond yields.",
        ),
        (
            "Inflation",
            "CORRELATES_WITH",
            "Rupee",
            "Inflation is often correlated with currency movements such as the Rupee.",
        ),
    ],
    "FR.INR.RINR": [
        (
            "Real Interest Rate",
            "AFFECTS",
            "Borrowing Cost",
            "Real interest rates influence borrowing costs for households and firms.",
        ),
        (
            "Real Interest Rate",
            "IMPACTS",
            "GDP",
            "Real interest rates can impact investment and GDP growth.",
        ),
        (
            "Real Interest Rate",
            "AFFECTS",
            "Investment",
            "Real interest rates affect investment decisions across the economy.",
        ),
        (
            "Real Interest Rate",
            "AFFECTS",
            "Credit Growth",
            "Real interest rates influence credit demand and growth.",
        ),
        (
            "Real Interest Rate",
            "IMPACTS",
            "Rupee",
            "Real interest rate differentials can impact Rupee valuation.",
        ),
    ],
    "SL.UEM.TOTL.ZS": [
        (
            "Unemployment",
            "IMPACTS",
            "GDP",
            "Higher unemployment tends to reduce GDP growth through weaker labor utilization.",
        ),
        (
            "Unemployment",
            "AFFECTS",
            "Consumption",
            "Unemployment can lower household consumption.",
        ),
        (
            "Unemployment",
            "CORRELATES_WITH",
            "Inflation",
            "Unemployment and inflation often move together through business cycles.",
        ),
    ],
    "NE.EXP.GNFS.ZS": [
        (
            "Exports",
            "IMPACTS",
            "GDP",
            "Stronger exports can support GDP growth.",
        ),
        (
            "Exports",
            "LEADS_TO",
            "Trade Balance",
            "Exports contribute to the trade balance position.",
        ),
        (
            "Exports",
            "CORRELATES_WITH",
            "Rupee",
            "Export performance is often correlated with the Rupee exchange rate.",
        ),
    ],
    "NE.IMP.GNFS.ZS": [
        (
            "Imports",
            "IMPACTS",
            "GDP",
            "Imports reflect domestic demand conditions that affect GDP.",
        ),
        (
            "Imports",
            "LEADS_TO",
            "Trade Balance",
            "Imports influence the trade balance position.",
        ),
        (
            "Imports",
            "CORRELATES_WITH",
            "Rupee",
            "Import costs are sensitive to Rupee movements.",
        ),
    ],
    "BN.CAB.XOKA.GD.ZS": [
        (
            "Current Account",
            "AFFECTS",
            "Rupee",
            "Current account balances can affect Rupee valuation.",
        ),
        (
            "Current Account",
            "IMPACTS",
            "FX Reserves",
            "Current account deficits can draw down FX reserves.",
        ),
        (
            "Current Account",
            "CORRELATES_WITH",
            "Trade Balance",
            "The current account often correlates with the trade balance.",
        ),
    ],
    "GC.BAL.CASH.GD.ZS": [
        (
            "Fiscal Balance",
            "IMPACTS",
            "Government Debt",
            "Fiscal deficits can increase government debt levels.",
        ),
        (
            "Fiscal Balance",
            "AFFECTS",
            "Bond Yield",
            "Larger deficits can push up government borrowing costs and bond yields.",
        ),
        (
            "Fiscal Balance",
            "AFFECTS",
            "Inflation",
            "Persistent deficits can add inflationary pressures.",
        ),
    ],
    "GC.DOD.TOTL.GD.ZS": [
        (
            "Government Debt",
            "IMPACTS",
            "Bond Yield",
            "Higher debt levels can raise bond yields through risk premia.",
        ),
        (
            "Government Debt",
            "AFFECTS",
            "Borrowing Cost",
            "Debt burdens can lift overall borrowing costs in the economy.",
        ),
        (
            "Government Debt",
            "IMPACTS",
            "GDP",
            "High debt can weigh on long-term GDP growth.",
        ),
    ],
    "BX.KLT.DINV.WD.GD.ZS": [
        (
            "FDI",
            "IMPACTS",
            "GDP",
            "FDI inflows support capital formation and GDP growth.",
        ),
        (
            "FDI",
            "AFFECTS",
            "Investment",
            "FDI contributes to investment activity in the economy.",
        ),
        (
            "FDI",
            "CORRELATES_WITH",
            "Rupee",
            "FDI flows often correlate with Rupee stability and sentiment.",
        ),
    ],
    "NY.GNS.ICTR.ZS": [
        (
            "Savings",
            "AFFECTS",
            "Investment",
            "Higher savings can finance higher investment.",
        ),
        (
            "Savings",
            "IMPACTS",
            "GDP",
            "Savings rates can influence long-term GDP growth.",
        ),
        (
            "Savings",
            "AFFECTS",
            "Borrowing Cost",
            "Higher savings can ease funding pressures and borrowing costs.",
        ),
    ],
    "FS.AST.PRVT.GD.ZS": [
        (
            "Credit Growth",
            "IMPACTS",
            "GDP",
            "Private sector credit growth supports economic expansion.",
        ),
        (
            "Credit Growth",
            "AFFECTS",
            "Investment",
            "Credit availability drives investment activity.",
        ),
        (
            "Credit Growth",
            "AFFECTS",
            "Borrowing Cost",
            "Credit conditions influence borrowing costs across sectors.",
        ),
    ],
    "PA.NUS.FCRF": [
        (
            "Exchange Rate",
            "AFFECTS",
            "Inflation",
            "A weaker exchange rate can raise imported inflation pressures.",
        ),
        (
            "Exchange Rate",
            "IMPACTS",
            "Exports",
            "Exchange rate moves impact export competitiveness.",
        ),
        (
            "Exchange Rate",
            "IMPACTS",
            "Imports",
            "Exchange rate moves affect import costs.",
        ),
    ],
    "FI.RES.TOTL.CD": [
        (
            "FX Reserves",
            "IMPACTS",
            "Rupee",
            "Stronger FX reserves can support currency stability.",
        ),
        (
            "FX Reserves",
            "AFFECTS",
            "Current Account",
            "Reserve adequacy interacts with current account dynamics.",
        ),
        (
            "FX Reserves",
            "AFFECTS",
            "Imports",
            "FX reserves influence the ability to cover imports.",
        ),
    ],
    "NE.TRD.GNFS.ZS": [
        (
            "Trade",
            "IMPACTS",
            "GDP",
            "Trade intensity contributes to overall GDP performance.",
        ),
        (
            "Trade",
            "CORRELATES_WITH",
            "Rupee",
            "Trade flows often correlate with Rupee movements.",
        ),
        (
            "Trade",
            "AFFECTS",
            "Current Account",
            "Trade balance shifts affect the current account.",
        ),
    ],
    "EG.USE.PCAP.KG.OE": [
        (
            "Energy Use",
            "IMPACTS",
            "GDP",
            "Energy consumption per capita is linked to economic output.",
        ),
        (
            "Energy Use",
            "CORRELATES_WITH",
            "Inflation",
            "Energy use is often correlated with energy price inflation.",
        ),
    ],
    "EG.ELC.ACCS.ZS": [
        (
            "Electricity Access",
            "IMPACTS",
            "GDP",
            "Wider electricity access supports productivity and GDP growth.",
        ),
        (
            "Electricity Access",
            "AFFECTS",
            "Employment",
            "Electricity access enables job creation and formal employment.",
        ),
    ],
}

_WORLDBANK_ENTITY_MAP: dict[str, str] = {
    "NY.GDP.MKTP.KD.ZG": "GDP",
    "FP.CPI.TOTL.ZG": "Inflation",
    "FR.INR.RINR": "Real Interest Rate",
    "SL.UEM.TOTL.ZS": "Unemployment",
    "NE.EXP.GNFS.ZS": "Exports",
    "NE.IMP.GNFS.ZS": "Imports",
    "BN.CAB.XOKA.GD.ZS": "Current Account",
    "GC.BAL.CASH.GD.ZS": "Fiscal Balance",
    "GC.DOD.TOTL.GD.ZS": "Government Debt",
    "BX.KLT.DINV.WD.GD.ZS": "FDI",
    "NY.GNS.ICTR.ZS": "Savings",
    "FS.AST.PRVT.GD.ZS": "Credit Growth",
    "PA.NUS.FCRF": "Exchange Rate",
    "FI.RES.TOTL.CD": "FX Reserves",
    "NE.TRD.GNFS.ZS": "Trade",
    "EG.USE.PCAP.KG.OE": "Energy Use",
    "EG.ELC.ACCS.ZS": "Electricity Access",
}

_TREND_THRESHOLD_BY_UNIT: dict[str, float] = {
    "%": 0.2,
    "INR per USD": 0.5,
    "USD": 1.0,
    "kg oil eq": 5.0,
}


def _neo4j_label(spacy_label: str) -> str:
    return _LABEL_MAP.get(spacy_label, "EconomicEntity")


class GraphBuilder:
    """High-level API for writing extracted knowledge to Neo4j."""

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    # ── Node operations ─────────────────────────────────────────────────────

    def create_entity_node(self, entity: Entity) -> None:
        """MERGE a node for the given entity. Adds both a domain label
        and the generic ``EconomicEntity`` label."""
        label = _neo4j_label(entity.label)
        cypher = (
            f"MERGE (n:{label} {{name: $name}}) "
            f"ON CREATE SET n.label = $label, n.created_at = datetime() "
            f"ON MATCH SET n.updated_at = datetime()"
        )
        self._client.run_query(
            cypher,
            {"name": entity.text, "label": entity.label},
            write=True,
        )

    def create_entity_by_name(self, name: str, label: str = "EconomicEntity") -> None:
        """MERGE a node by name (used during relationship creation to
        ensure both endpoints exist)."""
        cypher = (
            f"MERGE (n:{label} {{name: $name}}) "
            f"ON CREATE SET n.created_at = datetime() "
            f"ON MATCH SET n.updated_at = datetime()"
        )
        self._client.run_query(cypher, {"name": name}, write=True)

    # ── Relationship operations ─────────────────────────────────────────────

    def create_relationship(self, rel: Relationship) -> None:
        """MERGE a directed relationship between two entity nodes.

        Both source and target nodes are MERGE'd first to guarantee they
        exist.  The relationship carries the supporting sentence as a
        property.
        """
        # Ensure endpoints exist
        self.create_entity_by_name(rel.source)
        self.create_entity_by_name(rel.target)

        # MERGE the relationship
        cypher = (
            "MATCH (a {name: $source}), (b {name: $target}) "
            f"MERGE (a)-[r:{rel.relation}]->(b) "
            "ON CREATE SET r.sentence = $sentence, r.created_at = datetime() "
            "ON MATCH SET r.updated_at = datetime()"
        )
        self._client.run_query(
            cypher,
            {
                "source": rel.source,
                "target": rel.target,
                "sentence": rel.sentence,
            },
            write=True,
        )

    def _build_worldbank_trend_relationships(self, docs: list[Document]) -> int:
        """Create trend relationships from recent World Bank indicator changes."""
        grouped: dict[str, list[tuple[int, float, str]]] = {}
        for doc in docs:
            if doc.metadata.get("source") != "worldbank":
                continue
            indicator_id = doc.metadata.get("indicator_id")
            year = int(doc.metadata.get("year", 0))
            value = float(doc.metadata.get("value", 0.0))
            unit = str(doc.metadata.get("unit", "%"))
            grouped.setdefault(indicator_id, []).append((year, value, unit))

        created = 0
        for indicator_id, series in grouped.items():
            if len(series) < 2:
                continue
            series_sorted = sorted(series, key=lambda x: x[0])
            prev_year, prev_value, unit = series_sorted[-2]
            last_year, last_value, _ = series_sorted[-1]
            delta = last_value - prev_value
            threshold = _TREND_THRESHOLD_BY_UNIT.get(unit, 0.2)
            if abs(delta) < threshold:
                continue

            entity = _WORLDBANK_ENTITY_MAP.get(indicator_id)
            if not entity:
                continue

            direction = "Up" if delta > 0 else "Down"
            direction_word = "rose" if delta > 0 else "fell"
            trend_node = f"{entity} Trend {direction}"
            sentence = (
                f"World Bank data shows {entity} {direction_word} "
                f"from {prev_value} {unit} in {prev_year} to {last_value} {unit} in {last_year}."
            )
            rel = Relationship(
                source=entity,
                relation="LEADS_TO",
                target=trend_node,
                sentence=sentence,
            )
            self.create_relationship(rel)
            created += 1

        return created

    # ── Pipeline ────────────────────────────────────────────────────────────

    def build_from_document(self, doc: Document) -> dict[str, int]:
        """
        Full extraction-to-graph pipeline for a single document.

        1. Extract entities via SpaCy
        2. Extract relationships via LLM (or rules)
        3. Write everything to Neo4j

        Returns a summary dict with counts.
        """
        text = doc.page_content

        # Step 1: entities
        entities = extract_entities(text)
        for entity in entities:
            self.create_entity_node(entity)

        # Step 2: relationships
        relationships = extract_relationships(text)
        if doc.metadata.get("source") == "worldbank":
            indicator_id = doc.metadata.get("indicator_id")
            for src, rel, tgt, sentence in _WORLDBANK_RELATION_HINTS.get(indicator_id, []):
                relationships.append(Relationship(source=src, relation=rel, target=tgt, sentence=sentence))
        for rel in relationships:
            self.create_relationship(rel)

        logger.info(
            "Processed document (%d chars): %d entities, %d relationships.",
            len(text),
            len(entities),
            len(relationships),
        )

        return {
            "entities_found": len(entities),
            "relationships_created": len(relationships),
        }

    def build_from_documents(self, docs: list[Document]) -> dict[str, int]:
        """Process multiple documents and return aggregate counts."""
        totals = {"documents_processed": 0, "entities_found": 0, "relationships_created": 0}

        for doc in docs:
            result = self.build_from_document(doc)
            totals["documents_processed"] += 1
            totals["entities_found"] += result["entities_found"]
            totals["relationships_created"] += result["relationships_created"]

        trend_created = self._build_worldbank_trend_relationships(docs)
        totals["relationships_created"] += trend_created

        return totals
