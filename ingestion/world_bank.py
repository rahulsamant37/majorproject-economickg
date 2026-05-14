"""
Fetch India macroeconomic indicators from the World Bank Open Data API.

Indicators:
    NY.GDP.MKTP.KD.ZG  — GDP growth (annual %)
    FP.CPI.TOTL.ZG     — Consumer price inflation (annual %)
    FR.INR.RINR        — Real interest rate (%)
    SL.UEM.TOTL.ZS     — Unemployment, total (% of total labor force)
    NE.EXP.GNFS.ZS     — Exports of goods and services (% of GDP)
    NE.IMP.GNFS.ZS     — Imports of goods and services (% of GDP)
    BN.CAB.XOKA.GD.ZS  — Current account balance (% of GDP)
    GC.BAL.CASH.GD.ZS  — Fiscal balance, cash (% of GDP)
    GC.DOD.TOTL.GD.ZS  — Central government debt, total (% of GDP)
    BX.KLT.DINV.WD.GD.ZS — FDI net inflows (% of GDP)
    NY.GNS.ICTR.ZS     — Gross savings (% of GDP)
    FS.AST.PRVT.GD.ZS  — Domestic credit to private sector (% of GDP)
    PA.NUS.FCRF        — Official exchange rate (LCU per US$, period average)
    FI.RES.TOTL.CD     — Total reserves (includes gold, current US$)
    NE.TRD.GNFS.ZS     — Trade (% of GDP)
    EG.USE.PCAP.KG.OE  — Energy use (kg of oil equivalent per capita)
    EG.ELC.ACCS.ZS     — Access to electricity (% of population)

If the API is unreachable or returns errors the module falls back to
hardcoded sample documents so the rest of the pipeline can still run.
"""

from __future__ import annotations

import requests
from dataclasses import dataclass, field
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Indicator catalogue ─────────────────────────────────────────────────────

INDICATORS: dict[str, str] = {
    "NY.GDP.MKTP.KD.ZG": "GDP Growth (annual %)",
    "FP.CPI.TOTL.ZG": "CPI Inflation (annual %)",
    "FR.INR.RINR": "Real Interest Rate (%)",
    "SL.UEM.TOTL.ZS": "Unemployment, total (% of total labor force)",
    "NE.EXP.GNFS.ZS": "Exports of goods and services (% of GDP)",
    "NE.IMP.GNFS.ZS": "Imports of goods and services (% of GDP)",
    "BN.CAB.XOKA.GD.ZS": "Current account balance (% of GDP)",
    "GC.BAL.CASH.GD.ZS": "Fiscal balance, cash (% of GDP)",
    "GC.DOD.TOTL.GD.ZS": "Central government debt, total (% of GDP)",
    "BX.KLT.DINV.WD.GD.ZS": "FDI net inflows (% of GDP)",
    "NY.GNS.ICTR.ZS": "Gross savings (% of GDP)",
    "FS.AST.PRVT.GD.ZS": "Domestic credit to private sector (% of GDP)",
    "PA.NUS.FCRF": "Official exchange rate (LCU per US$, period average)",
    "FI.RES.TOTL.CD": "Total reserves (includes gold, current US$)",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP)",
    "EG.USE.PCAP.KG.OE": "Energy use (kg of oil equivalent per capita)",
    "EG.ELC.ACCS.ZS": "Access to electricity (% of population)",
}

INDICATOR_UNITS: dict[str, str] = {
    "NY.GDP.MKTP.KD.ZG": "%",
    "FP.CPI.TOTL.ZG": "%",
    "FR.INR.RINR": "%",
    "SL.UEM.TOTL.ZS": "%",
    "NE.EXP.GNFS.ZS": "%",
    "NE.IMP.GNFS.ZS": "%",
    "BN.CAB.XOKA.GD.ZS": "%",
    "GC.BAL.CASH.GD.ZS": "%",
    "GC.DOD.TOTL.GD.ZS": "%",
    "BX.KLT.DINV.WD.GD.ZS": "%",
    "NY.GNS.ICTR.ZS": "%",
    "FS.AST.PRVT.GD.ZS": "%",
    "PA.NUS.FCRF": "INR per USD",
    "FI.RES.TOTL.CD": "USD",
    "NE.TRD.GNFS.ZS": "%",
    "EG.USE.PCAP.KG.OE": "kg oil eq",
    "EG.ELC.ACCS.ZS": "%",
}

COUNTRY_CODE = "IND"
BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"


# ── Document model ──────────────────────────────────────────────────────────

@dataclass
class Document:
    """Lightweight document container used across the ingestion pipeline."""
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Hardcoded fallback data ─────────────────────────────────────────────────

_FALLBACK_DATA: list[dict[str, Any]] = [
    {
        "indicator": "GDP Growth (annual %)",
        "indicator_id": "NY.GDP.MKTP.KD.ZG",
        "year": 2023,
        "value": 7.2,
    },
    {
        "indicator": "GDP Growth (annual %)",
        "indicator_id": "NY.GDP.MKTP.KD.ZG",
        "year": 2022,
        "value": 7.0,
    },
    {
        "indicator": "CPI Inflation (annual %)",
        "indicator_id": "FP.CPI.TOTL.ZG",
        "year": 2023,
        "value": 5.7,
    },
    {
        "indicator": "CPI Inflation (annual %)",
        "indicator_id": "FP.CPI.TOTL.ZG",
        "year": 2022,
        "value": 6.7,
    },
    {
        "indicator": "Real Interest Rate (%)",
        "indicator_id": "FR.INR.RINR",
        "year": 2023,
        "value": 1.3,
    },
    {
        "indicator": "Real Interest Rate (%)",
        "indicator_id": "FR.INR.RINR",
        "year": 2022,
        "value": -0.7,
    },
    {
        "indicator": "Unemployment, total (% of total labor force)",
        "indicator_id": "SL.UEM.TOTL.ZS",
        "year": 2023,
        "value": 7.1,
    },
    {
        "indicator": "Unemployment, total (% of total labor force)",
        "indicator_id": "SL.UEM.TOTL.ZS",
        "year": 2022,
        "value": 7.4,
    },
    {
        "indicator": "Exports of goods and services (% of GDP)",
        "indicator_id": "NE.EXP.GNFS.ZS",
        "year": 2023,
        "value": 22.3,
    },
    {
        "indicator": "Exports of goods and services (% of GDP)",
        "indicator_id": "NE.EXP.GNFS.ZS",
        "year": 2022,
        "value": 23.1,
    },
    {
        "indicator": "Imports of goods and services (% of GDP)",
        "indicator_id": "NE.IMP.GNFS.ZS",
        "year": 2023,
        "value": 25.8,
    },
    {
        "indicator": "Imports of goods and services (% of GDP)",
        "indicator_id": "NE.IMP.GNFS.ZS",
        "year": 2022,
        "value": 26.5,
    },
    {
        "indicator": "Current account balance (% of GDP)",
        "indicator_id": "BN.CAB.XOKA.GD.ZS",
        "year": 2023,
        "value": -1.6,
    },
    {
        "indicator": "Current account balance (% of GDP)",
        "indicator_id": "BN.CAB.XOKA.GD.ZS",
        "year": 2022,
        "value": -2.1,
    },
    {
        "indicator": "Fiscal balance, cash (% of GDP)",
        "indicator_id": "GC.BAL.CASH.GD.ZS",
        "year": 2023,
        "value": -6.2,
    },
    {
        "indicator": "Fiscal balance, cash (% of GDP)",
        "indicator_id": "GC.BAL.CASH.GD.ZS",
        "year": 2022,
        "value": -6.8,
    },
    {
        "indicator": "Central government debt, total (% of GDP)",
        "indicator_id": "GC.DOD.TOTL.GD.ZS",
        "year": 2023,
        "value": 58.4,
    },
    {
        "indicator": "Central government debt, total (% of GDP)",
        "indicator_id": "GC.DOD.TOTL.GD.ZS",
        "year": 2022,
        "value": 59.7,
    },
    {
        "indicator": "FDI net inflows (% of GDP)",
        "indicator_id": "BX.KLT.DINV.WD.GD.ZS",
        "year": 2023,
        "value": 2.1,
    },
    {
        "indicator": "FDI net inflows (% of GDP)",
        "indicator_id": "BX.KLT.DINV.WD.GD.ZS",
        "year": 2022,
        "value": 2.3,
    },
    {
        "indicator": "Gross savings (% of GDP)",
        "indicator_id": "NY.GNS.ICTR.ZS",
        "year": 2023,
        "value": 30.1,
    },
    {
        "indicator": "Gross savings (% of GDP)",
        "indicator_id": "NY.GNS.ICTR.ZS",
        "year": 2022,
        "value": 29.6,
    },
    {
        "indicator": "Domestic credit to private sector (% of GDP)",
        "indicator_id": "FS.AST.PRVT.GD.ZS",
        "year": 2023,
        "value": 55.2,
    },
    {
        "indicator": "Domestic credit to private sector (% of GDP)",
        "indicator_id": "FS.AST.PRVT.GD.ZS",
        "year": 2022,
        "value": 54.1,
    },
    {
        "indicator": "Official exchange rate (LCU per US$, period average)",
        "indicator_id": "PA.NUS.FCRF",
        "year": 2023,
        "value": 82.5,
    },
    {
        "indicator": "Official exchange rate (LCU per US$, period average)",
        "indicator_id": "PA.NUS.FCRF",
        "year": 2022,
        "value": 78.6,
    },
    {
        "indicator": "Total reserves (includes gold, current US$)",
        "indicator_id": "FI.RES.TOTL.CD",
        "year": 2023,
        "value": 590.2,
    },
    {
        "indicator": "Total reserves (includes gold, current US$)",
        "indicator_id": "FI.RES.TOTL.CD",
        "year": 2022,
        "value": 562.3,
    },
    {
        "indicator": "Trade (% of GDP)",
        "indicator_id": "NE.TRD.GNFS.ZS",
        "year": 2023,
        "value": 48.1,
    },
    {
        "indicator": "Trade (% of GDP)",
        "indicator_id": "NE.TRD.GNFS.ZS",
        "year": 2022,
        "value": 49.6,
    },
    {
        "indicator": "Energy use (kg of oil equivalent per capita)",
        "indicator_id": "EG.USE.PCAP.KG.OE",
        "year": 2023,
        "value": 662.0,
    },
    {
        "indicator": "Energy use (kg of oil equivalent per capita)",
        "indicator_id": "EG.USE.PCAP.KG.OE",
        "year": 2022,
        "value": 651.0,
    },
    {
        "indicator": "Access to electricity (% of population)",
        "indicator_id": "EG.ELC.ACCS.ZS",
        "year": 2023,
        "value": 99.6,
    },
    {
        "indicator": "Access to electricity (% of population)",
        "indicator_id": "EG.ELC.ACCS.ZS",
        "year": 2022,
        "value": 99.2,
    },
]


def _build_document(
    indicator_name: str,
    indicator_id: str,
    year: int,
    value: float,
    unit: str,
) -> Document:
    """Convert a single data-point into a Document."""
    if unit == "%":
        value_text = f"{value}%"
    else:
        value_text = f"{value} {unit}"
    content = (
        f"India's {indicator_name} was {value_text} in {year}. "
        f"(Source: World Bank, indicator {indicator_id})"
    )
    return Document(
        page_content=content,
        metadata={
            "source": "worldbank",
            "indicator": indicator_name,
            "indicator_id": indicator_id,
            "year": year,
            "value": value,
            "unit": unit,
        },
    )


def _fallback_documents() -> list[Document]:
    """Return hardcoded sample documents when the API is unavailable."""
    logger.warning("World Bank API unavailable — returning hardcoded sample data.")
    return [
        _build_document(
            d["indicator"],
            d["indicator_id"],
            d["year"],
            d["value"],
            INDICATOR_UNITS.get(d["indicator_id"], "%"),
        )
        for d in _FALLBACK_DATA
    ]


# ── Public API ──────────────────────────────────────────────────────────────

def fetch_world_bank_indicators(
    country: str = COUNTRY_CODE,
    date_range: str = "2018:2024",
    per_page: int = 50,
) -> list[Document]:
    """
    Fetch macroeconomic indicator data from the World Bank API.

    Returns a list of ``Document`` objects, each describing one
    indicator-year data point, with full metadata.

    Falls back to hardcoded sample data if the API call fails.
    """
    documents: list[Document] = []

    for indicator_id, indicator_name in INDICATORS.items():
        url = BASE_URL.format(country=country, indicator=indicator_id)
        params = {
            "format": "json",
            "date": date_range,
            "per_page": per_page,
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # World Bank JSON response has two elements: [metadata, records]
            if not isinstance(data, list) or len(data) < 2 or data[1] is None:
                logger.warning(
                    "Unexpected response structure for %s — skipping.",
                    indicator_id,
                )
                continue

            records = data[1]
            for record in records:
                value = record.get("value")
                year = record.get("date")
                if value is None or year is None:
                    continue

                doc = _build_document(
                    indicator_name=indicator_name,
                    indicator_id=indicator_id,
                    year=int(year),
                    value=round(float(value), 2),
                    unit=INDICATOR_UNITS.get(indicator_id, "%"),
                )
                documents.append(doc)

            logger.info(
                "Fetched %d records for %s (%s).",
                len(records),
                indicator_name,
                indicator_id,
            )

        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.error("Error fetching %s: %s", indicator_id, exc)

    if not documents:
        return _fallback_documents()

    return documents
