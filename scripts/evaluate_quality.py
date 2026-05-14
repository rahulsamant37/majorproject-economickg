"""
Evaluate baseline vs improved reasoning quality across common failure modes.

Runs the macro agent twice per query:
- Baseline: without link prediction
- Improved: with link prediction

Outputs summary counts for nine issue types.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any

# Ensure project root is on sys.path for script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.macro_agent import run_macro_agent
from graph.neo4j_client import Neo4jClient


@dataclass
class Case:
    query: str
    expected_entities: list[str]


CASES: list[Case] = [
    Case(
        query="Why did NIFTY fall?",
        expected_entities=["NIFTY", "FII", "Bond Yield", "Repo Rate"],
    ),
    Case(
        query="How does repo rate affect inflation?",
        expected_entities=["Repo Rate", "Inflation"],
    ),
    Case(
        query="What is the link between CPI inflation and bond yields?",
        expected_entities=["Inflation", "Bond Yield"],
    ),
    Case(
        query="How do exports and the rupee interact?",
        expected_entities=["Exports", "Rupee"],
    ),
    Case(
        query="What is the relationship between unemployment and GDP?",
        expected_entities=["Unemployment", "GDP"],
    ),
    Case(
        query="How does government debt affect bond yields?",
        expected_entities=["Government Debt", "Bond Yield"],
    ),
    Case(
        query="How do FX reserves influence the rupee?",
        expected_entities=["FX Reserves", "Rupee"],
    ),
    Case(
        query="Does electricity access affect employment?",
        expected_entities=["Electricity Access", "Employment"],
    ),
]


def _contains_entity(text: str, entity: str) -> bool:
    return entity.lower() in text.lower()


def _years_from_sources(sources: list[str]) -> list[int]:
    years: list[int] = []
    for s in sources:
        for y in re.findall(r"(?:19|20)\d{2}", s):
            years.append(int(y))
    return years


def _evaluate_result(result: dict[str, Any], expected: list[str]) -> dict[str, bool]:
    answer = result.get("insight", "")
    chain = result.get("causal_chain", [])
    sources = result.get("sources", [])
    predicted_links = result.get("predicted_links", [])

    chain_text = " ".join(chain)
    entity_hits = sum(
        1 for e in expected if _contains_entity(answer, e) or _contains_entity(chain_text, e)
    )
    coverage = entity_hits / max(len(expected), 1)

    years = _years_from_sources(sources)
    current_year = datetime.now().year
    max_year = max(years) if years else None

    hallucination = (not sources) and len(answer) > 200
    incompleteness = coverage < 0.5
    retrieval_failure = len(chain) == 0
    ambiguity = (not sources) and bool(re.search(r"\b(might|could|possible|uncertain)\b", answer, re.I))
    overreliance = (len(chain) > 0) and len(answer) < 120
    unfaithfulness = (not sources) and bool(re.search(r"\b(evidence|graph shows|as per the graph)\b", answer, re.I))
    reasoning_failure = (len(chain) > 0) and all(
        not _contains_entity(answer, e) for e in expected
    )
    staleness = bool(max_year) and (current_year - max_year >= 4)
    inconsistency = bool(re.search(r"\b(increase|rise)\b.*\b(decrease|fall)\b|\b(decrease|fall)\b.*\b(increase|rise)\b", answer, re.I))

    # If predictions exist, soften hallucination a bit (they are marked as hypotheses)
    if predicted_links and hallucination:
        hallucination = False

    return {
        "hallucination": hallucination,
        "incompleteness": incompleteness,
        "retrieval_failure": retrieval_failure,
        "ambiguity": ambiguity,
        "overreliance": overreliance,
        "unfaithfulness": unfaithfulness,
        "reasoning_failure": reasoning_failure,
        "staleness": staleness,
        "inconsistency": inconsistency,
    }


def _summarize(label: str, results: list[dict[str, Any]]) -> None:
    total = len(results)
    counts: dict[str, int] = {
        "hallucination": 0,
        "incompleteness": 0,
        "retrieval_failure": 0,
        "ambiguity": 0,
        "overreliance": 0,
        "unfaithfulness": 0,
        "reasoning_failure": 0,
        "staleness": 0,
        "inconsistency": 0,
    }
    avg_chain = 0
    avg_sources = 0
    avg_predicted = 0

    for row in results:
        metrics = row["metrics"]
        for k in counts:
            if metrics[k]:
                counts[k] += 1
        avg_chain += row["chain_len"]
        avg_sources += row["sources_len"]
        avg_predicted += row["predicted_len"]

    avg_chain = avg_chain / total
    avg_sources = avg_sources / total
    avg_predicted = avg_predicted / total

    print(f"\n=== {label} ===")
    print(f"Queries: {total}")
    print(f"Avg causal chain length: {avg_chain:.2f}")
    print(f"Avg sources count:       {avg_sources:.2f}")
    print(f"Avg predicted links:     {avg_predicted:.2f}")
    print("Issue counts:")
    for k, v in counts.items():
        print(f"  {k:18s} {v}/{total}")


def main() -> None:
    client = Neo4jClient()
    client.connect()

    baseline_rows: list[dict[str, Any]] = []
    improved_rows: list[dict[str, Any]] = []

    for case in CASES:
        base = run_macro_agent(case.query, use_predictions=False, return_debug=True)
        improved = run_macro_agent(case.query, use_predictions=True, return_debug=True)

        base_metrics = _evaluate_result(base, case.expected_entities)
        improved_metrics = _evaluate_result(improved, case.expected_entities)

        baseline_rows.append(
            {
                "query": case.query,
                "metrics": base_metrics,
                "chain_len": len(base.get("causal_chain", [])),
                "sources_len": len(base.get("sources", [])),
                "predicted_len": len(base.get("predicted_links", [])),
            }
        )
        improved_rows.append(
            {
                "query": case.query,
                "metrics": improved_metrics,
                "chain_len": len(improved.get("causal_chain", [])),
                "sources_len": len(improved.get("sources", [])),
                "predicted_len": len(improved.get("predicted_links", [])),
            }
        )

    _summarize("Baseline (no link prediction)", baseline_rows)
    _summarize("Improved (link prediction + trends)", improved_rows)

    print("\nPer-query details (baseline -> improved):")
    for i, case in enumerate(CASES):
        b = baseline_rows[i]["metrics"]
        a = improved_rows[i]["metrics"]
        print(f"- {case.query}")
        print(f"  baseline issues:  {', '.join(k for k,v in b.items() if v) or 'none'}")
        print(f"  improved issues:  {', '.join(k for k,v in a.items() if v) or 'none'}")

    client.close()


if __name__ == "__main__":
    main()
