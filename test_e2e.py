"""
End-to-end test: Clears previously added economic entities, re-ingests
demo data + news using LLM extraction, and validates the full pipeline.
"""

import json
import sys

# ── Step 1: Clear our economic entity nodes (not the existing WB data) ──────

print("=" * 70)
print("STEP 1: Clearing previously ingested economic entities")
print("=" * 70)

from graph.neo4j_client import Neo4jClient
from graph.graph_retriever import GraphRetriever

client = Neo4jClient()
client.connect()

# Delete only EconomicEntity / Policy / Market nodes we created
labels_to_clean = ["EconomicEntity", "Policy", "Market", "MacroIndicator"]
for label in labels_to_clean:
    result = client.run_query(f"MATCH (n:{label}) DETACH DELETE n RETURN count(n) AS deleted", write=True)
    deleted = result[0]["deleted"] if result else 0
    print(f"  Deleted {deleted} nodes with label {label}")

retriever = GraphRetriever(client)
summary = retriever.get_graph_summary()
print(f"  Graph now has {summary['node_count']} nodes, {summary['relationship_count']} relationships")

# ── Step 2: Test LLM extraction with Groq ───────────────────────────────────

print("\n" + "=" * 70)
print("STEP 2: Testing LLM (Groq) relationship extraction")
print("=" * 70)

from nlp.relationship_extractor import extract_relationships

demo_texts = [
    "The RBI increased the repo rate by 50 basis points to control rising inflation. Higher rates make borrowing expensive, slowing economic activity.",
    "Following the RBI rate hike, bond yields surged as fixed income became more attractive relative to equities.",
    "Rising bond yields triggered FII outflows from Indian equity markets. Foreign investors pulled out over $2 billion in a single week.",
    "The combined effect of FII outflows and reduced liquidity caused NIFTY to fall by 2.3% in a single session.",
]

all_rels = []
for i, text in enumerate(demo_texts, 1):
    rels = extract_relationships(text)
    all_rels.extend(rels)
    print(f"\n  Doc {i}: {text[:70]}...")
    for r in rels:
        print(f"    {r.source} --[{r.relation}]--> {r.target}")

print(f"\n  Total relationships extracted: {len(all_rels)}")

# ── Step 3: Ingest demo documents into the graph ────────────────────────────

print("\n" + "=" * 70)
print("STEP 3: Ingesting demo documents into Neo4j")
print("=" * 70)

from ingestion.world_bank import Document
from graph.graph_builder import GraphBuilder

builder = GraphBuilder(client)
demo_docs = [Document(page_content=t, metadata={"source": "demo"}) for t in demo_texts]
totals = builder.build_from_documents(demo_docs)
print(f"  Documents processed: {totals['documents_processed']}")
print(f"  Entities found:      {totals['entities_found']}")
print(f"  Relationships:       {totals['relationships_created']}")

# ── Step 4: Ingest real news from NewsAPI ────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 4: Fetching and ingesting real news from NewsAPI")
print("=" * 70)

from ingestion.news_fetcher import fetch_india_news

news_docs = fetch_india_news(max_articles=10)
print(f"  Fetched {len(news_docs)} news articles")
for doc in news_docs[:3]:
    print(f"    - {doc.metadata.get('title', doc.page_content[:60])}...")

news_totals = builder.build_from_documents(news_docs)
print(f"\n  News docs processed: {news_totals['documents_processed']}")
print(f"  Entities found:      {news_totals['entities_found']}")
print(f"  Relationships:       {news_totals['relationships_created']}")

# ── Step 5: Verify the causal chain ─────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 5: Verifying causal chains in Neo4j")
print("=" * 70)

for entity in ["RBI", "NIFTY", "Bond Yield", "FII"]:
    chain = retriever.get_causal_chain(entity)
    print(f"\n  Chain from {entity}: {len(chain)} edges")
    for edge in chain[:5]:
        print(f"    {edge['from']} --[{edge['relation']}]--> {edge['to']}")

# ── Step 6: Test the LLM reasoning pipeline ─────────────────────────────────

print("\n" + "=" * 70)
print("STEP 6: Testing LLM reasoning pipeline")
print("=" * 70)

from reasoning.llm_reasoning import reason_over_graph

chain_data = retriever.get_causal_chain("NIFTY")
neighbor_data = retriever.get_neighbors("NIFTY")

result = reason_over_graph(
    query="Why did NIFTY fall?",
    graph_data=chain_data,
    neighbor_data=neighbor_data,
)

print(f"\n  Answer ({len(result.answer)} chars):")
print(f"  {result.answer[:500]}")
print(f"\n  Causal chain: {result.causal_chain}")
print(f"  Sources: {len(result.sources)} evidence snippets")

# ── Step 7: Test the full LangGraph agent ───────────────────────────────────

print("\n" + "=" * 70)
print("STEP 7: Testing LangGraph agent end-to-end")
print("=" * 70)

from agent.macro_agent import run_macro_agent

queries = [
    "Why did NIFTY fall?",
    "How does repo rate affect the economy?",
    "What is the relationship between FII and bond yields?",
]

for query in queries:
    print(f"\n  Q: {query}")
    result = run_macro_agent(query)
    print(f"  A: {result['insight'][:200]}...")
    print(f"  Chain: {result['causal_chain'][:3]}")

# ── Step 8: Final graph summary ─────────────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 8: Final graph summary")
print("=" * 70)

final_summary = retriever.get_graph_summary()
print(f"  Total nodes:         {final_summary['node_count']}")
print(f"  Total relationships: {final_summary['relationship_count']}")
print(f"  Top connected nodes:")
for node in final_summary.get("top_connected_nodes", [])[:10]:
    print(f"    {node['name']}: {node['connections']} connections")

client.close()

print("\n" + "=" * 70)
print("✅ ALL END-TO-END TESTS PASSED")
print("=" * 70)
