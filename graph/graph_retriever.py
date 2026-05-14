"""
Graph retriever — read-only queries against the Neo4j knowledge graph.

Provides causal-chain traversal, neighbour lookup, entity search,
and summary statistics.
"""

from __future__ import annotations

from typing import Any

from graph.neo4j_client import Neo4jClient
from utils.logger import get_logger

logger = get_logger(__name__)


class GraphRetriever:
    """Read-only facade over the Neo4j knowledge graph."""

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    # ── Causal chain ────────────────────────────────────────────────────────

    def get_causal_chain(self, entity: str, max_depth: int = 4) -> list[dict[str, Any]]:
        """
        Follow relationships up to *max_depth* hops from *entity* in both
        directions (outgoing and incoming).

        Returns a list of dicts with keys: ``from``, ``relation``, ``to``,
        ``sentence``, ``depth``.
        """
        # Outgoing chain: entity → … → terminal
        outgoing_cypher = """
        MATCH path = (n)-[rels*1..4]->(m)
        WHERE toLower(n.name) = toLower($entity)
        UNWIND range(0, size(rels)-1) AS idx
        WITH nodes(path) AS ns, rels[idx] AS r, idx
        RETURN ns[idx].name   AS `from`,
               type(r)        AS relation,
               ns[idx+1].name AS `to`,
               r.sentence     AS sentence,
               idx + 1        AS depth
        ORDER BY depth
        """

        # Incoming chain: root → … → entity  (reverse traversal)
        incoming_cypher = """
        MATCH path = (m)-[rels*1..4]->(n)
        WHERE toLower(n.name) = toLower($entity)
        UNWIND range(0, size(rels)-1) AS idx
        WITH nodes(path) AS ns, rels[idx] AS r, idx
        RETURN ns[idx].name   AS `from`,
               type(r)        AS relation,
               ns[idx+1].name AS `to`,
               r.sentence     AS sentence,
               idx + 1        AS depth
        ORDER BY depth
        """

        params = {"entity": entity}
        outgoing = self._client.run_query(outgoing_cypher, params)
        incoming = self._client.run_query(incoming_cypher, params)

        # Merge and deduplicate while preserving order
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in incoming + outgoing:
            key = f"{row['from']}|{row['relation']}|{row['to']}"
            if key not in seen:
                seen.add(key)
                unique.append(row)
        logger.debug("Causal chain for '%s': %d edges.", entity, len(unique))
        return unique

    # ── Neighbours ──────────────────────────────────────────────────────────

    def get_neighbors(self, entity: str) -> list[dict[str, Any]]:
        """
        Get all directly connected nodes (both directions).

        Returns a list of dicts: ``neighbor``, ``relation``, ``direction``.
        """
        cypher = """
        MATCH (n)-[r]-(m)
        WHERE toLower(n.name) = toLower($entity)
        RETURN m.name                                 AS neighbor,
               type(r)                                AS relation,
               CASE WHEN startNode(r) = n THEN 'OUT' ELSE 'IN' END AS direction,
               r.sentence                             AS sentence
        """
        results = self._client.run_query(cypher, {"entity": entity})
        logger.debug("Neighbors of '%s': %d.", entity, len(results))
        return results

    # ── Link prediction (heuristic) ───────────────────────────────────────

    def predict_links(self, entity: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Predict likely links using shared-neighbor counts (two-hop paths).

        Returns a list of dicts with keys: ``from``, ``relation``, ``to``,
        ``score``, ``sentence``.
        """
        cypher = """
        MATCH (n {name: $entity})-[]-(m)-[]-(cand)
        WHERE n <> cand AND n.name IS NOT NULL AND cand.name IS NOT NULL
        WITH cand, collect(DISTINCT m.name) AS shared, count(DISTINCT m) AS shared_count
        RETURN cand.name AS candidate,
               shared AS shared_neighbors,
               shared_count AS score
        ORDER BY score DESC, candidate ASC
        LIMIT $limit
        """

        rows = self._client.run_query(cypher, {"entity": entity, "limit": limit})
        predictions: list[dict[str, Any]] = []
        for row in rows:
            shared = row.get("shared_neighbors", [])
            predictions.append(
                {
                    "from": entity,
                    "relation": "PREDICTED_LINK",
                    "to": row.get("candidate"),
                    "score": row.get("score", 0),
                    "sentence": (
                        "Predicted link based on shared neighbors: "
                        + ", ".join(shared[:5])
                    ),
                }
            )

        logger.debug("Predicted %d links for '%s'.", len(predictions), entity)
        return predictions

    # ── Summary statistics ──────────────────────────────────────────────────

    def get_graph_summary(self) -> dict[str, Any]:
        """
        Return high-level graph stats: node count, relationship count,
        and the top-10 most-connected nodes.
        """
        node_count_q = "MATCH (n) RETURN count(n) AS count"
        rel_count_q = "MATCH ()-[r]->() RETURN count(r) AS count"
        top_nodes_q = """
        MATCH (n)-[r]-()
        RETURN n.name AS name, count(r) AS connections
        ORDER BY connections DESC
        LIMIT 10
        """

        node_count = self._client.run_query(node_count_q)
        rel_count = self._client.run_query(rel_count_q)
        top_nodes = self._client.run_query(top_nodes_q)

        return {
            "node_count": node_count[0]["count"] if node_count else 0,
            "relationship_count": rel_count[0]["count"] if rel_count else 0,
            "top_connected_nodes": top_nodes,
        }

    # ── Keyword search ──────────────────────────────────────────────────────

    def search_entities(self, keyword: str) -> list[dict[str, Any]]:
        """
        Case-insensitive keyword search across all node names.

        Returns list of dicts with ``name`` and ``labels``.
        """
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($keyword)
        RETURN n.name AS name, labels(n) AS labels
        LIMIT 25
        """
        results = self._client.run_query(cypher, {"keyword": keyword})
        logger.debug("Entity search '%s': %d matches.", keyword, len(results))
        return results

    # ── Visualization data ──────────────────────────────────────────────────

    def get_visualization_data(self, limit: int = 100) -> dict[str, Any]:
        """
        Return nodes and edges for D3.js force-directed graph visualization.

        Only includes our economic entity nodes (EconomicEntity, Policy,
        Market, MacroIndicator) to keep the graph readable.
        """
        nodes_cypher = """
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN
            ['EconomicEntity', 'Policy', 'Market', 'MacroIndicator'])
        CALL (n) {
            MATCH (n)-[]-()
            RETURN count(*) AS connections
        }
        RETURN n.name AS name,
               labels(n) AS labels,
               connections
        ORDER BY connections DESC
        LIMIT $limit
        """

        edges_cypher = """
        MATCH (a)-[r]->(b)
        WHERE any(label IN labels(a) WHERE label IN
                ['EconomicEntity', 'Policy', 'Market', 'MacroIndicator'])
          AND any(label IN labels(b) WHERE label IN
                ['EconomicEntity', 'Policy', 'Market', 'MacroIndicator'])
        RETURN a.name AS source,
               type(r) AS relation,
               b.name AS target,
               r.sentence AS sentence
        LIMIT $limit
        """

        nodes = self._client.run_query(nodes_cypher, {"limit": limit})
        edges = self._client.run_query(edges_cypher, {"limit": limit})

        # Build unique node list
        node_set: set[str] = set()
        node_list: list[dict] = []
        for n in nodes:
            if n["name"] and n["name"] not in node_set:
                node_set.add(n["name"])
                node_list.append({
                    "id": n["name"],
                    "labels": n["labels"],
                    "connections": n["connections"],
                })

        # Ensure edge endpoints are in node_set
        for e in edges:
            for endpoint in [e["source"], e["target"]]:
                if endpoint and endpoint not in node_set:
                    node_set.add(endpoint)
                    node_list.append({
                        "id": endpoint,
                        "labels": ["EconomicEntity"],
                        "connections": 1,
                    })

        edge_list = [
            {
                "source": e["source"],
                "target": e["target"],
                "relation": e["relation"],
                "sentence": e.get("sentence", ""),
            }
            for e in edges
            if e["source"] and e["target"]
        ]

        return {"nodes": node_list, "edges": edge_list}

