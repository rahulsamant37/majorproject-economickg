"""
Neo4j driver singleton.

Provides a thin wrapper around the official ``neo4j`` Python driver
with connect / close / run_query / ping semantics.
"""

from __future__ import annotations

from typing import Any, Optional

import neo4j
from neo4j import GraphDatabase

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """Thread-safe singleton wrapper around the Neo4j Python driver."""

    _instance: Optional["Neo4jClient"] = None
    _driver: Optional[neo4j.Driver] = None

    def __new__(cls) -> "Neo4jClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Establish the driver connection using settings from .env."""
        if self._driver is not None:
            return

        settings = get_settings()
        logger.info("Connecting to Neo4j at %s …", settings.neo4j_uri)

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

        # Fail fast if credentials or network are wrong
        if not self.ping():
            self.close()
            raise ConnectionError(
                f"Cannot reach Neo4j at {settings.neo4j_uri}. "
                "Check NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in .env"
            )

        logger.info("Neo4j connection verified ✓")

    def close(self) -> None:
        """Gracefully close the driver."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")

    # ── Health check ────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if the database is reachable."""
        if self._driver is None:
            logger.error("ping() called but driver is not initialised.")
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.error("Neo4j ping failed: %s", exc)
            return False

    # ── Query execution ─────────────────────────────────────────────────────

    def run_query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        write: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return the results as a list of dicts.

        Parameters
        ----------
        cypher : str
            The Cypher statement.
        params : dict, optional
            Parameters to bind into the query.
        write : bool
            If True the query is sent in a *write* transaction,
            otherwise a *read* transaction is used.
        """
        if self._driver is None:
            raise RuntimeError("Neo4j driver is not connected. Call connect() first.")

        settings = get_settings()
        database = settings.neo4j_database

        try:
            with self._driver.session(database=database) as session:
                if write:
                    result = session.run(cypher, parameters=params or {})
                else:
                    result = session.run(cypher, parameters=params or {})
                records = [record.data() for record in result]
                return records
        except Exception as exc:
            logger.error("Cypher query failed: %s\nQuery: %s", exc, cypher)
            raise
