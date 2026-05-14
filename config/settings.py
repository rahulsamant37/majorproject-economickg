"""
Centralized application settings loaded from .env with sensible defaults.
LLM provider is selected automatically: OpenAI if key present, else Groq.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM keys ────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    # ── Data API keys ───────────────────────────────────────────────────────
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")

    # ── Neo4j ───────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="neo4j://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # ── App ─────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    demo_seed: bool = Field(default=True, alias="DEMO_SEED")

    # ── Derived helpers ─────────────────────────────────────────────────────
    @property
    def llm_provider(self) -> Literal["openai", "groq", "none"]:
        """Select LLM provider based on available API keys."""
        if self.openai_api_key:
            return "openai"
        if self.groq_api_key:
            return "groq"
        return "none"

    @property
    def has_llm(self) -> bool:
        return self.llm_provider != "none"

    @property
    def has_news_api(self) -> bool:
        return bool(self.news_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor for application settings."""
    return Settings()
