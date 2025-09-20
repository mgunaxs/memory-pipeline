"""
Configuration management using Pydantic Settings.

Handles environment variables, validation, and application settings
with proper type checking and default values.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Production application settings with comprehensive environment variable support.

    All settings can be overridden via environment variables.
    See .env.example for all available configuration options.
    """

    # ================================
    # LLM API Configuration
    # ================================
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-1.5-flash", description="Gemini model name")
    gemini_max_retries: int = Field(default=3, description="Gemini API max retries")
    gemini_timeout: int = Field(default=30, description="Gemini API timeout in seconds")

    # ================================
    # PostgreSQL Database Configuration
    # ================================
    database_url: str = Field(..., description="PostgreSQL database connection URL")

    # Connection pool settings
    db_pool_size: int = Field(default=20, description="Database connection pool size")
    db_max_overflow: int = Field(default=40, description="Database max overflow connections")
    db_pool_timeout: int = Field(default=30, description="Database pool timeout")
    db_pool_recycle: int = Field(default=3600, description="Database pool recycle time")

    # Query settings
    db_statement_timeout: int = Field(default=30000, description="Database statement timeout (ms)")
    db_connect_timeout: int = Field(default=10, description="Database connect timeout")

    # ================================
    # ChromaDB Cloud Configuration
    # ================================
    chroma_api_key: str = Field(..., description="ChromaDB API key")
    chroma_api_url: str = Field(default="https://api.trychroma.com", description="ChromaDB API URL")
    chroma_tenant: str = Field(..., description="ChromaDB tenant ID")
    chroma_database: str = Field(..., description="ChromaDB database name")
    chroma_collection_name: str = Field(default="memories", description="ChromaDB collection name")
    chroma_batch_size: int = Field(default=100, description="ChromaDB batch size")
    chroma_max_retries: int = Field(default=3, description="ChromaDB max retries")
    chroma_timeout: int = Field(default=30, description="ChromaDB timeout")

    # Embedding configuration
    chroma_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    chroma_embedding_dimension: int = Field(default=384, description="Embedding dimension")

    # ================================
    # RAG Configuration
    # ================================
    default_relevance_threshold: float = Field(
        default=0.75,
        description="Default relevance threshold for memory retrieval"
    )
    min_confidence_score: float = Field(
        default=0.6,
        description="Minimum confidence score for memories"
    )
    max_memories_per_query: int = Field(
        default=10,
        description="Maximum memories per query"
    )

    # Feature toggles
    enable_context_filtering: bool = Field(
        default=True,
        description="Enable context-aware memory filtering"
    )
    enable_deduplication: bool = Field(
        default=True,
        description="Enable memory deduplication"
    )
    enable_relevance_validation: bool = Field(
        default=True,
        description="Enable LLM-based relevance validation"
    )
    enable_smart_summarization: bool = Field(
        default=True,
        description="Enable smart summarization"
    )

    # Performance settings
    token_budget_per_retrieval: int = Field(
        default=500,
        description="Token budget per retrieval"
    )
    max_content_length: int = Field(
        default=2000,
        description="Maximum content length"
    )
    max_text_length: int = Field(
        default=5000,
        description="Maximum text length for memory extraction"
    )
    retrieval_timeout_ms: int = Field(
        default=5000,
        description="Retrieval timeout in milliseconds"
    )

    # ================================
    # Application Configuration
    # ================================
    log_level: str = Field(default="INFO", description="Logging level")
    api_version: str = Field(default="1.0.0", description="API version")
    debug: bool = Field(default=False, description="Debug mode")

    # Rate limiting
    rate_limit_per_minute: int = Field(
        default=100,
        description="Rate limit per minute"
    )
    rate_limit_burst: int = Field(
        default=10,
        description="Rate limit burst"
    )

    # CORS settings
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="Comma-separated list of allowed CORS origins"
    )

    # ================================
    # Monitoring and Performance
    # ================================
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    slow_query_threshold_ms: int = Field(
        default=1000,
        description="Slow query threshold in milliseconds"
    )
    enable_performance_metrics: bool = Field(
        default=True,
        description="Enable performance metrics"
    )

    # ================================
    # Legacy/Optional Settings
    # ================================
    # Keep some old settings for backward compatibility during migration
    api_title: str = Field(
        default="Memory Pipeline API",
        description="FastAPI application title"
    )
    api_description: str = Field(
        default="Proactive AI Companion Memory Pipeline",
        description="API description"
    )

    # Optional Redis (for future caching)
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL for caching"
    )

    # Legacy embedding cache
    embedding_cache_size: int = Field(
        default=1000,
        description="Maximum number of embeddings to cache"
    )

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """
    Get application settings.

    Returns:
        Settings: Validated application settings

    Example:
        >>> settings = get_settings()
        >>> print(settings.gemini_api_key)
        AIzaSy...
    """
    return Settings()


# Global settings instance
settings = get_settings()


def create_directories() -> None:
    """
    Create necessary directories for production data storage.

    Creates logs and temporary directories as PostgreSQL and ChromaDB
    are now handled externally in the cloud.
    """
    # Create logs directory for application logging
    os.makedirs("logs", exist_ok=True)

    # Create data directory for any local temporary files
    os.makedirs("data", exist_ok=True)

    # Create backup directory for migration scripts
    os.makedirs("scripts", exist_ok=True)