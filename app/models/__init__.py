"""Database models for the memory pipeline."""

from .memory import User, Memory, MemoryConnection, Conversation
from .schemas import (
    MemoryCreate,
    MemoryResponse,
    MemoryUpdate,
    MemorySearchRequest,
    MemorySearchResponse,
    APIResponse,
    HealthResponse,
    MemoryType,
    ConnectionType
)

__all__ = [
    # SQLAlchemy models
    "User",
    "Memory",
    "MemoryConnection",
    "Conversation",
    # Pydantic schemas
    "MemoryCreate",
    "MemoryResponse",
    "MemoryUpdate",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "APIResponse",
    "HealthResponse",
    "MemoryType",
    "ConnectionType"
]