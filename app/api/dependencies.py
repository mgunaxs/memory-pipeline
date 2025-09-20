"""
FastAPI dependencies for dependency injection.

Provides reusable dependencies for database sessions,
services, and authentication.
"""

import logging
from typing import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database_prod import get_db
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)

# Global service instance (singleton pattern)
_memory_service = None


def get_memory_service() -> MemoryService:
    """
    Get memory service instance.

    Returns:
        MemoryService: Singleton memory service instance

    Example:
        >>> @app.get("/test")
        >>> def test(service: MemoryService = Depends(get_memory_service)):
        ...     return service.get_user_stats("user123", db)
    """
    global _memory_service
    if _memory_service is None:
        logger.info("Initializing memory service")
        _memory_service = MemoryService()
    return _memory_service


# Database dependency is now imported from database_prod
# get_database = get_db  # Use the production version