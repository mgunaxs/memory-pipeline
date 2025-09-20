"""
Pytest configuration and fixtures for production PostgreSQL testing.

Provides reusable test fixtures using production PostgreSQL database
with isolated test schemas.
"""

import os
import pytest
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database_prod import Base, engine, SessionLocal
from app.core.config import settings
from app.models.memory import User, Memory
from app.memory.extractor import MemoryExtractor
from app.memory.embedder import EmbeddingService
from app.services.memory_service import MemoryService


@pytest.fixture(scope="session")
def test_db():
    """Create test database using production PostgreSQL with test schema."""
    # Create test schema
    test_schema = "test_memory_pipeline"

    with engine.connect() as conn:
        # Drop and recreate test schema
        conn.execute(text(f"DROP SCHEMA IF EXISTS {test_schema} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {test_schema}"))
        conn.execute(text(f"SET search_path TO {test_schema}, public"))
        conn.commit()

    # Create test engine with test schema
    test_database_url = settings.database_url + f"?options=-csearch_path%3D{test_schema}%2Cpublic"
    test_engine = create_engine(test_database_url)

    # Create tables in test schema
    Base.metadata.create_all(test_engine)

    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    yield TestSessionLocal

    # Cleanup: drop test schema
    with engine.connect() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {test_schema} CASCADE"))
        conn.commit()

    test_engine.dispose()


@pytest.fixture
def db_session(test_db):
    """Create test database session."""
    session = test_db()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(db_session):
    """Create test user."""
    user = User(
        user_id="test_user_123",
        settings={"test": True},
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def memory_service():
    """Create memory service instance."""
    return MemoryService()


@pytest.fixture
def memory_extractor():
    """Create memory extractor instance."""
    return MemoryExtractor()


@pytest.fixture
def embedding_service():
    """Create embedding service instance."""
    return EmbeddingService()


@pytest.fixture
def sample_memories():
    """Sample memory data for testing."""
    return [
        {
            "content": "I love pizza with mushrooms and pepperoni",
            "memory_type": "preference",
            "category": "food",
            "importance_score": 0.8
        },
        {
            "content": "Meeting scheduled for Monday at 3 PM with the marketing team",
            "memory_type": "event",
            "category": "work",
            "importance_score": 0.9
        },
        {
            "content": "John works in the engineering department and likes coffee",
            "memory_type": "fact",
            "category": "people",
            "importance_score": 0.7
        }
    ]


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()