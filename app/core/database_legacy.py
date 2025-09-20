"""
Database connection and session management.

Provides SQLAlchemy engine, session factory, and database initialization
with proper connection pooling and error handling.
"""

import logging
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)

# SQLAlchemy base class for models
Base = declarative_base()

# Database engine with connection pooling
engine = create_engine(
    settings.database_url,
    # SQLite specific configurations
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    # Connection pool settings
    pool_pre_ping=True,
    echo=settings.log_level == "DEBUG"
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Set SQLite pragma settings for better performance and data integrity.

    Args:
        dbapi_connection: Database connection
        connection_record: Connection record
    """
    if "sqlite" in settings.database_url:
        cursor = dbapi_connection.cursor()
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys=ON")
        # Set journal mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        # Set synchronous mode for better performance
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.

    Yields:
        Session: SQLAlchemy database session

    Example:
        >>> from fastapi import Depends
        >>> def my_endpoint(db: Session = Depends(get_db)):
        ...     # Use db session here
        ...     pass
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_database() -> bool:
    """
    Initialize database by creating all tables.

    Returns:
        bool: True if successful, False otherwise

    Example:
        >>> success = init_database()
        >>> if success:
        ...     print("Database initialized successfully")
    """
    try:
        # Import models to ensure they're registered
        from app.models.memory import User, Memory, MemoryConnection, Conversation

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def check_database_connection() -> bool:
    """
    Check if database connection is working.

    Returns:
        bool: True if connection is healthy, False otherwise

    Example:
        >>> if check_database_connection():
        ...     print("Database is healthy")
    """
    try:
        from sqlalchemy import text
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()  # Actually fetch the result
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False