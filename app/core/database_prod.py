"""
Production PostgreSQL database configuration.
Completely replaces SQLite with PostgreSQL + connection pooling.
"""

import logging
import os
import asyncio
from typing import AsyncGenerator, Generator, Optional, Dict, Any
from contextlib import asynccontextmanager, contextmanager

import asyncpg
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)

# SQLAlchemy base class for models
Base = declarative_base()

# ================================
# PostgreSQL Connection Configuration
# ================================

def get_database_config() -> Dict[str, Any]:
    """
    Get optimized PostgreSQL configuration for production.

    Returns:
        Database configuration dictionary
    """
    return {
        'pool_size': getattr(settings, 'db_pool_size', 20),
        'max_overflow': getattr(settings, 'db_max_overflow', 40),
        'pool_timeout': getattr(settings, 'db_pool_timeout', 30),
        'pool_recycle': getattr(settings, 'db_pool_recycle', 3600),
        'pool_pre_ping': True,
        'echo': settings.log_level == "DEBUG",
        'echo_pool': settings.log_level == "DEBUG",
        'connect_args': {
            'statement_timeout': getattr(settings, 'db_statement_timeout', 30000),
            'connect_timeout': getattr(settings, 'db_connect_timeout', 10),
            'server_settings': {
                'jit': 'off',  # Disable JIT for predictable performance
                'application_name': 'memory_pipeline'
            }
        }
    }

# Create synchronous engine with connection pooling
config = get_database_config()
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    **config
)

# Create async engine for async operations
async_engine = create_async_engine(
    settings.database_url.replace('postgresql://', 'postgresql+asyncpg://'),
    pool_size=config['pool_size'],
    max_overflow=config['max_overflow'],
    pool_timeout=config['pool_timeout'],
    pool_recycle=config['pool_recycle'],
    pool_pre_ping=True,
    echo=config['echo']
)

# Session factories
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Prevent expired object issues
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# ================================
# Connection Event Handlers
# ================================

@event.listens_for(Engine, "connect")
def set_postgresql_pragma(dbapi_connection, connection_record):
    """
    Configure PostgreSQL connection parameters for optimal performance.

    Args:
        dbapi_connection: Database connection
        connection_record: Connection record
    """
    try:
        cursor = dbapi_connection.cursor()

        # Set search path to use memory_pipeline schema
        cursor.execute("SET search_path TO memory_pipeline, public")

        # Set connection-level optimizations
        cursor.execute("SET timezone = 'UTC'")
        cursor.execute("SET statement_timeout = '30s'")
        cursor.execute("SET lock_timeout = '10s'")
        cursor.execute("SET idle_in_transaction_session_timeout = '60s'")

        # Optimize for read-heavy workloads
        cursor.execute("SET default_statistics_target = 100")
        cursor.execute("SET random_page_cost = 1.1")  # SSD optimization

        cursor.close()
        logger.debug("PostgreSQL connection configured")

    except Exception as e:
        logger.warning(f"Failed to configure PostgreSQL connection: {e}")

@event.listens_for(Engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log connection checkout for monitoring."""
    logger.debug("Connection checked out from pool")

@event.listens_for(Engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log connection checkin for monitoring."""
    logger.debug("Connection returned to pool")

# ================================
# Dependency Functions
# ================================

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

@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Yields:
        AsyncSession: Async SQLAlchemy database session

    Example:
        >>> async with get_async_db() as db:
        ...     result = await db.execute(select(User))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Async database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()

# ================================
# Database Initialization
# ================================

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

async def init_async_database() -> bool:
    """
    Initialize database asynchronously.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import models to ensure they're registered
        from app.models.memory import User, Memory, MemoryConnection, Conversation

        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Async database tables created successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize async database: {e}")
        return False

# ================================
# Health Check Functions
# ================================

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
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()
        return True

    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False

async def check_async_database_connection() -> bool:
    """
    Check async database connection.

    Returns:
        bool: True if connection is healthy, False otherwise
    """
    try:
        async with async_engine.connect() as connection:
            result = await connection.execute(text("SELECT 1"))
            await result.fetchone()
        return True

    except Exception as e:
        logger.error(f"Async database connection check failed: {e}")
        return False

def get_database_stats() -> Dict[str, Any]:
    """
    Get database connection pool statistics.

    Returns:
        Dict with database statistics
    """
    try:
        pool = engine.pool
        return {
            'pool_size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid(),
            'status': 'healthy' if check_database_connection() else 'unhealthy'
        }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {'status': 'error', 'error': str(e)}

# ================================
# Transaction Management
# ================================

@contextmanager
def get_db_transaction():
    """
    Context manager for database transactions with automatic rollback.

    Example:
        >>> with get_db_transaction() as db:
        ...     db.add(new_user)
        ...     # Automatically commits or rolls back
    """
    db = SessionLocal()
    try:
        db.begin()
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Transaction failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()

@asynccontextmanager
async def get_async_db_transaction():
    """
    Async context manager for database transactions.

    Example:
        >>> async with get_async_db_transaction() as db:
        ...     db.add(new_user)
        ...     # Automatically commits or rolls back
    """
    async with AsyncSessionLocal() as session:
        try:
            async with session.begin():
                yield session
        except Exception as e:
            logger.error(f"Async transaction failed: {e}")
            await session.rollback()
            raise

# ================================
# Connection Pool Management
# ================================

def get_connection_pool_status() -> Dict[str, Any]:
    """
    Get detailed connection pool status for monitoring.

    Returns:
        Dict with detailed pool information
    """
    try:
        pool = engine.pool

        status = {
            'total_connections': pool.size(),
            'available_connections': pool.checkedin(),
            'active_connections': pool.checkedout(),
            'overflow_connections': pool.overflow(),
            'invalid_connections': pool.invalid(),
            'pool_timeout': config['pool_timeout'],
            'max_overflow': config['max_overflow'],
            'health_status': 'healthy' if check_database_connection() else 'unhealthy'
        }

        # Calculate utilization percentage
        total_possible = pool.size() + config['max_overflow']
        used_connections = pool.checkedout() + pool.overflow()
        status['utilization_percent'] = (used_connections / total_possible) * 100 if total_possible > 0 else 0

        return status

    except Exception as e:
        logger.error(f"Failed to get connection pool status: {e}")
        return {'health_status': 'error', 'error': str(e)}

def close_all_connections():
    """
    Close all database connections (for graceful shutdown).
    """
    try:
        engine.dispose()
        logger.info("All database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

async def close_all_async_connections():
    """
    Close all async database connections (for graceful shutdown).
    """
    try:
        await async_engine.dispose()
        logger.info("All async database connections closed")
    except Exception as e:
        logger.error(f"Error closing async database connections: {e}")

# ================================
# Performance Monitoring
# ================================

def execute_with_timing(query_func, *args, **kwargs):
    """
    Execute a database query with timing metrics.

    Args:
        query_func: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple of (result, execution_time_ms)
    """
    import time

    start_time = time.time()
    try:
        result = query_func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000

        # Log slow queries
        if execution_time > getattr(settings, 'slow_query_threshold_ms', 1000):
            logger.warning(f"Slow query detected: {execution_time:.2f}ms")

        return result, execution_time

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"Query failed after {execution_time:.2f}ms: {e}")
        raise

# ================================
# Startup and Shutdown Handlers
# ================================

async def startup_database():
    """
    Database startup handler for FastAPI application.
    """
    logger.info("Starting database connections...")

    # Test connections
    if not check_database_connection():
        raise RuntimeError("Failed to connect to database")

    if not await check_async_database_connection():
        raise RuntimeError("Failed to connect to async database")

    logger.info("Database connections established successfully")

async def shutdown_database():
    """
    Database shutdown handler for FastAPI application.
    """
    logger.info("Closing database connections...")

    close_all_connections()
    await close_all_async_connections()

    logger.info("Database connections closed")

# ================================
# Query Helpers
# ================================

def execute_raw_sql(sql: str, params: Optional[Dict] = None) -> Any:
    """
    Execute raw SQL with proper error handling.

    Args:
        sql: SQL query string
        params: Query parameters

    Returns:
        Query result
    """
    try:
        with engine.connect() as connection:
            if params:
                result = connection.execute(text(sql), params)
            else:
                result = connection.execute(text(sql))

            return result.fetchall()

    except Exception as e:
        logger.error(f"Raw SQL execution failed: {e}")
        raise

async def execute_async_raw_sql(sql: str, params: Optional[Dict] = None) -> Any:
    """
    Execute raw SQL asynchronously.

    Args:
        sql: SQL query string
        params: Query parameters

    Returns:
        Query result
    """
    try:
        async with async_engine.connect() as connection:
            if params:
                result = await connection.execute(text(sql), params)
            else:
                result = await connection.execute(text(sql))

            return await result.fetchall()

    except Exception as e:
        logger.error(f"Async raw SQL execution failed: {e}")
        raise