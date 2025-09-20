#!/usr/bin/env python3
"""
Production PostgreSQL database setup script.

Sets up the memory_pipeline schema with all tables, indexes, and constraints.
Uses dedicated schema instead of public for better organization and security.
"""

import os
import sys
import logging
import asyncio
import asyncpg
from typing import Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_connection() -> bool:
    """Test database connection."""
    try:
        conn = await asyncpg.connect(settings.database_url)
        result = await conn.fetchrow("SELECT version()")
        await conn.close()
        logger.info(f"PostgreSQL version: {result['version']}")
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


async def run_sql_file(file_path: str) -> Dict[str, Any]:
    """
    Run SQL file with proper error handling and reporting.

    Args:
        file_path: Path to SQL file

    Returns:
        Dict with execution results
    """
    start_time = asyncio.get_event_loop().time()

    try:
        # Read SQL file
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Connect to database
        conn = await asyncpg.connect(settings.database_url)

        # Split SQL into individual statements (simple approach)
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

        executed_statements = 0
        failed_statements = 0

        for i, statement in enumerate(statements):
            try:
                # Skip comments and empty statements
                if statement.startswith('--') or not statement:
                    continue

                await conn.execute(statement)
                executed_statements += 1

            except Exception as e:
                failed_statements += 1
                logger.warning(f"Statement {i+1} failed: {e}")
                logger.debug(f"Failed statement: {statement[:100]}...")

        await conn.close()

        execution_time = asyncio.get_event_loop().time() - start_time

        return {
            'success': failed_statements == 0,
            'executed_statements': executed_statements,
            'failed_statements': failed_statements,
            'execution_time_ms': round(execution_time * 1000, 2),
            'file_path': file_path
        }

    except Exception as e:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"Failed to execute SQL file {file_path}: {e}")
        return {
            'success': False,
            'error': str(e),
            'execution_time_ms': round(execution_time * 1000, 2),
            'file_path': file_path
        }


async def verify_schema() -> Dict[str, Any]:
    """Verify schema was created successfully."""
    try:
        conn = await asyncpg.connect(settings.database_url)

        # Check schema exists
        schema_result = await conn.fetchrow(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'memory_pipeline'"
        )

        if not schema_result:
            await conn.close()
            return {'success': False, 'error': 'Schema memory_pipeline not found'}

        # Count tables
        tables_result = await conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'memory_pipeline'
            ORDER BY table_name
            """
        )

        # Count indexes
        indexes_result = await conn.fetch(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'memory_pipeline'
            ORDER BY indexname
            """
        )

        # Check types
        types_result = await conn.fetch(
            """
            SELECT typname
            FROM pg_type t
            JOIN pg_namespace n ON t.typnamespace = n.oid
            WHERE n.nspname = 'memory_pipeline'
            ORDER BY typname
            """
        )

        await conn.close()

        return {
            'success': True,
            'schema_name': 'memory_pipeline',
            'tables': [row['table_name'] for row in tables_result],
            'table_count': len(tables_result),
            'indexes': [row['indexname'] for row in indexes_result],
            'index_count': len(indexes_result),
            'types': [row['typname'] for row in types_result],
            'type_count': len(types_result)
        }

    except Exception as e:
        logger.error(f"Schema verification failed: {e}")
        return {'success': False, 'error': str(e)}


async def setup_database() -> bool:
    """
    Main database setup function.

    Returns:
        bool: True if setup was successful
    """
    logger.info("Starting production database setup...")

    # Test connection first
    logger.info("Testing database connection...")
    if not await test_connection():
        logger.error("Database connection test failed. Cannot proceed.")
        return False

    logger.info("Database connection successful!")

    # Run production schema
    schema_file = Path(__file__).parent / "schema_production.sql"
    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        return False

    logger.info("Executing production schema...")
    schema_result = await run_sql_file(str(schema_file))

    if not schema_result['success']:
        logger.error("Schema execution failed!")
        logger.error(f"Result: {schema_result}")
        return False

    logger.info(f"Schema executed successfully!")
    logger.info(f"  - Executed statements: {schema_result['executed_statements']}")
    logger.info(f"  - Failed statements: {schema_result['failed_statements']}")
    logger.info(f"  - Execution time: {schema_result['execution_time_ms']}ms")

    # Verify schema
    logger.info("Verifying schema creation...")
    verification = await verify_schema()

    if not verification['success']:
        logger.error("Schema verification failed!")
        logger.error(f"Error: {verification.get('error', 'Unknown error')}")
        return False

    logger.info("Schema verification successful!")
    logger.info(f"  - Schema: {verification['schema_name']}")
    logger.info(f"  - Tables: {verification['table_count']}")
    logger.info(f"  - Indexes: {verification['index_count']}")
    logger.info(f"  - Types: {verification['type_count']}")

    # Print table list
    logger.info("Created tables:")
    for table in verification['tables']:
        logger.info(f"  - {table}")

    logger.info("Database setup completed successfully!")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(setup_database())
        if success:
            logger.info("✅ Production database setup completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Database setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)