#!/usr/bin/env python3
"""
Run all PostgreSQL setup scripts in the correct order.
Creates the memory_pipeline schema with all tables, indexes, and constraints.
"""

import os
import sys
import logging
import asyncio
import asyncpg
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the setup files in order
SETUP_FILES = [
    "01_drop_schema.sql",
    "02_create_schema.sql",
    "03_create_types.sql",
    "04_create_tables.sql",
    "05_create_indexes.sql",
    "06_create_functions.sql",
    "07_create_triggers.sql",
    "08_create_constraints.sql",
    "09_verify_setup.sql"
]

async def run_sql_file_simple(file_path: str) -> bool:
    """
    Run SQL file using asyncpg - handles complex SQL properly.

    Args:
        file_path: Path to SQL file

    Returns:
        bool: True if successful
    """
    try:
        # Read SQL file
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Connect to database
        conn = await asyncpg.connect(settings.database_url)

        try:
            # Execute the entire file content as one transaction
            await conn.execute(sql_content)
            logger.info(f"✅ {Path(file_path).name} executed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ {Path(file_path).name} failed: {e}")
            return False

        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"❌ Failed to process {file_path}: {e}")
        return False


async def run_all_setup():
    """Run all setup files in order."""
    setup_dir = Path(__file__).parent

    logger.info("🚀 Starting PostgreSQL schema setup...")
    logger.info(f"Using database: {settings.database_url.split('@')[1] if '@' in settings.database_url else 'localhost'}")

    # Test connection first
    try:
        conn = await asyncpg.connect(settings.database_url)
        version = await conn.fetchrow("SELECT version()")
        await conn.close()
        logger.info(f"📀 PostgreSQL version: {version['version']}")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

    success_count = 0

    for sql_file in SETUP_FILES:
        file_path = setup_dir / sql_file

        if not file_path.exists():
            logger.error(f"❌ File not found: {sql_file}")
            continue

        logger.info(f"🔄 Running {sql_file}...")

        if await run_sql_file_simple(str(file_path)):
            success_count += 1
        else:
            logger.error(f"❌ Setup failed at {sql_file}")
            return False

    if success_count == len(SETUP_FILES):
        logger.info(f"🎉 All {success_count} setup files completed successfully!")
        logger.info("📊 Schema 'memory_pipeline' is ready for use")
        return True
    else:
        logger.error(f"❌ Setup incomplete: {success_count}/{len(SETUP_FILES)} files succeeded")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_setup())
        if success:
            logger.info("✅ Database setup completed successfully!")
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