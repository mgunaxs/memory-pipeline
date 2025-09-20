#!/usr/bin/env python3
"""
Quick setup script for PostgreSQL database.
Run this after ensuring dependencies are installed.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    """Test basic PostgreSQL connection."""
    try:
        import psycopg2
        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"✅ PostgreSQL Connection: {version[0]}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL Connection Failed: {e}")
        return False

def test_chromadb():
    """Test ChromaDB cloud connection."""
    try:
        from app.core.chromadb_prod import test_chromadb_connection
        if test_chromadb_connection():
            logger.info("✅ ChromaDB Cloud: Connected")
            return True
        else:
            logger.error("❌ ChromaDB Cloud: Failed")
            return False
    except Exception as e:
        logger.error(f"❌ ChromaDB Error: {e}")
        return False

def run_sql_files():
    """Run PostgreSQL setup files."""
    try:
        import asyncio
        from setup.run_all_setup import run_all_setup

        logger.info("🚀 Running PostgreSQL setup...")
        success = asyncio.run(run_all_setup())

        if success:
            logger.info("✅ Database setup completed")
            return True
        else:
            logger.error("❌ Database setup failed")
            return False

    except Exception as e:
        logger.error(f"❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔍 Memory Pipeline Setup Check")
    logger.info("=" * 50)

    # Test connections
    pg_ok = test_connection()
    chroma_ok = test_chromadb()

    if pg_ok:
        setup_ok = run_sql_files()
    else:
        setup_ok = False

    # Summary
    logger.info("=" * 50)
    logger.info("📊 SETUP SUMMARY:")
    logger.info(f"PostgreSQL: {'✅' if pg_ok else '❌'}")
    logger.info(f"ChromaDB:   {'✅' if chroma_ok else '❌'}")
    logger.info(f"Schema:     {'✅' if setup_ok else '❌'}")

    if pg_ok and chroma_ok and setup_ok:
        logger.info("🎉 ALL SYSTEMS READY!")
        sys.exit(0)
    else:
        logger.info("❌ Setup incomplete. Check errors above.")
        sys.exit(1)