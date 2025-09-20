#!/usr/bin/env python3
"""
Simple connection test script.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings

def test_postgresql():
    """Test PostgreSQL connection using psycopg2."""
    try:
        import psycopg2
        print("Testing PostgreSQL connection...")

        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()

        # Test basic query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"SUCCESS: PostgreSQL Connected: {version[:50]}...")

        # Check if our schema exists
        cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'memory_pipeline';")
        schema_exists = cursor.fetchone()

        if schema_exists:
            print("SUCCESS: Schema 'memory_pipeline' exists")

            # Check tables
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'memory_pipeline'
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()

            if tables:
                print(f"SUCCESS: Found {len(tables)} tables: {', '.join([t[0] for t in tables])}")
            else:
                print("WARNING: Schema exists but no tables found")
        else:
            print("ERROR: Schema 'memory_pipeline' does not exist - needs setup")

        cursor.close()
        conn.close()
        return True, schema_exists is not None

    except Exception as e:
        print(f"ERROR: PostgreSQL Connection Failed: {e}")
        return False, False

def run_database_setup():
    """Run database setup using asyncpg."""
    try:
        import asyncio
        from setup.run_all_setup import run_all_setup

        print("Running database setup...")
        success = asyncio.run(run_all_setup())

        if success:
            print("SUCCESS: Database setup completed successfully")
            return True
        else:
            print("ERROR: Database setup failed")
            return False

    except Exception as e:
        print(f"ERROR: Setup error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Database Connections")
    print("=" * 50)

    # Test PostgreSQL
    pg_connected, schema_exists = test_postgresql()

    # Run setup if needed
    setup_success = True
    if pg_connected and not schema_exists:
        setup_success = run_database_setup()

        # Re-test after setup
        if setup_success:
            print("\nRe-testing after setup...")
            pg_connected, schema_exists = test_postgresql()

    print("\n" + "=" * 50)
    print("CONNECTION SUMMARY:")
    print(f"PostgreSQL: {'SUCCESS' if pg_connected else 'FAILED'}")
    print(f"Schema:     {'SUCCESS' if schema_exists else 'FAILED'}")
    print(f"Setup:      {'SUCCESS' if setup_success else 'FAILED'}")

    if pg_connected and schema_exists:
        print("DATABASE READY!")
        sys.exit(0)
    else:
        print("Database not ready")
        sys.exit(1)