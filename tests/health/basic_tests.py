#!/usr/bin/env python3
"""
Basic connection tests without SQLAlchemy.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_basic_postgresql():
    """Test PostgreSQL using raw psycopg2."""
    try:
        import psycopg2
        from app.core.config import settings

        print("Testing basic PostgreSQL connection...")

        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()

        # Test connection
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"SUCCESS: PostgreSQL version: {version[:60]}...")

        # Check tables
        cursor.execute("""
            SELECT table_name, table_schema
            FROM information_schema.tables
            WHERE table_schema = 'memory_pipeline'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()

        if tables:
            print(f"SUCCESS: Found {len(tables)} tables in memory_pipeline schema:")
            for table_name, schema in tables:
                print(f"  - {schema}.{table_name}")
        else:
            print("WARNING: No tables found in memory_pipeline schema")

        # Check if we can insert/select (basic test)
        cursor.execute("""
            INSERT INTO memory_pipeline.users (user_id, settings, is_active)
            VALUES ('test_basic_user', '{}', true)
            ON CONFLICT (user_id) DO NOTHING;
        """)

        cursor.execute("""
            SELECT user_id, created_at, is_active
            FROM memory_pipeline.users
            WHERE user_id = 'test_basic_user';
        """)
        user = cursor.fetchone()

        if user:
            print(f"SUCCESS: Database operations working. User: {user[0]}, Active: {user[2]}")
        else:
            print("ERROR: Could not insert/retrieve test user")

        conn.commit()
        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"ERROR: Basic PostgreSQL test failed: {e}")
        return False


def test_basic_chromadb_config():
    """Test ChromaDB configuration without imports."""
    try:
        from app.core.config import settings

        print("Testing ChromaDB configuration...")

        # Check if all required settings exist
        required_settings = [
            'chroma_api_key', 'chroma_api_url', 'chroma_tenant',
            'chroma_database', 'chroma_collection_name'
        ]

        all_present = True
        for setting in required_settings:
            if hasattr(settings, setting):
                value = getattr(settings, setting)
                if value:
                    if setting == 'chroma_api_key':
                        print(f"  {setting}: {value[:10]}...")
                    else:
                        print(f"  {setting}: {value}")
                else:
                    print(f"  ERROR: {setting} is empty")
                    all_present = False
            else:
                print(f"  ERROR: {setting} not found")
                all_present = False

        if all_present:
            print("SUCCESS: All ChromaDB settings configured")
            return True
        else:
            print("ERROR: Missing ChromaDB settings")
            return False

    except Exception as e:
        print(f"ERROR: ChromaDB config test failed: {e}")
        return False


def test_basic_app_structure():
    """Test basic app structure and imports."""
    try:
        print("Testing app structure...")

        # Test basic imports
        from app.core.config import settings
        print("SUCCESS: Config import works")

        from app.models.schemas import MemoryType
        print("SUCCESS: Schema imports work")

        # Check main API files exist
        files_to_check = [
            'app/main.py',
            'app/api/endpoints/memory.py',
            'app/services/memory_service.py',
            'app/memory/extractor.py',
            'app/memory/embedder.py'
        ]

        all_exist = True
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (missing)")
                all_exist = False

        if all_exist:
            print("SUCCESS: All critical files present")
            return True
        else:
            print("ERROR: Missing critical files")
            return False

    except Exception as e:
        print(f"ERROR: App structure test failed: {e}")
        return False


if __name__ == "__main__":
    print("Basic System Tests")
    print("=" * 50)

    # Run tests
    tests = [
        ("App Structure", test_basic_app_structure),
        ("PostgreSQL Connection", test_basic_postgresql),
        ("ChromaDB Configuration", test_basic_chromadb_config)
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        results[test_name] = test_func()

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    overall_success = all(results.values())
    print(f"\nOverall: {'PASS' if overall_success else 'FAIL'}")

    sys.exit(0 if overall_success else 1)