"""
System health tests without SQLAlchemy dependencies.
Tests PostgreSQL, ChromaDB, and API health using direct connections.
"""

import pytest
import asyncio
import time
import psycopg2
import asyncpg
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.core.config import settings


class TestPostgreSQLHealth:
    """PostgreSQL database health tests."""

    def test_postgresql_basic_connection(self):
        """Test basic PostgreSQL connection."""
        try:
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()

            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            assert "PostgreSQL" in version
            print(f"✅ PostgreSQL version: {version[:50]}...")

            cursor.close()
            conn.close()

        except Exception as e:
            pytest.fail(f"PostgreSQL connection failed: {e}")

    def test_postgresql_schema_exists(self):
        """Test that memory_pipeline schema exists."""
        try:
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name = 'memory_pipeline'
            """)

            schema = cursor.fetchone()
            assert schema is not None, "memory_pipeline schema does not exist"

            print(f"✅ Schema 'memory_pipeline' exists")

            cursor.close()
            conn.close()

        except Exception as e:
            pytest.fail(f"Schema check failed: {e}")

    def test_postgresql_tables_exist(self):
        """Test that all required tables exist."""
        required_tables = [
            'users', 'memories', 'conversations', 'memory_connections',
            'memory_access_log', 'rag_metrics', 'chroma_sync_status'
        ]

        try:
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'memory_pipeline'
                ORDER BY table_name
            """)

            existing_tables = [row[0] for row in cursor.fetchall()]

            for table in required_tables:
                assert table in existing_tables, f"Required table '{table}' missing"

            print(f"✅ All {len(required_tables)} required tables exist")

            cursor.close()
            conn.close()

        except Exception as e:
            pytest.fail(f"Table existence check failed: {e}")

    def test_postgresql_crud_operations(self):
        """Test basic CRUD operations."""
        try:
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()

            # Test user insertion
            test_user_id = f"health_test_user_{int(time.time())}"

            cursor.execute("""
                INSERT INTO memory_pipeline.users (user_id, settings, is_active)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
            """, (test_user_id, '{"test": true}', True))

            # Test user selection
            cursor.execute("""
                SELECT user_id, is_active
                FROM memory_pipeline.users
                WHERE user_id = %s
            """, (test_user_id,))

            user = cursor.fetchone()
            assert user is not None, "User insertion/selection failed"
            assert user[1] is True, "User active flag incorrect"

            # Test user update
            cursor.execute("""
                UPDATE memory_pipeline.users
                SET total_memories = 1
                WHERE user_id = %s
            """, (test_user_id,))

            # Test user deletion
            cursor.execute("""
                DELETE FROM memory_pipeline.users
                WHERE user_id = %s
            """, (test_user_id,))

            conn.commit()
            print("✅ PostgreSQL CRUD operations working")

            cursor.close()
            conn.close()

        except Exception as e:
            pytest.fail(f"CRUD operations test failed: {e}")

    def test_postgresql_performance(self):
        """Test PostgreSQL query performance."""
        try:
            times = []

            for _ in range(5):
                start_time = time.time()

                conn = psycopg2.connect(settings.database_url)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                conn.close()

                query_time = (time.time() - start_time) * 1000
                times.append(query_time)

            avg_time = sum(times) / len(times)
            max_time = max(times)

            print(f"✅ PostgreSQL performance - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")

            # Performance assertions
            assert avg_time < 50, f"Average query time too slow: {avg_time:.2f}ms"
            assert max_time < 100, f"Max query time too slow: {max_time:.2f}ms"

        except Exception as e:
            pytest.fail(f"Performance test failed: {e}")

    @pytest.mark.asyncio
    async def test_postgresql_async_connection(self):
        """Test PostgreSQL async connection."""
        try:
            conn = await asyncpg.connect(settings.database_url)

            version = await conn.fetchval("SELECT version()")
            assert "PostgreSQL" in version

            await conn.close()
            print("✅ PostgreSQL async connection working")

        except Exception as e:
            pytest.fail(f"Async PostgreSQL test failed: {e}")


class TestChromaDBHealth:
    """ChromaDB cloud health tests."""

    def test_chromadb_configuration(self):
        """Test ChromaDB configuration."""
        required_settings = [
            'chroma_api_key', 'chroma_api_url', 'chroma_tenant',
            'chroma_database', 'chroma_collection_name'
        ]

        for setting in required_settings:
            value = getattr(settings, setting, None)
            assert value is not None, f"ChromaDB setting '{setting}' not configured"
            assert len(str(value)) > 0, f"ChromaDB setting '{setting}' is empty"

        print("✅ ChromaDB configuration complete")

    def test_chromadb_credentials(self):
        """Test ChromaDB credentials format."""
        # Test API key format
        api_key = settings.chroma_api_key
        assert api_key.startswith('ck-'), "ChromaDB API key should start with 'ck-'"
        assert len(api_key) > 20, "ChromaDB API key seems too short"

        # Test URL format
        api_url = settings.chroma_api_url
        assert api_url.startswith('https://'), "ChromaDB API URL should be HTTPS"
        assert 'chroma' in api_url.lower(), "ChromaDB URL should contain 'chroma'"

        print("✅ ChromaDB credentials format valid")

    def test_chromadb_import_compatibility(self):
        """Test ChromaDB package compatibility."""
        try:
            import numpy as np
            numpy_version = np.__version__

            # Check if NumPy version is compatible
            major_version = int(numpy_version.split('.')[0])

            if major_version >= 2:
                print(f"⚠️ NumPy {numpy_version} detected - ChromaDB may have compatibility issues")
                pytest.skip("NumPy 2.0+ compatibility issue with ChromaDB")

            # Try to import ChromaDB
            import chromadb
            print(f"✅ ChromaDB package imports successfully with NumPy {numpy_version}")

        except ImportError as e:
            if "np.float_" in str(e):
                pytest.skip("ChromaDB NumPy 2.0 compatibility issue - skipping import test")
            else:
                pytest.fail(f"ChromaDB import failed: {e}")


class TestAPIHealth:
    """API health tests using direct imports."""

    def test_settings_loading(self):
        """Test that settings load correctly."""
        # Test required settings
        assert settings.database_url is not None
        assert settings.gemini_api_key is not None
        assert settings.chroma_api_key is not None

        # Test database URL format
        assert 'postgresql://' in settings.database_url
        assert 'memory-pipeline' in settings.database_url

        print("✅ Settings loaded correctly")

    def test_gemini_api_configuration(self):
        """Test Gemini API configuration."""
        api_key = settings.gemini_api_key
        assert api_key.startswith('AIza'), "Gemini API key should start with 'AIza'"
        assert len(api_key) > 30, "Gemini API key seems too short"

        model = settings.gemini_model
        assert 'gemini' in model.lower(), "Gemini model name should contain 'gemini'"

        print("✅ Gemini API configuration valid")

    def test_directory_structure(self):
        """Test critical directory structure."""
        critical_dirs = [
            'app',
            'app/core',
            'app/api',
            'app/memory',
            'app/services',
            'tests',
            'setup'
        ]

        project_root = os.path.join(os.path.dirname(__file__), '..', '..')

        for directory in critical_dirs:
            dir_path = os.path.join(project_root, directory)
            assert os.path.exists(dir_path), f"Critical directory '{directory}' missing"

        print("✅ Directory structure valid")

    def test_critical_files_exist(self):
        """Test that critical files exist."""
        critical_files = [
            'app/main.py',
            'app/core/config.py',
            'app/core/database_prod.py',
            'app/core/chromadb_prod.py',
            'app/services/memory_service.py',
            'app/api/endpoints/memory.py',
            'requirements.txt',
            '.env'
        ]

        project_root = os.path.join(os.path.dirname(__file__), '..', '..')

        for file_path in critical_files:
            full_path = os.path.join(project_root, file_path)
            assert os.path.exists(full_path), f"Critical file '{file_path}' missing"

        print("✅ All critical files exist")


class TestIntegrationHealth:
    """Integration health tests."""

    def test_database_chromadb_sync_table(self):
        """Test ChromaDB sync table operations."""
        try:
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()

            # Test inserting sync status
            test_memory_id = 'test_sync_memory_123'

            cursor.execute("""
                INSERT INTO memory_pipeline.memories
                (memory_id, user_id, content, memory_type, category, importance_score)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (memory_id) DO NOTHING
            """, (test_memory_id, 'test_user', 'test content', 'fact', 'test', 0.5))

            cursor.execute("""
                INSERT INTO memory_pipeline.chroma_sync_status
                (memory_id, chroma_id, collection_name, sync_status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (memory_id) DO UPDATE SET sync_status = EXCLUDED.sync_status
            """, (test_memory_id, 'chroma_123', 'test_collection', 'synced'))

            # Verify sync status
            cursor.execute("""
                SELECT sync_status FROM memory_pipeline.chroma_sync_status
                WHERE memory_id = %s
            """, (test_memory_id,))

            status = cursor.fetchone()
            assert status is not None, "Sync status not found"
            assert status[0] == 'synced', "Sync status incorrect"

            # Cleanup
            cursor.execute("DELETE FROM memory_pipeline.chroma_sync_status WHERE memory_id = %s", (test_memory_id,))
            cursor.execute("DELETE FROM memory_pipeline.memories WHERE memory_id = %s", (test_memory_id,))

            conn.commit()
            print("✅ Database-ChromaDB sync table working")

            cursor.close()
            conn.close()

        except Exception as e:
            pytest.fail(f"Sync table test failed: {e}")

    def test_environment_production_ready(self):
        """Test production readiness indicators."""
        # Check debug mode
        assert not settings.debug, "Debug mode should be disabled in production"

        # Check log level
        assert settings.log_level.upper() in ['INFO', 'WARNING', 'ERROR'], "Log level should be production-appropriate"

        # Check database pool settings
        assert settings.db_pool_size >= 10, "Database pool size should be adequate for production"
        assert settings.db_max_overflow >= 20, "Database max overflow should be adequate"

        print("✅ Production readiness checks passed")


if __name__ == "__main__":
    print("System Health Tests")
    print("=" * 50)

    # Run tests with pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])

    sys.exit(exit_code)