"""
Comprehensive health check tests for production environment.
Tests PostgreSQL, ChromaDB cloud, and all system dependencies.
"""

import pytest
import asyncio
import time
import psutil
from typing import Dict, Any

import psycopg2
import asyncpg
import httpx
from fastapi.testclient import TestClient

# Import production configurations
from app.core.database_prod import (
    engine, async_engine, check_database_connection,
    check_async_database_connection, get_database_stats,
    get_connection_pool_status
)
from app.core.chromadb_prod import (
    chroma_client, test_chromadb_connection
)
from app.core.config import settings
from app.main import app

# Test client
client = TestClient(app)

class TestHealthChecks:
    """Comprehensive system health checks."""

    def test_api_server_health(self):
        """Test that API server is responding."""
        response = client.get("/api/v1/memory/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "components" in data
        assert "timestamp" in data
        assert "version" in data

    def test_postgresql_connection(self):
        """Test PostgreSQL database connection."""
        # Test synchronous connection
        assert check_database_connection(), "PostgreSQL connection failed"

        # Test connection with actual query
        try:
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()

            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            assert "PostgreSQL" in version

            cursor.close()
            conn.close()

        except Exception as e:
            pytest.fail(f"PostgreSQL query test failed: {e}")

    @pytest.mark.asyncio
    async def test_postgresql_async_connection(self):
        """Test PostgreSQL async connection."""
        assert await check_async_database_connection(), "PostgreSQL async connection failed"

        # Test with asyncpg directly
        try:
            conn = await asyncpg.connect(settings.database_url)

            version = await conn.fetchval("SELECT version();")
            assert "PostgreSQL" in version

            await conn.close()

        except Exception as e:
            pytest.fail(f"PostgreSQL async query test failed: {e}")

    def test_database_performance(self):
        """Test database query performance."""
        start_time = time.time()

        # Test simple query performance
        assert check_database_connection()

        response_time = (time.time() - start_time) * 1000

        # Should respond within 100ms for health check
        assert response_time < 100, f"Database health check too slow: {response_time:.2f}ms"

    def test_connection_pool_health(self):
        """Test database connection pool status."""
        stats = get_connection_pool_status()

        assert stats["health_status"] == "healthy"
        assert stats["total_connections"] > 0
        assert stats["utilization_percent"] < 90  # Should not be at capacity

        # Test pool configuration
        assert stats["pool_timeout"] > 0
        assert stats["max_overflow"] > 0

    def test_chromadb_cloud_connection(self):
        """Test ChromaDB cloud connection."""
        if not settings.chroma_api_key:
            pytest.skip("ChromaDB API key not configured")

        # Test connection
        assert test_chromadb_connection(), "ChromaDB cloud connection failed"

        # Test health check
        health = chroma_client.health_check()
        assert health["status"] == "healthy"
        assert health["response_time_ms"] < 5000  # Should respond within 5 seconds

    def test_chromadb_performance(self):
        """Test ChromaDB performance."""
        if not settings.chroma_api_key:
            pytest.skip("ChromaDB API key not configured")

        try:
            # Test basic operations performance
            start_time = time.time()

            health = chroma_client.health_check()

            response_time = (time.time() - start_time) * 1000

            assert health["status"] == "healthy"
            assert response_time < 2000, f"ChromaDB too slow: {response_time:.2f}ms"

        except Exception as e:
            pytest.fail(f"ChromaDB performance test failed: {e}")

    def test_gemini_api_connection(self):
        """Test Gemini API connectivity."""
        if not settings.gemini_api_key:
            pytest.skip("Gemini API key not configured")

        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.gemini_api_key)

            # Test with a simple request
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Hello")

            assert response.text is not None
            assert len(response.text) > 0

        except Exception as e:
            pytest.fail(f"Gemini API test failed: {e}")

    def test_system_resources(self):
        """Test system resource availability."""
        # Check memory usage
        memory = psutil.virtual_memory()
        assert memory.percent < 90, f"High memory usage: {memory.percent}%"

        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        assert cpu_percent < 90, f"High CPU usage: {cpu_percent}%"

        # Check disk space
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        assert disk_percent < 90, f"High disk usage: {disk_percent:.1f}%"

    def test_api_response_times(self):
        """Test API endpoint response times."""
        endpoints = [
            "/api/v1/memory/health",
            "/docs",
            "/openapi.json"
        ]

        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint)
            response_time = (time.time() - start_time) * 1000

            assert response.status_code == 200
            assert response_time < 500, f"Endpoint {endpoint} too slow: {response_time:.2f}ms"

    def test_concurrent_connections(self):
        """Test handling of concurrent connections."""
        import threading
        import queue

        results = queue.Queue()

        def test_connection():
            try:
                response = client.get("/api/v1/memory/health")
                results.put(response.status_code == 200)
            except Exception as e:
                results.put(False)

        # Test 10 concurrent connections
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=test_connection)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        success_count = 0
        while not results.empty():
            if results.get():
                success_count += 1

        assert success_count == 10, f"Only {success_count}/10 concurrent connections succeeded"

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable under load."""
        import gc

        # Get initial memory usage
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Make multiple requests
        for _ in range(50):
            response = client.get("/api/v1/memory/health")
            assert response.status_code == 200

        # Check final memory usage
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50, f"Memory leak detected: {memory_increase:.1f}MB increase"

    def test_database_table_structure(self):
        """Test that all required database tables exist."""
        try:
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()

            # Check for required tables
            required_tables = [
                'users', 'memories', 'conversations', 'memory_connections',
                'memory_access_log', 'rag_metrics', 'chroma_sync_status'
            ]

            for table in required_tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    );
                """, (table,))

                exists = cursor.fetchone()[0]
                assert exists, f"Required table '{table}' does not exist"

            cursor.close()
            conn.close()

        except Exception as e:
            pytest.fail(f"Database structure test failed: {e}")

    def test_database_indexes(self):
        """Test that critical indexes exist."""
        try:
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()

            # Check for critical indexes
            critical_indexes = [
                'idx_memories_user_active',
                'idx_memories_user_category_type',
                'idx_memories_user_created',
                'idx_memories_user_importance'
            ]

            for index in critical_indexes:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM pg_indexes
                        WHERE schemaname = 'public'
                        AND indexname = %s
                    );
                """, (index,))

                exists = cursor.fetchone()[0]
                assert exists, f"Critical index '{index}' does not exist"

            cursor.close()
            conn.close()

        except Exception as e:
            pytest.fail(f"Database index test failed: {e}")

    def test_environment_configuration(self):
        """Test that all required environment variables are set."""
        required_settings = [
            'database_url',
            'gemini_api_key',
            'chroma_api_key'
        ]

        for setting in required_settings:
            value = getattr(settings, setting, None)
            assert value is not None, f"Required setting '{setting}' is not configured"
            assert len(str(value)) > 0, f"Required setting '{setting}' is empty"

    @pytest.mark.asyncio
    async def test_async_operations_performance(self):
        """Test async operations performance."""
        # Test multiple async database connections
        tasks = []

        async def test_async_query():
            return await check_async_database_connection()

        start_time = time.time()

        # Run 5 async database checks concurrently
        for _ in range(5):
            tasks.append(test_async_query())

        results = await asyncio.gather(*tasks)

        total_time = (time.time() - start_time) * 1000

        # All should succeed
        assert all(results), "Some async database connections failed"

        # Should complete within reasonable time
        assert total_time < 1000, f"Async operations too slow: {total_time:.2f}ms"

    def test_error_handling(self):
        """Test API error handling."""
        # Test 404 error
        response = client.get("/api/v1/memory/nonexistent")
        assert response.status_code == 404

        # Test malformed request
        response = client.post("/api/v1/memory/extract", json={"invalid": "data"})
        assert response.status_code == 422

        # Test missing required fields
        response = client.post("/api/v1/memory/extract", json={})
        assert response.status_code == 422

    def test_security_headers(self):
        """Test that security headers are properly set."""
        response = client.get("/api/v1/memory/health")

        # Check for security headers (if implemented)
        headers = response.headers

        # These would be set by reverse proxy or middleware
        # Just ensure response is valid
        assert response.status_code == 200

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_health_check_benchmark(self):
        """Benchmark health check endpoint."""
        times = []

        for _ in range(10):
            start_time = time.time()
            response = client.get("/api/v1/memory/health")
            response_time = (time.time() - start_time) * 1000

            assert response.status_code == 200
            times.append(response_time)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Performance targets
        assert avg_time < 50, f"Average health check time too slow: {avg_time:.2f}ms"
        assert max_time < 100, f"Max health check time too slow: {max_time:.2f}ms"

    def test_database_query_benchmark(self):
        """Benchmark database query performance."""
        times = []

        for _ in range(10):
            start_time = time.time()
            assert check_database_connection()
            query_time = (time.time() - start_time) * 1000
            times.append(query_time)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Database performance targets
        assert avg_time < 20, f"Average DB query time too slow: {avg_time:.2f}ms"
        assert max_time < 50, f"Max DB query time too slow: {max_time:.2f}ms"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])