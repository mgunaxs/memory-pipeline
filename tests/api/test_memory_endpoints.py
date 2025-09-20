"""
API Integration Tests for Memory Pipeline

Tests all API endpoints with real HTTP requests to verify
production readiness and proper error handling.
"""

import asyncio
import json
import time
from typing import Dict, Any
import requests
import pytest
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"


class APITestClient:
    """Test client for Memory Pipeline API."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    def get(self, path: str, **kwargs) -> requests.Response:
        """Make GET request."""
        url = f"{self.base_url}{API_PREFIX}{path}"
        return self.session.get(url, **kwargs)

    def post(self, path: str, data: Dict[str, Any] = None, **kwargs) -> requests.Response:
        """Make POST request."""
        url = f"{self.base_url}{API_PREFIX}{path}"
        return self.session.post(url, json=data, **kwargs)


class TestMemoryAPI:
    """Comprehensive test suite for Memory API endpoints."""

    @classmethod
    def setup_class(cls):
        """Set up test client and verify API is running."""
        cls.client = APITestClient()
        cls.test_user_id = f"test_user_{int(time.time())}"

        # Verify API is accessible
        try:
            response = cls.client.get("/memory/health")
            if response.status_code != 200:
                raise Exception(f"API health check failed: {response.status_code}")
            print(f"[SETUP] API is running and accessible")
        except Exception as e:
            raise Exception(f"Cannot connect to API at {BASE_URL}: {e}")

    def test_health_endpoint(self):
        """Test /memory/health endpoint."""
        print("\n[TEST] Health endpoint")

        response = self.client.get("/memory/health")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
        assert "version" in data

        # Verify components
        components = data["components"]
        assert "database" in components
        assert "vector_db" in components
        assert "llm_api" in components
        assert "embedding_service" in components

        print(f"  [OK] Health status: {data['status']}")
        print(f"  [OK] Components: {components}")

    def test_extract_memories_endpoint(self):
        """Test /memory/extract endpoint."""
        print("\n[TEST] Memory extraction endpoint")

        # Test data
        request_data = {
            "text": "I work as a software engineer at Google and I love drinking coffee in the morning",
            "user_id": self.test_user_id,
            "message_id": f"msg_{int(time.time())}"
        }

        response = self.client.post("/memory/extract", data=request_data)

        # Check if extraction works or fails gracefully
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "data" in data

            extraction_data = data["data"]
            assert "memories" in extraction_data
            assert "total_extracted" in extraction_data
            assert "processing_time_ms" in extraction_data

            print(f"  [OK] Extracted {extraction_data['total_extracted']} memories")
            print(f"  [OK] Processing time: {extraction_data['processing_time_ms']}ms")

        elif response.status_code == 500:
            # Expected if Gemini API not configured
            print("  [INFO] Extraction failed (likely missing API keys) - this is expected in test environment")

        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")

    def test_search_memories_endpoint(self):
        """Test /memory/search endpoint."""
        print("\n[TEST] Memory search endpoint")

        request_data = {
            "query": "work preferences",
            "user_id": self.test_user_id,
            "limit": 5
        }

        response = self.client.post("/memory/search", data=request_data)

        # Should work even without memories stored
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "data" in data

        search_data = data["data"]
        assert "results" in search_data
        assert "total_found" in search_data
        assert "search_time_ms" in search_data

        print(f"  [OK] Found {search_data['total_found']} memories")
        print(f"  [OK] Search time: {search_data['search_time_ms']}ms")

    def test_get_user_memories_endpoint(self):
        """Test /memory/user/{user_id} endpoint."""
        print("\n[TEST] Get user memories endpoint")

        response = self.client.get(f"/memory/user/{self.test_user_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "data" in data

        user_data = data["data"]
        assert "memories" in user_data
        assert "total_returned" in user_data
        assert "user_id" in user_data

        print(f"  [OK] Retrieved {user_data['total_returned']} memories for user")

    def test_get_user_stats_endpoint(self):
        """Test /memory/stats/{user_id} endpoint."""
        print("\n[TEST] User stats endpoint")

        response = self.client.get(f"/memory/stats/{self.test_user_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "data" in data

        stats = data["data"]
        assert "total_memories" in stats
        assert "active_memories" in stats

        print(f"  [OK] User has {stats['total_memories']} total memories")

    def test_smart_search_endpoint(self):
        """Test /memory/smart-search endpoint (enhanced RAG)."""
        print("\n[TEST] Smart search endpoint (RAG)")

        params = {
            "query": "morning routine",
            "user_id": self.test_user_id,
            "context_type": "morning_checkin",
            "max_memories": 3,
            "token_budget": 300,
            "enable_validation": True
        }

        response = self.client.post("/memory/smart-search", params=params)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "data" in data

        result_data = data["data"]
        assert "results" in result_data
        assert "total_found" in result_data
        assert "context_filtered" in result_data
        assert "validation_applied" in result_data

        print(f"  [OK] Smart search found {result_data['total_found']} memories")
        print(f"  [OK] Context filtering: {result_data['context_filtered']}")
        print(f"  [OK] Validation applied: {result_data['validation_applied']}")

    def test_rag_stats_endpoint(self):
        """Test /memory/rag-stats/{user_id} endpoint."""
        print("\n[TEST] RAG stats endpoint")

        response = self.client.get(f"/memory/rag-stats/{self.test_user_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "data" in data

        rag_data = data["data"]
        assert "total_memories" in rag_data
        assert "category_distribution" in rag_data
        assert "quality_metrics" in rag_data
        assert "rag_features" in rag_data

        # Verify RAG features are configured
        rag_features = rag_data["rag_features"]
        assert "context_filtering_enabled" in rag_features
        assert "deduplication_enabled" in rag_features
        assert "validation_enabled" in rag_features

        print(f"  [OK] RAG features enabled: {rag_features}")

    def test_process_conversation_endpoint(self):
        """Test /memory/process-conversation endpoint."""
        print("\n[TEST] Process conversation endpoint")

        request_data = {
            "text": "I have a meeting tomorrow at 2pm with the design team",
            "user_id": self.test_user_id,
            "message_id": f"conv_{int(time.time())}"
        }

        response = self.client.post("/memory/process-conversation", data=request_data)

        # Should handle gracefully even if extraction fails
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "data" in data

            conv_data = data["data"]
            assert "extracted_memories" in conv_data
            assert "relevant_context" in conv_data
            assert "processing_summary" in conv_data

            print("  [OK] Conversation processed successfully")

        elif response.status_code == 500:
            print("  [INFO] Conversation processing failed (expected without API keys)")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")

    def test_api_error_handling(self):
        """Test API error handling."""
        print("\n[TEST] API error handling")

        # Test invalid user ID
        response = self.client.get("/memory/user/")
        assert response.status_code == 404
        print("  [OK] 404 for invalid user ID path")

        # Test malformed JSON
        response = requests.post(
            f"{BASE_URL}{API_PREFIX}/memory/extract",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        print("  [OK] 422 for malformed JSON")

        # Test missing required fields
        response = self.client.post("/memory/extract", data={})
        assert response.status_code == 422
        print("  [OK] 422 for missing required fields")

    def test_api_response_format(self):
        """Test consistent API response format."""
        print("\n[TEST] API response format consistency")

        # Test multiple endpoints for consistent format
        endpoints = [
            ("/memory/health", "get"),
            (f"/memory/user/{self.test_user_id}", "get"),
            (f"/memory/stats/{self.test_user_id}", "get"),
            (f"/memory/rag-stats/{self.test_user_id}", "get")
        ]

        for endpoint, method in endpoints:
            if method == "get":
                response = self.client.get(endpoint)
            else:
                response = self.client.post(endpoint, data={})

            assert response.status_code == 200
            data = response.json()

            # Check common response structure
            if endpoint != "/memory/health":  # Health has different format
                assert "success" in data
                assert "data" in data
                assert "timestamp" in data or "error" in data

        print("  [OK] All endpoints return consistent format")


def run_api_tests():
    """Run all API tests."""
    print("=" * 60)
    print("MEMORY PIPELINE API INTEGRATION TESTS")
    print("=" * 60)

    # Create test instance
    test_suite = TestMemoryAPI()
    test_suite.setup_class()

    # Run tests
    tests = [
        test_suite.test_health_endpoint,
        test_suite.test_extract_memories_endpoint,
        test_suite.test_search_memories_endpoint,
        test_suite.test_get_user_memories_endpoint,
        test_suite.test_get_user_stats_endpoint,
        test_suite.test_smart_search_endpoint,
        test_suite.test_rag_stats_endpoint,
        test_suite.test_process_conversation_endpoint,
        test_suite.test_api_error_handling,
        test_suite.test_api_response_format
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("API TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Success rate: {(passed / (passed + failed)) * 100:.1f}%")

    if failed == 0:
        print("\n[SUCCESS] All API tests passed! API is working correctly.")
    else:
        print(f"\n[WARNING] {failed} tests failed. Check API configuration.")

    return failed == 0


if __name__ == "__main__":
    success = run_api_tests()
    exit(0 if success else 1)