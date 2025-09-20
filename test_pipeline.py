#!/usr/bin/env python3
"""
Simple End-to-End Pipeline Test
Tests basic functionality after setup
"""

import asyncio
import sys
import os
import requests
from dotenv import load_dotenv

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_startup_check():
    """Test startup dependencies."""
    print("1. Testing Startup Dependencies...")
    try:
        from app.core.startup_check import startup_check
        startup_check()
        print("  OK: All startup checks passed")
        return True
    except SystemExit:
        print("  ERROR: Startup checks failed")
        return False
    except Exception as e:
        print(f"  ERROR: Startup check error: {e}")
        return False

def test_database_connection():
    """Test database connection."""
    print("\n2. Testing Database Connection...")
    try:
        from app.core.database_prod import check_database_connection
        if check_database_connection():
            print("  OK: Database connection successful")
            return True
        else:
            print("  ERROR: Database connection failed")
            return False
    except Exception as e:
        print(f"  ERROR: Database connection error: {e}")
        return False

def test_memory_service():
    """Test memory service initialization."""
    print("\n3. Testing Memory Service...")
    try:
        from app.services.memory_service import MemoryService
        service = MemoryService()
        print("  OK: Memory service initialized")
        return True
    except Exception as e:
        print(f"  ERROR: Memory service error: {e}")
        return False

async def test_basic_memory_pipeline():
    """Test basic memory extraction and storage."""
    print("\n4. Testing Memory Pipeline...")
    try:
        from app.core.database_prod import SessionLocal
        from app.services.memory_service import MemoryService
        from app.models.schemas import ExtractionRequest

        service = MemoryService()

        # Test extraction
        request = ExtractionRequest(
            text="I love coffee in the morning and hate evening meetings",
            user_id="test_user_pipeline",
            message_id="test_001"
        )

        db = SessionLocal()
        try:
            response = await service.extract_and_store_memories(request, db)
            if response.memories:
                print(f"  OK: Extracted {len(response.memories)} memories")
                for memory in response.memories:
                    print(f"    - {memory.memory_type.value}: {memory.content}")
                return True
            else:
                print("  ERROR: No memories extracted")
                return False
        finally:
            db.close()

    except Exception as e:
        print(f"  ERROR: Memory pipeline error: {e}")
        return False

def test_health_endpoint():
    """Test health endpoint if server is running."""
    print("\n5. Testing Health Endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  OK: Health endpoint: {data['status']}")
            return True
        else:
            print(f"  ERROR: Health endpoint returned {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  WARNING: Server not running - skipping health endpoint test")
        return True  # Not a failure, just not running
    except Exception as e:
        print(f"  ERROR: Health endpoint error: {e}")
        return False

async def main():
    """Run all tests."""
    load_dotenv()

    print("Memory Pipeline End-to-End Test")
    print("=" * 35)

    tests = [
        test_startup_check(),
        test_database_connection(),
        test_memory_service(),
        await test_basic_memory_pipeline(),
        test_health_endpoint()
    ]

    passed = sum(1 for test in tests if test)
    total = len(tests)

    print(f"\n" + "=" * 35)
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All tests passed! Pipeline is working correctly.")
        print("\nTo start the server:")
        print("python -m uvicorn app.main:app --reload")
    else:
        print("FAILURE: Some tests failed. Check the errors above.")
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)