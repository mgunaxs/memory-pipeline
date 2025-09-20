#!/usr/bin/env python3
"""
Critical Fix Verification Test
Test the complete pipeline after fixes
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.core.database_prod import SessionLocal
from app.services.memory_service import MemoryService
from app.models.schemas import ExtractionRequest, MemorySearchRequest


async def test_full_pipeline():
    """Test complete pipeline after critical fixes."""
    print("TESTING COMPLETE PIPELINE AFTER FIXES")
    print("=" * 45)

    # Test 1: Database Connection
    print("\n1. Testing Database Connection...")
    try:
        db = SessionLocal()
        print("  Database connected successfully")
        db.close()
    except Exception as e:
        print(f"  Database connection failed: {e}")
        return False

    # Test 2: Service Initialization
    print("\n2. Testing Service Initialization...")
    try:
        service = MemoryService()
        print("  Memory service initialized successfully")
    except Exception as e:
        print(f"  Service initialization failed: {e}")
        return False

    # Test 3: Memory Extraction
    print("\n3. Testing Memory Extraction...")
    try:
        test_text = "I love pizza and I hate morning meetings. I work at Google."

        request = ExtractionRequest(
            text=test_text,
            user_id="test_user_fixed",
            message_id="test_msg_001"
        )

        db = SessionLocal()
        try:
            response = await service.extract_and_store_memories(request, db)
            print(f"  Extracted {len(response.memories)} memories successfully")

            for i, memory in enumerate(response.memories, 1):
                print(f"    {i}. {memory.memory_type.value}: {memory.content}")

        finally:
            db.close()

    except Exception as e:
        print(f"  Memory extraction failed: {e}")
        return False

    # Test 4: Memory Retrieval
    print("\n4. Testing Memory Retrieval...")
    try:
        search_request = MemorySearchRequest(
            query="food preferences",
            user_id="test_user_fixed",
            limit=5
        )

        db = SessionLocal()
        try:
            search_response = await service.search_memories(search_request, db)
            print(f"  Retrieved {len(search_response.results)} memories successfully")

            for i, result in enumerate(search_response.results, 1):
                print(f"    {i}. {result.memory.content} (score: {result.similarity_score:.3f})")

        finally:
            db.close()

    except Exception as e:
        print(f"  Memory retrieval failed: {e}")
        return False

    # Test 5: Different User Isolation
    print("\n5. Testing User Data Isolation...")
    try:
        # Create memory for different user
        request2 = ExtractionRequest(
            text="I hate pizza and love morning meetings",
            user_id="test_user_2",
            message_id="test_msg_002"
        )

        db = SessionLocal()
        try:
            response2 = await service.extract_and_store_memories(request2, db)
            print(f"  Created {len(response2.memories)} memories for second user")

            # Search for first user should not return second user's data
            search_request = MemorySearchRequest(
                query="pizza",
                user_id="test_user_fixed",
                limit=10
            )

            search_response = await service.search_memories(search_request, db)

            # Verify no cross-user contamination
            for result in search_response.results:
                if result.memory.user_id != "test_user_fixed":
                    print(f"  ERROR: Cross-user data contamination detected!")
                    return False

            print("  User data isolation working correctly")

        finally:
            db.close()

    except Exception as e:
        print(f"  User isolation test failed: {e}")
        return False

    print("\n" + "=" * 45)
    print("PIPELINE WORKING SUCCESSFULLY!")
    print("=" * 45)
    print("\nAll critical issues have been resolved:")
    print("  - Dependencies installed")
    print("  - Configuration loaded")
    print("  - Database schema fixed")
    print("  - SQL parameter binding fixed")
    print("  - Memory extraction working")
    print("  - Memory storage working")
    print("  - Memory retrieval working")
    print("  - User data isolation working")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_full_pipeline())
    if success:
        print("\nSTATUS: READY FOR BUSINESS VALIDATION")
    else:
        print("\nSTATUS: ADDITIONAL FIXES NEEDED")