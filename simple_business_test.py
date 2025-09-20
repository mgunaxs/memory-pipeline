#!/usr/bin/env python3
"""
Simple Business Validation Test
Quick test to validate core functionality before full harness
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.core.database_prod import SessionLocal, init_database
from app.services.memory_service import MemoryService
from app.models.schemas import ExtractionRequest, MemorySearchRequest


async def quick_business_test():
    """Run a quick business validation test."""
    print("MEMORY PIPELINE BUSINESS VALIDATION - QUICK TEST")
    print("=" * 55)

    # Initialize
    init_database()
    service = MemoryService()

    # Test conversations from different user types
    test_scenarios = {
        "startup_founder": [
            "Just closed our Series A for $10M. Exhausted but excited.",
            "Meeting with Google tomorrow at 2pm about acquisition",
            "I can't do morning meetings anymore, too many late nights",
            "Board meeting next Thursday, need revenue projections"
        ],
        "busy_parent": [
            "Emma is allergic to peanuts, this is critical to remember",
            "Kids have soccer practice every Tuesday and Thursday 5pm",
            "Date night with husband this Saturday, need babysitter",
            "Parent teacher conference moved to Monday 3pm"
        ],
        "remote_worker": [
            "Living in Bali this month, wifi is terrible",
            "Daily standup at 9am PST, that's midnight here",
            "Missing my cat back home while traveling",
            "Client presentation next week, nervous about connection"
        ]
    }

    results = {}

    for scenario_name, conversations in test_scenarios.items():
        print(f"\nTesting: {scenario_name.replace('_', ' ').title()}")
        print("-" * 30)

        user_id = f"test_{scenario_name}_user"
        extracted_count = 0
        start_time = time.time()

        # Extract memories
        for i, conversation in enumerate(conversations):
            try:
                request = ExtractionRequest(
                    text=conversation,
                    user_id=user_id,
                    message_id=f"{scenario_name}_{i}"
                )

                db = SessionLocal()
                try:
                    response = await service.extract_and_store_memories(request, db)
                    extracted_count += len(response.memories)
                    print(f"  Processed: {conversation[:50]}... -> {len(response.memories)} memories")
                finally:
                    db.close()

            except Exception as e:
                print(f"  ERROR: {conversation[:30]}... -> {str(e)}")

        # Test retrieval
        print(f"\n  Total memories extracted: {extracted_count}")

        # Test search
        test_queries = [
            "important meetings and appointments",
            "food preferences and allergies",
            "work schedule and routines"
        ]

        for query in test_queries:
            try:
                search_request = MemorySearchRequest(
                    query=query,
                    user_id=user_id,
                    limit=3
                )

                db = SessionLocal()
                try:
                    search_response = await service.search_memories(search_request, db)
                    print(f"  Search '{query}': {len(search_response.results)} results")
                finally:
                    db.close()

            except Exception as e:
                print(f"  Search ERROR for '{query}': {str(e)}")

        processing_time = time.time() - start_time
        print(f"  Processing time: {processing_time:.2f}s")

        results[scenario_name] = {
            "messages": len(conversations),
            "memories_extracted": extracted_count,
            "processing_time": processing_time,
            "extraction_rate": extracted_count / len(conversations)
        }

    # Summary
    print("\n" + "=" * 55)
    print("BUSINESS VALIDATION SUMMARY")
    print("=" * 55)

    total_messages = sum(r["messages"] for r in results.values())
    total_memories = sum(r["memories_extracted"] for r in results.values())
    avg_extraction_rate = sum(r["extraction_rate"] for r in results.values()) / len(results)

    print(f"Total messages processed: {total_messages}")
    print(f"Total memories extracted: {total_memories}")
    print(f"Average extraction rate: {avg_extraction_rate:.2f} memories/message")

    # Business assessment
    if avg_extraction_rate >= 1.5:
        print("ASSESSMENT: GOOD extraction rate for business use")
    elif avg_extraction_rate >= 1.0:
        print("ASSESSMENT: ACCEPTABLE extraction rate")
    else:
        print("ASSESSMENT: LOW extraction rate, needs improvement")

    return results


if __name__ == "__main__":
    asyncio.run(quick_business_test())