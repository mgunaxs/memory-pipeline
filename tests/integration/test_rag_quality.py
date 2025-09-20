"""
RAG Quality Test Suite

Comprehensive tests to ensure the enhanced RAG system prevents
irrelevant retrieval and maintains high quality for proactive AI.
"""

import asyncio
import sys
import os
from typing import List, Dict

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.core.config import settings, create_directories
from app.core.database_prod import init_database, SessionLocal
from app.services.memory_service import MemoryService
from app.models.schemas import (
    ExtractionRequest, MemorySearchRequest, MemoryType,
    ContextType, MemoryCategory
)
from app.memory.smart_retriever import SmartRetriever
from app.memory.deduplicator import MemoryDeduplicator
from app.memory.validator import MemoryValidator
from app.config.rag_config import RAG_CONFIG

import logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests


class RAGQualityTester:
    """Test suite for RAG quality improvements."""

    def __init__(self):
        """Initialize test suite."""
        self.service = MemoryService()
        self.user_id = "rag_test_user"
        self.test_results = []

    async def setup_test_data(self):
        """Set up comprehensive test data."""
        print("📋 Setting up test data...")

        test_messages = [
            # Work-related memories
            "I'm a software engineer at Microsoft and I have daily standup meetings at 9am",
            "I hate morning meetings and prefer to work in the afternoon",
            "My manager Sarah is really supportive and helps with career growth",

            # Food preferences
            "I'm vegetarian and love Italian food, especially pasta",
            "I can't eat gluten due to celiac disease",
            "I usually skip breakfast and have a big lunch around 1pm",

            # Health and fitness
            "I go to the gym on Tuesdays and Thursdays at 6pm",
            "I have chronic back pain and need to do stretching exercises",
            "I take vitamin D supplements every morning",

            # Social and relationships
            "I'm dating Alex and we've been together for 2 years",
            "I have dinner with my parents every Sunday",
            "My best friend Emma lives in Seattle",

            # Schedule and events
            "I have a dentist appointment next Friday at 2pm",
            "Weekly team meeting every Monday at 10am",
            "Planning a vacation to Japan in March",

            # Emotional states
            "I'm feeling stressed about the upcoming project deadline",
            "Really excited about learning Python programming",
            "Sometimes I feel lonely working from home",

            # Entertainment preferences
            "I love watching sci-fi movies, especially Star Wars",
            "I play guitar in the evenings to relax",
            "I read fantasy novels before bed",
        ]

        # Store all test memories
        for i, message in enumerate(test_messages):
            request = ExtractionRequest(
                text=message,
                user_id=self.user_id,
                message_id=f"test_msg_{i}"
            )

            db = SessionLocal()
            try:
                await self.service.extract_and_store_memories(request, db)
            finally:
                db.close()

        print(f"✅ Created test memories from {len(test_messages)} messages")

    async def test_context_filtering(self):
        """Test 1: Context-aware filtering prevents irrelevant retrieval."""
        print("\n🎯 Test 1: Context-Aware Filtering")
        print("-" * 40)

        test_cases = [
            {
                "name": "Morning checkin shouldn't retrieve entertainment",
                "query": "morning routine",
                "context": ContextType.MORNING_CHECKIN,
                "should_exclude_categories": [MemoryCategory.ENTERTAINMENT, MemoryCategory.SOCIAL],
                "should_include_categories": [MemoryCategory.SCHEDULE, MemoryCategory.HEALTH]
            },
            {
                "name": "Food query shouldn't retrieve work stress",
                "query": "what to eat",
                "context": ContextType.MEAL_SUGGESTION,
                "should_exclude_categories": [MemoryCategory.WORK, MemoryCategory.EMOTIONAL],
                "should_include_categories": [MemoryCategory.FOOD, MemoryCategory.HEALTH]
            },
            {
                "name": "Weekend planning shouldn't include work meetings",
                "query": "weekend plans",
                "context": ContextType.WEEKEND_PLANNING,
                "should_exclude_categories": [MemoryCategory.WORK],
                "should_include_categories": [MemoryCategory.SOCIAL, MemoryCategory.ENTERTAINMENT]
            }
        ]

        passed_tests = 0
        for test_case in test_cases:
            try:
                # Use smart retriever with context
                smart_retriever = SmartRetriever(
                    self.service.retriever,
                    self.service.embedding_service
                )

                results = await smart_retriever.retrieve_memories(
                    query=test_case["query"],
                    user_id=self.user_id,
                    context_type=test_case["context"],
                    max_memories=10
                )

                # Analyze results
                excluded_found = []
                included_missing = []

                for result in results:
                    # Extract category from memory (this would need to be implemented)
                    memory_category = self._extract_category_from_memory(result.memory)

                    if memory_category in test_case["should_exclude_categories"]:
                        excluded_found.append(result.memory.content)

                # Check if we have appropriate memories
                has_appropriate = any(
                    self._extract_category_from_memory(result.memory) in test_case["should_include_categories"]
                    for result in results
                )

                success = len(excluded_found) == 0 and has_appropriate

                print(f"   {'✅' if success else '❌'} {test_case['name']}")
                if not success:
                    print(f"      Excluded found: {excluded_found}")
                    print(f"      Has appropriate: {has_appropriate}")

                if success:
                    passed_tests += 1

            except Exception as e:
                print(f"   ❌ {test_case['name']} - Error: {e}")

        self.test_results.append(("Context Filtering", passed_tests, len(test_cases)))
        return passed_tests == len(test_cases)

    async def test_relevance_thresholds(self):
        """Test 2: Relevance thresholds prevent low-quality matches."""
        print("\n📊 Test 2: Relevance Threshold Filtering")
        print("-" * 45)

        test_queries = [
            ("specific work preference", 0.8, "Should find work-related memories with high confidence"),
            ("random unrelated query xyz", 0.75, "Should return few or no results for random query"),
            ("food Italian pasta", 0.7, "Should find food preferences"),
        ]

        passed_tests = 0
        for query, min_threshold, description in test_queries:
            try:
                search_request = MemorySearchRequest(
                    query=query,
                    user_id=self.user_id,
                    limit=10
                )

                db = SessionLocal()
                try:
                    response = await self.service.search_memories(search_request, db)
                finally:
                    db.close()

                # Check that all results meet threshold
                high_quality_results = [
                    r for r in response.results
                    if r.similarity_score >= min_threshold
                ]

                # For random query, should have few results
                if "random" in query:
                    success = len(high_quality_results) <= 1
                    print(f"   {'✅' if success else '❌'} {description}")
                    print(f"      Found {len(high_quality_results)} high-quality results (expected ≤1)")
                else:
                    success = len(high_quality_results) > 0
                    print(f"   {'✅' if success else '❌'} {description}")
                    print(f"      Found {len(high_quality_results)} results above {min_threshold}")

                if success:
                    passed_tests += 1

            except Exception as e:
                print(f"   ❌ {description} - Error: {e}")

        self.test_results.append(("Relevance Thresholds", passed_tests, len(test_queries)))
        return passed_tests == len(test_queries)

    async def test_memory_deduplication(self):
        """Test 3: Deduplication prevents duplicate memories."""
        print("\n🔄 Test 3: Memory Deduplication")
        print("-" * 35)

        test_cases = [
            {
                "original": "I have a meeting tomorrow at 3pm",
                "duplicate": "I have a meeting tomorrow at 3pm",
                "expected_action": "duplicate"
            },
            {
                "original": "I have a meeting tomorrow at 3pm",
                "update": "My meeting got moved to 4pm tomorrow",
                "expected_action": "update"
            },
            {
                "original": "I love pizza",
                "similar": "I really enjoy Italian food",
                "expected_action": "new"  # Similar but different enough
            }
        ]

        passed_tests = 0
        deduplicator = MemoryDeduplicator(self.service.embedding_service)

        for i, test_case in enumerate(test_cases):
            try:
                # First, store the original
                original_request = ExtractionRequest(
                    text=test_case["original"],
                    user_id=f"dedup_test_user_{i}",
                    message_id=f"original_{i}"
                )

                db = SessionLocal()
                try:
                    await self.service.extract_and_store_memories(original_request, db)

                    # Then test the duplicate/update
                    test_key = "duplicate" if "duplicate" in test_case else ("update" if "update" in test_case else "similar")
                    test_text = test_case[test_key]

                    # Extract the new memory
                    test_request = ExtractionRequest(
                        text=test_text,
                        user_id=f"dedup_test_user_{i}",
                        message_id=f"test_{i}"
                    )

                    # Get extracted memories
                    extracted = await self.service.extractor.extract_memories(
                        test_text, f"dedup_test_user_{i}", f"test_{i}"
                    )

                    if extracted:
                        # Test deduplication logic
                        existing, action = await deduplicator.check_for_duplicates(
                            extracted[0], f"dedup_test_user_{i}", db
                        )

                        expected = test_case["expected_action"]
                        success = action == expected

                        print(f"   {'✅' if success else '❌'} Test case {i+1}: Expected '{expected}', got '{action}'")

                        if success:
                            passed_tests += 1

                finally:
                    db.close()

            except Exception as e:
                print(f"   ❌ Test case {i+1} - Error: {e}")

        self.test_results.append(("Deduplication", passed_tests, len(test_cases)))
        return passed_tests == len(test_cases)

    async def test_token_budget_management(self):
        """Test 4: Token budget prevents overflow."""
        print("\n💰 Test 4: Token Budget Management")
        print("-" * 38)

        test_cases = [
            {"budget": 200, "max_memories": 5, "description": "Small budget should limit results"},
            {"budget": 1000, "max_memories": 10, "description": "Large budget allows more results"},
        ]

        passed_tests = 0
        for test_case in test_cases:
            try:
                smart_retriever = SmartRetriever(
                    self.service.retriever,
                    self.service.embedding_service
                )

                results = await smart_retriever.retrieve_memories(
                    query="tell me everything about user preferences",
                    user_id=self.user_id,
                    token_budget=test_case["budget"],
                    max_memories=test_case["max_memories"]
                )

                # Estimate token usage
                estimated_tokens = smart_retriever.estimate_token_usage(results)

                success = estimated_tokens <= test_case["budget"]

                print(f"   {'✅' if success else '❌'} {test_case['description']}")
                print(f"      Budget: {test_case['budget']}, Used: {estimated_tokens}, Results: {len(results)}")

                if success:
                    passed_tests += 1

            except Exception as e:
                print(f"   ❌ {test_case['description']} - Error: {e}")

        self.test_results.append(("Token Budget", passed_tests, len(test_cases)))
        return passed_tests == len(test_cases)

    async def test_empty_retrieval_handling(self):
        """Test 5: Graceful handling of no results."""
        print("\n🔍 Test 5: Empty Retrieval Handling")
        print("-" * 38)

        empty_queries = [
            "quantum physics advanced mathematics",
            "completely unrelated topic xyz",
            "random words that mean nothing"
        ]

        passed_tests = 0
        for query in empty_queries:
            try:
                search_request = MemorySearchRequest(
                    query=query,
                    user_id=self.user_id,
                    limit=5
                )

                db = SessionLocal()
                try:
                    response = await self.service.search_memories(search_request, db)
                finally:
                    db.close()

                # Should return empty results, not crash
                success = len(response.results) == 0

                print(f"   {'✅' if success else '❌'} Query: '{query}' → {len(response.results)} results")

                if success:
                    passed_tests += 1

            except Exception as e:
                print(f"   ❌ Query '{query}' - Error: {e}")

        self.test_results.append(("Empty Retrieval", passed_tests, len(empty_queries)))
        return passed_tests == len(empty_queries)

    def _extract_category_from_memory(self, memory) -> MemoryCategory:
        """Extract category from memory (simplified)."""
        content_lower = memory.content.lower()

        if any(word in content_lower for word in ["work", "job", "meeting", "manager"]):
            return MemoryCategory.WORK
        elif any(word in content_lower for word in ["food", "eat", "vegetarian", "gluten"]):
            return MemoryCategory.FOOD
        elif any(word in content_lower for word in ["gym", "exercise", "health", "vitamin"]):
            return MemoryCategory.HEALTH
        elif any(word in content_lower for word in ["friend", "family", "dating", "dinner"]):
            return MemoryCategory.SOCIAL
        elif any(word in content_lower for word in ["appointment", "meeting", "vacation"]):
            return MemoryCategory.SCHEDULE
        elif any(word in content_lower for word in ["movie", "guitar", "read", "sci-fi"]):
            return MemoryCategory.ENTERTAINMENT
        elif any(word in content_lower for word in ["stress", "excited", "lonely", "feeling"]):
            return MemoryCategory.EMOTIONAL
        else:
            return MemoryCategory.SOCIAL  # Default

    async def test_rich_metadata_extraction(self):
        """Test 6: Rich metadata extraction works correctly."""
        print("\n📋 Test 6: Rich Metadata Extraction")
        print("-" * 40)

        test_message = "I'm a software engineer at Microsoft and I hate morning meetings but love afternoon coding sessions"

        try:
            request = ExtractionRequest(
                text=test_message,
                user_id="metadata_test_user",
                message_id="metadata_test"
            )

            db = SessionLocal()
            try:
                response = await self.service.extract_and_store_memories(request, db)

                # Check if memories have rich metadata
                success = True
                metadata_checks = []

                for memory in response.memories:
                    # Check if memory has required fields
                    has_category = hasattr(memory, 'category') and memory.category
                    has_entities = hasattr(memory, 'entities')
                    has_contexts = hasattr(memory, 'valid_contexts') or hasattr(memory, 'invalid_contexts')

                    metadata_checks.append({
                        "content": memory.content[:50],
                        "has_category": has_category,
                        "has_entities": has_entities,
                        "has_contexts": has_contexts
                    })

                    if not (has_category and has_entities):
                        success = False

                print(f"   {'✅' if success else '❌'} Rich metadata extraction")
                for check in metadata_checks:
                    print(f"      {check['content']}...")
                    print(f"        Category: {check['has_category']}, Entities: {check['has_entities']}, Contexts: {check['has_contexts']}")

                self.test_results.append(("Rich Metadata", 1 if success else 0, 1))
                return success

            finally:
                db.close()

        except Exception as e:
            print(f"   ❌ Rich metadata test failed: {e}")
            self.test_results.append(("Rich Metadata", 0, 1))
            return False

    async def run_all_tests(self):
        """Run all RAG quality tests."""
        print("🧪 RAG Quality Test Suite")
        print("=" * 50)

        # Initialize system
        create_directories()
        init_database()

        # Setup test data
        await self.setup_test_data()

        # Run all tests
        test_methods = [
            self.test_rich_metadata_extraction,
            self.test_context_filtering,
            self.test_relevance_thresholds,
            self.test_memory_deduplication,
            self.test_token_budget_management,
            self.test_empty_retrieval_handling,
        ]

        total_passed = 0
        total_tests = 0

        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                print(f"❌ Test {test_method.__name__} failed with error: {e}")

        # Summary
        print("\n📊 TEST SUMMARY")
        print("=" * 30)

        for test_name, passed, total in self.test_results:
            percentage = (passed / total * 100) if total > 0 else 0
            print(f"{test_name:.<25} {passed}/{total} ({percentage:.0f}%)")
            total_passed += passed
            total_tests += total

        overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
        print(f"{'OVERALL':.<25} {total_passed}/{total_tests} ({overall_percentage:.0f}%)")

        if overall_percentage >= 80:
            print("\n🎉 RAG QUALITY EXCELLENT! Ready for production.")
        elif overall_percentage >= 60:
            print("\n⚠️  RAG quality good but needs improvement.")
        else:
            print("\n🚨 RAG quality poor - needs significant work.")

        return overall_percentage >= 80


async def main():
    """Run RAG quality tests."""
    tester = RAGQualityTester()
    success = await tester.run_all_tests()
    return success


if __name__ == "__main__":
    print("RAG Quality Test Suite - Enhanced Memory Pipeline")
    print("This tests the critical improvements for proactive AI quality.")
    print()

    # Run tests
    success = asyncio.run(main())

    # Exit with appropriate code
    sys.exit(0 if success else 1)