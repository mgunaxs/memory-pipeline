"""
Usage Examples for Memory Pipeline

This file demonstrates how to use the Memory Pipeline components
in various scenarios and integration patterns.
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.config import settings, create_directories
from app.core.database_prod import init_database, SessionLocal
from app.services.memory_service import MemoryService
from app.models.schemas import (
    ExtractionRequest, MemorySearchRequest, MemoryType
)


class MemoryPipelineExamples:
    """Examples demonstrating Memory Pipeline usage patterns."""

    def __init__(self):
        """Initialize examples with service setup."""
        self.service = MemoryService()

    async def basic_extraction_example(self):
        """
        Example 1: Basic Memory Extraction

        Shows how to extract memories from user text.
        """
        print("Example 1: Basic Memory Extraction")
        print("-" * 40)

        # Sample user message
        user_message = "I'm a data scientist at OpenAI and I love working with Python. I have a team meeting every Tuesday at 10am."

        # Create extraction request
        request = ExtractionRequest(
            text=user_message,
            user_id="example_user_1",
            message_id="msg_001"
        )

        # Process the message
        db = SessionLocal()
        try:
            response = await self.service.extract_and_store_memories(request, db)

            print(f"Original message: {user_message}")
            print(f"Extracted {response.total_extracted} memories:")

            for memory in response.memories:
                print(f"  • Type: {memory.memory_type.value}")
                print(f"    Content: {memory.content}")
                print(f"    Confidence: {memory.confidence_score:.2f}")
                print(f"    Importance: {memory.importance_score:.1f}")
                print()

        finally:
            db.close()

    async def semantic_search_example(self):
        """
        Example 2: Semantic Search

        Shows how to search for relevant memories using natural language.
        """
        print("Example 2: Semantic Search")
        print("-" * 30)

        search_queries = [
            "work and job information",
            "programming languages and technologies",
            "meeting schedules and recurring events"
        ]

        for query in search_queries:
            print(f"Searching for: '{query}'")

            request = MemorySearchRequest(
                query=query,
                user_id="example_user_1",
                limit=3
            )

            db = SessionLocal()
            try:
                response = await self.service.search_memories(request, db)

                for result in response.results:
                    print(f"  • {result.memory.content}")
                    print(f"    Similarity: {result.similarity_score:.3f}")
                    print(f"    Type: {result.memory.memory_type.value}")

                print()

            finally:
                db.close()

    async def chatbot_integration_example(self):
        """
        Example 3: Chatbot Integration Pattern

        Shows how to integrate memory pipeline into a chatbot.
        """
        print("Example 3: Chatbot Integration")
        print("-" * 35)

        # Simulate a conversation
        conversation = [
            "Hi! I'm Sarah, a marketing manager at Spotify.",
            "I love listening to jazz music and playing guitar in my free time.",
            "I have a presentation tomorrow at 2pm about our new campaign.",
            "What do you know about me so far?"
        ]

        user_id = "chatbot_user_sarah"

        for i, message in enumerate(conversation[:-1], 1):
            print(f"User: {message}")

            # Extract and store memories
            request = ExtractionRequest(
                text=message,
                user_id=user_id,
                message_id=f"chat_msg_{i}"
            )

            db = SessionLocal()
            try:
                response = await self.service.extract_and_store_memories(request, db)
                print(f"Bot: Learned {response.total_extracted} new things about you!")

            finally:
                db.close()

            print()

        # Now answer the final question using memory search
        print(f"User: {conversation[-1]}")

        # Search for all user information
        search_request = MemorySearchRequest(
            query="everything about the user",
            user_id=user_id,
            limit=10
        )

        db = SessionLocal()
        try:
            search_response = await self.service.search_memories(search_request, db)

            print("Bot: Here's what I know about you:")
            for result in search_response.results:
                print(f"  • {result.memory.content}")

        finally:
            db.close()

    async def memory_type_filtering_example(self):
        """
        Example 4: Memory Type Filtering

        Shows how to filter memories by type for different use cases.
        """
        print("Example 4: Memory Type Filtering")
        print("-" * 35)

        user_id = "filter_example_user"

        # First, create some diverse memories
        sample_messages = [
            "I work as a software engineer at Microsoft",  # fact
            "I hate morning meetings and love afternoon coffee breaks",  # preferences
            "I have a doctor appointment next Friday at 3pm",  # event
            "I go to the gym every Monday, Wednesday, and Friday",  # routine
            "I'm feeling excited about my upcoming vacation"  # emotion
        ]

        # Store the memories
        for i, message in enumerate(sample_messages):
            request = ExtractionRequest(
                text=message,
                user_id=user_id,
                message_id=f"filter_msg_{i}"
            )

            db = SessionLocal()
            try:
                await self.service.extract_and_store_memories(request, db)
            finally:
                db.close()

        # Now demonstrate filtering by type
        memory_types = [MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.EVENT]

        for memory_type in memory_types:
            print(f"\n{memory_type.value.upper()} memories:")

            db = SessionLocal()
            try:
                memories = self.service.get_user_memories(
                    user_id=user_id,
                    db=db,
                    memory_type=memory_type,
                    limit=10
                )

                for memory in memories:
                    print(f"  • {memory.content}")

            finally:
                db.close()

    async def proactive_reminder_example(self):
        """
        Example 5: Proactive Reminder System

        Shows how to build a proactive reminder system using memory search.
        """
        print("Example 5: Proactive Reminder System")
        print("-" * 40)

        user_id = "reminder_user"

        # User mentions some events
        event_messages = [
            "I have a job interview on Thursday at 10am with TechCorp",
            "My mom's birthday is this weekend, I need to buy a gift",
            "I'm meeting my friend Alex for lunch tomorrow at noon"
        ]

        # Store the events
        for i, message in enumerate(event_messages):
            request = ExtractionRequest(
                text=message,
                user_id=user_id,
                message_id=f"event_msg_{i}"
            )

            db = SessionLocal()
            try:
                await self.service.extract_and_store_memories(request, db)
            finally:
                db.close()

        # Simulate checking for upcoming events
        print("\nChecking for upcoming events and important tasks...")

        # Search for events and time-sensitive information
        time_queries = [
            "tomorrow schedule appointments meetings",
            "upcoming events this week",
            "things to remember and do"
        ]

        for query in time_queries:
            search_request = MemorySearchRequest(
                query=query,
                user_id=user_id,
                memory_types=[MemoryType.EVENT],
                limit=5
            )

            db = SessionLocal()
            try:
                response = await self.service.search_memories(search_request, db)

                if response.results:
                    print(f"\nProactive reminder for '{query}':")
                    for result in response.results:
                        print(f"  🔔 {result.memory.content}")
                        print(f"     (Relevance: {result.similarity_score:.2f})")

            finally:
                db.close()

    async def analytics_and_insights_example(self):
        """
        Example 6: User Analytics and Insights

        Shows how to analyze user memory patterns for insights.
        """
        print("Example 6: User Analytics and Insights")
        print("-" * 40)

        user_id = "analytics_user"

        # Create a rich set of memories
        analysis_messages = [
            "I'm a software engineer who loves Python and AI",
            "I prefer working from home and dislike open offices",
            "I have weekly team meetings on Mondays at 9am",
            "I'm learning Spanish and practice every evening",
            "I feel most productive in the morning hours",
            "I have a gym membership but rarely use it",
            "My favorite cuisine is Italian, especially pasta dishes"
        ]

        # Store all memories
        for i, message in enumerate(analysis_messages):
            request = ExtractionRequest(
                text=message,
                user_id=user_id,
                message_id=f"analytics_msg_{i}"
            )

            db = SessionLocal()
            try:
                await self.service.extract_and_store_memories(request, db)
            finally:
                db.close()

        # Get comprehensive user statistics
        db = SessionLocal()
        try:
            stats = self.service.get_user_stats(user_id, db)

            print("User Memory Analysis:")
            print(f"  Total memories: {stats.get('total_memories', 0)}")
            print(f"  Active memories: {stats.get('active_memories', 0)}")

            vector_stats = stats.get('vector_stats', {})
            if vector_stats.get('type_distribution'):
                print("\n  Memory type distribution:")
                for mem_type, count in vector_stats['type_distribution'].items():
                    print(f"    {mem_type}: {count}")

            print(f"\n  Average importance: {vector_stats.get('average_importance', 0):.1f}")

        finally:
            db.close()

        # Search for specific patterns
        insight_queries = [
            "work preferences and habits",
            "food and eating preferences",
            "learning and personal development",
            "physical activities and health"
        ]

        print("\nPersonality insights based on memories:")
        for query in insight_queries:
            search_request = MemorySearchRequest(
                query=query,
                user_id=user_id,
                limit=3
            )

            db = SessionLocal()
            try:
                response = await self.service.search_memories(search_request, db)

                if response.results:
                    print(f"\n  {query.title()}:")
                    for result in response.results:
                        print(f"    • {result.memory.content}")

            finally:
                db.close()

    async def error_handling_example(self):
        """
        Example 7: Error Handling and Edge Cases

        Shows how the system handles various error conditions gracefully.
        """
        print("Example 7: Error Handling and Edge Cases")
        print("-" * 45)

        # Test empty input
        try:
            request = ExtractionRequest(
                text="",
                user_id="error_test_user"
            )
            print("Testing empty input...")
            # This should raise a validation error
        except Exception as e:
            print(f"✅ Properly caught empty input: {type(e).__name__}")

        # Test very long input
        try:
            long_text = "This is a test. " * 1000  # Very long text
            request = ExtractionRequest(
                text=long_text,
                user_id="error_test_user"
            )
            print("Testing very long input...")
            # This should be handled gracefully
        except Exception as e:
            print(f"✅ Properly handled long input: {type(e).__name__}")

        # Test search with no results
        search_request = MemorySearchRequest(
            query="completely unrelated quantum physics advanced mathematics",
            user_id="nonexistent_user",
            limit=5
        )

        db = SessionLocal()
        try:
            response = await self.service.search_memories(search_request, db)
            print(f"✅ Graceful handling of no results: {len(response.results)} results found")

        finally:
            db.close()

    async def run_all_examples(self):
        """Run all examples in sequence."""
        # Initialize system
        create_directories()
        init_database()

        examples = [
            self.basic_extraction_example,
            self.semantic_search_example,
            self.chatbot_integration_example,
            self.memory_type_filtering_example,
            self.proactive_reminder_example,
            self.analytics_and_insights_example,
            self.error_handling_example
        ]

        for example in examples:
            try:
                await example()
                print("\n" + "="*60 + "\n")
            except Exception as e:
                print(f"Example failed: {e}")
                continue

        print("🎉 All examples completed!")
        print("\nThese examples show how to:")
        print("  • Extract and store memories from text")
        print("  • Search memories semantically")
        print("  • Integrate with chatbots and AI assistants")
        print("  • Filter memories by type")
        print("  • Build proactive reminder systems")
        print("  • Analyze user patterns and insights")
        print("  • Handle errors gracefully")


async def main():
    """Run the examples."""
    examples = MemoryPipelineExamples()
    await examples.run_all_examples()


if __name__ == "__main__":
    print("Memory Pipeline - Usage Examples")
    print("This will demonstrate various integration patterns and use cases.")
    print()

    # Run examples
    asyncio.run(main())