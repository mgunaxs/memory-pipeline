"""
Quick Start Demo Script

Demonstrates the complete Memory Pipeline functionality with real examples.
Run this after successful basic tests to see the system in action.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.config import settings, create_directories
from app.core.database_prod import init_database, SessionLocal
from app.services.memory_service import MemoryService
from app.models.schemas import ExtractionRequest, MemorySearchRequest, MemoryType

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for demo


class MemoryPipelineDemo:
    """Demo class for Memory Pipeline functionality."""

    def __init__(self):
        """Initialize the demo."""
        self.service = None
        self.user_id = "demo_user"

    async def setup(self):
        """Set up the demo environment."""
        print("🚀 Memory Pipeline Quick Start Demo")
        print("=" * 50)

        # Create directories and initialize database
        create_directories()
        init_success = init_database()
        if not init_success:
            raise Exception("Failed to initialize database")

        # Create service
        self.service = MemoryService()
        print("✅ Memory Pipeline initialized successfully\n")

    async def demo_extraction(self):
        """Demonstrate memory extraction from various text types."""
        print("🧠 MEMORY EXTRACTION DEMO")
        print("-" * 30)

        demo_messages = [
            "I'm a software engineer at Microsoft and I hate morning meetings",
            "I love Thai food and really enjoy cooking on weekends",
            "I have a dentist appointment next Friday at 2pm",
            "I usually go to the gym on Tuesdays and Thursdays",
            "I'm feeling really stressed about the upcoming project deadline",
            "My birthday is on March 15th and I'm planning a big party"
        ]

        all_memories = []

        for i, message in enumerate(demo_messages, 1):
            print(f"\n📝 Processing message {i}: \"{message}\"")

            request = ExtractionRequest(
                text=message,
                user_id=self.user_id,
                message_id=f"demo_msg_{i}"
            )

            db = SessionLocal()
            try:
                response = await self.service.extract_and_store_memories(request, db)

                print(f"   ⏱️  Processing time: {response.processing_time_ms:.1f}ms")
                print(f"   📊 Extracted {response.total_extracted} memories:")

                for memory in response.memories:
                    print(f"      • [{memory.memory_type.value.upper()}] {memory.content}")
                    print(f"        Confidence: {memory.confidence_score:.2f} | Importance: {memory.importance_score:.1f}")

                all_memories.extend(response.memories)

            finally:
                db.close()

        print(f"\n✅ Total memories extracted: {len(all_memories)}")
        return all_memories

    async def demo_search(self):
        """Demonstrate semantic search functionality."""
        print("\n\n🔍 SEMANTIC SEARCH DEMO")
        print("-" * 30)

        search_queries = [
            "work and job preferences",
            "food and eating habits",
            "upcoming events and appointments",
            "exercise and fitness routine",
            "emotional state and feelings"
        ]

        for query in search_queries:
            print(f"\n🔎 Searching for: \"{query}\"")

            request = MemorySearchRequest(
                query=query,
                user_id=self.user_id,
                limit=3
            )

            db = SessionLocal()
            try:
                response = await self.service.search_memories(request, db)

                print(f"   ⏱️  Search time: {response.search_time_ms:.1f}ms")
                print(f"   📊 Found {response.total_found} relevant memories:")

                for i, result in enumerate(response.results, 1):
                    print(f"      {i}. [{result.memory.memory_type.value.upper()}] {result.memory.content}")
                    print(f"         Similarity: {result.similarity_score:.3f}")

            finally:
                db.close()

    async def demo_conversation_processing(self):
        """Demonstrate end-to-end conversation processing."""
        print("\n\n💬 CONVERSATION PROCESSING DEMO")
        print("-" * 35)

        conversation = "I just got promoted to Senior Engineer at Google! I'm excited but also nervous about the new responsibilities. I need to remember to schedule a celebration dinner this weekend."

        print(f"📝 Processing conversation: \"{conversation}\"")

        request = ExtractionRequest(
            text=conversation,
            user_id=self.user_id,
            message_id="conversation_demo"
        )

        db = SessionLocal()
        try:
            # Use the end-to-end processing method
            response = await self.service.extract_and_store_memories(request, db)

            print(f"\n📊 Extraction Results:")
            print(f"   ⏱️  Processing time: {response.processing_time_ms:.1f}ms")
            print(f"   🧠 New memories extracted: {response.total_extracted}")

            for memory in response.memories:
                print(f"      • [{memory.memory_type.value.upper()}] {memory.content}")

            # Now search for relevant context
            search_request = MemorySearchRequest(
                query="work career promotion",
                user_id=self.user_id,
                limit=5
            )

            search_response = await self.service.search_memories(search_request, db)

            print(f"\n🔍 Relevant Context Found:")
            print(f"   📊 {search_response.total_found} relevant memories:")

            for result in search_response.results:
                if result.memory.memory_id not in [m.memory_id for m in response.memories]:
                    print(f"      • [{result.memory.memory_type.value.upper()}] {result.memory.content}")
                    print(f"        Relevance: {result.similarity_score:.3f}")

        finally:
            db.close()

    async def demo_user_stats(self):
        """Demonstrate user statistics and analytics."""
        print("\n\n📈 USER ANALYTICS DEMO")
        print("-" * 25)

        db = SessionLocal()
        try:
            stats = self.service.get_user_stats(self.user_id, db)

            print(f"👤 User: {self.user_id}")
            print(f"📊 Total memories: {stats.get('total_memories', 0)}")
            print(f"✅ Active memories: {stats.get('active_memories', 0)}")

            vector_stats = stats.get('vector_stats', {})
            if vector_stats:
                print(f"🗃️  Vector database:")
                print(f"   • Collection: {vector_stats.get('collection_name', 'unknown')}")
                print(f"   • Average importance: {vector_stats.get('average_importance', 0):.1f}")

                type_dist = vector_stats.get('type_distribution', {})
                if type_dist:
                    print(f"   • Memory types:")
                    for mem_type, count in type_dist.items():
                        print(f"     - {mem_type}: {count}")

        finally:
            db.close()

    async def demo_type_filtering(self):
        """Demonstrate filtering by memory types."""
        print("\n\n🏷️  MEMORY TYPE FILTERING DEMO")
        print("-" * 35)

        memory_types = [MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.EVENT]

        for memory_type in memory_types:
            print(f"\n📋 {memory_type.value.upper()} memories:")

            db = SessionLocal()
            try:
                memories = self.service.get_user_memories(
                    user_id=self.user_id,
                    db=db,
                    limit=10,
                    memory_type=memory_type
                )

                if memories:
                    for memory in memories:
                        print(f"   • {memory.content}")
                        print(f"     Importance: {memory.importance_score:.1f} | Confidence: {memory.confidence_score:.2f}")
                else:
                    print(f"   No {memory_type.value} memories found")

            finally:
                db.close()

    async def run_demo(self):
        """Run the complete demo."""
        try:
            await self.setup()

            # Run all demo sections
            await self.demo_extraction()
            await self.demo_search()
            await self.demo_conversation_processing()
            await self.demo_user_stats()
            await self.demo_type_filtering()

            print("\n\n🎉 DEMO COMPLETE!")
            print("=" * 50)
            print("✅ Memory Pipeline is working perfectly!")
            print("\n🚀 Next Steps:")
            print("   • Start the API server: python -m uvicorn app.main:app --reload")
            print("   • Access the docs: http://localhost:8000/docs")
            print("   • Try the API endpoints with your own data")
            print("\n💡 Integration Ideas:")
            print("   • Connect to a chat application")
            print("   • Build a proactive reminder system")
            print("   • Create a personal AI assistant")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True


async def main():
    """Main demo function."""
    demo = MemoryPipelineDemo()
    success = await demo.run_demo()
    return success


if __name__ == "__main__":
    print("Memory Pipeline - Quick Start Demo")
    print("Make sure you've run test_basic.py successfully first!")
    print()

    # Ask for confirmation
    response = input("Continue with demo? (y/N): ").strip().lower()
    if response != 'y':
        print("Demo cancelled.")
        sys.exit(0)

    # Run demo
    success = asyncio.run(main())

    # Exit with appropriate code
    sys.exit(0 if success else 1)