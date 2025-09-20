"""
Comprehensive PostgreSQL unit tests.
Tests all database operations without SQLAlchemy dependencies.
"""

import sys
import os
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import psycopg2
import asyncpg
import pytest
from app.core.config import settings


class TestPostgreSQLOperations:
    """Test PostgreSQL database operations."""

    @classmethod
    def setup_class(cls):
        """Set up test class."""
        cls.test_user_id = f"test_user_{int(time.time())}"
        cls.test_memory_id = str(uuid.uuid4())

    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(settings.database_url)

    def test_user_crud_operations(self):
        """Test complete user CRUD operations."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # CREATE - Insert user
            cursor.execute("""
                INSERT INTO memory_pipeline.users (user_id, settings, is_active, total_memories)
                VALUES (%s, %s, %s, %s)
            """, (self.test_user_id, '{"test": true, "preferences": {"theme": "dark"}}', True, 0))

            # READ - Verify user exists
            cursor.execute("""
                SELECT user_id, settings, is_active, total_memories, created_at
                FROM memory_pipeline.users
                WHERE user_id = %s
            """, (self.test_user_id,))

            user = cursor.fetchone()
            assert user is not None, "User not found after insert"
            assert user[0] == self.test_user_id
            assert user[1]["test"] is True
            assert user[2] is True  # is_active
            assert user[3] == 0  # total_memories
            assert user[4] is not None  # created_at

            # UPDATE - Modify user
            cursor.execute("""
                UPDATE memory_pipeline.users
                SET total_memories = %s, settings = %s
                WHERE user_id = %s
            """, (5, '{"test": true, "updated": true}', self.test_user_id))

            # Verify update
            cursor.execute("""
                SELECT total_memories, settings
                FROM memory_pipeline.users
                WHERE user_id = %s
            """, (self.test_user_id,))

            updated_user = cursor.fetchone()
            assert updated_user[0] == 5
            assert updated_user[1]["updated"] is True

            conn.commit()
            print("User CRUD operations: PASS")

        except Exception as e:
            conn.rollback()
            raise AssertionError(f"User CRUD operations failed: {e}")
        finally:
            cursor.close()
            conn.close()

    def test_memory_crud_operations(self):
        """Test complete memory CRUD operations."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Ensure user exists
            cursor.execute("""
                INSERT INTO memory_pipeline.users (user_id, settings, is_active)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
            """, (self.test_user_id, '{"test": true}', True))

            # CREATE - Insert memory
            cursor.execute("""
                INSERT INTO memory_pipeline.memories
                (memory_id, user_id, content, memory_type, category, importance_score, confidence_score, metadata, entities)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.test_memory_id,
                self.test_user_id,
                "I love pizza with mushrooms and pepperoni",
                "preference",
                "food",
                0.85,
                0.92,
                '{"source": "conversation", "context": "food_discussion"}',
                ["pizza", "mushrooms", "pepperoni"]
            ))

            # READ - Verify memory exists
            cursor.execute("""
                SELECT memory_id, content, memory_type, category, importance_score, metadata, entities
                FROM memory_pipeline.memories
                WHERE memory_id = %s
            """, (self.test_memory_id,))

            memory = cursor.fetchone()
            assert memory is not None, "Memory not found after insert"
            assert memory[1] == "I love pizza with mushrooms and pepperoni"
            assert memory[2] == "preference"
            assert memory[3] == "food"
            assert memory[4] == 0.85
            assert memory[5]["source"] == "conversation"
            assert "pizza" in memory[6]

            # UPDATE - Modify memory
            cursor.execute("""
                UPDATE memory_pipeline.memories
                SET importance_score = %s, metadata = %s
                WHERE memory_id = %s
            """, (0.95, '{"source": "conversation", "updated": true}', self.test_memory_id))

            # Verify update
            cursor.execute("""
                SELECT importance_score, metadata
                FROM memory_pipeline.memories
                WHERE memory_id = %s
            """, (self.test_memory_id,))

            updated_memory = cursor.fetchone()
            assert updated_memory[0] == 0.95
            assert updated_memory[1]["updated"] is True

            conn.commit()
            print("Memory CRUD operations: PASS")

        except Exception as e:
            conn.rollback()
            raise AssertionError(f"Memory CRUD operations failed: {e}")
        finally:
            cursor.close()
            conn.close()

    def test_conversation_tracking(self):
        """Test conversation tracking functionality."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            conversation_id = str(uuid.uuid4())

            # Insert conversation
            cursor.execute("""
                INSERT INTO memory_pipeline.conversations
                (conversation_id, user_id, message_id, message, memories_extracted, processing_time_ms, context_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                conversation_id,
                self.test_user_id,
                "msg_123",
                "I love Italian food and prefer vegetarian options",
                2,
                150.5,
                "food_discussion"
            ))

            # Verify conversation
            cursor.execute("""
                SELECT conversation_id, message, memories_extracted, processing_time_ms, context_type
                FROM memory_pipeline.conversations
                WHERE conversation_id = %s
            """, (conversation_id,))

            conversation = cursor.fetchone()
            assert conversation is not None
            assert conversation[1] == "I love Italian food and prefer vegetarian options"
            assert conversation[2] == 2
            assert conversation[3] == 150.5
            assert conversation[4] == "food_discussion"

            conn.commit()
            print("Conversation tracking: PASS")

        except Exception as e:
            conn.rollback()
            raise AssertionError(f"Conversation tracking failed: {e}")
        finally:
            cursor.close()
            conn.close()

    def test_memory_connections(self):
        """Test memory connection functionality."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Create two memories to connect
            memory_id_1 = str(uuid.uuid4())
            memory_id_2 = str(uuid.uuid4())

            # Insert memories
            for memory_id, content in [(memory_id_1, "I work at Microsoft"), (memory_id_2, "I have daily standup meetings")]:
                cursor.execute("""
                    INSERT INTO memory_pipeline.memories
                    (memory_id, user_id, content, memory_type, category, importance_score)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (memory_id, self.test_user_id, content, "fact", "work", 0.8))

            # Create connection
            cursor.execute("""
                INSERT INTO memory_pipeline.memory_connections
                (source_memory_id, target_memory_id, connection_type, strength)
                VALUES (%s, %s, %s, %s)
            """, (memory_id_1, memory_id_2, "work_related", 0.9))

            # Verify connection
            cursor.execute("""
                SELECT source_memory_id, target_memory_id, connection_type, strength
                FROM memory_pipeline.memory_connections
                WHERE source_memory_id = %s AND target_memory_id = %s
            """, (memory_id_1, memory_id_2))

            connection = cursor.fetchone()
            assert connection is not None
            assert connection[2] == "work_related"
            assert connection[3] == 0.9

            conn.commit()
            print("Memory connections: PASS")

        except Exception as e:
            conn.rollback()
            raise AssertionError(f"Memory connections failed: {e}")
        finally:
            cursor.close()
            conn.close()

    def test_rag_metrics_tracking(self):
        """Test RAG metrics functionality."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Insert RAG metrics
            cursor.execute("""
                INSERT INTO memory_pipeline.rag_metrics
                (user_id, query_type, context_type, memories_retrieved, avg_relevance, response_time_ms, token_budget_used, deduplication_applied, validation_applied)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.test_user_id,
                "food_preference",
                "meal_suggestion",
                5,
                0.87,
                75.2,
                120,
                True,
                True
            ))

            # Verify metrics
            cursor.execute("""
                SELECT query_type, memories_retrieved, avg_relevance, response_time_ms, deduplication_applied
                FROM memory_pipeline.rag_metrics
                WHERE user_id = %s AND query_type = %s
            """, (self.test_user_id, "food_preference"))

            metrics = cursor.fetchone()
            assert metrics is not None
            assert metrics[0] == "food_preference"
            assert metrics[1] == 5
            assert metrics[2] == 0.87
            assert metrics[3] == 75.2
            assert metrics[4] is True

            conn.commit()
            print("RAG metrics tracking: PASS")

        except Exception as e:
            conn.rollback()
            raise AssertionError(f"RAG metrics tracking failed: {e}")
        finally:
            cursor.close()
            conn.close()

    def test_chromadb_sync_tracking(self):
        """Test ChromaDB sync status tracking."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Insert sync status
            cursor.execute("""
                INSERT INTO memory_pipeline.chroma_sync_status
                (memory_id, chroma_id, collection_name, embedding_model, sync_status)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (memory_id) DO UPDATE SET sync_status = EXCLUDED.sync_status
            """, (
                self.test_memory_id,
                "chroma_vec_123",
                "memories",
                "sentence-transformers/all-MiniLM-L6-v2",
                "synced"
            ))

            # Verify sync status
            cursor.execute("""
                SELECT chroma_id, collection_name, embedding_model, sync_status
                FROM memory_pipeline.chroma_sync_status
                WHERE memory_id = %s
            """, (self.test_memory_id,))

            sync_status = cursor.fetchone()
            assert sync_status is not None
            assert sync_status[0] == "chroma_vec_123"
            assert sync_status[1] == "memories"
            assert sync_status[2] == "sentence-transformers/all-MiniLM-L6-v2"
            assert sync_status[3] == "synced"

            conn.commit()
            print("ChromaDB sync tracking: PASS")

        except Exception as e:
            conn.rollback()
            raise AssertionError(f"ChromaDB sync tracking failed: {e}")
        finally:
            cursor.close()
            conn.close()

    def test_database_constraints(self):
        """Test database constraints and validations."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Test foreign key constraint
            try:
                cursor.execute("""
                    INSERT INTO memory_pipeline.memories
                    (memory_id, user_id, content, memory_type, category, importance_score)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (str(uuid.uuid4()), "nonexistent_user", "test content", "fact", "work", 0.5))
                conn.commit()
                assert False, "Foreign key constraint should have failed"
            except psycopg2.IntegrityError:
                conn.rollback()
                print("Foreign key constraint working")

            # Test unique constraint
            try:
                # Insert duplicate user_id
                cursor.execute("""
                    INSERT INTO memory_pipeline.users (user_id, settings, is_active)
                    VALUES (%s, %s, %s)
                """, (self.test_user_id, '{"test": true}', True))
                conn.commit()
                assert False, "Unique constraint should have failed"
            except psycopg2.IntegrityError:
                conn.rollback()
                print("Unique constraint working")

            # Test check constraint
            try:
                # Insert invalid importance_score
                cursor.execute("""
                    INSERT INTO memory_pipeline.memories
                    (memory_id, user_id, content, memory_type, category, importance_score)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (str(uuid.uuid4()), self.test_user_id, "test", "fact", "work", 1.5))  # > 1.0
                conn.commit()
                assert False, "Check constraint should have failed"
            except psycopg2.IntegrityError:
                conn.rollback()
                print("Check constraint working")

        except Exception as e:
            conn.rollback()
            raise AssertionError(f"Database constraints test failed: {e}")
        finally:
            cursor.close()
            conn.close()

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async database operations."""
        try:
            conn = await asyncpg.connect(settings.database_url)

            # Test async query
            result = await conn.fetchval("SELECT COUNT(*) FROM memory_pipeline.users")
            assert isinstance(result, int)
            assert result >= 0

            # Test async transaction
            async with conn.transaction():
                await conn.execute("""
                    INSERT INTO memory_pipeline.users (user_id, settings, is_active)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (user_id) DO NOTHING
                """, f"async_test_{int(time.time())}", '{"async": true}', True)

            await conn.close()
            print("Async operations: PASS")

        except Exception as e:
            raise AssertionError(f"Async operations failed: {e}")

    def test_database_performance(self):
        """Test database performance benchmarks."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Test query performance
            times = []
            for _ in range(10):
                start_time = time.time()
                cursor.execute("SELECT COUNT(*) FROM memory_pipeline.users")
                cursor.fetchone()
                query_time = (time.time() - start_time) * 1000
                times.append(query_time)

            avg_time = sum(times) / len(times)
            max_time = max(times)

            assert avg_time < 50, f"Average query time too slow: {avg_time:.2f}ms"
            assert max_time < 100, f"Max query time too slow: {max_time:.2f}ms"

            print(f"Database performance: PASS (avg: {avg_time:.2f}ms, max: {max_time:.2f}ms)")

        except Exception as e:
            raise AssertionError(f"Database performance test failed: {e}")
        finally:
            cursor.close()
            conn.close()

    @classmethod
    def teardown_class(cls):
        """Clean up test data."""
        try:
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()

            # Clean up test data
            cursor.execute("DELETE FROM memory_pipeline.chroma_sync_status WHERE memory_id = %s", (cls.test_memory_id,))
            cursor.execute("DELETE FROM memory_pipeline.memory_connections WHERE source_memory_id IN (SELECT memory_id FROM memory_pipeline.memories WHERE user_id = %s)", (cls.test_user_id,))
            cursor.execute("DELETE FROM memory_pipeline.rag_metrics WHERE user_id = %s", (cls.test_user_id,))
            cursor.execute("DELETE FROM memory_pipeline.conversations WHERE user_id = %s", (cls.test_user_id,))
            cursor.execute("DELETE FROM memory_pipeline.memories WHERE user_id = %s", (cls.test_user_id,))
            cursor.execute("DELETE FROM memory_pipeline.users WHERE user_id LIKE %s", (f"{cls.test_user_id}%",))

            conn.commit()
            cursor.close()
            conn.close()
            print("Test cleanup completed")

        except Exception as e:
            print(f"Test cleanup failed: {e}")


if __name__ == "__main__":
    # Run tests manually
    test_class = TestPostgreSQLOperations()
    test_class.setup_class()

    tests = [
        test_class.test_user_crud_operations,
        test_class.test_memory_crud_operations,
        test_class.test_conversation_tracking,
        test_class.test_memory_connections,
        test_class.test_rag_metrics_tracking,
        test_class.test_chromadb_sync_tracking,
        test_class.test_database_constraints,
        test_class.test_database_performance
    ]

    print("PostgreSQL Unit Tests")
    print("=" * 50)

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL {test.__name__}: {e}")
            failed += 1

    test_class.teardown_class()

    print(f"\nResults: {passed}/{passed + failed} tests passed")
    if failed == 0:
        print("All PostgreSQL unit tests passed!")
    else:
        print(f"{failed} tests failed")

    exit(0 if failed == 0 else 1)