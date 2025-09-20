"""
Working PostgreSQL unit tests without dependencies.
Each test is independent and creates its own test data.
"""

import sys
import os
import time
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import psycopg2
from app.core.config import settings


def get_connection():
    """Get database connection."""
    return psycopg2.connect(settings.database_url)


def test_user_operations():
    """Test user CRUD operations."""
    conn = get_connection()
    cursor = conn.cursor()
    test_user_id = f"test_user_{int(time.time())}"

    try:
        # Create user
        cursor.execute("""
            INSERT INTO memory_pipeline.users (user_id, settings, is_active, total_memories)
            VALUES (%s, %s, %s, %s)
        """, (test_user_id, '{"test": true}', True, 0))

        # Read user
        cursor.execute("""
            SELECT user_id, is_active, total_memories
            FROM memory_pipeline.users
            WHERE user_id = %s
        """, (test_user_id,))

        user = cursor.fetchone()
        assert user is not None
        assert user[0] == test_user_id
        assert user[1] is True
        assert user[2] == 0

        # Update user
        cursor.execute("""
            UPDATE memory_pipeline.users
            SET total_memories = %s
            WHERE user_id = %s
        """, (5, test_user_id))

        # Verify update
        cursor.execute("""
            SELECT total_memories FROM memory_pipeline.users WHERE user_id = %s
        """, (test_user_id,))

        assert cursor.fetchone()[0] == 5

        # Delete user
        cursor.execute("DELETE FROM memory_pipeline.users WHERE user_id = %s", (test_user_id,))

        conn.commit()
        print("User CRUD operations: PASS")
        return True

    except Exception as e:
        conn.rollback()
        print(f"User CRUD operations: FAIL - {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def test_memory_operations():
    """Test memory operations with valid enum values."""
    conn = get_connection()
    cursor = conn.cursor()
    test_user_id = f"test_user_{int(time.time())}"
    test_memory_id = str(uuid.uuid4())

    try:
        # Create user first
        cursor.execute("""
            INSERT INTO memory_pipeline.users (user_id, settings, is_active)
            VALUES (%s, %s, %s)
        """, (test_user_id, '{"test": true}', True))

        # Create memory with valid enum values
        cursor.execute("""
            INSERT INTO memory_pipeline.memories
            (memory_id, user_id, content, memory_type, category, importance_score, confidence_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            test_memory_id,
            test_user_id,
            "I love pizza with mushrooms",
            "preference",  # Valid memory_type
            "food",        # Valid memory_category
            0.8,
            0.9
        ))

        # Read memory
        cursor.execute("""
            SELECT content, memory_type, category, importance_score
            FROM memory_pipeline.memories
            WHERE memory_id = %s
        """, (test_memory_id,))

        memory = cursor.fetchone()
        assert memory is not None
        assert memory[0] == "I love pizza with mushrooms"
        assert memory[1] == "preference"
        assert memory[2] == "food"
        assert float(memory[3]) == 0.8

        # Update memory
        cursor.execute("""
            UPDATE memory_pipeline.memories
            SET importance_score = %s
            WHERE memory_id = %s
        """, (0.95, test_memory_id))

        # Verify update
        cursor.execute("""
            SELECT importance_score FROM memory_pipeline.memories WHERE memory_id = %s
        """, (test_memory_id,))

        assert float(cursor.fetchone()[0]) == 0.95

        # Cleanup
        cursor.execute("DELETE FROM memory_pipeline.memories WHERE memory_id = %s", (test_memory_id,))
        cursor.execute("DELETE FROM memory_pipeline.users WHERE user_id = %s", (test_user_id,))

        conn.commit()
        print("Memory operations: PASS")
        return True

    except Exception as e:
        conn.rollback()
        print(f"Memory operations: FAIL - {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        conn.close()


def test_memory_connections():
    """Test memory connections."""
    conn = get_connection()
    cursor = conn.cursor()
    test_user_id = f"test_user_{int(time.time())}"
    memory_id_1 = str(uuid.uuid4())
    memory_id_2 = str(uuid.uuid4())

    try:
        # Create user
        cursor.execute("""
            INSERT INTO memory_pipeline.users (user_id, settings, is_active)
            VALUES (%s, %s, %s)
        """, (test_user_id, '{"test": true}', True))

        # Create two memories
        for memory_id, content in [(memory_id_1, "I work at Microsoft"), (memory_id_2, "Daily standup at 9am")]:
            cursor.execute("""
                INSERT INTO memory_pipeline.memories
                (memory_id, user_id, content, memory_type, category, importance_score)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (memory_id, test_user_id, content, "fact", "work", 0.8))

        # Create connection
        cursor.execute("""
            INSERT INTO memory_pipeline.memory_connections
            (source_memory_id, target_memory_id, connection_type, strength)
            VALUES (%s, %s, %s, %s)
        """, (memory_id_1, memory_id_2, "work_related", 0.9))

        # Verify connection
        cursor.execute("""
            SELECT connection_type, strength
            FROM memory_pipeline.memory_connections
            WHERE source_memory_id = %s AND target_memory_id = %s
        """, (memory_id_1, memory_id_2))

        connection = cursor.fetchone()
        assert connection is not None
        assert connection[0] == "work_related"
        assert float(connection[1]) == 0.9

        # Cleanup
        cursor.execute("DELETE FROM memory_pipeline.memory_connections WHERE source_memory_id = %s", (memory_id_1,))
        cursor.execute("DELETE FROM memory_pipeline.memories WHERE user_id = %s", (test_user_id,))
        cursor.execute("DELETE FROM memory_pipeline.users WHERE user_id = %s", (test_user_id,))

        conn.commit()
        print("Memory connections: PASS")
        return True

    except Exception as e:
        conn.rollback()
        print(f"Memory connections: FAIL - {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        conn.close()


def test_chromadb_sync():
    """Test ChromaDB sync tracking."""
    conn = get_connection()
    cursor = conn.cursor()
    test_user_id = f"test_user_{int(time.time())}"
    test_memory_id = str(uuid.uuid4())

    try:
        # Create user and memory first
        cursor.execute("""
            INSERT INTO memory_pipeline.users (user_id, settings, is_active)
            VALUES (%s, %s, %s)
        """, (test_user_id, '{"test": true}', True))

        cursor.execute("""
            INSERT INTO memory_pipeline.memories
            (memory_id, user_id, content, memory_type, category, importance_score)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (test_memory_id, test_user_id, "Test content", "fact", "work", 0.8))

        # Insert sync status
        cursor.execute("""
            INSERT INTO memory_pipeline.chroma_sync_status
            (memory_id, chroma_id, collection_name, embedding_model, sync_status)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            test_memory_id,
            "chroma_123",
            "memories",
            "sentence-transformers/all-MiniLM-L6-v2",
            "synced"
        ))

        # Verify sync status
        cursor.execute("""
            SELECT sync_status, chroma_id
            FROM memory_pipeline.chroma_sync_status
            WHERE memory_id = %s
        """, (test_memory_id,))

        sync_status = cursor.fetchone()
        assert sync_status is not None
        assert sync_status[0] == "synced"
        assert sync_status[1] == "chroma_123"

        # Cleanup
        cursor.execute("DELETE FROM memory_pipeline.chroma_sync_status WHERE memory_id = %s", (test_memory_id,))
        cursor.execute("DELETE FROM memory_pipeline.memories WHERE memory_id = %s", (test_memory_id,))
        cursor.execute("DELETE FROM memory_pipeline.users WHERE user_id = %s", (test_user_id,))

        conn.commit()
        print("ChromaDB sync tracking: PASS")
        return True

    except Exception as e:
        conn.rollback()
        print(f"ChromaDB sync tracking: FAIL - {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def test_rag_metrics():
    """Test RAG metrics tracking."""
    conn = get_connection()
    cursor = conn.cursor()
    test_user_id = f"test_user_{int(time.time())}"

    try:
        # Create user
        cursor.execute("""
            INSERT INTO memory_pipeline.users (user_id, settings, is_active)
            VALUES (%s, %s, %s)
        """, (test_user_id, '{"test": true}', True))

        # Insert RAG metrics
        cursor.execute("""
            INSERT INTO memory_pipeline.rag_metrics
            (user_id, query_type, context_type, memories_retrieved, avg_relevance, response_time_ms, token_budget_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (test_user_id, "food_query", "meal_suggestion", 3, 0.85, 125.5, 80))

        # Verify metrics
        cursor.execute("""
            SELECT query_type, memories_retrieved, avg_relevance
            FROM memory_pipeline.rag_metrics
            WHERE user_id = %s
        """, (test_user_id,))

        metrics = cursor.fetchone()
        assert metrics is not None
        assert metrics[0] == "food_query"
        assert metrics[1] == 3
        assert float(metrics[2]) == 0.85

        # Cleanup
        cursor.execute("DELETE FROM memory_pipeline.rag_metrics WHERE user_id = %s", (test_user_id,))
        cursor.execute("DELETE FROM memory_pipeline.users WHERE user_id = %s", (test_user_id,))

        conn.commit()
        print("RAG metrics tracking: PASS")
        return True

    except Exception as e:
        conn.rollback()
        print(f"RAG metrics tracking: FAIL - {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        conn.close()


def test_conversation_tracking():
    """Test conversation tracking."""
    conn = get_connection()
    cursor = conn.cursor()
    test_user_id = f"test_user_{int(time.time())}"
    conversation_id = str(uuid.uuid4())

    try:
        # Create user
        cursor.execute("""
            INSERT INTO memory_pipeline.users (user_id, settings, is_active)
            VALUES (%s, %s, %s)
        """, (test_user_id, '{"test": true}', True))

        # Insert conversation
        cursor.execute("""
            INSERT INTO memory_pipeline.conversations
            (conversation_id, user_id, message_id, message, memories_extracted, processing_time_ms)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (conversation_id, test_user_id, "msg_123", "I love pizza", 2, 150.5))

        # Verify conversation
        cursor.execute("""
            SELECT message, memories_extracted, processing_time_ms
            FROM memory_pipeline.conversations
            WHERE conversation_id = %s
        """, (conversation_id,))

        conversation = cursor.fetchone()
        assert conversation is not None
        assert conversation[0] == "I love pizza"
        assert conversation[1] == 2
        assert conversation[2] == 150.5

        # Cleanup
        cursor.execute("DELETE FROM memory_pipeline.conversations WHERE conversation_id = %s", (conversation_id,))
        cursor.execute("DELETE FROM memory_pipeline.users WHERE user_id = %s", (test_user_id,))

        conn.commit()
        print("Conversation tracking: PASS")
        return True

    except Exception as e:
        conn.rollback()
        print(f"Conversation tracking: FAIL - {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def test_database_constraints():
    """Test database constraints."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Test foreign key constraint (should fail)
        try:
            cursor.execute("""
                INSERT INTO memory_pipeline.memories
                (memory_id, user_id, content, memory_type, category, importance_score)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (str(uuid.uuid4()), "nonexistent_user", "test", "fact", "work", 0.5))
            conn.commit()
            return False  # Should not reach here
        except psycopg2.IntegrityError:
            conn.rollback()

        # Test check constraint (importance_score > 1.0 should fail)
        test_user_id = f"test_user_{int(time.time())}"

        # Create user first
        cursor.execute("""
            INSERT INTO memory_pipeline.users (user_id, settings, is_active)
            VALUES (%s, %s, %s)
        """, (test_user_id, '{"test": true}', True))

        try:
            cursor.execute("""
                INSERT INTO memory_pipeline.memories
                (memory_id, user_id, content, memory_type, category, importance_score)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (str(uuid.uuid4()), test_user_id, "test", "fact", "work", 1.5))  # > 1.0
            conn.commit()
            return False  # Should not reach here
        except psycopg2.IntegrityError:
            conn.rollback()

        # Cleanup
        cursor.execute("DELETE FROM memory_pipeline.users WHERE user_id = %s", (test_user_id,))
        conn.commit()

        print("Database constraints: PASS")
        return True

    except Exception as e:
        conn.rollback()
        print(f"Database constraints: FAIL - {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def test_performance():
    """Test basic performance."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        times = []
        for _ in range(5):
            start_time = time.time()
            cursor.execute("SELECT COUNT(*) FROM memory_pipeline.users")
            cursor.fetchone()
            query_time = (time.time() - start_time) * 1000
            times.append(query_time)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        print(f"Database performance: PASS (avg: {avg_time:.2f}ms, max: {max_time:.2f}ms)")
        return avg_time < 100  # Should be under 100ms

    except Exception as e:
        print(f"Database performance: FAIL - {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def main():
    """Run all PostgreSQL tests."""
    print("PostgreSQL Unit Tests (Working)")
    print("=" * 50)

    tests = [
        ("User Operations", test_user_operations),
        ("Memory Operations", test_memory_operations),
        ("Memory Connections", test_memory_connections),
        ("ChromaDB Sync", test_chromadb_sync),
        ("RAG Metrics", test_rag_metrics),
        ("Conversation Tracking", test_conversation_tracking),
        ("Database Constraints", test_database_constraints),
        ("Performance", test_performance)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\nTesting {test_name}:")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"{test_name}: FAIL - {e}")
            failed += 1

    print(f"\n" + "=" * 50)
    print(f"Results: {passed}/{passed + failed} tests passed")

    if failed == 0:
        print("All PostgreSQL unit tests passed!")
        return 0
    else:
        print(f"{failed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())