"""
Basic test cases for Memory Pipeline components.

Simple unit tests that verify core functionality without
requiring external dependencies or complex async operations.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)


def test_imports():
    """Test that core modules can be imported successfully."""
    try:
        from app.core.config import settings
        from app.models.schemas import MemoryType, MemoryCreate
        from app.config.rag_config import ContextType, MemoryCategory

        assert settings is not None
        assert MemoryType.FACT == "fact"
        assert ContextType.MORNING_CHECKIN is not None
        assert MemoryCategory.WORK is not None

    except ImportError as e:
        assert False, f"Import failed: {e}"


def test_memory_types():
    """Test memory type enumeration."""
    from app.models.schemas import MemoryType

    expected_types = ["fact", "preference", "event", "routine", "emotion"]

    for expected_type in expected_types:
        assert hasattr(MemoryType, expected_type.upper())
        assert getattr(MemoryType, expected_type.upper()) == expected_type


def test_context_types():
    """Test context type enumeration."""
    from app.config.rag_config import ContextType

    # Test that key context types exist
    assert hasattr(ContextType, 'MORNING_CHECKIN')
    assert hasattr(ContextType, 'MEAL_SUGGESTION')
    assert hasattr(ContextType, 'WEEKEND_PLANNING')


def test_memory_categories():
    """Test memory category enumeration."""
    from app.config.rag_config import MemoryCategory

    expected_categories = [
        "WORK", "FOOD", "HEALTH", "SOCIAL",
        "SCHEDULE", "ENTERTAINMENT", "EMOTIONAL"
    ]

    for category in expected_categories:
        assert hasattr(MemoryCategory, category)


def test_rag_config_exists():
    """Test that RAG configuration is properly defined."""
    from app.config.rag_config import RAG_CONFIG, RETRIEVAL_STRATEGIES

    assert RAG_CONFIG is not None
    assert RETRIEVAL_STRATEGIES is not None
    assert len(RETRIEVAL_STRATEGIES) > 0


def test_memory_schema_creation():
    """Test creating memory schemas."""
    from app.models.schemas import MemoryCreate, MemoryType, TemporalRelevance
    from app.config.rag_config import MemoryCategory

    memory = MemoryCreate(
        content="I work as a software engineer",
        memory_type=MemoryType.FACT,
        category=MemoryCategory.WORK,
        entities=["software engineer"],
        temporal_relevance=TemporalRelevance.DAILY,
        importance_score=0.8,
        confidence_score=0.9
    )

    assert memory.content == "I work as a software engineer"
    assert memory.memory_type == MemoryType.FACT
    assert memory.category == MemoryCategory.WORK
    assert memory.importance_score == 0.8


def test_config_settings():
    """Test configuration settings."""
    from app.core.config import settings

    # Test that settings object exists and has expected attributes
    assert hasattr(settings, 'database_url')
    assert hasattr(settings, 'api_version')
    assert settings.api_version == "1.0.0"


def test_database_models_import():
    """Test that database models can be imported."""
    try:
        from app.models.memory import User, Memory, MemoryConnection, Conversation

        # Test that classes exist
        assert User is not None
        assert Memory is not None
        assert MemoryConnection is not None
        assert Conversation is not None

    except ImportError as e:
        assert False, f"Database model import failed: {e}"


if __name__ == "__main__":
    """Run tests directly when script is executed."""
    import traceback

    test_functions = [
        test_imports,
        test_memory_types,
        test_context_types,
        test_memory_categories,
        test_rag_config_exists,
        test_memory_schema_creation,
        test_config_settings,
        test_database_models_import
    ]

    passed = 0
    failed = 0

    print("BASIC UNIT TESTS")
    print("=" * 40)

    for test_func in test_functions:
        try:
            test_func()
            print(f"[OK] {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_func.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 40)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All basic tests passed!")
    else:
        print(f"{failed} tests failed")
        sys.exit(1)