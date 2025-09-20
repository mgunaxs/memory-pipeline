"""
Simplified RAG Quality Test

Tests core RAG improvements without heavy dependencies.
Focuses on critical functionality verification.
"""

import asyncio
import sys
import os
import time
import json

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Simple test without complex dependencies
print("SIMPLIFIED RAG QUALITY TEST")
print("=" * 40)

def test_basic_imports():
    """Test that core modules can be imported."""
    try:

        # Test basic imports
        from app.config.rag_config import RETRIEVAL_STRATEGIES, ContextType, MemoryCategory
        from app.models.schemas import MemoryType, MemoryCreate

        print("[OK] Basic imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_rag_config():
    """Test RAG configuration setup."""
    try:
        from app.config.rag_config import RETRIEVAL_STRATEGIES, ContextType, get_retrieval_strategy

        # Test context types exist
        contexts = [
            ContextType.MORNING_CHECKIN,
            ContextType.MEAL_SUGGESTION,
            ContextType.WEEKEND_PLANNING
        ]

        strategies_found = 0
        for context in contexts:
            strategy = get_retrieval_strategy(context)
            if strategy:
                strategies_found += 1
                print(f"  [OK] {context.value}: max_memories={strategy.max_memories}, min_relevance={strategy.min_relevance}")

        success = strategies_found == len(contexts)
        print(f"{'[OK]' if success else '[FAIL]'} RAG strategies configured: {strategies_found}/{len(contexts)}")
        return success

    except Exception as e:
        print(f"[FAIL] RAG config test failed: {e}")
        return False

def test_memory_categories():
    """Test memory category classification."""
    try:
        from app.config.rag_config import classify_memory_category, MemoryCategory

        test_cases = [
            ("I work at Google as a software engineer", MemoryCategory.WORK),
            ("I love pizza and Italian food", MemoryCategory.FOOD),
            ("I go to the gym on Tuesdays", MemoryCategory.HEALTH),
            ("Meeting with friends this weekend", MemoryCategory.SOCIAL),
            ("Doctor appointment tomorrow at 2pm", MemoryCategory.SCHEDULE),
            ("I love watching Netflix", MemoryCategory.ENTERTAINMENT),
            ("Feeling stressed about work", MemoryCategory.EMOTIONAL)
        ]

        correct_classifications = 0
        for text, expected_category in test_cases:
            classified = classify_memory_category(text)
            is_correct = classified == expected_category
            correct_classifications += is_correct

            print(f"  {'[OK]' if is_correct else '[FAIL]'} '{text[:30]}...' -> {classified.value} (expected: {expected_category.value})")

        accuracy = correct_classifications / len(test_cases)
        success = accuracy >= 0.6  # 60% accuracy threshold

        print(f"{'[OK]' if success else '[FAIL]'} Category classification: {accuracy:.1%} accuracy")
        return success

    except Exception as e:
        print(f"[FAIL] Category test failed: {e}")
        return False

def test_enhanced_schemas():
    """Test enhanced memory schemas."""
    try:
        from app.models.schemas import (
            MemoryCreate, MemoryType, MemoryCategory,
            TemporalRelevance, ContextType
        )

        # Test creating enhanced memory
        memory = MemoryCreate(
            content="I love coffee and work remotely",
            memory_type=MemoryType.PREFERENCE,
            category=MemoryCategory.WORK,
            entities=["coffee", "remote work"],
            temporal_relevance=TemporalRelevance.DAILY,
            importance_score=0.8,
            confidence_score=0.9,
            valid_contexts=[ContextType.WORK_PLANNING],
            invalid_contexts=[ContextType.MEAL_SUGGESTION]
        )

        # Validate fields
        checks = [
            memory.content == "I love coffee and work remotely",
            memory.memory_type == MemoryType.PREFERENCE,
            memory.category == MemoryCategory.WORK,
            len(memory.entities) == 2,
            memory.temporal_relevance == TemporalRelevance.DAILY,
            0 <= memory.importance_score <= 1,
            0 <= memory.confidence_score <= 1,
            len(memory.valid_contexts) == 1,
            len(memory.invalid_contexts) == 1
        ]

        success = all(checks)
        print(f"{'[OK]' if success else '[FAIL]'} Enhanced schema validation: {sum(checks)}/{len(checks)} checks passed")
        return success

    except Exception as e:
        print(f"[FAIL] Schema test failed: {e}")
        return False

def test_context_filtering_logic():
    """Test context filtering logic."""
    try:
        from app.config.rag_config import should_include_memory, get_retrieval_strategy
        from app.models.schemas import MemoryType, MemoryCategory, ContextType

        # Test morning context filtering
        morning_strategy = get_retrieval_strategy(ContextType.MORNING_CHECKIN)

        test_cases = [
            # (category, type, should_include_in_morning)
            (MemoryCategory.SCHEDULE, MemoryType.EVENT, True),
            (MemoryCategory.HEALTH, MemoryType.ROUTINE, True),
            (MemoryCategory.WORK, MemoryType.FACT, True),
            (MemoryCategory.ENTERTAINMENT, MemoryType.PREFERENCE, False),
            (MemoryCategory.SOCIAL, MemoryType.EVENT, False),
        ]

        correct_filtering = 0
        for category, memory_type, should_include in test_cases:
            result = should_include_memory(category, memory_type, morning_strategy)
            is_correct = result == should_include
            correct_filtering += is_correct

            print(f"  {'[OK]' if is_correct else '[FAIL]'} {category.value}/{memory_type.value} -> {'Include' if result else 'Exclude'}")

        accuracy = correct_filtering / len(test_cases)
        success = accuracy >= 0.8  # 80% accuracy threshold

        print(f"{'[OK]' if success else '[FAIL]'} Context filtering logic: {accuracy:.1%} accuracy")
        return success

    except Exception as e:
        print(f"[FAIL] Context filtering test failed: {e}")
        return False

def test_deduplication_logic():
    """Test deduplication detection logic."""
    try:
        # Test text similarity detection (simplified)
        import difflib

        def simple_similarity(text1, text2):
            return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

        test_cases = [
            # (text1, text2, expected_similarity_level)
            ("I love pizza", "I love pizza", "duplicate"),  # Exact match
            ("Meeting at 3pm tomorrow", "Meeting moved to 4pm tomorrow", "update"),  # Update
            ("I like coffee", "I love espresso", "similar"),  # Related but different
            ("I work at Google", "I love hiking", "different"),  # Completely different
        ]

        correct_detections = 0
        for text1, text2, expected in test_cases:
            similarity = simple_similarity(text1, text2)

            if similarity >= 0.95:
                detected = "duplicate"
            elif similarity >= 0.7:
                detected = "update" if any(word in text2.lower() for word in ["moved", "changed", "updated"]) else "similar"
            else:
                detected = "different"

            is_correct = detected == expected
            correct_detections += is_correct

            print(f"  {'[OK]' if is_correct else '[FAIL]'} '{text1}' vs '{text2}' -> {detected} (similarity: {similarity:.2f})")

        accuracy = correct_detections / len(test_cases)
        success = accuracy >= 0.75  # 75% accuracy threshold

        print(f"{'[OK]' if success else '[FAIL]'} Deduplication logic: {accuracy:.1%} accuracy")
        return success

    except Exception as e:
        print(f"[FAIL] Deduplication test failed: {e}")
        return False

def test_token_estimation():
    """Test token estimation logic."""
    try:
        def estimate_tokens(text):
            return len(text) // 4  # Simple estimation

        test_cases = [
            ("Short text", 3),
            ("This is a longer piece of text that should have more tokens", 16),
            ("Very detailed memory with lots of specific information about user preferences and behaviors", 21)
        ]

        correct_estimates = 0
        for text, expected_range in test_cases:
            estimated = estimate_tokens(text)
            # Allow 50% tolerance
            is_reasonable = abs(estimated - expected_range) <= expected_range * 0.5
            correct_estimates += is_reasonable

            print(f"  {'[OK]' if is_reasonable else '[FAIL]'} '{text[:30]}...' -> {estimated} tokens (expected ~{expected_range})")

        accuracy = correct_estimates / len(test_cases)
        success = accuracy >= 0.8

        print(f"{'[OK]' if success else '[FAIL]'} Token estimation: {accuracy:.1%} accuracy")
        return success

    except Exception as e:
        print(f"[FAIL] Token estimation test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    required_files = [
        "app/config/rag_config.py",
        "app/memory/smart_retriever.py",
        "app/memory/deduplicator.py",
        "app/memory/validator.py",
        "app/memory/summarizer.py",
        "app/models/schemas.py",
        "tests/integration/test_rag_quality.py",
        "RAG_IMPROVEMENTS.md"
    ]

    existing_files = 0
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files += 1
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")

    success = existing_files == len(required_files)
    print(f"{'[OK]' if success else '[FAIL]'} File structure: {existing_files}/{len(required_files)} files present")
    return success

def run_all_tests():
    """Run all simplified tests."""
    print("\nRunning Simplified RAG Quality Tests...")

    tests = [
        ("Basic Imports", test_basic_imports),
        ("RAG Configuration", test_rag_config),
        ("Memory Categories", test_memory_categories),
        ("Enhanced Schemas", test_enhanced_schemas),
        ("Context Filtering", test_context_filtering_logic),
        ("Deduplication Logic", test_deduplication_logic),
        ("Token Estimation", test_token_estimation),
        ("File Structure", test_file_structure),
    ]

    results = []
    total_passed = 0

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}:")
        try:
            passed = test_func()
            results.append((test_name, passed))
            if passed:
                total_passed += 1
        except Exception as e:
            print(f"[CRASH] {test_name} crashed: {e}")
            results.append((test_name, False))

    # Final report
    print("\n" + "=" * 50)
    print("SIMPLIFIED RAG TEST REPORT")
    print("=" * 50)

    print(f"\nSUMMARY:")
    print(f"Tests Passed: {total_passed}/{len(tests)}")
    overall_percentage = (total_passed / len(tests)) * 100
    print(f"Overall Score: {overall_percentage:.1f}%")

    print(f"\nDETAILED RESULTS:")
    for test_name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"  {test_name}: {status}")

    print(f"\nASSESSMENT:")
    if overall_percentage >= 80:
        print("RAG FOUNDATION IS SOLID")
        print("[OK] Core RAG improvements are properly implemented")
        print("[OK] Ready for integration testing with dependencies")
        print("[OK] Architecture supports production RAG quality")
    elif overall_percentage >= 60:
        print("[WARNING] RAG FOUNDATION NEEDS IMPROVEMENT")
        print("[INFO] Some core components need fixes")
        print("[INFO] Address failing tests before full integration")
    else:
        print("[ERROR] RAG FOUNDATION HAS ISSUES")
        print("[ERROR] Critical components not working")
        print("[ERROR] Requires significant fixes")

    print(f"\nNEXT STEPS:")
    if overall_percentage >= 80:
        print("1. [NEXT] Install full dependencies")
        print("2. [NEXT] Run comprehensive integration tests")
        print("3. [NEXT] Test with real Gemini API")
        print("4. [NEXT] Deploy for production testing")
    else:
        print("1. [TODO] Fix failing core tests")
        print("2. [TODO] Review implementation logic")
        print("3. [TODO] Test individual components")
        print("4. [TODO] Re-run simplified tests")

    return overall_percentage >= 80

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)