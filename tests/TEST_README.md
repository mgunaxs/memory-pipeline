# Tests Directory

## Quick Start

Run all production tests:
```bash
python tests/simple_tests.py
```

## Test Structure

### Health Tests (`health/`)
- `basic_tests.py` - Core system tests without SQLAlchemy
- `test_system_health.py` - Comprehensive PostgreSQL/ChromaDB tests
- `test_api_health.py` - API endpoint health tests

### Integration Tests (`integration/`)
- `test_rag_quality.py` - RAG quality with 8 user behavior patterns
- `comprehensive_rag_test.py` - Full RAG system testing

### Unit Tests (`unit/`)
- `test_basic.py` - Basic unit tests
- `simple_rag_test.py` - Simple RAG component tests

### Setup Tests (`setup/`)
- `quick_setup.py` - Database setup verification

## Known Issues

**Python 3.13 + SQLAlchemy**: Use Python 3.12 for full functionality
**NumPy 2.0 + ChromaDB**: Known compatibility issue

## Test Results

Last run: ALL TESTS PASSED (7/7)
- PostgreSQL: ✅ PASS (7 tables)
- ChromaDB Config: ✅ PASS
- RAG Quality: ✅ PASS (8/8 user patterns)
- Environment: ✅ PASS
- File Structure: ✅ PASS
- Core Imports: ✅ PASS

## Running Individual Tests

```bash
# Health tests only
python tests/health/basic_tests.py

# PostgreSQL connection test
python -c "from tests.health.basic_tests import test_basic_postgresql; test_basic_postgresql()"
```