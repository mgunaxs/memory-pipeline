# Memory Pipeline Architecture

This document explains the design decisions, architecture patterns, and technical implementation of the Memory Pipeline component.

## 🎯 Design Goals

1. **Cost Optimization**: Use free tier services exclusively (Gemini, ChromaDB, SQLite)
2. **Semantic Understanding**: Leverage vector embeddings for intelligent memory retrieval
3. **Scalability**: Design for 1-100 users initially, easy scaling later
4. **Reliability**: Proper error handling, rate limiting, and fallback mechanisms
5. **Performance**: <500ms memory retrieval, efficient caching
6. **Extensibility**: Easy to add new memory types and features

## 🏗️ High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│ Memory Pipeline │───▶│  Proactive AI   │
│   (Messages)    │    │   Component     │    │   (Future)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Three-Tier       │
                    │   Memory System     │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Hot Memory   │  │ Warm Memory  │  │ Cold Storage │
    │ (Redis)      │  │ (ChromaDB)   │  │ (SQLite)     │
    │ 24-48 hours  │  │ Semantic     │  │ Complete     │
    │              │  │ Search       │  │ History      │
    └──────────────┘  └──────────────┘  └──────────────┘
```

## 🧩 Component Architecture

### 1. Memory Extraction Layer

```python
Text Input → Gemini 1.5 Flash → Structured Memories
```

**Components:**
- `MemoryExtractor`: Prompts Gemini to extract structured memories
- `RateLimiter`: Manages 15 RPM limit with exponential backoff
- `MemoryTypes`: Classification system for different memory types

**Design Decisions:**
- **Gemini 1.5 Flash**: Chosen for balance of capability and cost
- **Structured Prompting**: JSON output format for reliable parsing
- **Confidence Scoring**: Each extraction includes confidence (0-1)
- **Retry Logic**: Exponential backoff for rate limit handling

### 2. Embedding Generation Layer

```python
Memory Text → Gemini Embeddings → 768D Vector
                     ↓
              Fallback to local
           sentence-transformers
```

**Components:**
- `EmbeddingService`: Manages embedding generation with caching
- `EmbeddingCache`: File-based cache to reduce API calls
- Fallback to `sentence-transformers` when rate limited

**Design Decisions:**
- **Gemini text-embedding-004**: High quality, 768 dimensions
- **Local Fallback**: `all-MiniLM-L6-v2` (384D) for rate limit scenarios
- **Persistent Caching**: Pickle-based cache survives restarts
- **Cache Eviction**: LRU-style eviction when cache fills

### 3. Vector Storage Layer

```python
Memory Vector → ChromaDB Collection → Semantic Search
```

**Components:**
- `MemoryRetriever`: Manages ChromaDB operations
- Per-user collections for data isolation
- Metadata filtering for memory types, importance, expiration

**Design Decisions:**
- **ChromaDB**: Fully local, no external dependencies
- **Per-User Collections**: Isolation and easier management
- **Metadata Storage**: Memory type, importance, timestamps
- **Relevance Scoring**: Combines similarity + importance + temporal decay

### 4. Database Layer

```python
Memory Metadata → SQLite → Relationships & History
```

**Components:**
- `Memory`: Core memory model with metadata
- `User`: User management and statistics
- `Conversation`: Message history tracking
- `MemoryConnection`: Relationship mapping between memories

**Design Decisions:**
- **SQLite**: Perfect for local development, easy to migrate
- **Proper Indexing**: Optimized queries for user_id, memory_type
- **Relationship Tracking**: Connections between related memories
- **Temporal Management**: Automatic expiration for events/emotions

## 🔄 Data Flow

### Memory Extraction Flow

```
1. User Message
2. Text Validation & Cleaning
3. Gemini API Call (with rate limiting)
4. JSON Response Parsing
5. Memory Object Creation
6. Deduplication Check
7. Database Storage (SQLite)
8. Embedding Generation
9. Vector Storage (ChromaDB)
10. Response Formation
```

### Memory Retrieval Flow

```
1. Search Query
2. Query Embedding Generation
3. Vector Similarity Search (ChromaDB)
4. Metadata Filtering
5. Relevance Score Calculation
6. Result Ranking & Limiting
7. Response Formation
```

## 🧠 Memory Type System

### Memory Classification

```python
class MemoryType(Enum):
    FACT = "fact"        # Permanent, high importance
    PREFERENCE = "preference"  # 6-month retention
    EVENT = "event"      # Auto-expires after date
    ROUTINE = "routine"  # Permanent, updateable
    EMOTION = "emotion"  # 7-day retention
```

### Type-Specific Behavior

| Type | Default Importance | Retention | Update Strategy |
|------|-------------------|-----------|-----------------|
| Fact | 8.0 | Permanent | Version (create new) |
| Preference | 6.0 | 6 months | Replace |
| Event | 7.0 | 30 days post-event | Merge details |
| Routine | 5.0 | Permanent | Replace |
| Emotion | 4.0 | 7 days | Replace |

## 🔧 Rate Limiting Strategy

### Gemini API Limits (Free Tier)
- **Rate**: 15 requests per minute
- **Daily**: 1,500 requests for embeddings
- **Monthly**: 1M tokens for LLM

### Implementation
```python
class RateLimiter:
    def __init__(self, max_requests_per_minute=15):
        self.min_interval = 60.0 / max_requests_per_minute  # 4 seconds

    async def wait_if_needed(self):
        # Ensure minimum 4-second intervals
```

### Optimization Strategies
1. **Batch Processing**: Multiple memories per API call
2. **Intelligent Caching**: Avoid redundant embeddings
3. **Fallback Models**: Local sentence-transformers
4. **Request Queuing**: Smooth rate limiting

## 📊 Performance Optimizations

### Caching Strategy

```python
# Three-Level Caching
1. In-Memory Cache (current session)
2. File Cache (persistent embeddings)
3. Database Cache (query results)
```

### Database Optimization

```sql
-- Strategic Indexes
CREATE INDEX idx_memory_user_type ON memories(user_id, memory_type);
CREATE INDEX idx_memory_user_active ON memories(user_id, is_active);
CREATE INDEX idx_memory_importance ON memories(importance_score);
```

### Vector Search Optimization

```python
# Relevance Scoring Formula
relevance = (similarity * 0.6) + (importance/10 * 0.3) + (temporal_boost * 0.1)

# Temporal Decay
recent_memories (≤7 days) → +0.1 boost
month_old (≤30 days) → +0.05 boost
older → no boost
```

## 🛡️ Error Handling Strategy

### Layered Error Handling

```python
1. Input Validation (Pydantic)
2. Business Logic Errors (Custom Exceptions)
3. External API Errors (Retry with backoff)
4. Database Errors (Transaction rollback)
5. System Errors (Graceful degradation)
```

### Specific Error Types

```python
class RateLimitError(Exception): pass
class APIQuotaError(Exception): pass
class VectorStoreError(Exception): pass
```

### Graceful Degradation

1. **API Failures**: Fall back to local models
2. **Vector DB Issues**: Use SQL-only search
3. **High Load**: Queue requests, async processing

## 🔍 Search & Retrieval Strategy

### Semantic Search Pipeline

```python
1. Query → Embedding
2. Vector Similarity Search
3. Metadata Filtering
4. Relevance Calculation
5. Ranking & Deduplication
6. Result Formatting
```

### Advanced Search Features

```python
# Multi-Filter Search
filters = {
    "memory_types": ["fact", "preference"],
    "min_importance": 5.0,
    "include_expired": False,
    "max_age_days": 30
}
```

### Search Result Ranking

```python
final_score = combine_scores(
    similarity_score=vector_distance,
    importance_weight=memory.importance_score,
    temporal_decay=calculate_age_decay(memory.extracted_at),
    user_feedback=future_learning_signals
)
```

## 🔮 Future Architecture Considerations

### Scaling Decisions

1. **1-10 Users**: Current SQLite + ChromaDB architecture
2. **10-100 Users**: Add Redis for hot memory layer
3. **100+ Users**: Migrate to PostgreSQL + cloud vector DB
4. **Enterprise**: Microservices, separate embedding service

### Integration Points

```python
# Future Components
class ProactiveEngine:
    """Decides when to message users based on memories"""

class ContextAssembler:
    """Combines memories with external data (weather, time)"""

class LearningLoop:
    """Updates importance scores based on user engagement"""
```

### Technology Evolution Path

```
Current: Gemini → ChromaDB → SQLite
Phase 2: Add Redis hot layer
Phase 3: PostgreSQL + Supabase
Phase 4: Cloud vector DBs (Pinecone, Weaviate)
Phase 5: Custom embedding models
```

## 🏗️ Design Patterns Used

### Repository Pattern
- `MemoryService`: Business logic abstraction
- `MemoryRetriever`: Data access abstraction

### Strategy Pattern
- Different embedding providers (Gemini, local)
- Multiple search strategies

### Observer Pattern
- Health monitoring and metrics collection

### Factory Pattern
- Memory type configuration
- Service initialization

## 📈 Monitoring & Observability

### Health Checks
```python
/api/v1/memory/health
- Database connectivity
- Vector DB status
- API quotas
- Cache performance
```

### Metrics to Track
- API response times
- Memory extraction accuracy
- Search relevance scores
- Cache hit rates
- Rate limit utilization

### Logging Strategy
```python
logger.info(f"Extracted {count} memories for user {user_id} in {time}ms")
logger.warning(f"Rate limit approaching: {current_usage}/{limit}")
logger.error(f"Memory extraction failed: {error}", exc_info=True)
```

## 🔧 Configuration Management

### Environment-Based Configuration
```python
class Settings(BaseSettings):
    # External APIs
    gemini_api_key: str

    # Performance Tuning
    rate_limit_per_minute: int = 15
    embedding_cache_size: int = 1000

    # Business Logic
    min_confidence_threshold: float = 0.3
    max_memories_per_user: int = 1000
```

### Feature Flags (Future)
```python
# Enable/disable features without code changes
ENABLE_FALLBACK_EMBEDDINGS = True
ENABLE_PROACTIVE_SUGGESTIONS = False
ENABLE_LEARNING_LOOPS = False
```

This architecture provides a solid foundation for a proactive AI companion while maintaining cost efficiency and scalability for future growth.