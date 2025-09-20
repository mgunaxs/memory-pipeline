"""
Pydantic schemas for API request/response validation.

Defines data models for API endpoints with proper validation,
documentation, and examples.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field, validator


class MemoryType(str, Enum):
    """
    Enumeration of supported memory types.

    Each type has different characteristics:
    - fact: Permanent information about the user
    - preference: User likes/dislikes with long retention
    - event: Temporal information with expiration
    - routine: Recurring patterns and habits
    - emotion: Short-term emotional states
    """
    FACT = "fact"
    PREFERENCE = "preference"
    EVENT = "event"
    ROUTINE = "routine"
    EMOTION = "emotion"


class ConnectionType(str, Enum):
    """
    Enumeration of memory connection types.

    Defines how memories relate to each other:
    - relates_to: General relationship
    - contradicts: Conflicting information
    - updates: One memory updates another
    - supports: One memory supports another
    """
    RELATES_TO = "relates_to"
    CONTRADICTS = "contradicts"
    UPDATES = "updates"
    SUPPORTS = "supports"


class MemoryCategory(str, Enum):
    """Memory categories for context-aware filtering."""
    WORK = "work"
    FOOD = "food"
    HEALTH = "health"
    SOCIAL = "social"
    SCHEDULE = "schedule"
    ENTERTAINMENT = "entertainment"
    EMOTIONAL = "emotional"
    LEARNING = "learning"
    TRAVEL = "travel"
    FINANCE = "finance"


class TemporalRelevance(str, Enum):
    """Temporal relevance for memory usage."""
    DAILY = "daily"
    WEEKLY = "weekly"
    ONCE = "once"
    EXPIRED = "expired"


class ContextType(str, Enum):
    """Context types for memory retrieval."""
    MORNING_CHECKIN = "morning_checkin"
    EVENING_CHECKIN = "evening_checkin"
    WEEKEND_PLANNING = "weekend_planning"
    WORK_PLANNING = "work_planning"
    MEAL_SUGGESTION = "meal_suggestion"
    EMOTIONAL_SUPPORT = "emotional_support"
    EVENT_FOLLOWUP = "event_followup"
    GENERAL_CHAT = "general_chat"


class MemoryCreate(BaseModel):
    """
    Schema for creating a new memory with rich metadata.

    Used when storing extracted memories from user messages.
    """
    content: str = Field(..., min_length=1, max_length=5000, description="Memory content")
    memory_type: MemoryType = Field(..., description="Type of memory")
    category: MemoryCategory = Field(..., description="Memory category for context filtering")
    entities: List[str] = Field(default_factory=list, description="Extracted entities (people, places, etc.)")
    temporal_relevance: TemporalRelevance = Field(default=TemporalRelevance.ONCE, description="Temporal relevance")
    importance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score (0-1)"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence (0-1)"
    )
    valid_contexts: List[ContextType] = Field(
        default_factory=list,
        description="Contexts where this memory is relevant"
    )
    invalid_contexts: List[ContextType] = Field(
        default_factory=list,
        description="Contexts where this memory should not be used"
    )
    source_message_id: Optional[str] = Field(
        None,
        description="ID of source conversation message"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional structured data"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Optional expiration timestamp"
    )

    @validator('content')
    def validate_content(cls, v):
        """Validate memory content is not empty."""
        if not v.strip():
            raise ValueError('Memory content cannot be empty')
        return v.strip()

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "content": "Works as a software engineer at Microsoft",
                "memory_type": "fact",
                "importance_score": 8.0,
                "confidence_score": 0.95,
                "source_message_id": "msg-123",
                "metadata": {"company": "Microsoft", "role": "software engineer"}
            }
        }


class MemoryUpdate(BaseModel):
    """
    Schema for updating an existing memory.

    Allows partial updates to memory fields.
    """
    content: Optional[str] = Field(None, min_length=1, max_length=5000)
    importance_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_active: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "importance_score": 9.0,
                "metadata": {"updated": True}
            }
        }


class MemoryResponse(BaseModel):
    """
    Schema for memory response data.

    Used when returning memory information to clients.
    """
    memory_id: str = Field(..., description="Unique memory identifier")
    user_id: str = Field(..., description="User identifier")
    content: str = Field(..., description="Memory content")
    memory_type: MemoryType = Field(..., description="Type of memory")
    importance_score: float = Field(..., description="Importance score (0-10)")
    confidence_score: float = Field(..., description="Extraction confidence (0-1)")
    extracted_at: datetime = Field(..., description="Extraction timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    is_active: bool = Field(..., description="Whether memory is active")
    source_message_id: Optional[str] = Field(None, description="Source message ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional data")
    version: int = Field(..., description="Memory version")

    class Config:
        """Pydantic configuration."""
        from_attributes = True
        schema_extra = {
            "example": {
                "memory_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "user123",
                "content": "Works as a software engineer at Microsoft",
                "memory_type": "fact",
                "importance_score": 8.0,
                "confidence_score": 0.95,
                "extracted_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
                "expires_at": None,
                "is_active": True,
                "source_message_id": "msg-123",
                "metadata": {"company": "Microsoft"},
                "version": 1
            }
        }


class MemorySearchRequest(BaseModel):
    """
    Schema for memory search requests.

    Defines parameters for semantic search across user memories.
    """
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    user_id: str = Field(..., description="User identifier")
    memory_types: Optional[List[MemoryType]] = Field(
        None,
        description="Filter by memory types"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results"
    )
    min_importance: Optional[float] = Field(
        None,
        ge=0.0,
        le=10.0,
        description="Minimum importance score"
    )
    include_expired: bool = Field(
        default=False,
        description="Include expired memories"
    )

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "query": "work preferences",
                "user_id": "user123",
                "memory_types": ["fact", "preference"],
                "limit": 5,
                "min_importance": 5.0,
                "include_expired": False
            }
        }


class MemorySearchResult(BaseModel):
    """
    Schema for individual search result.

    Contains memory data plus search-specific metadata.
    """
    memory: MemoryResponse = Field(..., description="Memory data")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score"
    )
    rank: int = Field(..., ge=1, description="Result rank")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "memory": {
                    "memory_id": "123e4567-e89b-12d3-a456-426614174000",
                    "content": "Works as a software engineer",
                    "memory_type": "fact",
                    "importance_score": 8.0
                },
                "similarity_score": 0.92,
                "rank": 1
            }
        }


class MemorySearchResponse(BaseModel):
    """
    Schema for memory search response.

    Contains search results and metadata about the search.
    """
    results: List[MemorySearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., ge=0, description="Total matching memories")
    query: str = Field(..., description="Original search query")
    search_time_ms: float = Field(..., ge=0, description="Search time in milliseconds")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "results": [
                    {
                        "memory": {"content": "Software engineer at Microsoft"},
                        "similarity_score": 0.92,
                        "rank": 1
                    }
                ],
                "total_found": 1,
                "query": "work preferences",
                "search_time_ms": 45.2
            }
        }


class ExtractionRequest(BaseModel):
    """
    Schema for memory extraction requests.

    Used when processing user messages to extract memories.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to extract memories from"
    )
    user_id: str = Field(..., description="User identifier")
    message_id: Optional[str] = Field(
        None,
        description="Optional message identifier"
    )

    @validator('text')
    def validate_text(cls, v):
        """Validate text is not empty."""
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "text": "I'm a software engineer at Microsoft and I hate morning meetings",
                "user_id": "user123",
                "message_id": "msg-456"
            }
        }


class ExtractionResponse(BaseModel):
    """
    Schema for memory extraction response.

    Contains extracted memories and processing metadata.
    """
    memories: List[MemoryResponse] = Field(..., description="Extracted memories")
    processing_time_ms: float = Field(..., ge=0, description="Processing time")
    message_id: Optional[str] = Field(None, description="Message identifier")
    total_extracted: int = Field(..., ge=0, description="Number of memories extracted")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "memories": [
                    {"content": "Software engineer at Microsoft", "memory_type": "fact"},
                    {"content": "Dislikes morning meetings", "memory_type": "preference"}
                ],
                "processing_time_ms": 1250.5,
                "message_id": "msg-456",
                "total_extracted": 2
            }
        }


class APIResponse(BaseModel):
    """
    Standard API response wrapper.

    Provides consistent response format across all endpoints.
    """
    success: bool = Field(..., description="Whether request was successful")
    data: Optional[Union[Dict[str, Any], List[Any], str, int, float]] = Field(
        None,
        description="Response data"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "success": True,
                "data": {"result": "operation completed"},
                "error": None,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class HealthResponse(BaseModel):
    """
    Schema for health check response.

    Provides status of all system components.
    """
    status: str = Field(..., description="Overall system status")
    components: Dict[str, bool] = Field(..., description="Component health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    version: str = Field(..., description="API version")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "status": "healthy",
                "components": {
                    "database": True,
                    "vector_db": True,
                    "llm_api": True
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0"
            }
        }