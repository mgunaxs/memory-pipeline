"""
Memory API endpoints.

Provides REST API endpoints for memory extraction, storage, retrieval,
and management with proper error handling and response formatting.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_memory_service
from app.core.database_prod import get_db
from app.models.schemas import (
    APIResponse, MemoryResponse, ExtractionRequest, ExtractionResponse,
    MemorySearchRequest, MemorySearchResponse, MemoryType, HealthResponse,
    ContextType
)
from app.services.memory_service import MemoryService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


def create_api_response(success: bool, data=None, error: str = None) -> APIResponse:
    """
    Create standardized API response.

    Args:
        success: Whether operation was successful
        data: Response data
        error: Error message if failed

    Returns:
        APIResponse: Standardized response
    """
    return APIResponse(
        success=success,
        data=data,
        error=error,
        timestamp=datetime.utcnow()
    )


@router.post("/extract", response_model=APIResponse)
async def extract_memories(
    request: ExtractionRequest,
    service: MemoryService = Depends(get_memory_service),
    db: Session = Depends(get_db)
):
    """
    Extract memories from user text and store them.

    Processes natural language text to identify and extract different types
    of memories (facts, preferences, events, routines, emotions) and stores
    them in both the relational database and vector database.

    Args:
        request: Text extraction request
        service: Memory service
        db: Database session

    Returns:
        APIResponse: Response with extracted memories

    Example:
        POST /memory/extract
        {
            "text": "I'm a software engineer at Microsoft and I hate morning meetings",
            "user_id": "user123",
            "message_id": "msg-456"
        }

        Response:
        {
            "success": true,
            "data": {
                "memories": [...],
                "total_extracted": 2,
                "processing_time_ms": 1250.5
            },
            "error": null,
            "timestamp": "2024-01-01T12:00:00Z"
        }
    """
    try:
        logger.info(f"Processing extraction request for user {request.user_id}")

        response = await service.extract_and_store_memories(request, db)

        return create_api_response(
            success=True,
            data=response.dict()
        )

    except ValueError as e:
        logger.warning(f"Invalid extraction request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Memory extraction failed")


@router.post("/search", response_model=APIResponse)
async def search_memories(
    request: MemorySearchRequest,
    service: MemoryService = Depends(get_memory_service),
    db: Session = Depends(get_db)
):
    """
    Search user memories using semantic similarity.

    Performs vector-based semantic search across user memories with support
    for filtering by memory type, importance, expiration, and other criteria.

    Args:
        request: Search request parameters
        service: Memory service
        db: Database session

    Returns:
        APIResponse: Search results with similarity scores

    Example:
        POST /memory/search
        {
            "query": "work preferences",
            "user_id": "user123",
            "memory_types": ["fact", "preference"],
            "limit": 5,
            "min_importance": 5.0
        }

        Response:
        {
            "success": true,
            "data": {
                "results": [...],
                "total_found": 3,
                "search_time_ms": 45.2
            }
        }
    """
    try:
        logger.info(f"Processing search request for user {request.user_id}")

        response = await service.search_memories(request, db)

        return create_api_response(
            success=True,
            data=response.dict()
        )

    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail="Memory search failed")


@router.get("/user/{user_id}", response_model=APIResponse)
async def get_user_memories(
    user_id: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of memories"),
    memory_type: Optional[MemoryType] = Query(default=None, description="Filter by memory type"),
    include_expired: bool = Query(default=False, description="Include expired memories"),
    service: MemoryService = Depends(get_memory_service),
    db: Session = Depends(get_db)
):
    """
    Get all memories for a user.

    Retrieves user memories from the database with optional filtering
    by type, expiration status, and result limits.

    Args:
        user_id: User identifier
        limit: Maximum number of memories to return
        memory_type: Optional memory type filter
        include_expired: Whether to include expired memories
        service: Memory service
        db: Database session

    Returns:
        APIResponse: User memories

    Example:
        GET /memory/user/user123?limit=10&memory_type=fact

        Response:
        {
            "success": true,
            "data": {
                "memories": [...],
                "total_returned": 8
            }
        }
    """
    try:
        logger.info(f"Getting memories for user {user_id}")

        memories = service.get_user_memories(
            user_id=user_id,
            db=db,
            limit=limit,
            memory_type=memory_type,
            include_expired=include_expired
        )

        return create_api_response(
            success=True,
            data={
                "memories": [memory.dict() for memory in memories],
                "total_returned": len(memories),
                "user_id": user_id
            }
        )

    except Exception as e:
        logger.error(f"Failed to get user memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memories")


@router.post("/process-conversation", response_model=APIResponse)
async def process_conversation(
    request: ExtractionRequest,
    service: MemoryService = Depends(get_memory_service),
    db: Session = Depends(get_db)
):
    """
    End-to-end conversation processing pipeline.

    Processes a user message through the complete memory pipeline:
    extraction, storage, and returns both the extracted memories
    and relevant existing memories for context.

    Args:
        request: Conversation processing request
        service: Memory service
        db: Database session

    Returns:
        APIResponse: Extraction results plus relevant context

    Example:
        POST /memory/process-conversation
        {
            "text": "I have a dentist appointment tomorrow at 2pm",
            "user_id": "user123"
        }

        Response:
        {
            "success": true,
            "data": {
                "extracted_memories": [...],
                "relevant_context": [...],
                "processing_summary": {...}
            }
        }
    """
    try:
        logger.info(f"Processing conversation for user {request.user_id}")

        # Extract and store new memories
        extraction_response = await service.extract_and_store_memories(request, db)

        # Search for relevant existing memories using the original text
        search_request = MemorySearchRequest(
            query=request.text,
            user_id=request.user_id,
            limit=5
        )
        search_response = await service.search_memories(search_request, db)

        # Filter out newly created memories from search results
        new_memory_ids = {mem.memory_id for mem in extraction_response.memories}
        relevant_context = [
            result for result in search_response.results
            if result.memory.memory_id not in new_memory_ids
        ]

        return create_api_response(
            success=True,
            data={
                "extracted_memories": extraction_response.dict(),
                "relevant_context": [result.dict() for result in relevant_context],
                "processing_summary": {
                    "new_memories_count": extraction_response.total_extracted,
                    "relevant_memories_count": len(relevant_context),
                    "total_processing_time_ms": extraction_response.processing_time_ms + search_response.search_time_ms
                }
            }
        )

    except Exception as e:
        logger.error(f"Conversation processing failed: {e}")
        raise HTTPException(status_code=500, detail="Conversation processing failed")


@router.get("/stats/{user_id}", response_model=APIResponse)
async def get_user_stats(
    user_id: str,
    service: MemoryService = Depends(get_memory_service),
    db: Session = Depends(get_db)
):
    """
    Get user memory statistics.

    Provides comprehensive statistics about a user's memories including
    counts, type distribution, importance metrics, and vector database stats.

    Args:
        user_id: User identifier
        service: Memory service
        db: Database session

    Returns:
        APIResponse: User memory statistics

    Example:
        GET /memory/stats/user123

        Response:
        {
            "success": true,
            "data": {
                "total_memories": 45,
                "active_memories": 42,
                "type_distribution": {...},
                "vector_stats": {...}
            }
        }
    """
    try:
        logger.info(f"Getting stats for user {user_id}")

        stats = service.get_user_stats(user_id, db)

        return create_api_response(
            success=True,
            data=stats
        )

    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: MemoryService = Depends(get_memory_service)
):
    """
    Health check endpoint.

    Checks the health of all system components including database,
    vector database, and external API connections.

    Returns:
        HealthResponse: System health status

    Example:
        GET /memory/health

        Response:
        {
            "status": "healthy",
            "components": {
                "database": true,
                "vector_db": true,
                "llm_api": true,
                "embedding_service": true
            },
            "timestamp": "2024-01-01T12:00:00Z",
            "version": "1.0.0"
        }
    """
    try:
        # Check database connection
        from app.core.database import check_database_connection
        db_healthy = check_database_connection()

        # Check vector database
        vector_health = service.vector_service.check_vector_health()
        vector_healthy = vector_health['status'] == 'healthy'

        # Check embedding service
        embedding_healthy = True  # Basic check - service exists

        # TODO: Add Gemini API health check
        llm_healthy = True  # For now, assume healthy

        components = {
            "database": db_healthy,
            "vector_db": vector_healthy,
            "llm_api": llm_healthy,
            "embedding_service": embedding_healthy
        }

        overall_status = "healthy" if all(components.values()) else "unhealthy"

        return HealthResponse(
            status=overall_status,
            components=components,
            timestamp=datetime.utcnow(),
            version=settings.api_version
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            components={
                "database": False,
                "vector_db": False,
                "llm_api": False,
                "embedding_service": False
            },
            timestamp=datetime.utcnow(),
            version=settings.api_version
        )


@router.post("/smart-search", response_model=APIResponse)
async def smart_search_memories(
    query: str = Query(..., description="Search query"),
    user_id: str = Query(..., description="User identifier"),
    context_type: Optional[ContextType] = Query(None, description="Context for smart filtering"),
    max_memories: Optional[int] = Query(5, ge=1, le=20, description="Maximum memories to return"),
    token_budget: Optional[int] = Query(None, ge=100, le=2000, description="Token budget for results"),
    enable_validation: bool = Query(True, description="Enable relevance validation"),
    service: MemoryService = Depends(get_memory_service)
):
    """
    Smart memory search with enhanced RAG features.

    Provides context-aware filtering, relevance validation, and token budget management
    to ensure only appropriate memories are retrieved for specific contexts.

    Args:
        query: Search query text
        user_id: User identifier
        context_type: Context type for smart filtering (e.g., morning_checkin, meal_suggestion)
        max_memories: Maximum number of memories to return
        token_budget: Token budget for memory content
        enable_validation: Whether to validate relevance with LLM
        service: Memory service

    Returns:
        APIResponse: Enhanced search results

    Example:
        POST /api/v1/memory/smart-search?query=morning routine&user_id=user123&context_type=morning_checkin&token_budget=300

        Response:
        {
            "success": true,
            "data": {
                "results": [...],
                "total_found": 3,
                "context_filtered": true,
                "validation_applied": true
            }
        }
    """
    try:
        logger.info(f"Smart search request: '{query}' for user {user_id} with context {context_type}")

        response = await service.smart_search_memories(
            query=query,
            user_id=user_id,
            context_type=context_type,
            max_memories=max_memories,
            token_budget=token_budget,
            enable_validation=enable_validation
        )

        return create_api_response(
            success=True,
            data={
                **response.dict(),
                "context_filtered": context_type is not None,
                "validation_applied": enable_validation,
                "token_budget_applied": token_budget is not None
            }
        )

    except Exception as e:
        logger.error(f"Smart search failed: {e}")
        raise HTTPException(status_code=500, detail="Smart search failed")


@router.get("/rag-stats/{user_id}", response_model=APIResponse)
async def get_rag_stats(
    user_id: str,
    service: MemoryService = Depends(get_memory_service),
    db: Session = Depends(get_db)
):
    """
    Get RAG quality statistics for a user.

    Provides insights into memory quality, distribution, and retrieval patterns
    to help understand and optimize the RAG system performance.

    Args:
        user_id: User identifier
        service: Memory service
        db: Database session

    Returns:
        APIResponse: RAG quality statistics

    Example:
        GET /api/v1/memory/rag-stats/user123

        Response:
        {
            "success": true,
            "data": {
                "total_memories": 45,
                "category_distribution": {...},
                "average_confidence": 0.87,
                "quality_metrics": {...}
            }
        }
    """
    try:
        logger.info(f"Getting RAG stats for user {user_id}")

        # Get basic user stats
        basic_stats = service.get_user_stats(user_id, db)

        # Get additional RAG-specific stats
        memories = service.get_user_memories(user_id, db, limit=1000)

        # Calculate quality metrics
        if memories:
            # Category distribution
            category_dist = {}
            confidence_scores = []
            importance_scores = []

            for memory in memories:
                # Extract category from metadata
                if memory.metadata_ and 'category' in memory.metadata_:
                    category = memory.metadata_['category']
                    category_dist[category] = category_dist.get(category, 0) + 1

                confidence_scores.append(memory.confidence_score)
                importance_scores.append(memory.importance_score)

            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            avg_importance = sum(importance_scores) / len(importance_scores)

            quality_metrics = {
                "average_confidence": round(avg_confidence, 3),
                "average_importance": round(avg_importance, 3),
                "high_confidence_count": len([c for c in confidence_scores if c >= 0.8]),
                "high_importance_count": len([i for i in importance_scores if i >= 0.7])
            }
        else:
            category_dist = {}
            quality_metrics = {
                "average_confidence": 0.0,
                "average_importance": 0.0,
                "high_confidence_count": 0,
                "high_importance_count": 0
            }

        rag_stats = {
            **basic_stats,
            "category_distribution": category_dist,
            "quality_metrics": quality_metrics,
            "rag_features": {
                "context_filtering_enabled": settings.enable_context_filtering,
                "deduplication_enabled": settings.enable_deduplication,
                "validation_enabled": settings.enable_relevance_validation
            }
        }

        return create_api_response(
            success=True,
            data=rag_stats
        )

    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve RAG statistics")