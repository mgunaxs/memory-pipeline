"""
Core memory service for business logic.

Coordinates between extraction, storage, and retrieval components
to provide high-level memory management operations.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.database_prod import get_db, SessionLocal
from app.core.config import settings
from app.models.memory import User, Memory, Conversation
from app.models.schemas import (
    MemoryCreate, MemoryResponse, MemoryUpdate, MemorySearchRequest,
    MemorySearchResponse, ExtractionRequest, ExtractionResponse,
    MemoryType
)
from app.memory.extractor import MemoryExtractor
from app.memory.embedder import EmbeddingService
from app.memory.retriever import MemoryRetriever
from app.memory.smart_retriever import SmartRetriever
from app.memory.deduplicator import MemoryDeduplicator
from app.memory.validator import MemoryValidator
from app.memory.summarizer import MemorySummarizer
from app.memory.types import get_memory_type_config
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


class MemoryService:
    """
    High-level memory management service.

    Provides business logic for memory extraction, storage, retrieval,
    and management with proper error handling and transaction management.
    """

    def __init__(self):
        """Initialize memory service with all enhanced components."""
        self.extractor = MemoryExtractor()
        self.embedding_service = EmbeddingService()
        self.retriever = MemoryRetriever(self.embedding_service)
        self.smart_retriever = SmartRetriever(self.retriever, self.embedding_service)
        self.deduplicator = MemoryDeduplicator(self.embedding_service)
        self.validator = MemoryValidator()
        self.summarizer = MemorySummarizer()
        self.vector_service = VectorService(self.embedding_service, self.retriever)

    async def extract_and_store_memories(
        self,
        request: ExtractionRequest,
        db: Session
    ) -> ExtractionResponse:
        """
        Extract memories from text and store them.

        Args:
            request: Extraction request with text and user info
            db: Database session

        Returns:
            ExtractionResponse: Response with extracted memories

        Example:
            >>> service = MemoryService()
            >>> request = ExtractionRequest(
            ...     text="I love coffee and work at Google",
            ...     user_id="user123"
            ... )
            >>> response = await service.extract_and_store_memories(request, db)
            >>> print(response.total_extracted)  # 2
        """
        start_time = time.time()
        logger.info(f"Processing memory extraction for user {request.user_id}")

        try:
            # Ensure user exists
            user = self._get_or_create_user(request.user_id, db)

            # Record conversation
            conversation = self._record_conversation(
                request.user_id,
                request.text,
                request.message_id,
                db
            )

            # Extract memories using Gemini
            memory_creates = await self.extractor.extract_memories(
                request.text,
                request.user_id,
                conversation.message_id
            )

            # Store memories
            stored_memories = []
            for memory_create in memory_creates:
                try:
                    memory_response = await self._store_single_memory(
                        memory_create,
                        request.user_id,
                        db
                    )
                    if memory_response:
                        stored_memories.append(memory_response)
                except Exception as e:
                    logger.warning(f"Failed to store memory: {e}")
                    continue

            # Update conversation with extraction count
            conversation.memories_extracted = len(stored_memories)
            conversation.processing_time = time.time() - start_time
            db.commit()

            processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Extracted and stored {len(stored_memories)} memories "
                f"for user {request.user_id} in {processing_time_ms:.1f}ms"
            )

            return ExtractionResponse(
                memories=stored_memories,
                processing_time_ms=processing_time_ms,
                message_id=conversation.message_id,
                total_extracted=len(stored_memories)
            )

        except Exception as e:
            logger.error(f"Memory extraction failed for user {request.user_id}: {e}")
            db.rollback()
            raise Exception(f"Memory extraction failed: {e}")

    async def _store_single_memory(
        self,
        memory_create: MemoryCreate,
        user_id: str,
        db: Session
    ) -> Optional[MemoryResponse]:
        """
        Store a single memory in database and vector store.

        Args:
            memory_create: Memory data to store
            user_id: User identifier
            db: Database session

        Returns:
            Optional[MemoryResponse]: Stored memory or None if failed
        """
        try:
            # Check for duplicates using enhanced deduplication
            if settings.enable_deduplication:
                existing_memory, action = await self.deduplicator.check_for_duplicates(
                    memory_create, user_id, db
                )

                if action == "duplicate":
                    logger.debug(f"Skipping duplicate memory: {memory_create.content[:50]}")
                    return None
                elif action == "update":
                    logger.debug(f"Updating existing memory: {existing_memory.memory_id}")
                    updated_memory = self.deduplicator.update_memory(existing_memory, memory_create, db)
                    return self._memory_to_response(updated_memory)
                elif action == "merge":
                    logger.debug(f"Merging with existing memory: {existing_memory.memory_id}")
                    merged_memory_create = self.deduplicator.merge_memories(existing_memory, memory_create)
                    # Continue with storing the merged memory
                    memory_create = merged_memory_create

            # Create new memory with enhanced metadata
            metadata = memory_create.metadata_ or {}

            # Add enhanced metadata fields
            metadata.update({
                'category': memory_create.category.value,
                'entities': memory_create.entities,
                'temporal_relevance': memory_create.temporal_relevance.value,
                'valid_contexts': [ctx.value for ctx in memory_create.valid_contexts],
                'invalid_contexts': [ctx.value for ctx in memory_create.invalid_contexts]
            })

            memory = Memory(
                user_id=user_id,
                memory_id=str(uuid.uuid4()),
                content=memory_create.content,
                memory_type=memory_create.memory_type.value,
                importance_score=memory_create.importance_score,
                confidence_score=memory_create.confidence_score,
                source_message_id=memory_create.source_message_id,
                metadata=metadata,
                expires_at=memory_create.expires_at
            )

            # Set expiration based on type if not provided
            if not memory.expires_at:
                memory.set_expiration_based_on_type()

            # Store in database
            db.add(memory)
            db.flush()  # Get the ID without committing

            # Generate embedding and store in vector database
            embedding, model_used = await self.embedding_service.get_embedding(memory.content)
            success = await self.retriever.store_memory(memory, embedding)

            if not success:
                logger.error(f"Failed to store memory {memory.memory_id} in vector database")
                db.rollback()
                return None

            db.commit()

            logger.debug(f"Stored memory {memory.memory_id} using {model_used} embeddings")

            # Convert to response
            return self._memory_to_response(memory)

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            db.rollback()
            return None

    def _find_duplicate_memory(
        self,
        memory_create: MemoryCreate,
        user_id: str,
        db: Session
    ) -> Optional[Memory]:
        """
        Find existing memory with similar content.

        Args:
            memory_create: New memory data
            user_id: User identifier
            db: Database session

        Returns:
            Optional[Memory]: Existing memory if found
        """
        try:
            # Look for memories with similar content and same type
            similar_memories = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.memory_type == memory_create.memory_type.value,
                Memory.is_active == True
            ).all()

            # Simple content similarity check
            for existing in similar_memories:
                if self._is_similar_content(existing.content, memory_create.content):
                    return existing

            return None

        except Exception as e:
            logger.warning(f"Error checking for duplicates: {e}")
            return None

    def _is_similar_content(self, content1: str, content2: str) -> bool:
        """
        Check if two memory contents are similar.

        Simple similarity check based on word overlap.

        Args:
            content1: First content
            content2: Second content

        Returns:
            bool: True if contents are similar
        """
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1.intersection(words2))
        total_unique = len(words1.union(words2))

        # Consider similar if >70% word overlap
        similarity = overlap / total_unique if total_unique > 0 else 0
        return similarity > 0.7

    async def _update_existing_memory(
        self,
        existing_memory: Memory,
        new_memory_create: MemoryCreate,
        db: Session
    ) -> Optional[MemoryResponse]:
        """
        Update existing memory with new information.

        Args:
            existing_memory: Existing memory to update
            new_memory_create: New memory data
            db: Database session

        Returns:
            Optional[MemoryResponse]: Updated memory
        """
        try:
            # Update fields
            existing_memory.content = new_memory_create.content
            existing_memory.importance_score = max(
                existing_memory.importance_score,
                new_memory_create.importance_score
            )
            existing_memory.confidence_score = max(
                existing_memory.confidence_score,
                new_memory_create.confidence_score
            )
            existing_memory.version += 1
            existing_memory.updated_at = datetime.utcnow()

            # Merge metadata
            if new_memory_create.metadata__:
                existing_metadata = existing_memory.get_metadata()
                existing_metadata.update(new_memory_create.metadata__)
                existing_memory.set_metadata(existing_metadata)

            # Update in vector database
            embedding, _ = await self.embedding_service.get_embedding(existing_memory.content)
            await self.retriever.update_memory(existing_memory, embedding)

            db.commit()

            logger.info(f"Updated existing memory {existing_memory.memory_id}")
            return self._memory_to_response(existing_memory)

        except Exception as e:
            logger.error(f"Failed to update existing memory: {e}")
            db.rollback()
            return None

    async def search_memories(
        self,
        request: MemorySearchRequest,
        db: Session
    ) -> MemorySearchResponse:
        """
        Search user memories semantically.

        Args:
            request: Search request parameters
            db: Database session

        Returns:
            MemorySearchResponse: Search results with metadata

        Example:
            >>> request = MemorySearchRequest(
            ...     query="work preferences",
            ...     user_id="user123",
            ...     limit=5
            ... )
            >>> response = await service.search_memories(request, db)
            >>> print(response.total_found)  # 3
        """
        start_time = time.time()

        try:
            # Perform vector search
            results = await self.retriever.search_memories(
                query=request.query,
                user_id=request.user_id,
                limit=request.limit,
                memory_types=request.memory_types,
                min_importance=request.min_importance,
                include_expired=request.include_expired
            )

            search_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Found {len(results)} memories for query '{request.query}' "
                f"by user {request.user_id} in {search_time_ms:.1f}ms"
            )

            return MemorySearchResponse(
                results=results,
                total_found=len(results),
                query=request.query,
                search_time_ms=search_time_ms
            )

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return MemorySearchResponse(
                results=[],
                total_found=0,
                query=request.query,
                search_time_ms=(time.time() - start_time) * 1000
            )

    async def smart_search_memories(
        self,
        query: str,
        user_id: str,
        context_type: Optional['ContextType'] = None,
        max_memories: Optional[int] = None,
        token_budget: Optional[int] = None,
        enable_validation: bool = True
    ) -> MemorySearchResponse:
        """
        Smart memory search with all RAG enhancements.

        Args:
            query: Search query
            user_id: User identifier
            context_type: Context for smart filtering
            max_memories: Maximum memories to return
            token_budget: Token budget for results
            enable_validation: Whether to validate relevance

        Returns:
            MemorySearchResponse: Enhanced search results

        Example:
            >>> response = await service.smart_search_memories(
            ...     "morning routine",
            ...     "user123",
            ...     context_type=ContextType.MORNING_CHECKIN,
            ...     token_budget=300
            ... )
            >>> # Returns contextually appropriate, validated results
        """
        start_time = time.time()

        try:
            from app.models.schemas import ContextType

            logger.info(f"Smart search for '{query}' with context {context_type}")

            # Use smart retriever with context awareness
            results = await self.smart_retriever.retrieve_memories(
                query=query,
                user_id=user_id,
                context_type=context_type,
                max_memories=max_memories,
                token_budget=token_budget
            )

            # Apply relevance validation if enabled
            if enable_validation and settings.enable_relevance_validation and results:
                validated_results = await self.validator.validate_memory_relevance(
                    query_context=query,
                    retrieved_memories=results,
                    context_type=context_type
                )
                results = validated_results

            # Apply token budget with summarization if needed
            if token_budget and results:
                preserved_memories, summary = await self.summarizer.summarize_within_budget(
                    results, token_budget, preserve_count=3
                )

                # If we have a summary, we could add it as metadata
                # For now, just use the preserved memories
                results = preserved_memories

            search_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Smart search completed: {len(results)} results in {search_time_ms:.1f}ms"
            )

            return MemorySearchResponse(
                results=results,
                total_found=len(results),
                query=query,
                search_time_ms=search_time_ms
            )

        except Exception as e:
            logger.error(f"Smart memory search failed: {e}")
            # Return empty results on error
            return MemorySearchResponse(
                results=[],
                total_found=0,
                query=query,
                search_time_ms=(time.time() - start_time) * 1000
            )

    def get_user_memories(
        self,
        user_id: str,
        db: Session,
        limit: int = 100,
        memory_type: Optional[MemoryType] = None,
        include_expired: bool = False
    ) -> List[MemoryResponse]:
        """
        Get all memories for a user.

        Args:
            user_id: User identifier
            db: Database session
            limit: Maximum number of memories
            memory_type: Filter by memory type
            include_expired: Include expired memories

        Returns:
            List[MemoryResponse]: User memories

        Example:
            >>> memories = service.get_user_memories("user123", db, limit=10)
            >>> print(len(memories))  # 8
        """
        try:
            query = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.is_active == True
            )

            if memory_type:
                query = query.filter(Memory.memory_type == memory_type.value)

            if not include_expired:
                current_time = datetime.utcnow()
                query = query.filter(
                    (Memory.expires_at.is_(None)) |
                    (Memory.expires_at > current_time)
                )

            memories = query.order_by(
                Memory.importance_score.desc(),
                Memory.updated_at.desc()
            ).limit(limit).all()

            return [self._memory_to_response(memory) for memory in memories]

        except Exception as e:
            logger.error(f"Failed to get user memories: {e}")
            return []

    def _get_or_create_user(self, user_id: str, db: Session) -> User:
        """
        Get existing user or create new one.

        Args:
            user_id: User identifier
            db: Database session

        Returns:
            User: User object
        """
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = User(user_id=user_id)
            db.add(user)
            try:
                db.commit()
                logger.info(f"Created new user: {user_id}")
            except IntegrityError:
                db.rollback()
                # User might have been created by another request
                user = db.query(User).filter(User.user_id == user_id).first()
        return user

    def _record_conversation(
        self,
        user_id: str,
        message: str,
        message_id: Optional[str],
        db: Session
    ) -> Conversation:
        """
        Record conversation message.

        Args:
            user_id: User identifier
            message: Message content
            message_id: Optional message ID
            db: Database session

        Returns:
            Conversation: Recorded conversation
        """
        conversation = Conversation(
            user_id=user_id,
            message_id=message_id or str(uuid.uuid4()),
            message=message
        )
        db.add(conversation)
        db.flush()
        return conversation

    def _memory_to_response(self, memory: Memory) -> MemoryResponse:
        """
        Convert Memory model to MemoryResponse.

        Args:
            memory: Memory model

        Returns:
            MemoryResponse: Response object
        """
        return MemoryResponse(
            memory_id=memory.memory_id,
            user_id=memory.user_id,
            content=memory.content,
            memory_type=MemoryType(memory.memory_type),
            importance_score=memory.importance_score,
            confidence_score=memory.confidence_score,
            extracted_at=memory.extracted_at,
            updated_at=memory.updated_at,
            expires_at=memory.expires_at,
            is_active=memory.is_active,
            source_message_id=memory.source_message_id,
            metadata=memory.get_metadata() if memory.metadata_ else None,
            version=memory.version
        )

    def get_user_stats(self, user_id: str, db: Session) -> Dict[str, any]:
        """
        Get user memory statistics.

        Args:
            user_id: User identifier
            db: Database session

        Returns:
            Dict: User statistics

        Example:
            >>> stats = service.get_user_stats("user123", db)
            >>> print(stats['total_memories'])  # 45
        """
        try:
            # Database stats
            total_memories = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.is_active == True
            ).count()

            active_memories = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.is_active == True,
                (Memory.expires_at.is_(None)) | (Memory.expires_at > datetime.utcnow())
            ).count()

            # Vector database stats
            vector_stats = self.retriever.get_collection_stats(user_id)

            return {
                'user_id': user_id,
                'total_memories': total_memories,
                'active_memories': active_memories,
                'vector_stats': vector_stats,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {'error': str(e)}