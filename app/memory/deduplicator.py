"""
Memory deduplication and update logic.

Prevents duplicate memories and handles intelligent updates
to existing memories when new similar information is provided.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import difflib

from sqlalchemy.orm import Session

from app.models.memory import Memory
from app.models.schemas import MemoryCreate, MemoryType
from app.config.rag_config import RAG_CONFIG

logger = logging.getLogger(__name__)


class MemoryDeduplicator:
    """
    Handles memory deduplication and intelligent updates.

    Prevents storage of duplicate memories and determines when
    to update existing memories vs creating new ones.
    """

    def __init__(self, embedding_service):
        """
        Initialize deduplicator.

        Args:
            embedding_service: Service for semantic similarity
        """
        self.embedding_service = embedding_service

    async def check_for_duplicates(
        self,
        new_memory: MemoryCreate,
        user_id: str,
        db: Session
    ) -> Tuple[Optional[Memory], str]:
        """
        Check if new memory is duplicate or update of existing memory.

        Args:
            new_memory: New memory to check
            user_id: User identifier
            db: Database session

        Returns:
            Tuple[Optional[Memory], str]: (existing_memory, action)
            where action is 'duplicate', 'update', 'merge', or 'new'

        Example:
            >>> existing, action = await dedup.check_for_duplicates(
            ...     new_memory, "user123", db
            ... )
            >>> if action == "duplicate":
            ...     # Skip storing
            >>> elif action == "update":
            ...     # Update existing memory
        """
        try:
            # Find similar memories of same type and category
            similar_memories = await self._find_similar_memories(
                new_memory, user_id, db
            )

            if not similar_memories:
                return None, "new"

            # Check each similar memory
            for existing_memory, similarity_score in similar_memories:
                action = await self._determine_action(
                    existing_memory, new_memory, similarity_score
                )

                if action != "new":
                    return existing_memory, action

            return None, "new"

        except Exception as e:
            logger.error(f"Duplicate check failed: {e}")
            return None, "new"  # Default to creating new memory

    async def _find_similar_memories(
        self,
        new_memory: MemoryCreate,
        user_id: str,
        db: Session
    ) -> List[Tuple[Memory, float]]:
        """
        Find memories similar to the new memory.

        Args:
            new_memory: New memory to compare
            user_id: User identifier
            db: Database session

        Returns:
            List[Tuple[Memory, float]]: List of (memory, similarity_score)
        """
        try:
            # Get memories of same type and category
            candidate_memories = db.query(Memory).filter(
                Memory.user_id == user_id,
                Memory.memory_type == new_memory.memory_type.value,
                Memory.is_active == True
            ).limit(50).all()  # Limit for performance

            if not candidate_memories:
                return []

            # Calculate semantic similarity
            new_embedding, _ = await self.embedding_service.get_embedding(new_memory.content)
            similar_memories = []

            for memory in candidate_memories:
                try:
                    # Get existing memory embedding
                    existing_embedding, _ = await self.embedding_service.get_embedding(memory.content)

                    # Calculate cosine similarity
                    similarity = self._calculate_cosine_similarity(new_embedding, existing_embedding)

                    # Only consider if above threshold
                    if similarity >= RAG_CONFIG["SIMILAR_THRESHOLD"]:
                        similar_memories.append((memory, similarity))

                except Exception as e:
                    logger.debug(f"Failed to calculate similarity for memory {memory.id}: {e}")
                    continue

            # Sort by similarity
            similar_memories.sort(key=lambda x: x[1], reverse=True)

            return similar_memories

        except Exception as e:
            logger.error(f"Finding similar memories failed: {e}")
            return []

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity (0-1)
        """
        try:
            import numpy as np

            v1 = np.array(vec1)
            v2 = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.debug(f"Cosine similarity calculation failed: {e}")
            # Fallback to text similarity
            return self._calculate_text_similarity(
                " ".join(str(x) for x in vec1[:10]),  # Use first 10 dimensions as text
                " ".join(str(x) for x in vec2[:10])
            )

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using difflib.

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Text similarity (0-1)
        """
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    async def _determine_action(
        self,
        existing_memory: Memory,
        new_memory: MemoryCreate,
        similarity_score: float
    ) -> str:
        """
        Determine what action to take with similar memories.

        Args:
            existing_memory: Existing memory in database
            new_memory: New memory being added
            similarity_score: Semantic similarity score

        Returns:
            str: Action to take ('duplicate', 'update', 'merge', 'new')
        """
        try:
            # Very high similarity = duplicate
            if similarity_score >= RAG_CONFIG["DUPLICATE_THRESHOLD"]:
                # Check if content is essentially identical
                text_similarity = self._calculate_text_similarity(
                    existing_memory.content, new_memory.content
                )
                if text_similarity >= 0.9:
                    return "duplicate"

            # High similarity = potential update
            if similarity_score >= RAG_CONFIG["UPDATE_THRESHOLD"]:
                return self._analyze_update_type(existing_memory, new_memory)

            # Medium similarity = check for merge
            if similarity_score >= RAG_CONFIG["SIMILAR_THRESHOLD"]:
                if self._should_merge(existing_memory, new_memory):
                    return "merge"

            return "new"

        except Exception as e:
            logger.error(f"Action determination failed: {e}")
            return "new"

    def _analyze_update_type(self, existing_memory: Memory, new_memory: MemoryCreate) -> str:
        """
        Analyze whether new memory is an update to existing memory.

        Args:
            existing_memory: Existing memory
            new_memory: New memory

        Returns:
            str: 'update' if it's an update, 'new' otherwise
        """
        # Check for update indicators
        update_indicators = [
            # Time-related updates
            ("tomorrow", "today"),
            ("next week", "this week"),
            ("later", "now"),
            # Status updates
            ("will", "did"),
            ("planning to", "finished"),
            ("going to", "went to"),
            # Correction words
            ("actually", "originally"),
            ("changed", "decided"),
            ("moved", "scheduled"),
        ]

        new_content_lower = new_memory.content.lower()
        existing_content_lower = existing_memory.content.lower()

        # Check for update indicators in new content
        for indicator, _ in update_indicators:
            if indicator in new_content_lower:
                return "update"

        # Check for contradictory information
        if self._has_contradictory_info(existing_memory, new_memory):
            return "update"

        # Check if new memory has more specific information
        if self._is_more_specific(new_memory.content, existing_memory.content):
            return "update"

        return "new"

    def _should_merge(self, existing_memory: Memory, new_memory: MemoryCreate) -> bool:
        """
        Determine if memories should be merged.

        Args:
            existing_memory: Existing memory
            new_memory: New memory

        Returns:
            bool: True if memories should be merged
        """
        # Don't merge different types
        if existing_memory.memory_type != new_memory.memory_type.value:
            return False

        # Check for complementary information
        existing_words = set(existing_memory.content.lower().split())
        new_words = set(new_memory.content.lower().split())

        # If new memory adds significant new information, consider merging
        unique_new_words = new_words - existing_words
        if len(unique_new_words) >= 3:  # Significant new information
            return True

        return False

    def _has_contradictory_info(self, existing_memory: Memory, new_memory: MemoryCreate) -> bool:
        """
        Check if new memory contradicts existing memory.

        Args:
            existing_memory: Existing memory
            new_memory: New memory

        Returns:
            bool: True if there's contradictory information
        """
        # Time contradictions
        time_words = ["morning", "afternoon", "evening", "night"]
        existing_times = [word for word in time_words if word in existing_memory.content.lower()]
        new_times = [word for word in time_words if word in new_memory.content.lower()]

        if existing_times and new_times and existing_times != new_times:
            return True

        # Day contradictions
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        existing_days = [day for day in days if day in existing_memory.content.lower()]
        new_days = [day for day in days if day in new_memory.content.lower()]

        if existing_days and new_days and existing_days != new_days:
            return True

        # Preference contradictions
        if existing_memory.memory_type == MemoryType.PREFERENCE.value:
            existing_lower = existing_memory.content.lower()
            new_lower = new_memory.content.lower()

            # Check for opposing sentiments
            if ("love" in existing_lower and "hate" in new_lower) or \
               ("hate" in existing_lower and "love" in new_lower) or \
               ("like" in existing_lower and "dislike" in new_lower) or \
               ("dislike" in existing_lower and "like" in new_lower):
                return True

        return False

    def _is_more_specific(self, new_content: str, existing_content: str) -> bool:
        """
        Check if new content is more specific than existing content.

        Args:
            new_content: New memory content
            existing_content: Existing memory content

        Returns:
            bool: True if new content is more specific
        """
        # Longer content might be more specific
        if len(new_content) > len(existing_content) * 1.5:
            return True

        # Check for specific details (numbers, times, places)
        import re

        # Numbers (times, dates, quantities)
        new_numbers = len(re.findall(r'\d+', new_content))
        existing_numbers = len(re.findall(r'\d+', existing_content))

        if new_numbers > existing_numbers:
            return True

        # Proper nouns (names, places)
        new_proper = len(re.findall(r'\b[A-Z][a-z]+\b', new_content))
        existing_proper = len(re.findall(r'\b[A-Z][a-z]+\b', existing_content))

        if new_proper > existing_proper:
            return True

        return False

    def merge_memories(
        self,
        existing_memory: Memory,
        new_memory: MemoryCreate
    ) -> MemoryCreate:
        """
        Merge two memories into a single enhanced memory.

        Args:
            existing_memory: Existing memory
            new_memory: New memory to merge

        Returns:
            MemoryCreate: Merged memory
        """
        try:
            # Combine content intelligently
            merged_content = self._merge_content(existing_memory.content, new_memory.content)

            # Use higher confidence and importance
            merged_confidence = max(existing_memory.confidence_score, new_memory.confidence_score)
            merged_importance = max(existing_memory.importance_score, new_memory.importance_score)

            # Merge entities
            existing_entities = existing_memory.get_metadata().get('entities', [])
            new_entities = new_memory.entities or []
            merged_entities = list(set(existing_entities + new_entities))

            # Merge metadata
            existing_metadata = existing_memory.get_metadata()
            new_metadata = new_memory.metadata_ or {}
            merged_metadata = {**existing_metadata, **new_metadata}

            return MemoryCreate(
                content=merged_content,
                memory_type=new_memory.memory_type,
                category=new_memory.category,
                entities=merged_entities,
                temporal_relevance=new_memory.temporal_relevance,
                importance_score=merged_importance,
                confidence_score=merged_confidence,
                valid_contexts=new_memory.valid_contexts,
                invalid_contexts=new_memory.invalid_contexts,
                source_message_id=new_memory.source_message_id,
                metadata=merged_metadata
            )

        except Exception as e:
            logger.error(f"Memory merging failed: {e}")
            return new_memory

    def _merge_content(self, existing_content: str, new_content: str) -> str:
        """
        Intelligently merge content from two memories.

        Args:
            existing_content: Existing memory content
            new_content: New memory content

        Returns:
            str: Merged content
        """
        # Simple merge strategy: combine with "and" if different enough
        if self._calculate_text_similarity(existing_content, new_content) < 0.7:
            return f"{existing_content} and {new_content}"
        else:
            # Use the longer, more detailed version
            return new_content if len(new_content) > len(existing_content) else existing_content

    def update_memory(
        self,
        existing_memory: Memory,
        new_memory: MemoryCreate,
        db: Session
    ) -> Memory:
        """
        Update existing memory with new information.

        Args:
            existing_memory: Memory to update
            new_memory: New memory with updated information
            db: Database session

        Returns:
            Memory: Updated memory
        """
        try:
            # Update core fields
            existing_memory.content = new_memory.content
            existing_memory.confidence_score = max(
                existing_memory.confidence_score,
                new_memory.confidence_score
            )
            existing_memory.importance_score = max(
                existing_memory.importance_score,
                new_memory.importance_score
            )

            # Update version and timestamp
            existing_memory.version += 1
            existing_memory.updated_at = datetime.utcnow()

            # Update metadata
            existing_metadata = existing_memory.get_metadata()
            new_metadata = new_memory.metadata_ or {}

            # Merge entities
            existing_entities = existing_metadata.get('entities', [])
            new_entities = new_memory.entities or []
            merged_entities = list(set(existing_entities + new_entities))

            # Update metadata
            updated_metadata = {
                **existing_metadata,
                **new_metadata,
                'entities': merged_entities,
                'updated_from': new_memory.source_message_id,
                'update_count': existing_metadata.get('update_count', 0) + 1
            }

            existing_memory.set_metadata(updated_metadata)

            # Set expiration if applicable
            if new_memory.expires_at:
                existing_memory.expires_at = new_memory.expires_at

            db.commit()

            logger.info(f"Updated memory {existing_memory.memory_id} to version {existing_memory.version}")
            return existing_memory

        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            db.rollback()
            raise