"""
Smart retrieval system with context-aware strategies.

Implements intelligent memory retrieval with relevance filtering,
context awareness, and token budget management to prevent
irrelevant or overwhelming memory retrieval.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set

from app.config.rag_config import (
    RAG_CONFIG, RETRIEVAL_STRATEGIES, ContextType, MemoryCategory,
    get_retrieval_strategy, should_include_memory
)
from app.models.schemas import MemoryType, MemoryResponse, MemorySearchResult
from app.memory.retriever import MemoryRetriever
from app.memory.embedder import EmbeddingService

logger = logging.getLogger(__name__)


class SmartRetriever:
    """
    Enhanced memory retriever with context-aware strategies.

    Provides intelligent filtering, relevance validation, and token management
    to ensure only appropriate memories are retrieved for specific contexts.
    """

    def __init__(self, base_retriever: MemoryRetriever, embedding_service: EmbeddingService):
        """
        Initialize smart retriever.

        Args:
            base_retriever: Base memory retriever
            embedding_service: Embedding service for query processing
        """
        self.base_retriever = base_retriever
        self.embedding_service = embedding_service

    async def retrieve_memories(
        self,
        query: str,
        user_id: str,
        context_type: Optional[ContextType] = None,
        max_memories: Optional[int] = None,
        min_relevance: Optional[float] = None,
        token_budget: Optional[int] = None,
        time_window_hours: Optional[int] = None
    ) -> List[MemorySearchResult]:
        """
        Retrieve memories with smart filtering and context awareness.

        Args:
            query: Search query
            user_id: User identifier
            context_type: Context for retrieval strategy
            max_memories: Maximum memories to return
            min_relevance: Minimum relevance threshold
            token_budget: Token budget for memories
            time_window_hours: Time window for temporal filtering

        Returns:
            List[MemorySearchResult]: Filtered and ranked memories

        Example:
            >>> retriever = SmartRetriever(base_retriever, embedding_service)
            >>> results = await retriever.retrieve_memories(
            ...     "morning routine",
            ...     "user123",
            ...     context_type=ContextType.MORNING_CHECKIN
            ... )
            >>> # Only returns morning-relevant memories
        """
        start_time = time.time()

        try:
            # Get retrieval strategy
            strategy = get_retrieval_strategy(context_type or ContextType.GENERAL_CHAT)

            # Apply strategy defaults
            max_memories = max_memories or strategy.max_memories
            min_relevance = min_relevance or strategy.min_relevance
            token_budget = token_budget or RAG_CONFIG["TOKEN_BUDGET_PER_RETRIEVAL"]
            time_window_hours = time_window_hours or strategy.time_window_hours

            logger.info(
                f"Smart retrieval for '{query}' with context {context_type}, "
                f"max_memories={max_memories}, min_relevance={min_relevance}"
            )

            # Step 1: Initial broad retrieval
            initial_results = await self._broad_retrieval(
                query, user_id, max_memories * 3, time_window_hours
            )

            if not initial_results:
                return await self._fallback_retrieval(user_id, max_memories)

            # Step 2: Context-aware filtering
            context_filtered = self._apply_context_filtering(initial_results, strategy)

            # Step 3: Relevance threshold filtering
            relevance_filtered = self._apply_relevance_filtering(context_filtered, min_relevance)

            # Step 4: Token budget management
            budget_managed = await self._apply_token_budget(
                relevance_filtered, token_budget, max_memories
            )

            # Step 5: Final ranking with strategy weights
            final_results = self._apply_strategy_ranking(budget_managed, strategy)

            duration = time.time() - start_time
            logger.info(
                f"Smart retrieval completed: {len(initial_results)} → "
                f"{len(context_filtered)} → {len(relevance_filtered)} → "
                f"{len(budget_managed)} → {len(final_results)} in {duration:.2f}s"
            )

            return final_results[:max_memories]

        except Exception as e:
            logger.error(f"Smart retrieval failed: {e}")
            # Fallback to basic retrieval
            return await self._emergency_fallback(query, user_id)

    async def _broad_retrieval(
        self,
        query: str,
        user_id: str,
        limit: int,
        time_window_hours: int
    ) -> List[MemorySearchResult]:
        """
        Initial broad retrieval without strict filtering.

        Args:
            query: Search query
            user_id: User identifier
            limit: Number of results to fetch
            time_window_hours: Time window for temporal filtering

        Returns:
            List[MemorySearchResult]: Initial results
        """
        try:
            # Calculate time cutoff
            if time_window_hours > 0:
                cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
                # This would need to be implemented in the base retriever
                # For now, we'll use the existing search

            results = await self.base_retriever.search_memories(
                query=query,
                user_id=user_id,
                limit=limit,
                include_expired=False
            )

            return results

        except Exception as e:
            logger.error(f"Broad retrieval failed: {e}")
            return []

    def _apply_context_filtering(
        self,
        results: List[MemorySearchResult],
        strategy: 'RetrievalStrategy'
    ) -> List[MemorySearchResult]:
        """
        Filter results based on context strategy.

        Args:
            results: Search results to filter
            strategy: Retrieval strategy with filtering rules

        Returns:
            List[MemorySearchResult]: Context-filtered results
        """
        filtered = []

        for result in results:
            memory = result.memory

            # Extract category from metadata or memory content
            category = self._extract_memory_category(memory)
            memory_type = memory.memory_type

            # Check if memory should be included
            if should_include_memory(category, memory_type, strategy):
                # Check context-specific exclusions
                if self._check_context_specific_rules(memory, strategy):
                    filtered.append(result)
                else:
                    logger.debug(f"Excluded by context rules: {memory.content[:50]}")
            else:
                logger.debug(f"Excluded by type/category filter: {memory.content[:50]}")

        logger.debug(f"Context filtering: {len(results)} → {len(filtered)}")
        return filtered

    def _extract_memory_category(self, memory: MemoryResponse) -> MemoryCategory:
        """
        Extract memory category from memory metadata.

        Args:
            memory: Memory response object

        Returns:
            MemoryCategory: Extracted or inferred category
        """
        from app.config.rag_config import classify_memory_category

        # Try to get from metadata first
        if memory.metadata_ and 'category' in memory.metadata_:
            try:
                return MemoryCategory(memory.metadata_['category'])
            except ValueError:
                pass

        # Fallback to classification
        return classify_memory_category(memory.content, {"memory_type": memory.memory_type.value})

    def _check_context_specific_rules(
        self,
        memory: MemoryResponse,
        strategy: 'RetrievalStrategy'
    ) -> bool:
        """
        Apply context-specific filtering rules.

        Args:
            memory: Memory to check
            strategy: Retrieval strategy

        Returns:
            bool: True if memory passes context rules
        """
        # Check valid/invalid contexts if available in metadata
        if memory.metadata_:
            valid_contexts = memory.metadata_.get('valid_contexts', [])
            invalid_contexts = memory.metadata_.get('invalid_contexts', [])

            # If invalid contexts are specified and current context is in them, exclude
            if invalid_contexts and any(ctx in invalid_contexts for ctx in strategy.include_types):
                return False

            # If valid contexts are specified and current context is not in them, check carefully
            if valid_contexts and not any(ctx in valid_contexts for ctx in strategy.include_types):
                # Allow if it's a high-importance memory
                if memory.importance_score < 0.8:
                    return False

        return True

    def _apply_relevance_filtering(
        self,
        results: List[MemorySearchResult],
        min_relevance: float
    ) -> List[MemorySearchResult]:
        """
        Filter results by relevance threshold.

        Args:
            results: Search results to filter
            min_relevance: Minimum relevance score

        Returns:
            List[MemorySearchResult]: Relevance-filtered results
        """
        filtered = [
            result for result in results
            if result.similarity_score >= min_relevance
        ]

        logger.debug(f"Relevance filtering: {len(results)} → {len(filtered)} (threshold: {min_relevance})")
        return filtered

    async def _apply_token_budget(
        self,
        results: List[MemorySearchResult],
        token_budget: int,
        max_memories: int
    ) -> List[MemorySearchResult]:
        """
        Manage token budget and summarize if needed.

        Args:
            results: Search results
            token_budget: Available token budget
            max_memories: Maximum number of memories

        Returns:
            List[MemorySearchResult]: Budget-managed results
        """
        if not results:
            return results

        # Estimate tokens per memory
        avg_tokens = RAG_CONFIG["AVERAGE_TOKENS_PER_MEMORY"]
        estimated_tokens = len(results) * avg_tokens

        if estimated_tokens <= token_budget:
            # Within budget, return as is
            return results[:max_memories]

        # Calculate how many memories fit in budget
        budget_count = min(max_memories, token_budget // avg_tokens)

        if budget_count <= 0:
            # Emergency case: return minimal set
            return results[:RAG_CONFIG["EMERGENCY_MAX_MEMORIES"]]

        # Prioritize by importance and relevance
        sorted_results = sorted(
            results,
            key=lambda r: (r.similarity_score * 0.6 + r.memory.importance_score * 0.4),
            reverse=True
        )

        return sorted_results[:budget_count]

    def _apply_strategy_ranking(
        self,
        results: List[MemorySearchResult],
        strategy: 'RetrievalStrategy'
    ) -> List[MemorySearchResult]:
        """
        Apply strategy-specific ranking weights.

        Args:
            results: Search results to rank
            strategy: Retrieval strategy with weights

        Returns:
            List[MemorySearchResult]: Re-ranked results
        """
        for result in results:
            # Calculate strategic score
            base_score = result.similarity_score
            importance_boost = result.memory.importance_score * strategy.importance_weight

            # Apply temporal boost for recent memories
            if strategy.temporal_boost > 0:
                age_hours = (datetime.utcnow() - result.memory.extracted_at).total_seconds() / 3600
                if age_hours <= strategy.time_window_hours:
                    temporal_boost = strategy.temporal_boost * (1 - age_hours / strategy.time_window_hours)
                else:
                    temporal_boost = 0
            else:
                temporal_boost = 0

            # Combine scores
            strategic_score = min(1.0, base_score + importance_boost + temporal_boost)

            # Update similarity score with strategic score
            result.similarity_score = strategic_score

        # Sort by strategic score
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        return results

    async def _fallback_retrieval(
        self,
        user_id: str,
        max_memories: int
    ) -> List[MemorySearchResult]:
        """
        Fallback retrieval when primary search fails.

        Args:
            user_id: User identifier
            max_memories: Maximum memories to return

        Returns:
            List[MemorySearchResult]: Fallback results
        """
        logger.info(f"Using fallback retrieval for user {user_id}")

        try:
            # Get recent important memories
            fallback_count = min(max_memories, RAG_CONFIG["FALLBACK_MEMORY_COUNT"])

            # This would need to be implemented as a new method in base retriever
            # For now, return empty to avoid errors
            return []

        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")
            return []

    async def _emergency_fallback(
        self,
        query: str,
        user_id: str
    ) -> List[MemorySearchResult]:
        """
        Emergency fallback when all retrieval fails.

        Args:
            query: Original query
            user_id: User identifier

        Returns:
            List[MemorySearchResult]: Emergency results
        """
        logger.warning(f"Using emergency fallback for query '{query}'")

        try:
            # Try basic retrieval with very permissive settings
            results = await self.base_retriever.search_memories(
                query=query,
                user_id=user_id,
                limit=RAG_CONFIG["EMERGENCY_MAX_MEMORIES"]
            )

            return results

        except Exception as e:
            logger.error(f"Emergency fallback failed: {e}")
            return []

    def estimate_token_usage(self, results: List[MemorySearchResult]) -> int:
        """
        Estimate token usage for memory results.

        Args:
            results: Search results

        Returns:
            int: Estimated token count
        """
        total_chars = sum(len(result.memory.content) for result in results)
        # Rough estimation: 4 characters per token
        return total_chars // 4

    async def validate_relevance_with_llm(
        self,
        query: str,
        results: List[MemorySearchResult],
        context_type: ContextType
    ) -> List[MemorySearchResult]:
        """
        Use LLM to validate relevance of retrieved memories.

        This is implemented as a separate method that can be called
        for high-stakes retrievals.

        Args:
            query: Original query
            results: Retrieved memories
            context_type: Context type

        Returns:
            List[MemorySearchResult]: LLM-validated results
        """
        # This would use Gemini to validate relevance
        # For now, return the results as-is
        logger.info(f"LLM validation not yet implemented for query: {query}")
        return results