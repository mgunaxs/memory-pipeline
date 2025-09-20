"""
Token-aware memory summarization.

Intelligently summarizes and prioritizes memories when token budgets
are exceeded, ensuring the most important information is preserved.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from app.core.rate_limiter import call_gemini_with_retry, RateLimitError
from app.models.schemas import MemorySearchResult, MemoryType, MemoryCategory
from app.config.rag_config import RAG_CONFIG

logger = logging.getLogger(__name__)


class MemorySummarizer:
    """
    Summarizes memories when token budgets are exceeded.

    Provides intelligent summarization that preserves the most important
    information while staying within token constraints.
    """

    def __init__(self):
        """Initialize memory summarizer."""
        self.model_name = "gemini-1.5-flash"

    async def summarize_within_budget(
        self,
        memories: List[MemorySearchResult],
        token_budget: int,
        preserve_count: int = 3
    ) -> Tuple[List[MemorySearchResult], Optional[str]]:
        """
        Summarize memories to fit within token budget.

        Args:
            memories: List of memories to potentially summarize
            token_budget: Available token budget
            preserve_count: Number of top memories to keep verbatim

        Returns:
            Tuple[List[MemorySearchResult], Optional[str]]:
            (preserved_memories, summary_of_remaining)

        Example:
            >>> summarizer = MemorySummarizer()
            >>> preserved, summary = await summarizer.summarize_within_budget(
            ...     memories, token_budget=300, preserve_count=2
            ... )
            >>> # Top 2 memories preserved, rest summarized
        """
        if not memories:
            return [], None

        try:
            # Estimate current token usage
            current_tokens = self._estimate_total_tokens(memories)

            if current_tokens <= token_budget:
                # Within budget, no summarization needed
                return memories, None

            logger.info(f"Token budget exceeded: {current_tokens} > {token_budget}, summarizing...")

            # Sort memories by importance and relevance
            sorted_memories = self._prioritize_memories(memories)

            # Preserve top N memories
            preserved_memories = sorted_memories[:preserve_count]
            remaining_memories = sorted_memories[preserve_count:]

            if not remaining_memories:
                return preserved_memories, None

            # Calculate remaining budget after preserved memories
            preserved_tokens = self._estimate_total_tokens(preserved_memories)
            remaining_budget = token_budget - preserved_tokens - 50  # Reserve for summary overhead

            if remaining_budget <= 0:
                # No room for summary, just return preserved
                return preserved_memories, None

            # Summarize remaining memories
            summary = await self._create_summary(remaining_memories, remaining_budget)

            logger.info(
                f"Summarization complete: {len(memories)} → {len(preserved_memories)} preserved + summary"
            )

            return preserved_memories, summary

        except Exception as e:
            logger.error(f"Memory summarization failed: {e}")
            # Fallback: return top memories within budget
            return self._emergency_truncate(memories, token_budget), None

    def _prioritize_memories(self, memories: List[MemorySearchResult]) -> List[MemorySearchResult]:
        """
        Prioritize memories by importance and relevance.

        Args:
            memories: Memories to prioritize

        Returns:
            List[MemorySearchResult]: Prioritized memories
        """
        def priority_score(memory_result: MemorySearchResult) -> float:
            memory = memory_result.memory

            # Base score from similarity and importance
            base_score = (memory_result.similarity_score * 0.6 +
                         memory.importance_score * 0.4)

            # Boost for certain types
            type_boost = {
                MemoryType.FACT: 0.1,      # Facts are important for context
                MemoryType.EVENT: 0.05,    # Events are time-sensitive
                MemoryType.PREFERENCE: 0.0, # Neutral
                MemoryType.ROUTINE: 0.0,   # Neutral
                MemoryType.EMOTION: -0.05  # Emotions less critical for summaries
            }.get(memory.memory_type, 0.0)

            # Recent memories get slight boost
            age_days = (datetime.utcnow() - memory.extracted_at).days
            recency_boost = max(0, 0.1 - (age_days / 30 * 0.1))  # Decays over 30 days

            return base_score + type_boost + recency_boost

        return sorted(memories, key=priority_score, reverse=True)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            int: Estimated token count
        """
        # Rough estimation: 4 characters per token on average
        return len(text) // 4

    def _estimate_total_tokens(self, memories: List[MemorySearchResult]) -> int:
        """
        Estimate total tokens for list of memories.

        Args:
            memories: Memories to estimate

        Returns:
            int: Total estimated tokens
        """
        return sum(self._estimate_tokens(memory.memory.content) for memory in memories)

    async def _create_summary(
        self,
        memories: List[MemorySearchResult],
        target_tokens: int
    ) -> str:
        """
        Create summary of memories within token budget.

        Args:
            memories: Memories to summarize
            target_tokens: Target token count for summary

        Returns:
            str: Summary text
        """
        try:
            # Group memories by category/type for better summarization
            grouped_memories = self._group_memories_for_summary(memories)

            # Create summarization prompt
            prompt = self._build_summarization_prompt(grouped_memories, target_tokens)

            # Call Gemini for summarization
            response = await call_gemini_with_retry(
                self.model_name,
                prompt,
                temperature=0.3,  # Moderate creativity for good summaries
                max_output_tokens=min(target_tokens + 50, 512)  # Allow some buffer
            )

            summary = response.text.strip()

            # Validate summary length
            if self._estimate_tokens(summary) > target_tokens * 1.2:  # 20% tolerance
                # Truncate if too long
                summary = self._truncate_to_budget(summary, target_tokens)

            logger.debug(f"Created summary: {len(summary)} chars, ~{self._estimate_tokens(summary)} tokens")
            return summary

        except RateLimitError:
            logger.warning("Rate limit hit during summarization, using fallback")
            return self._create_fallback_summary(memories, target_tokens)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return self._create_fallback_summary(memories, target_tokens)

    def _group_memories_for_summary(
        self,
        memories: List[MemorySearchResult]
    ) -> Dict[str, List[MemorySearchResult]]:
        """
        Group memories by category for structured summarization.

        Args:
            memories: Memories to group

        Returns:
            Dict[str, List[MemorySearchResult]]: Grouped memories
        """
        groups = {}

        for memory_result in memories:
            memory = memory_result.memory

            # Group by type primarily
            group_key = memory.memory_type.value

            # Sub-group by category if available in metadata
            if memory.metadata_ and 'category' in memory.metadata_:
                category = memory.metadata_['category']
                group_key = f"{memory.memory_type.value}_{category}"

            if group_key not in groups:
                groups[group_key] = []

            groups[group_key].append(memory_result)

        return groups

    def _build_summarization_prompt(
        self,
        grouped_memories: Dict[str, List[MemorySearchResult]],
        target_tokens: int
    ) -> str:
        """
        Build prompt for memory summarization.

        Args:
            grouped_memories: Memories grouped by category
            target_tokens: Target token count

        Returns:
            str: Summarization prompt
        """
        memories_text = ""
        for group_name, group_memories in grouped_memories.items():
            memories_text += f"\n{group_name.upper()}:\n"
            for memory_result in group_memories:
                memories_text += f"- {memory_result.memory.content}\n"

        prompt = f"""Summarize the following memories concisely while preserving key information.

TARGET LENGTH: Approximately {target_tokens} tokens ({target_tokens * 4} characters)

MEMORIES TO SUMMARIZE:
{memories_text}

SUMMARIZATION GUIDELINES:
1. Group related information together
2. Preserve specific facts, names, and important details
3. Use clear, concise language
4. Maintain the user's perspective (use "I" statements)
5. Prioritize actionable and contextually relevant information
6. Remove redundant information

OUTPUT: A well-structured summary that captures the essence of these memories in approximately {target_tokens} tokens.
"""

        return prompt

    def _create_fallback_summary(
        self,
        memories: List[MemorySearchResult],
        target_tokens: int
    ) -> str:
        """
        Create simple fallback summary without LLM.

        Args:
            memories: Memories to summarize
            target_tokens: Target token count

        Returns:
            str: Fallback summary
        """
        try:
            # Simple extraction of key information
            summary_parts = []

            # Group by type
            type_groups = {}
            for memory_result in memories:
                mem_type = memory_result.memory.memory_type.value
                if mem_type not in type_groups:
                    type_groups[mem_type] = []
                type_groups[mem_type].append(memory_result.memory.content)

            # Create summary for each type
            for mem_type, contents in type_groups.items():
                if mem_type == "fact":
                    summary_parts.append(f"Facts: {', '.join(contents[:3])}")
                elif mem_type == "preference":
                    summary_parts.append(f"Preferences: {', '.join(contents[:3])}")
                elif mem_type == "event":
                    summary_parts.append(f"Events: {', '.join(contents[:2])}")
                elif mem_type == "routine":
                    summary_parts.append(f"Routines: {', '.join(contents[:2])}")

            summary = ". ".join(summary_parts)

            # Truncate if too long
            return self._truncate_to_budget(summary, target_tokens)

        except Exception as e:
            logger.error(f"Fallback summary failed: {e}")
            return "Additional user context available."

    def _truncate_to_budget(self, text: str, target_tokens: int) -> str:
        """
        Truncate text to fit within token budget.

        Args:
            text: Text to truncate
            target_tokens: Target token count

        Returns:
            str: Truncated text
        """
        target_chars = target_tokens * 4  # Rough estimation

        if len(text) <= target_chars:
            return text

        # Truncate at sentence boundary if possible
        truncated = text[:target_chars]
        last_period = truncated.rfind('.')
        last_comma = truncated.rfind(',')

        if last_period > target_chars * 0.8:  # If period is reasonably close
            return truncated[:last_period + 1]
        elif last_comma > target_chars * 0.8:  # If comma is reasonably close
            return truncated[:last_comma + 1] + "..."
        else:
            return truncated + "..."

    def _emergency_truncate(
        self,
        memories: List[MemorySearchResult],
        token_budget: int
    ) -> List[MemorySearchResult]:
        """
        Emergency truncation when summarization fails.

        Args:
            memories: Memories to truncate
            token_budget: Available budget

        Returns:
            List[MemorySearchResult]: Truncated memory list
        """
        try:
            # Sort by priority
            prioritized = self._prioritize_memories(memories)

            # Add memories until budget is exceeded
            selected = []
            current_tokens = 0

            for memory_result in prioritized:
                memory_tokens = self._estimate_tokens(memory_result.memory.content)

                if current_tokens + memory_tokens <= token_budget:
                    selected.append(memory_result)
                    current_tokens += memory_tokens
                else:
                    break

            logger.warning(f"Emergency truncation: {len(memories)} → {len(selected)} memories")
            return selected

        except Exception as e:
            logger.error(f"Emergency truncation failed: {e}")
            # Return minimal set
            return memories[:RAG_CONFIG["EMERGENCY_MAX_MEMORIES"]]

    async def create_contextual_summary(
        self,
        memories: List[MemorySearchResult],
        context_description: str,
        token_budget: int
    ) -> str:
        """
        Create context-aware summary of memories.

        Args:
            memories: Memories to summarize
            context_description: Description of context for summary
            token_budget: Token budget for summary

        Returns:
            str: Context-aware summary
        """
        try:
            if not memories:
                return ""

            # Build context-aware prompt
            memories_text = "\n".join([
                f"- {memory.memory.content}" for memory in memories
            ])

            prompt = f"""Create a contextual summary for: {context_description}

MEMORIES:
{memories_text}

Create a summary that:
1. Focuses on information relevant to: {context_description}
2. Is approximately {token_budget} tokens long
3. Maintains user perspective (use "I" statements)
4. Prioritizes actionable information for this context

Summary:"""

            response = await call_gemini_with_retry(
                self.model_name,
                prompt,
                temperature=0.2,
                max_output_tokens=min(token_budget + 50, 512)
            )

            summary = response.text.strip()
            return self._truncate_to_budget(summary, token_budget)

        except Exception as e:
            logger.error(f"Contextual summary failed: {e}")
            return self._create_fallback_summary(memories, token_budget)