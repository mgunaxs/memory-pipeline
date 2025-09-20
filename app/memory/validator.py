"""
Memory relevance validation using Gemini.

Validates whether retrieved memories are actually relevant
for specific contexts to prevent inappropriate memory usage.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from app.core.rate_limiter import call_gemini_with_retry, RateLimitError
from app.models.schemas import MemorySearchResult, ContextType
from app.config.rag_config import RAG_CONFIG

logger = logging.getLogger(__name__)


class MemoryValidator:
    """
    Validates memory relevance using LLM-based second-pass filtering.

    Provides additional quality control to ensure retrieved memories
    are actually appropriate for the given context.
    """

    def __init__(self):
        """Initialize memory validator."""
        self.model_name = "gemini-1.5-flash"
        self.validation_enabled = RAG_CONFIG["VALIDATION_ENABLED"]
        self.validation_threshold = RAG_CONFIG["VALIDATION_THRESHOLD"]

    async def validate_memory_relevance(
        self,
        query_context: str,
        retrieved_memories: List[MemorySearchResult],
        context_type: Optional[ContextType] = None,
        validation_threshold: Optional[float] = None
    ) -> List[MemorySearchResult]:
        """
        Validate relevance of retrieved memories using Gemini.

        Args:
            query_context: The context/intent of the query
            retrieved_memories: List of retrieved memories
            context_type: Context type for validation
            validation_threshold: Custom validation threshold

        Returns:
            List[MemorySearchResult]: Validated and filtered memories

        Example:
            >>> validator = MemoryValidator()
            >>> validated = await validator.validate_memory_relevance(
            ...     "suggesting morning routine",
            ...     retrieved_memories,
            ...     ContextType.MORNING_CHECKIN
            ... )
            >>> # Only contextually appropriate memories returned
        """
        if not self.validation_enabled or not retrieved_memories:
            return retrieved_memories

        try:
            threshold = validation_threshold or self.validation_threshold

            # Limit validation to prevent excessive API calls
            sample_size = min(len(retrieved_memories), RAG_CONFIG["VALIDATION_SAMPLE_SIZE"])
            memories_to_validate = retrieved_memories[:sample_size]

            logger.info(f"Validating {len(memories_to_validate)} memories for context: {query_context}")

            # Build validation prompt
            prompt = self._build_validation_prompt(
                query_context, memories_to_validate, context_type
            )

            # Call Gemini for validation
            response = await call_gemini_with_retry(
                self.model_name,
                prompt,
                temperature=0.1,  # Low temperature for consistent validation
                max_output_tokens=1024
            )

            # Parse validation results
            validation_results = self._parse_validation_response(response.text)

            # Filter based on validation scores
            validated_memories = self._apply_validation_results(
                memories_to_validate, validation_results, threshold
            )

            # Add back memories that weren't validated (lower priority ones)
            remaining_memories = retrieved_memories[sample_size:]
            final_memories = validated_memories + remaining_memories

            logger.info(
                f"Validation completed: {len(memories_to_validate)} → "
                f"{len(validated_memories)} validated memories"
            )

            return final_memories

        except RateLimitError:
            logger.warning("Rate limit hit during validation, returning unvalidated memories")
            return retrieved_memories
        except Exception as e:
            logger.error(f"Memory validation failed: {e}")
            return retrieved_memories  # Return original memories on failure

    def _build_validation_prompt(
        self,
        query_context: str,
        memories: List[MemorySearchResult],
        context_type: Optional[ContextType]
    ) -> str:
        """
        Build validation prompt for Gemini.

        Args:
            query_context: Query context description
            memories: Memories to validate
            context_type: Context type

        Returns:
            str: Validation prompt
        """
        context_description = self._get_context_description(context_type)

        memories_text = ""
        for i, result in enumerate(memories):
            memories_text += f"Memory {i+1}: {result.memory.content}\n"
            memories_text += f"Type: {result.memory.memory_type.value}\n"
            memories_text += f"Similarity: {result.similarity_score:.3f}\n\n"

        prompt = f"""You are validating memory relevance for a proactive AI assistant.

CONTEXT: {query_context}
SITUATION: {context_description}

MEMORIES TO VALIDATE:
{memories_text}

VALIDATION CRITERIA:
Rate each memory's relevance (0.0-1.0) for this specific context. Consider:

1. APPROPRIATENESS: Is this memory helpful and appropriate for this context?
2. TIMING: Is this the right time to mention this memory?
3. USER EXPERIENCE: Would mentioning this memory feel natural and helpful, or weird/creepy?
4. CONTEXT FIT: Does this memory genuinely relate to the current situation?

INAPPROPRIATE EXAMPLES:
- Mentioning work stress during a food recommendation
- Bringing up personal relationships during morning routine planning
- Referencing old emotional states during unrelated activities
- Using outdated information when current info exists

OUTPUT FORMAT (JSON):
{{
  "validations": [
    {{
      "memory_index": 1,
      "relevance_score": 0.8,
      "reasoning": "Directly relevant to morning routine planning",
      "appropriate": true
    }},
    {{
      "memory_index": 2,
      "relevance_score": 0.2,
      "reasoning": "Work stress not relevant for meal suggestions",
      "appropriate": false
    }}
  ]
}}

Rate each memory honestly. It's better to exclude marginally relevant memories than to include inappropriate ones.
"""

        return prompt

    def _get_context_description(self, context_type: Optional[ContextType]) -> str:
        """
        Get human-readable context description.

        Args:
            context_type: Context type

        Returns:
            str: Context description
        """
        context_descriptions = {
            ContextType.MORNING_CHECKIN: "Starting the day, checking morning routine and schedule",
            ContextType.EVENING_CHECKIN: "End of day reflection and evening plans",
            ContextType.WEEKEND_PLANNING: "Planning weekend activities and leisure time",
            ContextType.WORK_PLANNING: "Work-related planning and productivity",
            ContextType.MEAL_SUGGESTION: "Food recommendations and meal planning",
            ContextType.EMOTIONAL_SUPPORT: "Providing emotional support and understanding",
            ContextType.EVENT_FOLLOWUP: "Following up on specific events or appointments",
            ContextType.GENERAL_CHAT: "General conversation and interaction"
        }

        return context_descriptions.get(context_type, "General conversation")

    def _parse_validation_response(self, response_text: str) -> Dict[int, Dict]:
        """
        Parse validation response from Gemini.

        Args:
            response_text: Raw response from Gemini

        Returns:
            Dict[int, Dict]: Validation results by memory index
        """
        try:
            # Clean response text
            cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text.strip())

            # Parse JSON
            data = json.loads(cleaned_text)

            if 'validations' not in data:
                raise ValueError("Invalid response format: missing 'validations' field")

            # Convert to dict by memory index
            validation_results = {}
            for validation in data['validations']:
                if 'memory_index' in validation and 'relevance_score' in validation:
                    index = validation['memory_index']
                    validation_results[index] = validation

            return validation_results

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            return {}
        except Exception as e:
            logger.error(f"Validation response parsing failed: {e}")
            return {}

    def _apply_validation_results(
        self,
        memories: List[MemorySearchResult],
        validation_results: Dict[int, Dict],
        threshold: float
    ) -> List[MemorySearchResult]:
        """
        Apply validation results to filter memories.

        Args:
            memories: Original memories
            validation_results: Validation scores by index
            threshold: Validation threshold

        Returns:
            List[MemorySearchResult]: Filtered memories
        """
        validated_memories = []

        for i, memory_result in enumerate(memories):
            memory_index = i + 1  # 1-based indexing in validation

            if memory_index in validation_results:
                validation = validation_results[memory_index]
                relevance_score = validation.get('relevance_score', 0.0)
                appropriate = validation.get('appropriate', False)

                # Apply threshold and appropriateness check
                if relevance_score >= threshold and appropriate:
                    # Update similarity score with validation score
                    # Combine original similarity with validation
                    combined_score = (memory_result.similarity_score * 0.6 + relevance_score * 0.4)
                    memory_result.similarity_score = combined_score
                    validated_memories.append(memory_result)

                    logger.debug(
                        f"Memory {memory_index} passed validation: "
                        f"relevance={relevance_score:.2f}, appropriate={appropriate}"
                    )
                else:
                    logger.debug(
                        f"Memory {memory_index} failed validation: "
                        f"relevance={relevance_score:.2f}, appropriate={appropriate}"
                    )
            else:
                # No validation result, be conservative
                logger.debug(f"Memory {memory_index} had no validation result, excluding")

        return validated_memories

    async def quick_relevance_check(
        self,
        query: str,
        memory_content: str,
        context_type: Optional[ContextType] = None
    ) -> float:
        """
        Quick relevance check for a single memory.

        Args:
            query: Search query
            memory_content: Memory content to check
            context_type: Context type

        Returns:
            float: Relevance score (0-1)
        """
        try:
            # Simple semantic relevance check without full validation
            # This could use embedding similarity or simple keyword matching

            query_lower = query.lower()
            memory_lower = memory_content.lower()

            # Basic keyword overlap
            query_words = set(query_lower.split())
            memory_words = set(memory_lower.split())

            overlap = len(query_words.intersection(memory_words))
            total_words = len(query_words.union(memory_words))

            if total_words == 0:
                return 0.0

            # Basic relevance score
            basic_score = overlap / len(query_words) if len(query_words) > 0 else 0.0

            return min(1.0, basic_score)

        except Exception as e:
            logger.debug(f"Quick relevance check failed: {e}")
            return 0.5  # Default neutral score

    def batch_validate_memories(
        self,
        validations: List[Tuple[str, List[MemorySearchResult], Optional[ContextType]]]
    ) -> List[List[MemorySearchResult]]:
        """
        Batch validate multiple sets of memories.

        Args:
            validations: List of (query_context, memories, context_type) tuples

        Returns:
            List[List[MemorySearchResult]]: Validated memory sets
        """
        # For now, process sequentially to avoid rate limits
        # Could be optimized with proper batching later
        results = []

        for query_context, memories, context_type in validations:
            try:
                validated = asyncio.run(
                    self.validate_memory_relevance(query_context, memories, context_type)
                )
                results.append(validated)
            except Exception as e:
                logger.error(f"Batch validation failed for query '{query_context}': {e}")
                results.append(memories)  # Return original on failure

        return results