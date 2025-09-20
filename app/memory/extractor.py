"""
Memory extraction from user text using Gemini 1.5 Flash.

Extracts structured memories from natural language text with confidence scoring,
type classification, and importance assessment.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

from app.core.config import settings
from app.core.rate_limiter import call_gemini_with_retry, RateLimitError, APIQuotaError
from app.models.schemas import MemoryType, MemoryCreate
from app.memory.types import get_memory_type_config, get_importance_score_for_type

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=settings.gemini_api_key)


class MemoryExtractor:
    """
    Extracts structured memories from user text.

    Uses Gemini 1.5 Flash to identify and extract different types of memories
    with confidence scoring and metadata extraction.
    """

    def __init__(self):
        """Initialize the memory extractor."""
        self.model_name = "gemini-1.5-flash"
        self.extraction_prompt_template = self._build_extraction_prompt()

    def _build_extraction_prompt(self) -> str:
        """
        Build the enhanced extraction prompt template with rich metadata.

        Returns:
            str: Prompt template for memory extraction
        """
        return """You are an advanced memory extraction system for a proactive AI companion. Extract structured memories with rich metadata to prevent irrelevant retrieval.

MEMORY TYPES:
- fact: Permanent information (job, location, family, personal details)
- preference: Likes, dislikes, opinions, values
- event: Scheduled events, appointments, future plans
- routine: Regular patterns, habits, recurring activities
- emotion: Current emotional states, feelings, mood

MEMORY CATEGORIES:
- work: Job, career, office, meetings, projects
- food: Meals, restaurants, cooking, dietary preferences
- health: Exercise, medical, sleep, wellness
- social: Friends, family, relationships, social activities
- schedule: Appointments, time-based activities
- entertainment: Movies, music, games, hobbies
- emotional: Feelings, moods, mental states
- learning: Education, skills, personal development
- travel: Trips, locations, transportation
- finance: Money, spending, financial goals

TEMPORAL RELEVANCE:
- daily: Relevant every day (routines, preferences)
- weekly: Relevant weekly (meetings, activities)
- once: One-time events
- expired: Past events no longer relevant

EXTRACTION RULES:
1. Extract specific, actionable memories
2. Add rich metadata for context-aware retrieval
3. Identify entities (people, places, times)
4. Classify temporal relevance
5. Define valid/invalid contexts for use
6. Rate importance and confidence

OUTPUT FORMAT (JSON):
{
  "memories": [
    {
      "content": "Clear, specific description",
      "type": "fact|preference|event|routine|emotion",
      "category": "work|food|health|social|schedule|entertainment|emotional|learning|travel|finance",
      "entities": ["entity1", "entity2"],
      "temporal_relevance": "daily|weekly|once|expired",
      "importance": 0.0-1.0,
      "confidence": 0.0-1.0,
      "valid_contexts": ["morning_checkin", "meal_suggestion", "work_planning"],
      "invalid_contexts": ["emotional_support", "weekend_planning"],
      "metadata": {
        "time_mentioned": "morning|afternoon|evening|specific_time",
        "frequency": "daily|weekly|monthly|once",
        "people": ["person1", "person2"],
        "places": ["place1", "place2"],
        "extracted_from": "original text snippet"
      }
    }
  ]
}

EXAMPLES:

Input: "I'm a software engineer at Microsoft and I hate morning meetings"
Output:
{
  "memories": [
    {
      "content": "Works as a software engineer at Microsoft",
      "type": "fact",
      "confidence": 1.0,
      "importance": 8,
      "metadata": {
        "company": "Microsoft",
        "role": "software engineer",
        "extracted_from": "I'm a software engineer at Microsoft"
      }
    },
    {
      "content": "Dislikes morning meetings",
      "type": "preference",
      "confidence": 1.0,
      "importance": 6,
      "metadata": {
        "category": "work_preferences",
        "extracted_from": "I hate morning meetings"
      }
    }
  ]
}

Input: "I usually go to the gym on Tuesdays and I have a dentist appointment next Friday at 2pm"
Output:
{
  "memories": [
    {
      "content": "Goes to gym on Tuesdays",
      "type": "routine",
      "confidence": 0.9,
      "importance": 5,
      "metadata": {
        "activity": "gym",
        "frequency": "weekly",
        "day": "Tuesday",
        "extracted_from": "I usually go to the gym on Tuesdays"
      }
    },
    {
      "content": "Dentist appointment Friday 2pm",
      "type": "event",
      "confidence": 1.0,
      "importance": 7,
      "metadata": {
        "appointment_type": "dentist",
        "day": "Friday",
        "time": "2pm",
        "extracted_from": "dentist appointment next Friday at 2pm"
      }
    }
  ]
}

Now extract memories from this text:
{text}

Respond with only the JSON output, no additional text."""

    async def extract_memories(
        self,
        text: str,
        user_id: str,
        source_message_id: Optional[str] = None
    ) -> List[MemoryCreate]:
        """
        Extract memories from user text.

        Args:
            text: User text to extract memories from
            user_id: User identifier
            source_message_id: Optional source message ID

        Returns:
            List[MemoryCreate]: List of extracted memories

        Raises:
            ValueError: If text is empty or too long
            RateLimitError: If API rate limit is exceeded
            APIQuotaError: If API quota is exhausted

        Example:
            >>> extractor = MemoryExtractor()
            >>> memories = await extractor.extract_memories(
            ...     "I love pizza and work at Google",
            ...     "user123"
            ... )
            >>> print(len(memories))  # 2
        """
        start_time = time.time()

        # Validate and sanitize input
        from app.utils.security import sanitize_memory_content, validate_user_id, validate_message_id

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Sanitize inputs
        text = sanitize_memory_content(text)
        user_id = validate_user_id(user_id)
        if source_message_id:
            source_message_id = validate_message_id(source_message_id)
        logger.info(f"Extracting memories for user {user_id} from {len(text)} characters")

        try:
            # Build prompt
            prompt = self.extraction_prompt_template.format(text=text)

            # Call Gemini API with retry logic
            response = await call_gemini_with_retry(
                self.model_name,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_output_tokens=2048,
                    candidate_count=1
                )
            )

            # Parse response
            raw_memories = self._parse_response(response.text)

            # Convert to MemoryCreate objects
            memories = self._create_memory_objects(
                raw_memories,
                user_id,
                source_message_id
            )

            processing_time = time.time() - start_time
            logger.info(
                f"Extracted {len(memories)} memories for user {user_id} "
                f"in {processing_time:.2f}s"
            )

            return memories

        except (RateLimitError, APIQuotaError):
            logger.warning("API limit reached during memory extraction")
            raise
        except Exception as e:
            logger.error(f"Memory extraction failed for user {user_id}: {e}")
            raise Exception(f"Memory extraction failed: {e}")

    def _parse_response(self, response_text: str) -> List[Dict]:
        """
        Parse Gemini response to extract memories.

        Args:
            response_text: Raw response from Gemini

        Returns:
            List[Dict]: Parsed memory data

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Clean response text (remove code blocks if present)
            cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text.strip())

            # Parse JSON
            data = json.loads(cleaned_text)

            if not isinstance(data, dict) or 'memories' not in data:
                raise ValueError("Invalid response format: missing 'memories' field")

            memories = data['memories']
            if not isinstance(memories, list):
                raise ValueError("Invalid response format: 'memories' must be a list")

            logger.debug(f"Parsed {len(memories)} memories from response")
            return memories

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            raise ValueError(f"Response parsing error: {e}")

    def _create_memory_objects(
        self,
        raw_memories: List[Dict],
        user_id: str,
        source_message_id: Optional[str]
    ) -> List[MemoryCreate]:
        """
        Convert raw memory data to MemoryCreate objects with rich metadata.

        Args:
            raw_memories: Raw memory data from Gemini
            user_id: User identifier
            source_message_id: Optional source message ID

        Returns:
            List[MemoryCreate]: Validated memory objects
        """
        from app.models.schemas import MemoryCategory, TemporalRelevance, ContextType
        from app.config.rag_config import classify_memory_category

        memories = []

        for raw_memory in raw_memories:
            try:
                # Validate required fields
                if not isinstance(raw_memory, dict):
                    logger.warning("Skipping invalid memory: not a dictionary")
                    continue

                content = raw_memory.get('content', '').strip()
                memory_type_str = raw_memory.get('type', '').lower()
                confidence = raw_memory.get('confidence', 0.5)

                if not content:
                    logger.warning("Skipping memory with empty content")
                    continue

                # Validate memory type
                try:
                    memory_type = MemoryType(memory_type_str)
                except ValueError:
                    logger.warning(f"Invalid memory type '{memory_type_str}', defaulting to 'fact'")
                    memory_type = MemoryType.FACT

                # Validate confidence
                if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    logger.warning(f"Invalid confidence {confidence}, defaulting to 0.5")
                    confidence = 0.5

                # Filter out low-confidence memories
                min_confidence = getattr(settings, 'min_confidence_threshold', settings.min_confidence_score)
                if confidence < min_confidence:
                    logger.debug(f"Skipping low-confidence memory: {confidence}")
                    continue

                # Extract and validate category
                category_str = raw_memory.get('category', '').lower()
                try:
                    category = MemoryCategory(category_str)
                except ValueError:
                    # Auto-classify if not provided or invalid
                    category = classify_memory_category(content, {"memory_type": memory_type.value})
                    logger.debug(f"Auto-classified category as {category.value} for: {content[:50]}")

                # Extract entities
                entities = raw_memory.get('entities', [])
                if not isinstance(entities, list):
                    entities = []

                # Extract temporal relevance
                temporal_str = raw_memory.get('temporal_relevance', 'once').lower()
                try:
                    temporal_relevance = TemporalRelevance(temporal_str)
                except ValueError:
                    temporal_relevance = TemporalRelevance.ONCE

                # Extract and validate importance (now 0-1 scale)
                raw_importance = raw_memory.get('importance', None)
                if isinstance(raw_importance, (int, float)) and 0 <= raw_importance <= 1:
                    importance_score = float(raw_importance)
                elif isinstance(raw_importance, (int, float)) and 0 <= raw_importance <= 10:
                    # Convert from 0-10 scale to 0-1 scale
                    importance_score = float(raw_importance) / 10.0
                else:
                    # Use type-based importance calculation
                    type_importance = get_importance_score_for_type(memory_type, confidence)
                    importance_score = min(1.0, type_importance / 10.0)  # Convert to 0-1 scale

                # Extract valid/invalid contexts
                valid_contexts_raw = raw_memory.get('valid_contexts', [])
                invalid_contexts_raw = raw_memory.get('invalid_contexts', [])

                valid_contexts = []
                for ctx in valid_contexts_raw:
                    try:
                        valid_contexts.append(ContextType(ctx))
                    except ValueError:
                        logger.debug(f"Invalid context type: {ctx}")

                invalid_contexts = []
                for ctx in invalid_contexts_raw:
                    try:
                        invalid_contexts.append(ContextType(ctx))
                    except ValueError:
                        logger.debug(f"Invalid context type: {ctx}")

                # Extract metadata
                metadata = raw_memory.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}

                # Create MemoryCreate object with rich metadata
                memory = MemoryCreate(
                    content=content,
                    memory_type=memory_type,
                    category=category,
                    entities=entities,
                    temporal_relevance=temporal_relevance,
                    importance_score=importance_score,
                    confidence_score=confidence,
                    valid_contexts=valid_contexts,
                    invalid_contexts=invalid_contexts,
                    source_message_id=source_message_id,
                    metadata=metadata
                )

                memories.append(memory)
                logger.debug(f"Created enhanced memory: {memory_type.value}/{category.value} - {content[:50]}...")

            except Exception as e:
                logger.warning(f"Failed to create memory object: {e}")
                continue

        return memories

    def validate_extraction_quality(self, memories: List[MemoryCreate]) -> Dict[str, any]:
        """
        Validate the quality of extracted memories.

        Args:
            memories: List of extracted memories

        Returns:
            Dict: Quality metrics and validation results

        Example:
            >>> metrics = extractor.validate_extraction_quality(memories)
            >>> print(metrics['average_confidence'])  # 0.85
        """
        if not memories:
            return {
                'total_memories': 0,
                'average_confidence': 0.0,
                'average_importance': 0.0,
                'type_distribution': {},
                'quality_score': 0.0
            }

        # Calculate metrics
        total_memories = len(memories)
        avg_confidence = sum(m.confidence_score for m in memories) / total_memories
        avg_importance = sum(m.importance_score for m in memories) / total_memories

        # Type distribution
        type_counts = {}
        for memory in memories:
            type_counts[memory.memory_type.value] = type_counts.get(memory.memory_type.value, 0) + 1

        # Quality score (0-1 based on confidence and diversity)
        confidence_score = avg_confidence
        diversity_score = min(1.0, len(type_counts) / 5)  # Max 5 types
        quality_score = (confidence_score * 0.7 + diversity_score * 0.3)

        return {
            'total_memories': total_memories,
            'average_confidence': round(avg_confidence, 3),
            'average_importance': round(avg_importance, 1),
            'type_distribution': type_counts,
            'quality_score': round(quality_score, 3)
        }