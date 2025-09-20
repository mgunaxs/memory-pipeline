"""
RAG-specific configuration for enhanced memory retrieval.

Defines context strategies, relevance thresholds, and retrieval policies
to prevent irrelevant memory retrieval in proactive AI scenarios.
"""

from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum

from app.models.schemas import MemoryType


class MemoryCategory(str, Enum):
    """Memory categories for better context filtering."""
    WORK = "work"
    FOOD = "food"
    HEALTH = "health"
    SOCIAL = "social"
    SCHEDULE = "schedule"
    ENTERTAINMENT = "entertainment"
    HOBBY = "hobby"
    FAMILY = "family"
    TRAVEL = "travel"
    FINANCE = "finance"
    LEARNING = "learning"
    EMOTIONAL = "emotional"


class ContextType(str, Enum):
    """Context types for retrieval strategies."""
    MORNING_CHECKIN = "morning_checkin"
    EVENING_CHECKIN = "evening_checkin"
    WEEKEND_PLANNING = "weekend_planning"
    WORK_PLANNING = "work_planning"
    MEAL_SUGGESTION = "meal_suggestion"
    EMOTIONAL_SUPPORT = "emotional_support"
    EVENT_FOLLOWUP = "event_followup"
    GENERAL_CHAT = "general_chat"


@dataclass
class RetrievalStrategy:
    """Configuration for context-aware retrieval."""
    include_types: Set[MemoryType]
    exclude_types: Set[MemoryType]
    include_categories: Set[MemoryCategory]
    exclude_categories: Set[MemoryCategory]
    time_window_hours: int  # How far back to look
    max_memories: int
    min_relevance: float
    temporal_boost: float  # Boost for recent memories
    importance_weight: float  # How much to weight importance


# Context-aware retrieval strategies
RETRIEVAL_STRATEGIES: Dict[ContextType, RetrievalStrategy] = {
    ContextType.MORNING_CHECKIN: RetrievalStrategy(
        include_types={MemoryType.EVENT, MemoryType.ROUTINE, MemoryType.PREFERENCE},
        exclude_types={MemoryType.EMOTION},  # Don't bring up old emotions in morning
        include_categories={MemoryCategory.SCHEDULE, MemoryCategory.HEALTH, MemoryCategory.WORK},
        exclude_categories={MemoryCategory.ENTERTAINMENT, MemoryCategory.SOCIAL},
        time_window_hours=48,  # Next 2 days
        max_memories=3,
        min_relevance=0.8,
        temporal_boost=0.3,
        importance_weight=0.4
    ),

    ContextType.EVENING_CHECKIN: RetrievalStrategy(
        include_types={MemoryType.EMOTION, MemoryType.EVENT, MemoryType.PREFERENCE},
        exclude_types=set(),
        include_categories={MemoryCategory.EMOTIONAL, MemoryCategory.SOCIAL, MemoryCategory.ENTERTAINMENT},
        exclude_categories={MemoryCategory.WORK},  # Don't stress about work in evening
        time_window_hours=24,  # Today's events and emotions
        max_memories=4,
        min_relevance=0.75,
        temporal_boost=0.4,
        importance_weight=0.3
    ),

    ContextType.WEEKEND_PLANNING: RetrievalStrategy(
        include_types={MemoryType.PREFERENCE, MemoryType.ROUTINE, MemoryType.EVENT},
        exclude_types=set(),
        include_categories={MemoryCategory.SOCIAL, MemoryCategory.ENTERTAINMENT, MemoryCategory.HOBBY},
        exclude_categories={MemoryCategory.WORK},
        time_window_hours=72,  # Weekend window
        max_memories=5,
        min_relevance=0.7,
        temporal_boost=0.2,
        importance_weight=0.5
    ),

    ContextType.MEAL_SUGGESTION: RetrievalStrategy(
        include_types={MemoryType.PREFERENCE, MemoryType.ROUTINE},
        exclude_types={MemoryType.EMOTION, MemoryType.EVENT},
        include_categories={MemoryCategory.FOOD, MemoryCategory.HEALTH},
        exclude_categories={MemoryCategory.WORK, MemoryCategory.SCHEDULE, MemoryCategory.EMOTIONAL},
        time_window_hours=168,  # Past week
        max_memories=4,
        min_relevance=0.85,  # Very high relevance for food
        temporal_boost=0.1,
        importance_weight=0.6
    ),

    ContextType.EMOTIONAL_SUPPORT: RetrievalStrategy(
        include_types={MemoryType.EMOTION, MemoryType.PREFERENCE, MemoryType.FACT},
        exclude_types={MemoryType.EVENT},  # Don't remind of stressful events
        include_categories={MemoryCategory.EMOTIONAL, MemoryCategory.HEALTH, MemoryCategory.SOCIAL},
        exclude_categories={MemoryCategory.WORK},
        time_window_hours=168,  # Past week's emotions
        max_memories=3,
        min_relevance=0.8,
        temporal_boost=0.5,
        importance_weight=0.3
    ),

    ContextType.EVENT_FOLLOWUP: RetrievalStrategy(
        include_types={MemoryType.EVENT, MemoryType.EMOTION},
        exclude_types=set(),
        include_categories=set(),  # Include all categories for events
        exclude_categories=set(),
        time_window_hours=72,  # Recent events
        max_memories=3,
        min_relevance=0.9,  # Very specific to the event
        temporal_boost=0.6,
        importance_weight=0.4
    ),

    ContextType.GENERAL_CHAT: RetrievalStrategy(
        include_types={MemoryType.FACT, MemoryType.PREFERENCE},
        exclude_types={MemoryType.EMOTION},  # Don't randomly bring up emotions
        include_categories=set(),
        exclude_categories={MemoryCategory.EMOTIONAL},
        time_window_hours=720,  # Past month
        max_memories=5,
        min_relevance=0.75,
        temporal_boost=0.2,
        importance_weight=0.5
    )
}


# RAG Quality Configuration
RAG_CONFIG = {
    # Relevance Filtering
    "DEFAULT_RELEVANCE_THRESHOLD": 0.75,
    "STRICT_RELEVANCE_THRESHOLD": 0.85,
    "LOOSE_RELEVANCE_THRESHOLD": 0.6,

    # Memory Limits
    "MAX_MEMORIES_PER_RETRIEVAL": 5,
    "MAX_MEMORIES_FOR_VALIDATION": 10,
    "EMERGENCY_MAX_MEMORIES": 3,  # Fallback when context is unclear

    # Token Management
    "TOKEN_BUDGET_PER_RETRIEVAL": 500,
    "TOKEN_BUDGET_STRICT": 300,  # For mobile/limited contexts
    "AVERAGE_TOKENS_PER_MEMORY": 50,
    "SUMMARIZATION_THRESHOLD": 400,  # Start summarizing above this

    # Deduplication
    "DUPLICATE_THRESHOLD": 0.95,
    "SIMILAR_THRESHOLD": 0.85,  # For updates vs new memories
    "UPDATE_THRESHOLD": 0.90,  # When to update vs create new

    # Validation
    "VALIDATION_THRESHOLD": 0.5,  # Gemini validation score
    "VALIDATION_ENABLED": True,
    "VALIDATION_SAMPLE_SIZE": 3,  # Validate top N results

    # Temporal Settings
    "MEMORY_DECAY_HALF_LIFE_DAYS": 30,
    "FACT_DECAY_DISABLED": True,  # Facts don't decay
    "EMOTION_DECAY_MULTIPLIER": 2.0,  # Emotions decay faster

    # Quality Metrics
    "MIN_IMPORTANCE_TO_KEEP": 0.3,
    "MAX_MEMORIES_PER_USER": 1000,
    "CLEANUP_THRESHOLD_DAYS": 90,  # Clean up old low-importance memories

    # Fallback Settings
    "ENABLE_FALLBACK_SEARCH": True,
    "FALLBACK_MEMORY_COUNT": 3,
    "FALLBACK_IMPORTANCE_THRESHOLD": 0.7,
}


# Category mapping for automatic classification
CATEGORY_KEYWORDS = {
    MemoryCategory.WORK: {
        "keywords": ["job", "work", "office", "meeting", "boss", "colleague", "project", "deadline", "career"],
        "patterns": ["at work", "work from", "office", "meeting", "coworker"]
    },
    MemoryCategory.FOOD: {
        "keywords": ["eat", "food", "meal", "restaurant", "cook", "recipe", "hungry", "dinner", "lunch", "breakfast"],
        "patterns": ["love eating", "favorite food", "restaurant", "cooking"]
    },
    MemoryCategory.HEALTH: {
        "keywords": ["health", "doctor", "medicine", "exercise", "gym", "sick", "tired", "energy", "sleep"],
        "patterns": ["feel", "health", "exercise", "doctor appointment"]
    },
    MemoryCategory.SOCIAL: {
        "keywords": ["friend", "family", "party", "social", "meet", "hang out", "date", "relationship"],
        "patterns": ["with friends", "social", "hanging out", "meeting people"]
    },
    MemoryCategory.SCHEDULE: {
        "keywords": ["appointment", "meeting", "schedule", "calendar", "time", "tomorrow", "next week"],
        "patterns": ["at", "on", "appointment", "meeting", "tomorrow", "next"]
    },
    MemoryCategory.ENTERTAINMENT: {
        "keywords": ["movie", "music", "game", "show", "entertainment", "fun", "hobby", "book", "watch"],
        "patterns": ["watching", "playing", "listening to", "reading"]
    },
    MemoryCategory.EMOTIONAL: {
        "keywords": ["feel", "emotion", "happy", "sad", "angry", "excited", "nervous", "stressed", "worried"],
        "patterns": ["feeling", "feel", "emotion", "mood"]
    }
}


def get_retrieval_strategy(context_type: ContextType) -> RetrievalStrategy:
    """
    Get retrieval strategy for context type.

    Args:
        context_type: The context for memory retrieval

    Returns:
        RetrievalStrategy: Strategy configuration
    """
    return RETRIEVAL_STRATEGIES.get(context_type, RETRIEVAL_STRATEGIES[ContextType.GENERAL_CHAT])


def classify_memory_category(content: str, metadata: dict = None) -> MemoryCategory:
    """
    Automatically classify memory category based on content.

    Args:
        content: Memory content text
        metadata: Additional metadata

    Returns:
        MemoryCategory: Classified category
    """
    content_lower = content.lower()

    # Score each category
    category_scores = {}

    for category, config in CATEGORY_KEYWORDS.items():
        score = 0

        # Check keywords
        for keyword in config["keywords"]:
            if keyword in content_lower:
                score += 1

        # Check patterns
        for pattern in config["patterns"]:
            if pattern in content_lower:
                score += 2  # Patterns are more specific

        category_scores[category] = score

    # Return highest scoring category, default to GENERAL
    if category_scores:
        best_category = max(category_scores, key=category_scores.get)
        if category_scores[best_category] > 0:
            return best_category

    # Default classification based on memory type if available
    if metadata and "memory_type" in metadata:
        memory_type = metadata["memory_type"]
        if memory_type == "emotion":
            return MemoryCategory.EMOTIONAL
        elif memory_type == "event":
            return MemoryCategory.SCHEDULE
        elif memory_type == "routine":
            return MemoryCategory.HEALTH  # Many routines are health-related

    # Default to SOCIAL as it's the safest general category
    return MemoryCategory.SOCIAL


def should_include_memory(
    memory_category: MemoryCategory,
    memory_type: MemoryType,
    strategy: RetrievalStrategy
) -> bool:
    """
    Check if memory should be included based on strategy.

    Args:
        memory_category: Memory category
        memory_type: Memory type
        strategy: Retrieval strategy

    Returns:
        bool: True if memory should be included
    """
    # Check type filters
    if strategy.exclude_types and memory_type in strategy.exclude_types:
        return False

    if strategy.include_types and memory_type not in strategy.include_types:
        return False

    # Check category filters
    if strategy.exclude_categories and memory_category in strategy.exclude_categories:
        return False

    if strategy.include_categories and memory_category not in strategy.include_categories:
        return False

    return True