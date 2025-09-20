"""
Memory type definitions and configurations.

Defines memory types, their characteristics, and processing rules.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional

from app.models.schemas import MemoryType


@dataclass
class MemoryTypeConfig:
    """
    Configuration for different memory types.

    Attributes:
        name: Memory type name
        default_importance: Default importance score (0-10)
        default_retention: Default retention period
        auto_expire: Whether memories auto-expire
        update_strategy: How to handle updates (replace, merge, version)
        description: Human-readable description
    """
    name: str
    default_importance: float
    default_retention: Optional[timedelta]
    auto_expire: bool
    update_strategy: str
    description: str


# Memory type configurations
MEMORY_TYPE_CONFIGS: Dict[MemoryType, MemoryTypeConfig] = {
    MemoryType.FACT: MemoryTypeConfig(
        name="fact",
        default_importance=8.0,
        default_retention=None,  # Permanent
        auto_expire=False,
        update_strategy="version",  # Create new version when updated
        description="Permanent facts about the user (job, location, family, etc.)"
    ),

    MemoryType.PREFERENCE: MemoryTypeConfig(
        name="preference",
        default_importance=6.0,
        default_retention=timedelta(days=180),  # 6 months default
        auto_expire=False,  # Manual expiration only
        update_strategy="replace",  # Replace with new preference
        description="User likes, dislikes, and preferences"
    ),

    MemoryType.EVENT: MemoryTypeConfig(
        name="event",
        default_importance=7.0,
        default_retention=timedelta(days=30),  # 30 days after event
        auto_expire=True,
        update_strategy="merge",  # Merge event details
        description="Scheduled events, appointments, and temporal information"
    ),

    MemoryType.ROUTINE: MemoryTypeConfig(
        name="routine",
        default_importance=5.0,
        default_retention=None,  # Permanent but updateable
        auto_expire=False,
        update_strategy="replace",  # Replace with updated routine
        description="Recurring patterns, habits, and scheduled activities"
    ),

    MemoryType.EMOTION: MemoryTypeConfig(
        name="emotion",
        default_importance=4.0,
        default_retention=timedelta(days=7),  # 7 days
        auto_expire=True,
        update_strategy="replace",  # Replace with current emotional state
        description="Emotional states and temporary feelings"
    )
}


def get_memory_type_config(memory_type: MemoryType) -> MemoryTypeConfig:
    """
    Get configuration for a memory type.

    Args:
        memory_type: The memory type to get config for

    Returns:
        MemoryTypeConfig: Configuration for the memory type

    Raises:
        ValueError: If memory type is not supported

    Example:
        >>> config = get_memory_type_config(MemoryType.FACT)
        >>> print(config.default_importance)  # 8.0
        >>> print(config.auto_expire)  # False
    """
    if memory_type not in MEMORY_TYPE_CONFIGS:
        raise ValueError(f"Unsupported memory type: {memory_type}")

    return MEMORY_TYPE_CONFIGS[memory_type]


def get_importance_score_for_type(memory_type: MemoryType, base_score: float = None) -> float:
    """
    Calculate importance score for a memory type.

    Args:
        memory_type: The memory type
        base_score: Base score from extraction (optional)

    Returns:
        float: Calculated importance score

    Example:
        >>> score = get_importance_score_for_type(MemoryType.FACT, 0.9)
        >>> print(score)  # Uses fact default importance
    """
    config = get_memory_type_config(memory_type)

    if base_score is None:
        return config.default_importance

    # Combine extraction confidence with type default
    # Higher confidence increases importance
    return min(10.0, config.default_importance + (base_score - 0.5) * 2)


def should_auto_expire(memory_type: MemoryType) -> bool:
    """
    Check if memory type should auto-expire.

    Args:
        memory_type: The memory type to check

    Returns:
        bool: True if memory type auto-expires

    Example:
        >>> should_auto_expire(MemoryType.EMOTION)  # True
        >>> should_auto_expire(MemoryType.FACT)     # False
    """
    config = get_memory_type_config(memory_type)
    return config.auto_expire


def get_default_retention(memory_type: MemoryType) -> Optional[timedelta]:
    """
    Get default retention period for memory type.

    Args:
        memory_type: The memory type

    Returns:
        Optional[timedelta]: Retention period or None for permanent

    Example:
        >>> retention = get_default_retention(MemoryType.EMOTION)
        >>> print(retention.days)  # 7
    """
    config = get_memory_type_config(memory_type)
    return config.default_retention