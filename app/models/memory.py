"""
SQLAlchemy models for memory storage.

Defines database tables for users, memories, connections, and conversations
with proper relationships, indexes, and constraints.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime,
    ForeignKey, JSON, Index
)
from sqlalchemy.orm import relationship

from app.core.database_prod import Base


class User(Base):
    """
    User model for storing user information.

    Attributes:
        id: Primary key
        user_id: Unique user identifier
        created_at: User creation timestamp
        updated_at: Last update timestamp
        memories: Related memories
        conversations: Related conversations
    """

    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    user_id = Column(String(255), primary_key=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    settings = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True, nullable=False)
    total_memories = Column(Integer, default=0, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow)

    # Relationships
    memories = relationship("Memory", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(user_id='{self.user_id}')>"


class Memory(Base):
    """
    Memory model for storing extracted memories.

    Attributes:
        id: Primary key
        user_id: Foreign key to users table
        memory_id: Unique UUID for Chroma reference
        content: Memory content text
        memory_type: Type of memory (fact, preference, event, routine, emotion)
        importance_score: Importance score (0-10)
        confidence_score: Extraction confidence (0-1)
        extracted_at: Memory extraction timestamp
        updated_at: Last update timestamp
        expires_at: Expiration timestamp (for events/emotions)
        is_active: Whether memory is active
        source_message_id: ID of source conversation message
        metadata: Additional structured data
        version: Memory version for updates
    """

    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.user_id"), nullable=False, index=True)
    memory_id = Column(String(36), unique=True, nullable=False, index=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    memory_type = Column(String(50), nullable=False, index=True)
    importance_score = Column(Float, default=5.0, nullable=False)
    confidence_score = Column(Float, nullable=False)
    extracted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    source_message_id = Column(String(36), nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)
    version = Column(Integer, default=1, nullable=False)

    # Relationships
    user = relationship("User", back_populates="memories")
    connections_from = relationship(
        "MemoryConnection",
        foreign_keys="MemoryConnection.from_memory_id",
        back_populates="from_memory"
    )
    connections_to = relationship(
        "MemoryConnection",
        foreign_keys="MemoryConnection.to_memory_id",
        back_populates="to_memory"
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_memory_user_type", "user_id", "memory_type"),
        Index("idx_memory_user_active", "user_id", "is_active"),
        Index("idx_memory_expires", "expires_at"),
        Index("idx_memory_importance", "importance_score"),
        {'extend_existing': True}
    )

    def set_expiration_based_on_type(self) -> None:
        """
        Set expiration date based on memory type.

        Events and emotions have automatic expiration dates.
        Other memory types remain permanent unless explicitly set.
        """
        if self.memory_type == "emotion":
            # Emotions expire after 7 days
            self.expires_at = datetime.utcnow() + timedelta(days=7)
        elif self.memory_type == "event":
            # Events expire 30 days after extraction
            self.expires_at = datetime.utcnow() + timedelta(days=30)
        # Facts, preferences, and routines don't auto-expire

    def is_expired(self) -> bool:
        """
        Check if memory has expired.

        Returns:
            bool: True if memory is expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata as dictionary.

        Returns:
            Dict: Memory metadata or empty dict if None
        """
        if self.metadata_ is None:
            return {}
        if isinstance(self.metadata_, str):
            return json.loads(self.metadata_)
        return self.metadata_

    def set_metadata(self, data: Dict[str, Any]) -> None:
        """
        Set metadata from dictionary.

        Args:
            data: Metadata dictionary to store
        """
        self.metadata_ = data

    def __repr__(self) -> str:
        return f"<Memory(memory_id='{self.memory_id}', type='{self.memory_type}')>"


class MemoryConnection(Base):
    """
    Model for relationships between memories.

    Attributes:
        id: Primary key
        from_memory_id: Source memory ID
        to_memory_id: Target memory ID
        connection_type: Type of connection
        strength: Connection strength (0-1)
        created_at: Connection creation timestamp
    """

    __tablename__ = "memory_connections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    from_memory_id = Column(String(36), ForeignKey("memories.memory_id"), nullable=False)
    to_memory_id = Column(String(36), ForeignKey("memories.memory_id"), nullable=False)
    connection_type = Column(String(50), nullable=False)  # relates_to, contradicts, updates
    strength = Column(Float, default=1.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    from_memory = relationship(
        "Memory",
        foreign_keys=[from_memory_id],
        back_populates="connections_from"
    )
    to_memory = relationship(
        "Memory",
        foreign_keys=[to_memory_id],
        back_populates="connections_to"
    )

    # Indexes for performance
    __table_args__ = (
        Index("idx_connection_from", "from_memory_id"),
        Index("idx_connection_to", "to_memory_id"),
        Index("idx_connection_type", "connection_type"),
        {'extend_existing': True}
    )

    def __repr__(self) -> str:
        return f"<MemoryConnection(from='{self.from_memory_id}', to='{self.to_memory_id}', type='{self.connection_type}')>"


class Conversation(Base):
    """
    Model for tracking conversation messages.

    Attributes:
        id: Primary key
        user_id: Foreign key to users table
        message_id: Unique message identifier
        message: Original message content
        processed_at: Processing timestamp
        memories_extracted: Number of memories extracted
        processing_time: Time taken to process (seconds)
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.user_id"), nullable=False, index=True)
    message_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    message = Column(Text, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    memories_extracted = Column(Integer, default=0, nullable=False)
    processing_time = Column(Float, nullable=True)  # Processing time in seconds

    # Relationships
    user = relationship("User", back_populates="conversations")

    # Indexes for performance
    __table_args__ = (
        Index("idx_conversation_user_time", "user_id", "processed_at"),
        {'extend_existing': True}
    )

    def __repr__(self) -> str:
        return f"<Conversation(message_id='{self.message_id}', user_id='{self.user_id}')>"