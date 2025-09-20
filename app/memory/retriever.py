"""
Memory retrieval using Chroma vector database.

Provides semantic search, filtering, and ranking for stored memories
with support for metadata filtering and temporal relevance.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from app.core.config import settings
from app.models.schemas import MemoryType, MemorySearchResult
from app.models.memory import Memory
from app.memory.embedder import EmbeddingService

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    Retrieves memories using semantic search and filtering.

    Provides advanced search capabilities including similarity search,
    metadata filtering, temporal decay, and importance weighting.
    """

    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize memory retriever.

        Args:
            embedding_service: Service for generating embeddings
        """
        self.embedding_service = embedding_service
        self.client = None
        self._init_chroma_client()

    def _init_chroma_client(self) -> None:
        """
        Initialize Chroma client with persistence.

        Creates client with proper configuration for local storage.
        """
        try:
            # Create Chroma client with persistence
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Chroma client initialized with path: {settings.chroma_persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise Exception(f"Vector database initialization failed: {e}")

    def _get_collection_name(self, user_id: str) -> str:
        """
        Get collection name for user.

        Args:
            user_id: User identifier

        Returns:
            str: Collection name
        """
        # Use user_id directly as collection name (sanitized)
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ["_", "-"])
        return f"memories_{safe_user_id}"

    def _get_or_create_collection(self, user_id: str):
        """
        Get or create collection for user.

        Args:
            user_id: User identifier

        Returns:
            Collection: Chroma collection
        """
        collection_name = self._get_collection_name(user_id)

        try:
            # Try to get existing collection
            collection = self.client.get_collection(collection_name)
            logger.debug(f"Retrieved existing collection: {collection_name}")
            return collection
        except NotFoundError:
            # Create new collection
            logger.info(f"Creating new collection: {collection_name}")
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"user_id": user_id, "created_at": datetime.utcnow().isoformat()}
            )
            return collection

    async def store_memory(
        self,
        memory: Memory,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Store memory in vector database.

        Args:
            memory: Memory object to store
            embedding: Pre-computed embedding (optional)

        Returns:
            bool: True if successful, False otherwise

        Example:
            >>> retriever = MemoryRetriever(embedding_service)
            >>> success = await retriever.store_memory(memory)
            >>> print(success)  # True
        """
        try:
            start_time = time.time()

            # Get or create collection
            collection = self._get_or_create_collection(memory.user_id)

            # Generate embedding if not provided
            if embedding is None:
                embedding, model_used = await self.embedding_service.get_embedding(memory.content)
                logger.debug(f"Generated embedding using {model_used}")

            # Prepare metadata
            metadata = {
                "memory_type": memory.memory_type,
                "importance_score": memory.importance_score,
                "confidence_score": memory.confidence_score,
                "extracted_at": memory.extracted_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
                "is_active": memory.is_active,
                "version": memory.version,
                "source_message_id": memory.source_message_id or "",
            }

            # Add expiration if present
            if memory.expires_at:
                metadata["expires_at"] = memory.expires_at.isoformat()

            # Add custom metadata
            if memory.metadata_:
                custom_metadata = memory.get_metadata()
                # Flatten custom metadata with prefix to avoid conflicts
                for key, value in custom_metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[f"custom_{key}"] = value

            # Store in Chroma
            collection.add(
                documents=[memory.content],
                metadatas=[metadata],
                ids=[memory.memory_id],
                embeddings=[embedding]
            )

            duration = time.time() - start_time
            logger.info(
                f"Stored memory {memory.memory_id} for user {memory.user_id} "
                f"in {duration:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store memory {memory.memory_id}: {e}")
            return False

    async def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: Optional[float] = None,
        include_expired: bool = False,
        max_age_days: Optional[int] = None
    ) -> List[MemorySearchResult]:
        """
        Search memories using semantic similarity.

        Args:
            query: Search query text
            user_id: User identifier
            limit: Maximum number of results
            memory_types: Filter by memory types
            min_importance: Minimum importance score
            include_expired: Include expired memories
            max_age_days: Maximum age in days

        Returns:
            List[MemorySearchResult]: Search results with similarity scores

        Example:
            >>> results = await retriever.search_memories(
            ...     "work preferences",
            ...     "user123",
            ...     limit=5
            ... )
            >>> print(len(results))  # 3
        """
        start_time = time.time()

        try:
            # Get collection
            collection = self._get_or_create_collection(user_id)

            # Check if collection is empty
            count = collection.count()
            if count == 0:
                logger.debug(f"No memories found for user {user_id}")
                return []

            # Generate query embedding
            query_embedding, model_used = await self.embedding_service.get_embedding(query)
            logger.debug(f"Generated query embedding using {model_used}")

            # Build metadata filter
            where_conditions = {"is_active": True}

            # Filter by memory types
            if memory_types:
                type_values = [mt.value for mt in memory_types]
                where_conditions["memory_type"] = {"$in": type_values}

            # Filter by importance
            if min_importance is not None:
                where_conditions["importance_score"] = {"$gte": min_importance}

            # Filter by expiration
            if not include_expired:
                current_time = datetime.utcnow().isoformat()
                # Include memories without expiration OR not yet expired
                where_conditions["$or"] = [
                    {"expires_at": {"$eq": ""}},  # No expiration
                    {"expires_at": {"$gte": current_time}}  # Not expired
                ]

            # Filter by age
            if max_age_days is not None:
                cutoff_date = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
                where_conditions["extracted_at"] = {"$gte": cutoff_date}

            # Perform vector search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(limit * 2, 100),  # Get extra results for post-processing
                where=where_conditions,
                include=["documents", "metadatas", "distances"]
            )

            # Process results
            search_results = self._process_search_results(
                results,
                query,
                user_id,
                limit
            )

            duration = time.time() - start_time
            logger.info(
                f"Found {len(search_results)} memories for query '{query}' "
                f"by user {user_id} in {duration:.2f}s"
            )

            return search_results

        except Exception as e:
            logger.error(f"Memory search failed for user {user_id}: {e}")
            return []

    def _process_search_results(
        self,
        raw_results: Dict,
        query: str,
        user_id: str,
        limit: int
    ) -> List[MemorySearchResult]:
        """
        Process raw Chroma results into MemorySearchResult objects.

        Args:
            raw_results: Raw results from Chroma
            query: Original search query
            user_id: User identifier
            limit: Maximum results to return

        Returns:
            List[MemorySearchResult]: Processed search results
        """
        search_results = []

        if not raw_results['ids'] or not raw_results['ids'][0]:
            return search_results

        documents = raw_results['documents'][0]
        metadatas = raw_results['metadatas'][0]
        distances = raw_results['distances'][0]
        ids = raw_results['ids'][0]

        for i, (doc, metadata, distance, memory_id) in enumerate(
            zip(documents, metadatas, distances, ids)
        ):
            try:
                # Convert distance to similarity (Chroma uses L2 distance)
                # Lower distance = higher similarity
                similarity_score = max(0.0, 1.0 - (distance / 2.0))

                # Apply temporal decay and importance weighting
                final_score = self._calculate_relevance_score(
                    similarity_score,
                    metadata
                )

                # Create memory response object
                memory_response = self._create_memory_response(
                    memory_id,
                    doc,
                    metadata,
                    user_id
                )

                # Create search result
                search_result = MemorySearchResult(
                    memory=memory_response,
                    similarity_score=round(final_score, 4),
                    rank=i + 1
                )

                search_results.append(search_result)

            except Exception as e:
                logger.warning(f"Failed to process search result {i}: {e}")
                continue

        # Sort by final relevance score and limit results
        search_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return search_results[:limit]

    def _calculate_relevance_score(
        self,
        similarity_score: float,
        metadata: Dict
    ) -> float:
        """
        Calculate final relevance score with temporal decay and importance weighting.

        Args:
            similarity_score: Base similarity score
            metadata: Memory metadata

        Returns:
            float: Final relevance score
        """
        try:
            # Base similarity weight
            relevance = similarity_score * 0.6

            # Importance weighting (0.3)
            importance = metadata.get('importance_score', 5.0)
            importance_weight = (importance / 10.0) * 0.3
            relevance += importance_weight

            # Temporal decay (0.1)
            extracted_at = datetime.fromisoformat(metadata.get('extracted_at', datetime.utcnow().isoformat()))
            days_old = (datetime.utcnow() - extracted_at).days

            # Decay function: newer memories get slight boost
            if days_old <= 7:
                temporal_weight = 0.1  # Full boost for recent memories
            elif days_old <= 30:
                temporal_weight = 0.05  # Half boost for memories within a month
            else:
                temporal_weight = 0.0  # No boost for older memories

            relevance += temporal_weight

            return min(1.0, relevance)

        except Exception as e:
            logger.warning(f"Failed to calculate relevance score: {e}")
            return similarity_score

    def _create_memory_response(
        self,
        memory_id: str,
        content: str,
        metadata: Dict,
        user_id: str
    ):
        """
        Create MemoryResponse object from search results.

        Args:
            memory_id: Memory identifier
            content: Memory content
            metadata: Memory metadata
            user_id: User identifier

        Returns:
            MemoryResponse: Memory response object
        """
        from app.models.schemas import MemoryResponse

        # Parse dates
        extracted_at = datetime.fromisoformat(metadata.get('extracted_at'))
        updated_at = datetime.fromisoformat(metadata.get('updated_at'))
        expires_at = None
        if metadata.get('expires_at'):
            expires_at = datetime.fromisoformat(metadata['expires_at'])

        # Extract custom metadata
        custom_metadata = {}
        for key, value in metadata.items():
            if key.startswith('custom_'):
                custom_key = key[7:]  # Remove 'custom_' prefix
                custom_metadata[custom_key] = value

        return MemoryResponse(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            memory_type=MemoryType(metadata.get('memory_type', 'fact')),
            importance_score=metadata.get('importance_score', 5.0),
            confidence_score=metadata.get('confidence_score', 0.5),
            extracted_at=extracted_at,
            updated_at=updated_at,
            expires_at=expires_at,
            is_active=metadata.get('is_active', True),
            source_message_id=metadata.get('source_message_id') or None,
            metadata=custom_metadata if custom_metadata else None,
            version=metadata.get('version', 1)
        )

    async def update_memory(
        self,
        memory: Memory,
        new_embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Update existing memory in vector database.

        Args:
            memory: Updated memory object
            new_embedding: New embedding if content changed

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove old version first
            success = await self.delete_memory(memory.memory_id, memory.user_id)
            if not success:
                logger.warning(f"Failed to delete old version of memory {memory.memory_id}")

            # Store new version
            return await self.store_memory(memory, new_embedding)

        except Exception as e:
            logger.error(f"Failed to update memory {memory.memory_id}: {e}")
            return False

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """
        Delete memory from vector database.

        Args:
            memory_id: Memory identifier
            user_id: User identifier

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection = self._get_or_create_collection(user_id)
            collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory {memory_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def get_collection_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for user's memory collection.

        Args:
            user_id: User identifier

        Returns:
            Dict: Collection statistics

        Example:
            >>> stats = retriever.get_collection_stats("user123")
            >>> print(stats['total_memories'])  # 45
        """
        try:
            collection = self._get_or_create_collection(user_id)
            count = collection.count()

            # Get sample of memories for type distribution
            if count > 0:
                sample_size = min(count, 100)
                sample = collection.get(limit=sample_size, include=["metadatas"])

                type_counts = {}
                importance_scores = []

                for metadata in sample['metadatas']:
                    mem_type = metadata.get('memory_type', 'unknown')
                    type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                    importance_scores.append(metadata.get('importance_score', 5.0))

                avg_importance = sum(importance_scores) / len(importance_scores) if importance_scores else 0

                return {
                    'total_memories': count,
                    'type_distribution': type_counts,
                    'average_importance': round(avg_importance, 2),
                    'collection_name': self._get_collection_name(user_id)
                }
            else:
                return {
                    'total_memories': 0,
                    'type_distribution': {},
                    'average_importance': 0.0,
                    'collection_name': self._get_collection_name(user_id)
                }

        except Exception as e:
            logger.error(f"Failed to get collection stats for user {user_id}: {e}")
            return {'error': str(e)}

    def check_connection(self) -> bool:
        """
        Check if Chroma connection is healthy.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            # Try to list collections
            collections = self.client.list_collections()
            logger.debug(f"Chroma connection healthy, {len(collections)} collections found")
            return True
        except Exception as e:
            logger.error(f"Chroma connection check failed: {e}")
            return False