"""
Vector database service for advanced operations.

Provides higher-level vector operations, batch processing,
and specialized search functionality.
"""

import logging
from typing import Dict, List, Optional, Tuple

from app.memory.embedder import EmbeddingService
from app.memory.retriever import MemoryRetriever
from app.models.schemas import MemoryResponse

logger = logging.getLogger(__name__)


class VectorService:
    """
    Advanced vector database operations.

    Provides batch operations, similarity analysis, and advanced
    search capabilities beyond basic retrieval.
    """

    def __init__(self, embedding_service: EmbeddingService, retriever: MemoryRetriever):
        """
        Initialize vector service.

        Args:
            embedding_service: Embedding generation service
            retriever: Memory retrieval service
        """
        self.embedding_service = embedding_service
        self.retriever = retriever

    async def find_similar_memories(
        self,
        target_memory: MemoryResponse,
        user_id: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Tuple[MemoryResponse, float]]:
        """
        Find memories similar to a target memory.

        Args:
            target_memory: Memory to find similarities for
            user_id: User identifier
            limit: Maximum number of similar memories
            min_similarity: Minimum similarity threshold

        Returns:
            List[Tuple[MemoryResponse, float]]: Similar memories with scores

        Example:
            >>> similar = await service.find_similar_memories(
            ...     target_memory,
            ...     "user123",
            ...     limit=3
            ... )
            >>> print(len(similar))  # 2
        """
        try:
            # Use target memory content as query
            search_results = await self.retriever.search_memories(
                query=target_memory.content,
                user_id=user_id,
                limit=limit + 1  # +1 to account for the target memory itself
            )

            # Filter out the target memory and apply similarity threshold
            similar_memories = []
            for result in search_results:
                # Skip the target memory itself
                if result.memory.memory_id == target_memory.memory_id:
                    continue

                # Apply similarity threshold
                if result.similarity_score >= min_similarity:
                    similar_memories.append((result.memory, result.similarity_score))

            # Limit results
            return similar_memories[:limit]

        except Exception as e:
            logger.error(f"Failed to find similar memories: {e}")
            return []

    async def get_memory_clusters(
        self,
        user_id: str,
        cluster_threshold: float = 0.8
    ) -> Dict[str, List[MemoryResponse]]:
        """
        Group user memories into clusters based on similarity.

        Args:
            user_id: User identifier
            cluster_threshold: Similarity threshold for clustering

        Returns:
            Dict[str, List[MemoryResponse]]: Memory clusters

        Example:
            >>> clusters = await service.get_memory_clusters("user123")
            >>> print(clusters.keys())  # ['work', 'hobbies', 'food']
        """
        try:
            # Get all user memories
            # Note: This would require integration with the memory service
            # For now, return empty clusters
            logger.info(f"Memory clustering not yet implemented for user {user_id}")
            return {}

        except Exception as e:
            logger.error(f"Failed to cluster memories: {e}")
            return {}

    async def analyze_memory_patterns(
        self,
        user_id: str
    ) -> Dict[str, any]:
        """
        Analyze patterns in user memories.

        Args:
            user_id: User identifier

        Returns:
            Dict: Pattern analysis results

        Example:
            >>> patterns = await service.analyze_memory_patterns("user123")
            >>> print(patterns['dominant_types'])  # ['fact', 'preference']
        """
        try:
            # Get collection stats
            stats = self.retriever.get_collection_stats(user_id)

            if stats.get('total_memories', 0) == 0:
                return {
                    'total_memories': 0,
                    'dominant_types': [],
                    'patterns': {}
                }

            # Analyze type distribution
            type_dist = stats.get('type_distribution', {})
            total = sum(type_dist.values())

            # Find dominant types (>20% of memories)
            dominant_types = [
                mem_type for mem_type, count in type_dist.items()
                if count / total > 0.2
            ]

            # Calculate patterns
            patterns = {
                'type_diversity': len(type_dist),
                'most_common_type': max(type_dist.items(), key=lambda x: x[1])[0] if type_dist else None,
                'type_percentages': {
                    mem_type: round((count / total) * 100, 1)
                    for mem_type, count in type_dist.items()
                } if total > 0 else {}
            }

            return {
                'total_memories': stats.get('total_memories', 0),
                'dominant_types': dominant_types,
                'patterns': patterns,
                'average_importance': stats.get('average_importance', 0.0)
            }

        except Exception as e:
            logger.error(f"Failed to analyze memory patterns: {e}")
            return {'error': str(e)}

    def check_vector_health(self) -> Dict[str, any]:
        """
        Check vector database health and performance.

        Returns:
            Dict: Health check results

        Example:
            >>> health = service.check_vector_health()
            >>> print(health['status'])  # 'healthy'
        """
        try:
            # Check Chroma connection
            chroma_healthy = self.retriever.check_connection()

            # Check embedding service
            cache_stats = self.embedding_service.get_cache_stats()

            return {
                'status': 'healthy' if chroma_healthy else 'unhealthy',
                'chroma_connection': chroma_healthy,
                'embedding_cache': cache_stats,
                'fallback_available': self.embedding_service.fallback_available
            }

        except Exception as e:
            logger.error(f"Vector health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }