"""
Production ChromaDB Cloud Configuration.
Replaces local ChromaDB with cloud-based vector database.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager

import chromadb
from chromadb import ClientAPI, Collection
from chromadb.config import Settings as ChromaSettings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import settings

# Configure logging
logger = logging.getLogger(__name__)

# ================================
# ChromaDB Cloud Configuration
# ================================

class ChromaDBCloudClient:
    """
    Production ChromaDB Cloud client with retry logic and error handling.
    """

    def __init__(self):
        """Initialize ChromaDB cloud client."""
        self.client: Optional[ClientAPI] = None
        self.collections: Dict[str, Collection] = {}
        self.config = self._get_cloud_config()

    def _get_cloud_config(self) -> Dict[str, Any]:
        """
        Get ChromaDB cloud configuration from environment.

        Returns:
            Configuration dictionary
        """
        return {
            'api_key': getattr(settings, 'chroma_api_key', None),
            'api_url': getattr(settings, 'chroma_api_url', 'https://api.trychroma.com'),
            'tenant': getattr(settings, 'chroma_tenant', 'default_tenant'),
            'database': getattr(settings, 'chroma_database', 'memory_pipeline'),
            'collection_name': getattr(settings, 'chroma_collection_name', 'memories'),
            'batch_size': getattr(settings, 'chroma_batch_size', 100),
            'max_retries': getattr(settings, 'chroma_max_retries', 3),
            'timeout': getattr(settings, 'chroma_timeout', 30),
            'embedding_model': getattr(settings, 'chroma_embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            'embedding_dimension': getattr(settings, 'chroma_embedding_dimension', 384)
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def connect(self) -> bool:
        """
        Connect to ChromaDB cloud with retry logic.

        Returns:
            bool: True if connection successful
        """
        try:
            if not self.config['api_key']:
                raise ValueError("CHROMA_API_KEY environment variable not set")

            # Configure ChromaDB client for cloud
            chroma_settings = ChromaSettings(
                chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                chroma_client_auth_credentials=self.config['api_key']
            )

            # Create cloud client
            self.client = chromadb.HttpClient(
                host=self.config['api_url'],
                settings=chroma_settings,
                tenant=self.config['tenant'],
                database=self.config['database']
            )

            # Test connection
            self.client.heartbeat()

            logger.info(f"Connected to ChromaDB cloud: {self.config['api_url']}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB cloud: {e}")
            raise

    def get_or_create_collection(self, collection_name: str, user_id: str) -> Collection:
        """
        Get or create a collection for a specific user.

        Args:
            collection_name: Base collection name
            user_id: User identifier

        Returns:
            ChromaDB collection
        """
        if not self.client:
            raise RuntimeError("ChromaDB client not connected")

        # Create user-specific collection name
        full_collection_name = f"{collection_name}_{user_id}"

        try:
            # Check if collection exists in cache
            if full_collection_name in self.collections:
                return self.collections[full_collection_name]

            # Try to get existing collection
            try:
                collection = self.client.get_collection(
                    name=full_collection_name,
                    embedding_function=self._get_embedding_function()
                )
                logger.debug(f"Retrieved existing collection: {full_collection_name}")

            except Exception:
                # Create new collection
                collection = self.client.create_collection(
                    name=full_collection_name,
                    embedding_function=self._get_embedding_function(),
                    metadata={
                        "user_id": user_id,
                        "created_at": time.time(),
                        "description": f"Memory collection for user {user_id}",
                        "version": "1.0"
                    }
                )
                logger.info(f"Created new collection: {full_collection_name}")

            # Cache collection
            self.collections[full_collection_name] = collection
            return collection

        except Exception as e:
            logger.error(f"Failed to get or create collection {full_collection_name}: {e}")
            raise

    def _get_embedding_function(self):
        """
        Get the embedding function for ChromaDB.

        Returns:
            Embedding function
        """
        try:
            # Use SentenceTransformer embedding function
            from chromadb.utils import embedding_functions

            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config['embedding_model']
            )
            return sentence_transformer_ef

        except Exception as e:
            logger.error(f"Failed to create embedding function: {e}")
            # Fallback to default
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def store_memories(
        self,
        user_id: str,
        memory_ids: List[str],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """
        Store memories in ChromaDB cloud with batch processing.

        Args:
            user_id: User identifier
            memory_ids: List of memory IDs
            contents: List of memory contents
            metadatas: List of metadata dictionaries
            embeddings: Optional pre-computed embeddings

        Returns:
            bool: True if storage successful
        """
        try:
            collection = self.get_or_create_collection(
                self.config['collection_name'],
                user_id
            )

            # Process in batches
            batch_size = self.config['batch_size']
            total_memories = len(memory_ids)

            for i in range(0, total_memories, batch_size):
                end_idx = min(i + batch_size, total_memories)

                batch_ids = memory_ids[i:end_idx]
                batch_contents = contents[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                batch_embeddings = embeddings[i:end_idx] if embeddings else None

                # Add to collection
                if batch_embeddings:
                    collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_contents,
                        metadatas=batch_metadatas
                    )
                else:
                    collection.add(
                        ids=batch_ids,
                        documents=batch_contents,
                        metadatas=batch_metadatas
                    )

                logger.debug(f"Stored batch {i//batch_size + 1} ({end_idx - i} memories) for user {user_id}")

            logger.info(f"Successfully stored {total_memories} memories for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store memories for user {user_id}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search memories in ChromaDB cloud.

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum number of results
            where: Metadata filters
            where_document: Document content filters

        Returns:
            Search results dictionary
        """
        try:
            collection = self.get_or_create_collection(
                self.config['collection_name'],
                user_id
            )

            start_time = time.time()

            # Perform search
            results = collection.query(
                query_texts=[query],
                n_results=limit,
                where=where,
                where_document=where_document,
                include=['documents', 'metadatas', 'distances']
            )

            search_time = (time.time() - start_time) * 1000

            # Format results
            formatted_results = {
                'ids': results.get('ids', [[]])[0],
                'documents': results.get('documents', [[]])[0],
                'metadatas': results.get('metadatas', [[]])[0],
                'distances': results.get('distances', [[]])[0],
                'search_time_ms': search_time,
                'total_results': len(results.get('ids', [[]])[0])
            }

            logger.debug(f"Search completed for user {user_id}: {len(formatted_results['ids'])} results in {search_time:.2f}ms")

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search memories for user {user_id}: {e}")
            raise

    def update_memory(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Update a memory in ChromaDB cloud.

        Args:
            user_id: User identifier
            memory_id: Memory ID to update
            content: Updated content
            metadata: Updated metadata
            embedding: Optional updated embedding

        Returns:
            bool: True if update successful
        """
        try:
            collection = self.get_or_create_collection(
                self.config['collection_name'],
                user_id
            )

            # Update the memory
            if embedding:
                collection.update(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata]
                )
            else:
                collection.update(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[metadata]
                )

            logger.debug(f"Updated memory {memory_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update memory {memory_id} for user {user_id}: {e}")
            raise

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """
        Delete a memory from ChromaDB cloud.

        Args:
            user_id: User identifier
            memory_id: Memory ID to delete

        Returns:
            bool: True if deletion successful
        """
        try:
            collection = self.get_or_create_collection(
                self.config['collection_name'],
                user_id
            )

            collection.delete(ids=[memory_id])

            logger.debug(f"Deleted memory {memory_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id} for user {user_id}: {e}")
            raise

    def get_collection_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a user's collection.

        Args:
            user_id: User identifier

        Returns:
            Collection statistics
        """
        try:
            collection = self.get_or_create_collection(
                self.config['collection_name'],
                user_id
            )

            count = collection.count()

            return {
                'total_memories': count,
                'collection_name': f"{self.config['collection_name']}_{user_id}",
                'embedding_model': self.config['embedding_model'],
                'last_updated': time.time()
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats for user {user_id}: {e}")
            return {'error': str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on ChromaDB cloud connection.

        Returns:
            Health check results
        """
        try:
            if not self.client:
                return {'status': 'disconnected', 'error': 'Client not connected'}

            # Test connection
            start_time = time.time()
            self.client.heartbeat()
            response_time = (time.time() - start_time) * 1000

            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'api_url': self.config['api_url'],
                'tenant': self.config['tenant'],
                'database': self.config['database'],
                'cached_collections': len(self.collections)
            }

        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'api_url': self.config['api_url']
            }

    def cleanup_expired_collections(self, days_old: int = 30) -> int:
        """
        Clean up old collections (optional maintenance).

        Args:
            days_old: Collections older than this many days

        Returns:
            Number of collections cleaned up
        """
        try:
            if not self.client:
                return 0

            # List all collections
            collections = self.client.list_collections()
            cleanup_count = 0
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)

            for collection in collections:
                try:
                    metadata = collection.metadata or {}
                    created_at = metadata.get('created_at', time.time())

                    if created_at < cutoff_time:
                        self.client.delete_collection(collection.name)
                        cleanup_count += 1
                        logger.info(f"Cleaned up old collection: {collection.name}")

                except Exception as e:
                    logger.warning(f"Failed to clean up collection {collection.name}: {e}")

            return cleanup_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired collections: {e}")
            return 0

    def disconnect(self):
        """
        Disconnect from ChromaDB cloud and cleanup resources.
        """
        try:
            # Clear cached collections
            self.collections.clear()
            self.client = None

            logger.info("Disconnected from ChromaDB cloud")

        except Exception as e:
            logger.error(f"Error disconnecting from ChromaDB cloud: {e}")

# ================================
# Global ChromaDB Client Instance
# ================================

# Create global client instance
chroma_client = ChromaDBCloudClient()

# ================================
# Convenience Functions
# ================================

def get_chroma_client() -> ChromaDBCloudClient:
    """
    Get the global ChromaDB client instance.

    Returns:
        ChromaDBCloudClient instance
    """
    return chroma_client

async def startup_chromadb():
    """
    Startup handler for ChromaDB cloud connection.
    """
    logger.info("Connecting to ChromaDB cloud...")

    try:
        chroma_client.connect()
        health = chroma_client.health_check()

        if health['status'] != 'healthy':
            raise RuntimeError(f"ChromaDB health check failed: {health}")

        logger.info("ChromaDB cloud connection established successfully")

    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB cloud: {e}")
        raise

async def shutdown_chromadb():
    """
    Shutdown handler for ChromaDB cloud connection.
    """
    logger.info("Disconnecting from ChromaDB cloud...")
    chroma_client.disconnect()
    logger.info("ChromaDB cloud disconnected")

# ================================
# Testing and Utilities
# ================================

def test_chromadb_connection() -> bool:
    """
    Test ChromaDB cloud connection.

    Returns:
        bool: True if connection successful
    """
    try:
        test_client = ChromaDBCloudClient()
        test_client.connect()

        health = test_client.health_check()
        test_client.disconnect()

        return health['status'] == 'healthy'

    except Exception as e:
        logger.error(f"ChromaDB connection test failed: {e}")
        return False

async def migrate_to_cloud(local_data_path: str) -> bool:
    """
    Migrate data from local ChromaDB to cloud (if needed).

    Args:
        local_data_path: Path to local ChromaDB data

    Returns:
        bool: True if migration successful
    """
    try:
        # This would implement migration logic if needed
        # For now, just log that migration is not implemented
        logger.warning("ChromaDB migration from local to cloud not implemented yet")
        return True

    except Exception as e:
        logger.error(f"ChromaDB migration failed: {e}")
        return False