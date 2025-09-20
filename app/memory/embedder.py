"""
Embedding service for converting text to vectors.

Provides embedding generation using Gemini text-embedding-004 with caching,
fallback to local models, and performance optimization.
"""

import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple
import pickle
import os

import google.generativeai as genai

from app.core.config import settings
from app.core.rate_limiter import get_embedding_with_retry, RateLimitError, APIQuotaError

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=settings.gemini_api_key)


class EmbeddingCache:
    """
    Simple file-based cache for embeddings.

    Caches embeddings to avoid redundant API calls and improve performance.
    """

    def __init__(self, cache_size: int = 1000):
        """
        Initialize embedding cache.

        Args:
            cache_size: Maximum number of embeddings to cache
        """
        self.cache_size = cache_size
        self.cache_file = os.path.join(settings.chroma_persist_directory, "embedding_cache.pkl")
        self.cache: Dict[str, List[float]] = {}
        self._load_cache()

    def _get_cache_key(self, text: str, model: str = "gemini") -> str:
        """
        Generate cache key for text and model.

        Args:
            text: Text to hash
            model: Model name

        Returns:
            str: Cache key
        """
        content = f"{model}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _load_cache(self) -> None:
        """Load cache from disk if it exists."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.debug(f"Loaded {len(self.cache)} embeddings from cache")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self.cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.debug(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def get(self, text: str, model: str = "gemini") -> Optional[List[float]]:
        """
        Get embedding from cache.

        Args:
            text: Text to look up
            model: Model name

        Returns:
            Optional[List[float]]: Cached embedding or None
        """
        key = self._get_cache_key(text, model)
        return self.cache.get(key)

    def put(self, text: str, embedding: List[float], model: str = "gemini") -> None:
        """
        Store embedding in cache.

        Args:
            text: Text being embedded
            embedding: Embedding vector
            model: Model name
        """
        key = self._get_cache_key(text, model)

        # Evict oldest entries if cache is full
        if len(self.cache) >= self.cache_size:
            # Remove 10% of cache entries (simple LRU approximation)
            keys_to_remove = list(self.cache.keys())[:int(self.cache_size * 0.1)]
            for old_key in keys_to_remove:
                del self.cache[old_key]

        self.cache[key] = embedding
        self._save_cache()

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("Embedding cache cleared")


class EmbeddingService:
    """
    Service for generating text embeddings.

    Provides embeddings using Gemini API with caching and fallback options.
    Handles rate limiting and error recovery automatically.
    """

    def __init__(self):
        """Initialize embedding service."""
        self.cache = EmbeddingCache(cache_size=settings.embedding_cache_size)
        self.gemini_model = "models/text-embedding-004"
        self.embedding_dimension = 768  # Gemini embedding dimension
        self.fallback_available = False

        # Try to initialize fallback model
        self._init_fallback()

    def _init_fallback(self) -> None:
        """
        Initialize fallback sentence-transformers model.

        This provides a local alternative when API limits are reached.
        """
        try:
            import sentence_transformers
            self.fallback_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
            self.fallback_dimension = 384  # MiniLM dimension
            self.fallback_available = True
            logger.info("Fallback embedding model initialized")
        except ImportError:
            logger.warning(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize fallback model: {e}")

    async def get_embedding(self, text: str, use_cache: bool = True) -> Tuple[List[float], str]:
        """
        Get embedding for text.

        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            Tuple[List[float], str]: (embedding_vector, model_used)

        Raises:
            ValueError: If text is empty
            Exception: If all embedding methods fail

        Example:
            >>> service = EmbeddingService()
            >>> embedding, model = await service.get_embedding("I love coffee")
            >>> print(len(embedding))  # 768 (Gemini) or 384 (fallback)
            >>> print(model)  # "gemini" or "sentence-transformers"
        """
        start_time = time.time()

        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text = text.strip()

        # Check cache first
        if use_cache:
            cached_embedding = self.cache.get(text, "gemini")
            if cached_embedding is not None:
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached_embedding, "gemini"

        # Try Gemini API first
        try:
            embedding = await self._get_gemini_embedding(text)

            # Cache the result
            if use_cache:
                self.cache.put(text, embedding, "gemini")

            duration = time.time() - start_time
            logger.debug(f"Generated Gemini embedding in {duration:.2f}s")
            return embedding, "gemini"

        except (RateLimitError, APIQuotaError) as e:
            logger.warning(f"Gemini API limit reached: {e}")

            # Fall back to local model if available
            if self.fallback_available:
                logger.info("Falling back to sentence-transformers")
                return await self._get_fallback_embedding(text, use_cache), "sentence-transformers"
            else:
                raise Exception(
                    "Gemini API limit reached and no fallback model available. "
                    "Install sentence-transformers: pip install sentence-transformers"
                )

        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")

            # Fall back to local model if available
            if self.fallback_available:
                logger.info("Falling back to sentence-transformers due to error")
                return await self._get_fallback_embedding(text, use_cache), "sentence-transformers"
            else:
                raise Exception(f"Embedding generation failed: {e}")

    async def _get_gemini_embedding(self, text: str) -> List[float]:
        """
        Get embedding from Gemini API.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector

        Raises:
            RateLimitError: If rate limit is exceeded
            APIQuotaError: If quota is exhausted
        """
        embedding = await get_embedding_with_retry(text)
        return embedding

    async def _get_fallback_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Get embedding from fallback model.

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            List[float]: Embedding vector
        """
        if not self.fallback_available:
            raise Exception("Fallback model not available")

        # Check cache for fallback model
        if use_cache:
            cached_embedding = self.cache.get(text, "sentence-transformers")
            if cached_embedding is not None:
                logger.debug(f"Fallback cache hit for text: {text[:50]}...")
                return cached_embedding

        # Generate embedding
        embedding = self.fallback_model.encode(text).tolist()

        # Cache the result
        if use_cache:
            self.cache.put(text, embedding, "sentence-transformers")

        return embedding

    async def get_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[Tuple[List[float], str]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings

        Returns:
            List[Tuple[List[float], str]]: List of (embedding, model_used) tuples

        Example:
            >>> service = EmbeddingService()
            >>> embeddings = await service.get_embeddings_batch([
            ...     "I love coffee",
            ...     "I work at Google"
            ... ])
            >>> print(len(embeddings))  # 2
        """
        start_time = time.time()
        results = []

        logger.info(f"Generating embeddings for {len(texts)} texts")

        for i, text in enumerate(texts):
            try:
                embedding, model = await self.get_embedding(text, use_cache)
                results.append((embedding, model))

                # Log progress for large batches
                if len(texts) > 5 and (i + 1) % 5 == 0:
                    logger.debug(f"Processed {i + 1}/{len(texts)} embeddings")

            except Exception as e:
                logger.error(f"Failed to embed text {i}: {e}")
                # Continue with other texts
                results.append(([], "error"))

        duration = time.time() - start_time
        successful = sum(1 for embedding, model in results if model != "error")
        logger.info(
            f"Generated {successful}/{len(texts)} embeddings in {duration:.2f}s"
        )

        return results

    def get_embedding_dimension(self, model: str = "gemini") -> int:
        """
        Get embedding dimension for model.

        Args:
            model: Model name ("gemini" or "sentence-transformers")

        Returns:
            int: Embedding dimension

        Example:
            >>> service = EmbeddingService()
            >>> dim = service.get_embedding_dimension("gemini")
            >>> print(dim)  # 768
        """
        if model == "gemini":
            return self.embedding_dimension
        elif model == "sentence-transformers":
            return self.fallback_dimension
        else:
            raise ValueError(f"Unknown model: {model}")

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dict: Cache statistics

        Example:
            >>> stats = service.get_cache_stats()
            >>> print(stats['total_entries'])  # 150
        """
        return {
            'total_entries': len(self.cache.cache),
            'max_size': self.cache.cache_size,
            'cache_file': self.cache.cache_file,
            'fallback_available': self.fallback_available
        }

    def clear_cache(self) -> None:
        """
        Clear embedding cache.

        Removes all cached embeddings to free memory and storage.
        """
        self.cache.clear()
        logger.info("Embedding cache cleared")