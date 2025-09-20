"""
Rate limiting for external API calls.

Implements rate limiting with exponential backoff for Gemini API
to stay within free tier limits (15 RPM).
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Custom exception for rate limiting errors."""
    pass


class APIQuotaError(Exception):
    """Custom exception for API quota errors."""
    pass


class RateLimiter:
    """
    Rate limiter for API calls with request tracking.

    Tracks API calls and enforces rate limits to prevent quota exhaustion.
    Uses token bucket algorithm for smooth rate limiting.
    """

    def __init__(self, max_requests_per_minute: int = 15):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.requests: Dict[int, int] = {}  # minute -> request_count
        self.last_request_time = 0.0
        self.min_interval = 60.0 / max_requests_per_minute  # Seconds between requests

    def _get_current_minute(self) -> int:
        """Get current minute as integer for tracking."""
        return int(time.time() // 60)

    def _cleanup_old_requests(self) -> None:
        """Remove request counts older than 1 minute."""
        current_minute = self._get_current_minute()
        # Keep only current minute
        self.requests = {minute: count for minute, count in self.requests.items()
                        if minute >= current_minute}

    def get_requests_this_minute(self) -> int:
        """
        Get number of requests made in current minute.

        Returns:
            int: Number of requests in current minute
        """
        self._cleanup_old_requests()
        current_minute = self._get_current_minute()
        return self.requests.get(current_minute, 0)

    def can_make_request(self) -> bool:
        """
        Check if a request can be made without exceeding rate limit.

        Returns:
            bool: True if request can be made, False otherwise
        """
        current_requests = self.get_requests_this_minute()
        return current_requests < self.max_requests_per_minute

    async def wait_if_needed(self) -> None:
        """
        Wait if necessary to respect rate limits.

        Implements smooth rate limiting by ensuring minimum interval
        between requests.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()

    def record_request(self) -> None:
        """Record that a request was made."""
        current_minute = self._get_current_minute()
        self.requests[current_minute] = self.requests.get(current_minute, 0) + 1
        logger.debug(f"Recorded request. Count this minute: {self.requests[current_minute]}")

    def get_wait_time_until_available(self) -> float:
        """
        Get time to wait until next request can be made.

        Returns:
            float: Seconds to wait, 0 if request can be made immediately
        """
        if self.can_make_request():
            return 0.0

        # Calculate time until next minute starts
        current_time = time.time()
        seconds_in_minute = current_time % 60
        return 60.0 - seconds_in_minute


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests_per_minute=settings.rate_limit_per_minute)


def handle_gemini_errors(func: Callable) -> Callable:
    """
    Decorator to handle common Gemini API errors.

    Args:
        func: Function to wrap

    Returns:
        Callable: Wrapped function with error handling
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_message = str(e).lower()

            # Check for rate limiting
            if "quota" in error_message or "rate limit" in error_message:
                logger.warning(f"API rate limit hit: {e}")
                raise RateLimitError(f"API rate limit exceeded: {e}")

            # Check for quota exhaustion
            if "quota exceeded" in error_message or "billing" in error_message:
                logger.error(f"API quota exhausted: {e}")
                raise APIQuotaError(f"API quota exhausted: {e}")

            # Check for authentication errors
            if "api key" in error_message or "authentication" in error_message:
                logger.error(f"API authentication error: {e}")
                raise Exception(f"API authentication error: {e}")

            # Re-raise other errors
            logger.error(f"Unexpected API error: {e}")
            raise

    return wrapper


@retry(
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
@handle_gemini_errors
async def call_gemini_with_retry(
    model_name: str,
    prompt: str,
    **kwargs
) -> Any:
    """
    Call Gemini API with retry logic and rate limiting.

    Args:
        model_name: Gemini model name
        prompt: Text prompt to send
        **kwargs: Additional model parameters

    Returns:
        Any: Model response

    Raises:
        RateLimitError: If rate limit is exceeded after retries
        APIQuotaError: If API quota is exhausted
        Exception: For other API errors

    Example:
        >>> response = await call_gemini_with_retry(
        ...     "gemini-1.5-flash",
        ...     "Extract memories from: I love pizza"
        ... )
    """
    # Wait for rate limiting
    await rate_limiter.wait_if_needed()

    # Check if we can make request
    if not rate_limiter.can_make_request():
        wait_time = rate_limiter.get_wait_time_until_available()
        logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
        await asyncio.sleep(wait_time)

    # Record the request
    rate_limiter.record_request()

    try:
        # Initialize the model
        model = genai.GenerativeModel(model_name)

        # Make the API call
        logger.debug(f"Calling Gemini API: {model_name}")
        response = await model.generate_content_async(prompt, **kwargs)

        logger.debug("Gemini API call successful")
        return response

    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise


async def get_embedding_with_retry(text: str) -> list:
    """
    Get text embedding with retry logic and rate limiting.

    Args:
        text: Text to embed

    Returns:
        list: Embedding vector

    Raises:
        RateLimitError: If rate limit is exceeded after retries
        APIQuotaError: If API quota is exhausted

    Example:
        >>> embedding = await get_embedding_with_retry("I love coffee")
        >>> print(len(embedding))  # 768
    """
    # Wait for rate limiting
    await rate_limiter.wait_if_needed()

    # Check if we can make request
    if not rate_limiter.can_make_request():
        wait_time = rate_limiter.get_wait_time_until_available()
        logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
        await asyncio.sleep(wait_time)

    # Record the request
    rate_limiter.record_request()

    try:
        logger.debug("Calling Gemini embedding API")
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )

        logger.debug("Gemini embedding API call successful")
        return response['embedding']

    except Exception as e:
        error_message = str(e).lower()

        if "quota" in error_message or "rate limit" in error_message:
            logger.warning(f"Embedding API rate limit hit: {e}")
            raise RateLimitError(f"Embedding API rate limit exceeded: {e}")

        logger.error(f"Embedding API call failed: {e}")
        raise