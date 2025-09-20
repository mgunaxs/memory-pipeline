"""
Security utilities for input validation and sanitization.

Provides functions to sanitize user input and prevent security vulnerabilities
like XSS, injection attacks, and other malicious input.
"""

import html
import re
from typing import Optional


def sanitize_text_input(text: str, max_length: int = 5000) -> str:
    """
    Sanitize text input to prevent XSS and injection attacks.

    Args:
        text: Raw text input from user
        max_length: Maximum allowed length

    Returns:
        Sanitized text safe for processing

    Raises:
        ValueError: If input is too long or contains dangerous content
    """
    if not text or not isinstance(text, str):
        return ""

    # Check length
    if len(text) > max_length:
        raise ValueError(f"Input too long. Maximum {max_length} characters allowed.")

    # Strip and normalize whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces

    # HTML escape to prevent XSS
    text = html.escape(text)

    # Remove potentially dangerous sequences
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'data:text/html',            # Data URLs
        r'vbscript:',                 # VBScript URLs
    ]

    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Limit special characters that could be used for injection
    if re.search(r'[<>"\'\`\$\{\}]', text):
        # Allow some basic punctuation but escape dangerous chars
        text = re.sub(r'[<>"\'\`\$\{\}]', '', text)

    return text


def validate_user_id(user_id: str) -> str:
    """
    Validate and sanitize user ID to ensure it's safe for database operations.

    Args:
        user_id: User identifier

    Returns:
        Sanitized user ID

    Raises:
        ValueError: If user_id is invalid
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("User ID is required")

    user_id = user_id.strip()

    # Check length
    if len(user_id) > 100:
        raise ValueError("User ID too long. Maximum 100 characters allowed.")

    # Only allow alphanumeric, hyphens, underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        raise ValueError("User ID can only contain letters, numbers, hyphens, and underscores")

    return user_id


def validate_message_id(message_id: str) -> str:
    """
    Validate and sanitize message ID.

    Args:
        message_id: Message identifier

    Returns:
        Sanitized message ID

    Raises:
        ValueError: If message_id is invalid
    """
    if not message_id or not isinstance(message_id, str):
        raise ValueError("Message ID is required")

    message_id = message_id.strip()

    # Check length
    if len(message_id) > 200:
        raise ValueError("Message ID too long. Maximum 200 characters allowed.")

    # Only allow alphanumeric, hyphens, underscores, dots
    if not re.match(r'^[a-zA-Z0-9._-]+$', message_id):
        raise ValueError("Message ID can only contain letters, numbers, hyphens, underscores, and dots")

    return message_id


def validate_search_query(query: str) -> str:
    """
    Validate and sanitize search query.

    Args:
        query: Search query string

    Returns:
        Sanitized search query

    Raises:
        ValueError: If query is invalid
    """
    if not query or not isinstance(query, str):
        raise ValueError("Search query is required")

    query = query.strip()

    # Check length
    if len(query) > 500:
        raise ValueError("Search query too long. Maximum 500 characters allowed.")

    # Remove dangerous characters but allow more flexibility for search
    query = html.escape(query)

    # Remove control characters
    query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)

    return query


def check_content_safety(content: str) -> bool:
    """
    Check if content appears to be safe (not spam, not malicious).

    Args:
        content: Content to check

    Returns:
        True if content appears safe, False otherwise
    """
    if not content:
        return True

    # Check for excessive repetition (possible spam)
    words = content.lower().split()
    if len(words) > 10:
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1

        # If any word appears more than 30% of the time, likely spam
        max_frequency = max(word_count.values())
        if max_frequency / len(words) > 0.3:
            return False

    # Check for suspicious patterns
    suspicious_patterns = [
        r'(viagra|cialis|casino|lottery|winner|congratulations).{0,20}(free|win|click)',
        r'urgent.{0,20}(action|response|reply)',
        r'(click|visit).{0,20}(here|now|link)',
        r'\$\d+.{0,20}(free|earn|make)',
    ]

    content_lower = content.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, content_lower):
            return False

    return True


def sanitize_memory_content(content: str) -> str:
    """
    Sanitize memory content with specific rules for memory storage.

    Args:
        content: Raw memory content

    Returns:
        Sanitized memory content

    Raises:
        ValueError: If content is invalid or unsafe
    """
    # Basic sanitization
    content = sanitize_text_input(content, max_length=2000)

    # Check safety
    if not check_content_safety(content):
        raise ValueError("Content appears to be spam or malicious")

    # Ensure minimum content length
    if len(content.strip()) < 3:
        raise ValueError("Memory content too short. Minimum 3 characters required.")

    return content