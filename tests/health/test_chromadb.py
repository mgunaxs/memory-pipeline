#!/usr/bin/env python3
"""
Test ChromaDB cloud connection.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_chromadb_simple():
    """Simple ChromaDB test to avoid NumPy issues."""
    try:
        from app.core.config import settings

        print("Testing ChromaDB configuration...")
        print(f"API URL: {settings.chroma_api_url}")
        print(f"Tenant: {settings.chroma_tenant}")
        print(f"Database: {settings.chroma_database}")

        # Test if API key is configured
        if hasattr(settings, 'chroma_api_key') and settings.chroma_api_key:
            print("SUCCESS: ChromaDB API key configured")
            key_preview = f"{settings.chroma_api_key[:10]}..." if len(settings.chroma_api_key) > 10 else "SHORT_KEY"
            print(f"API Key: {key_preview}")
        else:
            print("ERROR: ChromaDB API key not configured")
            return False

        # Try basic import
        try:
            import chromadb
            print("SUCCESS: ChromaDB package imported")
        except Exception as e:
            print(f"ERROR: ChromaDB import failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"ERROR: ChromaDB test failed: {e}")
        return False

def test_chromadb_connection():
    """Test actual ChromaDB connection."""
    try:
        from app.core.chromadb_prod import test_chromadb_connection

        print("Testing ChromaDB cloud connection...")
        if test_chromadb_connection():
            print("SUCCESS: ChromaDB cloud connection established")
            return True
        else:
            print("ERROR: ChromaDB cloud connection failed")
            return False

    except Exception as e:
        print(f"ERROR: ChromaDB connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing ChromaDB Connection")
    print("=" * 50)

    # Test configuration first
    config_ok = test_chromadb_simple()

    # Test connection if config is OK
    connection_ok = False
    if config_ok:
        connection_ok = test_chromadb_connection()

    print("\n" + "=" * 50)
    print("CHROMADB SUMMARY:")
    print(f"Configuration: {'SUCCESS' if config_ok else 'FAILED'}")
    print(f"Connection:    {'SUCCESS' if connection_ok else 'FAILED'}")

    if config_ok and connection_ok:
        print("CHROMADB READY!")
        sys.exit(0)
    else:
        print("ChromaDB not ready")
        sys.exit(1)