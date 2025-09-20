"""
Startup Check - Verify all dependencies are working before starting server
"""

import os
import sys
from dotenv import load_dotenv

def startup_check():
    """Verify all critical dependencies before starting the server."""
    print("Starting dependency checks...")

    # Load environment
    load_dotenv()

    # Check 1: PostgreSQL connection
    try:
        from app.core.database_prod import check_database_connection
        if not check_database_connection():
            print("ERROR: PostgreSQL connection failed")
            print("   Fix: Check DATABASE_URL in .env file")
            sys.exit(1)
        print("OK: PostgreSQL connected")
    except Exception as e:
        print(f"ERROR: PostgreSQL check failed: {e}")
        sys.exit(1)

    # Check 2: ChromaDB import
    try:
        import chromadb
        print("OK: ChromaDB available")
    except ImportError:
        print("ERROR: ChromaDB not installed")
        print("   Fix: pip install chromadb")
        sys.exit(1)

    # Check 3: Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not found")
        print("   Fix: Add GEMINI_API_KEY to .env file")
        sys.exit(1)
    print("OK: Gemini API key configured")

    print("OK: All startup checks passed!")

if __name__ == "__main__":
    startup_check()