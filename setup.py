#!/usr/bin/env python3
"""
Simple Setup Script for Memory Pipeline
Checks and installs dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"  Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"  OK: {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {description} - Failed: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("Memory Pipeline Setup")
    print("=" * 30)

    # Check if .env file exists
    if not Path(".env").exists():
        print("\nWARNING: .env file not found!")
        print("  Please copy .env.example to .env and configure your settings")
        if Path(".env.example").exists():
            print("  Run: copy .env.example .env")
        return False

    print("OK: .env file found")

    # Check Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ required")
        return False

    print(f"OK: Python {sys.version.split()[0]} detected")

    # Install dependencies
    print("\nInstalling dependencies...")

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    if not in_venv:
        print("WARNING: Not in virtual environment. Consider using venv.")

    # Install main requirements
    if not run_command("pip install -r requirements.txt", "Installing main requirements"):
        return False

    # Install sentence-transformers specifically (common issue)
    if not run_command("pip install sentence-transformers", "Installing sentence-transformers"):
        return False

    print("\nOK: Setup completed successfully!")
    print("\nNext steps:")
    print("1. Configure your .env file with API keys")
    print("2. Run: python test_pipeline.py")
    print("3. Start server: python -m uvicorn app.main:app --reload")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)