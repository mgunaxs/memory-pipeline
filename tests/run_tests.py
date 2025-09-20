#!/usr/bin/env python3
"""
Test Runner for Memory Pipeline

Provides standardized way to run different test suites
with proper configuration and reporting.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    print("Running Unit Tests...")
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_integration_tests(verbose=False):
    """Run integration tests."""
    print("Running Integration Tests...")
    cmd = ["python", "-m", "pytest", "tests/integration/"]
    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_api_tests(verbose=False):
    """Run API tests."""
    print("Running API Tests...")
    cmd = ["python", "tests/api/test_memory_endpoints.py"]
    if verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def run_all_tests(verbose=False, coverage=False):
    """Run all test suites."""
    print("=" * 60)
    print("MEMORY PIPELINE COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    results = []

    # Run unit tests
    results.append(("Unit Tests", run_unit_tests(verbose, coverage)))

    # Run integration tests
    results.append(("Integration Tests", run_integration_tests(verbose)))

    # Run API tests
    results.append(("API Tests", run_api_tests(verbose)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_suite, success in results:
        status = "[PASSED]" if success else "[FAILED]"
        print(f"{test_suite}: {status}")
        if success:
            passed += 1

    total = len(results)
    print(f"\nOverall: {passed}/{total} test suites passed")

    if passed == total:
        print("\n[SUCCESS] All test suites passed!")
        return True
    else:
        print(f"\n[FAILURE] {total - passed} test suite(s) failed")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Memory Pipeline Test Runner")
    parser.add_argument("--suite", choices=["unit", "integration", "api", "all"],
                       default="all", help="Test suite to run")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true",
                       help="Generate coverage report")

    args = parser.parse_args()

    if args.suite == "unit":
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.suite == "integration":
        success = run_integration_tests(args.verbose)
    elif args.suite == "api":
        success = run_api_tests(args.verbose)
    else:
        success = run_all_tests(args.verbose, args.coverage)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()