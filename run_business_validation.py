#!/usr/bin/env python3
"""
Business Validation Runner
Execute comprehensive business validation tests and generate report
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from validation.test_harness import BusinessValidationHarness


async def main():
    """Run business validation and generate report."""
    print("MEMORY PIPELINE BUSINESS VALIDATION")
    print("Testing real-world scenarios for production readiness")
    print("=" * 60)

    try:
        # Initialize and run validation
        harness = BusinessValidationHarness()
        results = await harness.run_comprehensive_validation()

        print("\n✅ Business validation completed successfully!")
        print("📊 Check validation_business_report.json for detailed results")

        return results

    except Exception as e:
        print(f"\n❌ Business validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())