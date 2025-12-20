# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-20
# Description: IFUHealthService.py
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from typing import AnyStr, Any, Dict

from health.TestRunner import TestRunner
from api.schemas.health import DeepHealthResponse, SmokeTestSummary


@dataclass
class IFUHealthService:
    """
    Wraps TestRunner class which operates smoke tests
    on cloud infrastructure components.
    Returns DeepHealthResponse for API layer
    """

    test_runner: TestRunner
    def deep_health(self, run_heavy_openai: bool = False) -> DeepHealthResponse:

        results = self.test_runner.run_all(run_heavy_openai=run_heavy_openai)

        total = len(results)
        passed = sum(1 for ok in results.values() if ok)
        failed = total - passed

        overall_status = "ok" if failed == 0 else "error"

        summary = SmokeTestSummary(
            total=total,
            passed=passed,
            failed=failed,
        )

        return DeepHealthResponse(
            status=overall_status,
            results=results,
            summary=summary,
        )




