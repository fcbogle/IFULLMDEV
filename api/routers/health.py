# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: health.py
# -----------------------------------------------------------------------------
import logging

from fastapi import APIRouter, Depends

from health.TestRunner import TestRunner
from config.Config import Config
from api.schemas.health import HealthResponse, DeepHealthResponse, SmokeTestSummary

logger = logging.getLogger("health_logger")

router = APIRouter(
    prefix="/health",
    tags=["health"]
)

@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Lightweight liveness endpoint.
    Returns static metadata confirming the service is running.
    """
    return HealthResponse(status="ok", message="IFULLMDEV API running")

def get_cfg() -> Config:
    return Config.from_env()

@router.get("/deep", response_model=DeepHealthResponse)
async def deep_health_check(
        cfg: Config = Depends(get_cfg),
        run_heavy_openai: bool = False,
) -> DeepHealthResponse:

    """
    Deep health check using the existing TestRunner:

      - BlobHealth   (Blob Storage round-trip)
      - ChromaHealth (Chroma Cloud R/W)
      - EmbeddingHealth (Azure OpenAI embeddings)
      - OpenAIHealth (OpenAI chat)
      - Optional heavy OpenAI test

    Returns a summary + per-test boolean results.
    """
    runner = TestRunner(cfg)
    results = runner.run_all(run_heavy_openai=run_heavy_openai)

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