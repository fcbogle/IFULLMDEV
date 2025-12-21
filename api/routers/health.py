# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: health.py
# -----------------------------------------------------------------------------
import logging
from fastapi import APIRouter, Depends, Query

from api.schemas.health import HealthResponse, DeepHealthResponse
from api.dependencies import get_health_service
from services.IFUHealthService import IFUHealthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", message="IFULLMDEV API running")


@router.get("/deep", response_model=DeepHealthResponse)
async def deep_health_check(
    svc: IFUHealthService = Depends(get_health_service),
    run_heavy_openai: bool = Query(False, description="Run heavier OpenAI test"),
) -> DeepHealthResponse:
    logger.info("GET /health/deep called (run_heavy_openai=%s)",run_heavy_openai)
    try:
        result = svc.deep_health(run_heavy_openai=run_heavy_openai)

        logger.info("GET /health/deep completed successfully")
        return result

    except Exception as e:
        logger.error("GET /health/deep failed: %s", e)

