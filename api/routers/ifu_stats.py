# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: ifu_stats.py
# -----------------------------------------------------------------------------
import logging

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_stats_service
from api.schemas.ifu_stats import IFUStatsResponse
from services.IFUStatsService import IFUStatsService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/stats",
    tags=["stats"]
)

@router.get("", response_model=IFUStatsResponse)
def get_ifu_stats(
    svc: IFUStatsService = Depends(get_stats_service),
    blob_container: str = Query("ifu-docs-test"),
) -> IFUStatsResponse:
    logger.info("Getting stats for blob_container='%s'", blob_container)
    return svc.get_stats(blob_container=blob_container)
