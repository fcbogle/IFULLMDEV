# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-20
# Description: query router
# -----------------------------------------------------------------------------
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_query_service
from api.schemas.query import QueryRequest, QueryResponse, QueryHit
from services.IFUQueryService import IFUQueryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def post_query(
    req: QueryRequest,
    svc: IFUQueryService = Depends(get_query_service),
) -> QueryResponse:
    query_text = (req.query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        raw: Dict[str, Any] = svc.query(
            query_text=query_text,
            n_results=req.n_results,
            where=req.where,
        )
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    # Convert Chroma-style result -> list of QueryHit
    try:
        hits_dicts = svc.to_hits(
            raw,
            include_text=req.include_text,
            include_scores=req.include_scores,
            include_metadata=req.include_metadata,
        )
        hits = [QueryHit(**h) for h in hits_dicts]
    except Exception as e:
        logger.exception("Failed to parse query results: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to parse results: {e}")

    return QueryResponse(
        query=query_text,
        n_results=req.n_results,
        results=hits,
    )
