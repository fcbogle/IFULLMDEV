# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-20
# Description: query.py
# -----------------------------------------------------------------------------
from typing import Optional, Any, Dict, List

from pydantic import Field, BaseModel

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    n_results: int = Field(5, ge=1, le=20)
    where: Optional[Dict[str, Any]] = None
    include_text: bool = True
    include_scores: bool = True
    include_metadata: bool = True

class QueryHit(BaseModel):
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    n_results: int
    results: List[QueryHit]