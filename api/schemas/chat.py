# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-28
# Description: api/schemas/chat.py
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from api.schemas.query import QueryHit


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)

    # Retrieval controls (mirror /query)
    n_results: int = Field(5, ge=1, le=20)
    where: Optional[Dict[str, Any]] = None

    # Prompt / model controls
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=4096)

    # Optional: pass conversation history in later iterations
    history: Optional[List[Dict[str, str]]] = None


class ChatSource(BaseModel):
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    text: Optional[str] = None  # include if you want UI to show it


class ChatResponse(BaseModel):
    question: str
    answer: str
    n_results: int
    sources: List[ChatSource] = Field(default_factory=list)

    # helpful for debugging / telemetry
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
