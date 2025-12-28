# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-27
# Description: chat.py
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    n_results: int = Field(5, ge=1, le=20)
    where: Optional[Dict[str, Any]] = None

    # optional multi-turn
    conversation: List[ChatMessage] = Field(default_factory=list)

    # optional tuning
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(700, ge=1, le=4000)


class ChatSource(BaseModel):
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[ChatSource] = Field(default_factory=list)
    model: Optional[str] = None
    usage: Optional[Any] = None