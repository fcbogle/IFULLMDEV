# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: health.py
# -----------------------------------------------------------------------------
from typing import Dict

from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    message: str

class SmokeTestSummary(BaseModel):
    total: int
    passed: int
    failed: int

class DeepHealthResponse(BaseModel):
    status: str
    results: Dict[str, bool]
    summary: SmokeTestSummary