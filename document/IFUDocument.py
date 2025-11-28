# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-28
# Description: IFUDocument
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class IFUDocument:
    path: Path
    doc_id: str
    doc_name: str
    version: Optional[str] = None
    region: Optional[str] = None
    is_primary_language: bool = True
    blob_container: Optional[str] = None
    blob_name: Optional[str] = None
