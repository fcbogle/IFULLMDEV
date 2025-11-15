# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-14
# Description: EmbeddingRecord
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

@dataclass
class EmbeddingRecord:
    """Embedding vector + original text + searchable metadata."""
    chunk_id: str
    vector: np.ndarray
    text: str
    metadata: Dict[str, Any]