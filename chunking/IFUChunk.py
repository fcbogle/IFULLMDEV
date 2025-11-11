# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-10
# Description: IFUChunk
# -----------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
@dataclass
class IFUChunk:
    """
    Represents a single semantically coherent chunk of text extracted from an AFU document.
    Includes language information and metadata for traceability, filtering, and vector storage.
    """

    # Core identifiers
    chunk_id: str
    doc_id: str
    doc_name: str

    # Text and position information
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int

    # Language information
    lang: str
    lang_confidence: float
    script: Optional[str] = None

    # IFU Specific Metadata
    section_type: Optional[str] = None
    version: Optional[str] = None
    region: Optional[str] = None
    is_primary_language: bool = False
    translation_group_id: Optional[str] = None

    # Embedding metadata
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        # Convert the chunk to a metadata dictionary suitable for Chroma or JSON storage.
        # Embedding is intentionally excluded to avoid large payloads
        r"""
        Converts the chunk into a metadata dictionary suitable for Chroma/JSON storage.
        The embedding is omitted to reduce payload size.
        """
        base_meta = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "lang": self.lang,
            "lang_confidence": self.lang_confidence,
            "script": self.script,
            "version": self.version,
            "region": self.region,
            "section_type": self.section_type,
            "is_primary_language": self.is_primary_language,
            "translation_group_id": self.translation_group_id,
        }
        base_meta.update(self.metadata)
        return base_meta

    def short_preview(self, n: int = 120) -> str:
        """Return a compact text preview for logging/debugging."""
        clean = " ".join(self.text.split())
        preview = (clean[:n] + "...") if len(clean) > n else clean
        return f"[{self.lang.upper()} | p{self.page_start}] {preview}"

