# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-15
# Updated: 2026-01-18
# Description: IFUVectorStore (Protocol)
# -----------------------------------------------------------------------------
from typing import Protocol, Sequence, Dict, Any, Optional, runtime_checkable, List

from chunking.IFUChunk import IFUChunk
from embedding.EmbeddingRecord import EmbeddingRecord


@runtime_checkable
class IFUVectorStore(Protocol):
    def test_connection(self) -> bool:
        ...

    def list_documents(self, *, limit: int = 1000) -> List[Dict[str, Any]]:
        ...

    def upsert_chunk_embeddings(
        self,
        doc_id: str,
        chunks: Sequence[IFUChunk],
        records: Sequence[EmbeddingRecord],
        *,
        corpus_id: str | None = None,
        collection: Optional[str] = None,
    ) -> None:
        ...

    # NEW: unified query API used by IFUQueryService
    def query(
        self,
        *,
        collection: Optional[str] = None,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ...

    # Backwards compatible: older API, still supported
    def query_text(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        *,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...

    def get_doc_sample_chunks(
        self,
        *,
        doc_id: str,
        corpus_id: Optional[str] = None,
        container: Optional[str] = None,
        lang: Optional[str] = None,
        max_chunks: int = 5,
        max_chars_per_chunk: int = 2000,
        collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def delete_by_doc_id(self, doc_id: str) -> int:
        ...
