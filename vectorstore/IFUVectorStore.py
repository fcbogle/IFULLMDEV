# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-15
# Description: IFUVectorStore
# -----------------------------------------------------------------------------

from typing import Protocol, Sequence, Dict, Any, runtime_checkable

from chunking.IFUChunk import IFUChunk
from embedding.EmbeddingRecord import EmbeddingRecord


@runtime_checkable
class IFUVectorStore(Protocol):
    def test_connection(self) -> bool:
        ...

    def upsert_chunk_embeddings(
            self,
            doc_id: str,
            chunks: Sequence[IFUChunk],
            records: Sequence[EmbeddingRecord],
    ) -> None:
        ...

    def query_text(
            self,
            query_text: str,
            n_results: int = 5,
            where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        ...

    def delete_by_doc_id(self, doc_id: str) -> int:
        ...
