# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-15
# Description: IFUVectorStore
# -----------------------------------------------------------------------------
from langchain_core.vectorstores import VectorStore


class IFUVectorStore:
    def test_connection(self): ...

    def upsert_chunk_embeddings(self, chunks, records): ...

    def query_text(self, query, n_results=5, where=None): ...

    def delete_by_doc_id(self, doc_id): ...
