# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-16
# Description: ChromaIFUVectorStore
# -----------------------------------------------------------------------------
from typing import Sequence, Any

import chromadb

from embedding.EmbeddingRecord import EmbeddingRecord
from chunking.IFUChunk import IFUChunk
from config.Config import Config
from utility.logging_utils import get_class_logger
from vectorstore.IFUVectorStore import IFUVectorStore

class ChromaIFUVectorStore(IFUVectorStore):
    """
        Chroma-backed implementation of IFUVectorStore.
        Stores IFU chunk embeddings and metadata for semantic search over IFUs.
    """

    def __init__(self, cfg: Config, collection_name: str = "ifu_chunks"):
        self.cfg = cfg
        self.logger = get_class_logger(self.__class__)

        # Connect to ChromaDB
        self.client = chromadb.HttpClient(
            host=cfg.chroma_endpoint,
            headers={"x-api-key": cfg.chroma_api_key} if cfg.chroma_api_key else None,
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "tenant": cfg.chroma_tenant,
                "database": cfg.chroma_database,
            },
        )

        self.logger.info(
            "ChromaIFUVectorStore initialised (collection=%s)", collection_name
        )

    # Implement Protocol Methods
    def test_connection(self) -> bool:
        """Verify that the collection is reachable."""
        try:
            _ = self.collection.count()
            self.logger.info(
                "Chroma collection '%s' is reachable", self.collection.name
            )
            return True
        except Exception as e:
            self.logger.error("Chroma connection failed: %s", e)
            return False

    def upsert_chunk_embeddings(
            self,
            chunks: Sequence[IFUChunk],
            records: Sequence[EmbeddingRecord],
    ) -> None:
        if len(chunks) != len(records):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(records)} embedding records"
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]

        metadatas: list[dict[str, Any]] = [
            {
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "lang": c.lang,
                "version": getattr(c, "version", None),
                "region": getattr(c, "region", None),
            }
            for c in chunks
        ]

        embeddings: list[list[float]] = []
        for rec in records:
            vec = rec.vector  # check EmbeddingRecord dataclass for attribute name
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            embeddings.append(vec)

        self.logger.info("Upserting %d embeddings to Chroma…", len(ids))

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query_text(
            self,
            query: str,
            n_results: int = 5,
            where: dict[str, Any] | None = None,
    ) -> dict:
        self.logger.debug(
            "Querying Chroma: %r (n_results=%d, where=%s)",
            query,
            n_results,
            where,
        )

        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks for the given doc_id."""
        res = self.collection.get(
            where={"doc_id": doc_id},
            include=["ids"],
        )
        ids = res.get("ids", [])
        if not ids:
            return 0

        self.logger.info("Deleting %d chunks for doc_id=%s", len(ids), doc_id)
        self.collection.delete(ids=ids)
        return len(ids)

