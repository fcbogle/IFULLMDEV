# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-16
# Description: ChromaIFUVectorStore
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Sequence, Dict, Any, List

import chromadb
from chromadb import ClientAPI
from chromadb.api.models import Collection

from chunking.IFUChunk import IFUChunk
from config.Config import Config
from embedding.IFUEmbedder import EmbeddingRecord, IFUEmbedder
from utility.logging_utils import get_class_logger
from vectorstore.IFUVectorStore import IFUVectorStore


@dataclass
class ChromaIFUVectorStore(IFUVectorStore):
    cfg: Config
    embedder: IFUEmbedder
    collection_name: str = "ifu_chunks"
    logger: Any = None

    def __post_init__(self) -> None:
        self.logger = self.logger or get_class_logger(self.__class__)

        self.logger.info(
            "Initialising Chroma Cloud client "
            f"(tenant={self.cfg.chroma_tenant}, database={self.cfg.chroma_database})"
        )

        self.client: ClientAPI = chromadb.CloudClient(
            tenant=self.cfg.chroma_tenant,
            database=self.cfg.chroma_database,
            api_key=self.cfg.chroma_api_key,
        )

        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        self.logger.info(
            "Chroma collection ready: '%s' (tenant=%s, db=%s)",
            self.collection_name,
            self.cfg.chroma_tenant,
            self.cfg.chroma_database,
        )

    def test_connection(self) -> bool:
        """
        Simple health check: can we talk to Chroma and our collection?
        """
        try:
            # count() is cheap and exercises the connection + auth
            _ = self.collection.count()
            return True
        except Exception as e:
            self.logger.error("Chroma connection failed: %s", e)
            return False

    def upsert_chunk_embeddings(
            self,
            doc_id: str,
            chunks: Sequence[IFUChunk],
            records: Sequence[EmbeddingRecord],
    ) -> None:
        if len(chunks) != len(records):
            raise ValueError(
                f"chunks ({len(chunks)}) and records ({len(records)}) length mismatch"
            )

        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk, rec in zip(chunks, records):
            # Optional safety check – catches accidental mismatches early
            if getattr(chunk, "doc_id", doc_id) != doc_id:
                raise ValueError(
                    f"Chunk doc_id '{chunk.doc_id}' does not match upsert doc_id '{doc_id}'"
                )

            vec = rec.vector
            if hasattr(vec, "tolist"):
                vec = vec.tolist()

            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            embeddings.append(vec)
            metadatas.append({
                "doc_id": doc_id,
                "doc_name": chunk.doc_name,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "lang": chunk.lang,
                "version": getattr(chunk, "version", None),
                "region": getattr(chunk, "region", None),
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self.logger.info(
            "Upserted %d chunks for doc_id '%s' into Chroma collection '%s'",
            len(chunks),
            doc_id,
            self.collection_name,
        )

    def query_text(
            self,
            query_text: str,
            n_results: int = 5,
            where: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        # High-level log for pipeline visibility
        self.logger.info(
            "Querying Chroma collection '%s' with query=%r (n_results=%d, where=%s)",
            self.collection_name,
            query_text,
            n_results,
            where,
        )

        try:
            # 1) Embed query text
            self.logger.debug("Embedding query text using embedder=%r", self.embedder)
            query_vectors = self.embedder.embed_texts([query_text])  # List[List[float]]
            self.logger.debug(
                "Query embedding generated: vector_length=%d",
                len(query_vectors[0]) if query_vectors else -1,
            )

            # 2) Build Chroma query parameters
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": query_vectors,
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                self.logger.debug("Applying metadata filter (where=%s)", where)
                query_kwargs["where"] = where

            self.logger.debug(
                "Final Chroma query kwargs: keys=%s",
                list(query_kwargs.keys())
            )

            # 3) Execute Chroma query
            self.logger.debug("Issuing Chroma query against collection '%s'", self.collection_name)
            res = self.collection.query(**query_kwargs)

            # Log summary of results
            returned = len(res.get("ids", []))
            self.logger.info(
                "Chroma search complete: returned %d results (requested %d)",
                returned,
                n_results,
            )
            self.logger.debug(
                "Chroma result keys: %s",
                list(res.keys())
            )

            return res

        except Exception as e:
            self.logger.error(
                "Error during query_text execution: %s",
                str(e),
                exc_info=True
            )
            raise

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunks in this collection that belong to the given doc_id.
        Returns the number of chunks actually deleted.
        """
        self.logger.info(
            "Deleting all chunks for doc_id '%s' from collection '%s'",
            doc_id,
            self.collection_name,
        )

        # 1) Fetch all IDs for this doc_id (no embeddings, no NumResults quota)
        try:
            res: Dict[str, Any] = self.collection.get(
                where={"doc_id": {"$eq": doc_id}},
                include=[],  # we only care about ids
            )
        except Exception as e:
            self.logger.error(
                "Failed to get chunks for doc_id '%s' from collection '%s': %s",
                doc_id,
                self.collection_name,
                e,
            )
            raise

        ids: List[str] = res.get("ids", []) or []
        if not ids:
            self.logger.info(
                "No chunks found for doc_id '%s' in collection '%s'",
                doc_id,
                self.collection_name,
            )
            return 0

        # 2) Ensure IDs are unique (Chroma requires this)
        # preserves order while de-duplicating
        unique_ids = list(dict.fromkeys(ids))
        if len(unique_ids) != len(ids):
            self.logger.info(
                "Found %d duplicate IDs for doc_id '%s'; deduping to %d IDs",
                len(ids) - len(unique_ids),
                doc_id,
                len(unique_ids),
            )

        # 3) Delete those specific IDs
        try:
            self.collection.delete(ids=unique_ids)
        except Exception as e:
            self.logger.error(
                "Failed to delete %d chunks for doc_id '%s' from collection '%s': %s",
                len(unique_ids),
                doc_id,
                self.collection_name,
                e,
            )
            raise

        deleted_count = len(unique_ids)
        self.logger.info(
            "Deleted %d chunks for doc_id '%s' from collection '%s'",
            deleted_count,
            doc_id,
            self.collection_name,
        )
        return deleted_count
