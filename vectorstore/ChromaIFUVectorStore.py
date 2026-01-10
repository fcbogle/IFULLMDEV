# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-16
# Description: ChromaIFUVectorStore
# -----------------------------------------------------------------------------
import time
from dataclasses import dataclass
from typing import Sequence, Dict, Any, List, Counter

import chromadb
from chromadb import ClientAPI
from chromadb.api.models import Collection

from chunking.IFUChunk import IFUChunk
from config.Config import Config
from embedding.IFUEmbedder import EmbeddingRecord, IFUEmbedder
from utility.logging_utils import get_class_logger
from vectorstore.IFUVectorStore import IFUVectorStore

from settings import ACTIVE_CORPUS_ID


@dataclass(kw_only=True)
class ChromaIFUVectorStore(IFUVectorStore):
    cfg: Config
    embedder: IFUEmbedder
    collection_name: str = "ifu-docs-test"
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
            *,
            corpus_id: str
    ) -> None:
        if len(chunks) != len(records):
            raise ValueError(
                f"chunks ({len(chunks)}) and records ({len(records)}) length mismatch"
            )

        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, (chunk, rec) in enumerate(zip(chunks, records)):
            # Safety check
            if getattr(chunk, "doc_id", doc_id) != doc_id:
                raise ValueError(
                    f"Chunk doc_id '{chunk.doc_id}' does not match upsert doc_id '{doc_id}'"
                )

            vec = rec.vector
            if hasattr(vec, "tolist"):
                vec = vec.tolist()

            # Start from embedding record metadata (this should include page_count)
            meta: Dict[str, Any] = dict(getattr(rec, "metadata", {}) or {})

            # Fill required/standard fields if missing
            meta.setdefault("doc_id", chunk.doc_id)
            meta.setdefault("doc_name", chunk.doc_name)

            # Fill corpus_id for accurate context and queries
            meta.setdefault("corpus_id", corpus_id)


            # Helpful aliases (pick one display field and keep it consistent)
            meta.setdefault("file_name", chunk.doc_name)
            meta.setdefault("source", chunk.doc_name)

            meta.setdefault("chunk_id", chunk.chunk_id)
            meta.setdefault("chunk_index", idx)

            meta.setdefault("section_type", chunk.section_type)
            meta.setdefault("page_start", chunk.page_start)
            meta.setdefault("page_end", chunk.page_end)

            meta.setdefault("lang", chunk.lang)
            meta.setdefault("version", chunk.version)
            meta.setdefault("region", chunk.region)

            meta.setdefault("ingested_at", int(time.time()))

            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            embeddings.append(vec)
            metadatas.append(meta)

        # Optional: log a sample to verify page_count shows up
        if metadatas:
            self.logger.info("Sample metadata sent to Chroma: %r", metadatas[0])

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

            # 2) Build and merge Chroma query parameters
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": query_vectors,
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }

            # Enforce corpus scoping (server-side safety)
            merged_where: Dict[str, Any] = dict(where or {})
            merged_where.setdefault("corpus_id", ACTIVE_CORPUS_ID)

            # Only attach where if it has something in it
            if merged_where:
                self.logger.debug("Applying metadata filter (where=%s)", merged_where)
                query_kwargs["where"] = merged_where

            # 3) Execute Chroma query
            self.logger.debug("Issuing Chroma query against collection '%s'", self.collection_name)
            res = self.collection.query(**query_kwargs)

            # Log summary of results
            ids = res.get("ids") or [[]]
            returned = len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
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

    @staticmethod
    def _flatten_ids(ids: Any) -> List[str]:
        if ids is None:
            return []

        # already flat: ["a","b"]
        if isinstance(ids, list) and (not ids or isinstance(ids[0], str)):
            return [x for x in ids if isinstance(x, str)]

        # nested: [["a","b"], ["c"]]
        if isinstance(ids, list):
            out: List[str] = []
            for item in ids:
                if isinstance(item, list):
                    out.extend([x for x in item if isinstance(x, str)])
                elif isinstance(item, str):
                    out.append(item)
            return out

        return []

    def delete_by_doc_id(self, doc_id: str) -> int:
        self.logger.info("delete_by_doc_id: doc_id='%s' (start)", doc_id)

        # get ids for doc_id
        res: Dict[str, Any] = self.collection.get(
            where={"doc_id": {"$eq": doc_id}},
            include=[],
        )

        ids = self._flatten_ids(res.get("ids"))
        if not ids:
            self.logger.info("delete_by_doc_id: doc_id='%s' -> 0 (nothing to delete)", doc_id)
            return 0

        unique_ids = list(dict.fromkeys(ids))
        self.logger.info(
            "delete_by_doc_id: doc_id='%s' found=%d unique=%d (deleting)",
            doc_id, len(ids), len(unique_ids)
        )

        # delete ids
        self.collection.delete(ids=unique_ids)

        # poll until gone (handles eventual consistency)
        max_wait_s = 3.0
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            check = self.collection.get(where={"doc_id": {"$eq": doc_id}}, include=[])
            remaining = self._flatten_ids(check.get("ids"))
            if not remaining:
                self.logger.info(
                    "delete_by_doc_id: doc_id='%s' deleted=%d (confirmed gone)",
                    doc_id, len(unique_ids)
                )
                return len(unique_ids)
            time.sleep(0.2)

        # if still visible, log it (but still return deleted count)
        self.logger.warning(
            "delete_by_doc_id: doc_id='%s' delete issued, but %d ids still visible after %.1fs (eventual consistency?)",
            doc_id, len(remaining), max_wait_s
        )
        return len(unique_ids)
    def list_documents(self, *, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Return a list of documents present in this collection,
        aggregated from chunk metadata.

        We page through Chroma in batches (max 300 per request) so we
        don't hit the 'Limit value' quota and still see all chunks.

        Each entry:
            {
              "doc_id": "BMK2IFU.pdf",
              "chunk_count": 289,
              "page_count": 164
            }
        """
        if not self.collection:
            return []

        # Chroma Cloud quota: per-request limit <= 300
        page_size = 300

        counts = Counter()
        page_counts: Dict[str, int] = {}
        doc_types: Dict[str, str] = {}
        last_modifieds: Dict[str, str] = {}

        offset = 0
        seen = 0

        while seen < limit:
            # Don't ask for more than remaining "limit" overall
            batch_limit = min(page_size, limit - seen)
            if batch_limit <= 0:
                break

            resp = self.collection.get(
                limit=batch_limit,
                offset=offset,
                include=["metadatas"],
            )

            metadatas = resp.get("metadatas") or []
            if not metadatas:
                break  # no more data

            # Log a sample on first page for debugging
            if offset == 0 and metadatas:
                self.logger.info("list_documents sample md: %r", metadatas[0])

            for md in metadatas:
                if not isinstance(md, dict):
                    continue

                doc_id = md.get("doc_id") or md.get("source_name")
                if not doc_id:
                    continue

                # Increment chunk count
                counts[doc_id] += 1

                # Capture page_count once per doc_id (if present)
                pc_raw = md.get("page_count")
                if pc_raw is not None and doc_id not in page_counts:
                    try:
                        page_counts[doc_id] = int(pc_raw)
                    except (TypeError, ValueError):
                        pass

                # document_type (new)
                dt = md.get("document_type")
                if isinstance(dt, str) and dt.strip() and doc_id not in doc_types:
                    doc_types[doc_id] = dt.strip()

                # last_modified (new) - keep ISO string
                lm = md.get("last_modified")
                if isinstance(lm, str) and lm.strip() and doc_id not in last_modifieds:
                    last_modifieds[doc_id] = lm.strip()

            batch_len = len(metadatas)
            seen += batch_len
            offset += batch_len

            # If we got fewer items than requested, we've exhausted the collection
            if batch_len < batch_limit:
                break

        docs: List[Dict[str, Any]] = []
        for doc_id, chunk_count in counts.items():
            docs.append(
                {
                    "doc_id": doc_id,
                    "chunk_count": chunk_count,
                    "page_count": page_counts.get(doc_id),
                    "document_type": doc_types.get(doc_id),
                    "last_modified": last_modifieds.get(doc_id),
                }
            )

        self.logger.info("list_documents docs summary: %r", docs[:2])
        return docs

