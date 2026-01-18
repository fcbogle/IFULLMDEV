# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-16
# Updated: 2026-01-18
# Description: ChromaIFUVectorStore
# -----------------------------------------------------------------------------
import time
from dataclasses import dataclass
from typing import Sequence, Dict, Any, List, Counter, Optional

import chromadb
from chromadb import ClientAPI
from chromadb.api.models import Collection

from chunking.IFUChunk import IFUChunk
from config.Config import Config
from embedding.IFUEmbedder import EmbeddingRecord, IFUEmbedder
from utility.logging_utils import get_class_logger
from vectorstore.IFUVectorStore import IFUVectorStore

from settings import ACTIVE_CORPUS_ID, VECTOR_COLLECTION_DEFAULT


@dataclass(kw_only=True)
class ChromaIFUVectorStore(IFUVectorStore):
    """
    Chroma Cloud-backed vector store.

    Backwards compatible:
    - `self.collection` remains the default collection created at init.
    - Existing methods continue to operate against the default collection.

    New capabilities:
    - `get_collection(name)` returns collection by name (cached).
    - `query(collection=..., ...)` used by IFUQueryService to support per-request collection selection.
    """
    cfg: Config
    embedder: IFUEmbedder
    collection_name: str = VECTOR_COLLECTION_DEFAULT
    logger: Any = None

    # Lazy cache of collections by name (supports multi-collection usage)
    _collections_cache: Optional[Dict[str, Collection]] = None

    def __post_init__(self) -> None:
        self.logger = self.logger or get_class_logger(self.__class__)

        self.collection_name = (self.collection_name or VECTOR_COLLECTION_DEFAULT).strip()
        if not self.collection_name:
            raise RuntimeError("ChromaIFUVectorStore.collection_name resolved to empty value")

        self.logger.info(
            "Initialising Chroma Cloud client (tenant=%s, database=%s)",
            self.cfg.chroma_tenant,
            self.cfg.chroma_database,
        )

        self.client: ClientAPI = chromadb.CloudClient(
            tenant=self.cfg.chroma_tenant,
            database=self.cfg.chroma_database,
            api_key=self.cfg.chroma_api_key,
        )

        # Default collection (backwards compatibility)
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        # Seed cache with default collection
        self._collections_cache = {self.collection_name: self.collection}

        self.logger.info(
            "Chroma collection ready: '%s' (tenant=%s, db=%s)",
            self.collection_name,
            self.cfg.chroma_tenant,
            self.cfg.chroma_database,
        )

    # -------------------------------------------------------------------------
    # New: multi-collection support
    # -------------------------------------------------------------------------
    def get_collection(self, name: Optional[str] = None) -> Collection:
        """
        Resolve a Chroma collection by name (get-or-create).
        If name is None/empty, returns the default collection created at init.
        """
        if not name or not str(name).strip():
            return self.collection

        col_name = str(name).strip()

        if self._collections_cache is None:
            self._collections_cache = {}

        cached = self._collections_cache.get(col_name)
        if cached is not None:
            return cached

        col = self.client.get_or_create_collection(name=col_name)
        self._collections_cache[col_name] = col
        return col

    def query(
            self,
            *,
            collection: Optional[str] = None,
            query_text: str,
            n_results: int = 5,
            where: Dict[str, Any] | None = None,
            include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query a Chroma collection using *local embeddings* (embedder),
        ensuring embedding dimension matches what you ingested.

        Returns raw Chroma query output: ids, documents, metadatas, distances
        """
        q = (query_text or "").strip()
        if not q:
            raise ValueError("query_text must not be empty")

        col = self.get_collection(collection)
        inc = include or ["documents", "metadatas", "distances"]

        # Embed locally to guarantee consistent dimensionality
        query_vectors = self.embedder.embed_texts([q])

        return col.query(
            query_embeddings=query_vectors,
            n_results=int(n_results),
            where=where,
            include=inc,
        )

    # -------------------------------------------------------------------------
    # Existing functionality
    # -------------------------------------------------------------------------
    def test_connection(self) -> bool:
        """
        Simple health check: can we talk to Chroma and our collection?
        """
        try:
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
            corpus_id: str | None = None,
            collection: Optional[str] = None,
    ) -> None:
        """
        Upsert embeddings into Chroma.

        Backwards compatible: if `collection` is None, uses default `self.collection`.
        """
        if len(chunks) != len(records):
            raise ValueError(
                f"chunks ({len(chunks)}) and records ({len(records)}) length mismatch"
            )

        col = self.get_collection(collection)

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
            meta.setdefault("corpus_id", corpus_id or ACTIVE_CORPUS_ID)

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

        if metadatas:
            self.logger.info("Sample metadata sent to Chroma: %r", metadatas[0])

        col.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self.logger.info(
            "Upserted %d chunks for doc_id '%s' into Chroma collection '%s'",
            len(chunks),
            doc_id,
            (collection or self.collection_name),
        )

    def query_text(
            self,
            query_text: str,
            n_results: int = 5,
            where: Dict[str, Any] | None = None,
            *,
            collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Backwards compatible wrapper around the new query() API.
        Uses text embeddings (existing approach) and supports collection selection.

        NOTE: Your IFUQueryService now uses ChromaIFUVectorStore.query(...),
        but other parts of the code may still call query_text().
        """
        self.logger.info(
            "Querying Chroma collection '%s' with query=%r (n_results=%d, where=%s)",
            (collection or self.collection_name),
            query_text,
            n_results,
            where,
        )

        try:
            self.logger.debug("Embedding query text using embedder=%r", self.embedder)
            query_vectors = self.embedder.embed_texts([query_text])

            query_kwargs: Dict[str, Any] = {
                "query_embeddings": query_vectors,
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }

            def _as_chroma_where(filters: Dict[str, Any]) -> Dict[str, Any]:
                if not filters:
                    return {}
                if len(filters) == 1:
                    return filters
                return {"$and": [{k: v} for k, v in filters.items()]}

            merged_where = dict(where or {})
            merged_where.setdefault("corpus_id", ACTIVE_CORPUS_ID)

            chroma_where = _as_chroma_where(merged_where)
            if chroma_where:
                self.logger.debug("Applying metadata filter (where=%s)", chroma_where)
                query_kwargs["where"] = chroma_where

            col = self.get_collection(collection)

            self.logger.debug("Issuing Chroma query against collection '%s'", (collection or self.collection_name))
            res = col.query(**query_kwargs)

            ids = res.get("ids") or [[]]
            returned = len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)
            self.logger.info(
                "Chroma search complete: returned %d results (requested %d)",
                returned,
                n_results,
            )
            self.logger.debug("Chroma result keys: %s", list(res.keys()))

            return res

        except Exception as e:
            self.logger.error("Error during query_text execution: %s", str(e), exc_info=True)
            raise

    @staticmethod
    def _flatten_ids(ids: Any) -> List[str]:
        if ids is None:
            return []

        if isinstance(ids, list) and (not ids or isinstance(ids[0], str)):
            return [x for x in ids if isinstance(x, str)]

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

        self.collection.delete(ids=unique_ids)

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
        """
        if not self.collection:
            return []

        page_size = 300

        counts = Counter()
        page_counts: Dict[str, int] = {}
        doc_types: Dict[str, str] = {}
        last_modifieds: Dict[str, str] = {}

        offset = 0
        seen = 0

        while seen < limit:
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
                break

            if offset == 0 and metadatas:
                self.logger.info("list_documents sample md: %r", metadatas[0])

            for md in metadatas:
                if not isinstance(md, dict):
                    continue

                doc_id = md.get("doc_id") or md.get("source_name")
                if not doc_id:
                    continue

                counts[doc_id] += 1

                pc_raw = md.get("page_count")
                if pc_raw is not None and doc_id not in page_counts:
                    try:
                        page_counts[doc_id] = int(pc_raw)
                    except (TypeError, ValueError):
                        pass

                dt = md.get("document_type")
                if isinstance(dt, str) and dt.strip() and doc_id not in doc_types:
                    doc_types[doc_id] = dt.strip()

                lm = md.get("last_modified")
                if isinstance(lm, str) and lm.strip() and doc_id not in last_modifieds:
                    last_modifieds[doc_id] = lm.strip()

            batch_len = len(metadatas)
            seen += batch_len
            offset += batch_len

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

    def _as_chroma_where(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        if not filters:
            return {}
        if len(filters) == 1:
            return filters
        return {"$and": [{k: v} for k, v in filters.items()]}

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
        """
        Fetch a small, deterministic sample of chunks for a given doc_id.

        Filters:
          - corpus_id (defaults to ACTIVE_CORPUS_ID)
          - optional container
          - optional lang (e.g., "en", "pt", "de")
        """
        corpus = corpus_id or ACTIVE_CORPUS_ID
        lang_norm = (lang or "").strip().lower() or None

        filters: Dict[str, Any] = {"doc_id": doc_id, "corpus_id": corpus}
        if container:
            filters["container"] = container
        if lang_norm:
            filters["lang"] = lang_norm

        where = self._as_chroma_where(filters)

        col = self.get_collection(collection)
        res = col.get(where=where, include=["documents", "metadatas"])
        docs: List[str] = res.get("documents") or []
        metas: List[Dict[str, Any]] = res.get("metadatas") or []

        pairs = list(zip(docs, metas))

        def _sort_key(p: Any) -> int:
            md = p[1] or {}
            try:
                return int(md.get("chunk_index", 10 ** 9))
            except Exception:
                return 10 ** 9

        pairs.sort(key=_sort_key)

        out: List[Dict[str, Any]] = []
        for text, md in pairs[:max_chunks]:
            md = md or {}
            t = (text or "").strip()
            if max_chars_per_chunk and len(t) > max_chars_per_chunk:
                t = t[:max_chars_per_chunk] + "…"

            out.append(
                {
                    "doc_id": md.get("doc_id"),
                    "doc_name": md.get("doc_name") or md.get("source") or md.get("file_name"),
                    "chunk_id": md.get("chunk_id"),
                    "chunk_index": md.get("chunk_index"),
                    "page_start": md.get("page_start"),
                    "page_end": md.get("page_end"),
                    "lang": md.get("lang"),
                    "section_type": md.get("section_type"),
                    "text": t,
                }
            )

        return out
