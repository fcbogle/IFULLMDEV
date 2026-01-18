# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-21
# Description: IFUStatsService.py
# -----------------------------------------------------------------------------

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from api.schemas.ifu_stats import IFUStatsResponse, DocumentStats, BlobStats
from settings import ACTIVE_CORPUS_ID


class IFUStatsService:
    """
    Stats service over the vector index, with optional storage-side (blob) enrichment.

    IMPORTANT DESIGN:
    - The vector collection is selected by `vector_collection` (namespace), NOT by metadata filtering.
    - `blob_container` is only used for optional storage-side stats/delta.
    """

    def __init__(self, *, vector_store, document_loader=None):
        self.vector_store = vector_store
        self.document_loader = document_loader

    # ----------------------------
    # Vector helpers
    # ----------------------------
    def _get_collection(self, vector_collection: str):
        """
        Return the underlying Chroma collection for the given name.

        You MUST implement ONE of these on your vector_store:
          - vector_store.get_collection(name)
          - vector_store.client.get_collection(name)
          - vector_store.collection(name)
        """
        # Try a few common patterns without forcing one implementation
        if hasattr(self.vector_store, "get_collection"):
            return self.vector_store.get_collection(vector_collection)

        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "get_collection"):
            return self.vector_store.client.get_collection(name=vector_collection)

        if callable(getattr(self.vector_store, "collection", None)):
            return self.vector_store.collection(vector_collection)

        raise AttributeError(
            "vector_store must expose get_collection(name) or client.get_collection(name=...) "
            "or collection(name)."
        )

    def _get_metadatas_for(self, *, vector_collection: str, corpus_id: str) -> List[Dict[str, Any]]:
        """
        Fetch chunk-level metadatas for a given corpus within a specific vector collection.
        """
        col = self._get_collection(vector_collection)

        where = {"corpus_id": corpus_id}

        try:
            res = col.get(where=where, include=["metadatas"])
            return res.get("metadatas") or []
        except (TypeError, ValueError):
            # Fallback: fetch all metadatas and filter locally
            res = col.get(include=["metadatas"])
            metadatas = res.get("metadatas") or []
            return [m for m in metadatas if isinstance(m, dict) and m.get("corpus_id") == corpus_id]

    # ----------------------------
    # Public API
    # ----------------------------
    def get_stats(
        self,
        *,
        vector_collection: str,
        corpus_id: Optional[str] = None,
        blob_container: Optional[str] = None,  # optional enrichment only
    ) -> IFUStatsResponse:
        corpus = (corpus_id or ACTIVE_CORPUS_ID).strip()
        vc = (vector_collection or "").strip()
        if not vc:
            raise ValueError("vector_collection must not be empty")

        # --- 1) Vector stats (corpus-aware within a specific collection) ---
        metadatas = self._get_metadatas_for(vector_collection=vc, corpus_id=corpus)
        total_chunks = len(metadatas)

        # Group by doc_id
        doc_chunk_counts: Dict[str, int] = defaultdict(int)
        doc_page_count: Dict[str, Optional[int]] = {}
        doc_last_modified: Dict[str, Optional[datetime]] = {}
        doc_type: Dict[str, Optional[str]] = {}
        doc_name: Dict[str, Optional[str]] = {}
        lang_counts_by_doc = defaultdict(Counter)

        for m in metadatas:
            if not isinstance(m, dict):
                continue

            doc_id = m.get("doc_id") or "UNKNOWN_DOC_ID"
            doc_chunk_counts[doc_id] += 1

            # prefer doc_name/file_name/source for display
            dn = m.get("doc_name") or m.get("file_name") or m.get("source") or doc_id
            doc_name.setdefault(doc_id, dn)

            # page_count (doc-level metadata you already attach)
            if doc_id not in doc_page_count:
                pc = m.get("page_count")
                doc_page_count[doc_id] = int(pc) if isinstance(pc, (int, float)) else None

            # document_type
            if doc_id not in doc_type:
                dt = m.get("document_type")
                doc_type[doc_id] = str(dt) if dt is not None else None

            # last_modified (stored as ISO string in your metadata)
            if doc_id not in doc_last_modified:
                lm = m.get("last_modified")
                if isinstance(lm, str) and lm.strip():
                    try:
                        doc_last_modified[doc_id] = datetime.fromisoformat(lm.replace("Z", "+00:00"))
                    except Exception:
                        doc_last_modified[doc_id] = None
                else:
                    doc_last_modified[doc_id] = None

            # languages per doc
            lang = m.get("lang") or "und"
            lang_counts_by_doc[doc_id][lang] += 1

        documents: List[DocumentStats] = []
        for doc_id, cnt in doc_chunk_counts.items():
            lc = dict(lang_counts_by_doc[doc_id])
            primary_lang = max(lc, key=lc.get) if lc else None

            documents.append(
                DocumentStats(
                    doc_id=doc_id,
                    doc_name=doc_name.get(doc_id),
                    chunk_count=cnt,
                    page_count=doc_page_count.get(doc_id),
                    last_modified=doc_last_modified.get(doc_id),
                    document_type=doc_type.get(doc_id),
                    primary_lang=primary_lang,
                    lang_counts=lc or None,
                )
            )

        documents.sort(key=lambda d: d.chunk_count, reverse=True)

        # --- 2) Optional blob stats (only if configured) ---
        blobs: List[BlobStats] = []
        total_blobs = 0

        if self.document_loader is not None and blob_container:
            blob_rows = self.document_loader.get_blob_details(container=blob_container)
            for b in blob_rows:
                blobs.append(
                    BlobStats(
                        blob_name=b.get("blob_name"),
                        size=b.get("size"),
                        content_type=b.get("content_type"),
                        last_modified=b.get("last_modified"),
                    )
                )
            total_blobs = len(blobs)

        return IFUStatsResponse(
            collection_name=vc,
            corpus_id=corpus,
            total_chunks=total_chunks,
            total_documents=len(documents),
            documents=documents,
            blob_container=blob_container,  # may be None
            total_blobs=total_blobs,
            blobs=blobs,
        )

    def get_indexed_doc_samples(
        self,
        *,
        vector_collection: str,
        corpus_id: Optional[str] = None,
        lang: Optional[str] = None,
        max_docs: int = 10,
        chunks_per_doc: int = 3,
    ) -> List[Dict[str, Any]]:
        corpus = (corpus_id or ACTIVE_CORPUS_ID).strip()
        vc = (vector_collection or "").strip()
        if not vc:
            raise ValueError("vector_collection must not be empty")

        lang_norm = (lang or "").strip().lower() or None

        stats = self.get_stats(vector_collection=vc, corpus_id=corpus)
        docs = (stats.documents or [])[:max_docs]

        results: List[Dict[str, Any]] = []
        for d in docs:
            # IMPORTANT: ensure your vector_store implementation samples from the correct collection
            samples = self.vector_store.get_doc_sample_chunks(
                collection=vc,         # ✅ new: collection namespace
                doc_id=d.doc_id,
                corpus_id=corpus,
                lang=lang_norm,
                max_chunks=chunks_per_doc,
            )

            results.append(
                {
                    "doc_id": d.doc_id,
                    "doc_name": getattr(d, "doc_name", None) or d.doc_id,
                    "chunk_count": d.chunk_count,
                    "page_count": d.page_count,
                    "primary_lang": getattr(d, "primary_lang", None),
                    "requested_lang": lang_norm,
                    "sample_chunk_count": len(samples),
                    "sample_chunks": samples,
                }
            )

        return results

    def get_storage_index_delta(self, *, vector_collection: str, blob_container: str, corpus_id: str) -> dict:
        """
        Optional: storage vs index delta. Requires document_loader.
        """
        if self.document_loader is None:
            raise RuntimeError("document_loader is not configured; cannot compute storage/index delta")

        vc = (vector_collection or "").strip()
        if not vc:
            raise ValueError("vector_collection must not be empty")

        # --- storage side ---
        stats = self.get_stats(vector_collection=vc, blob_container=blob_container, corpus_id=corpus_id)
        storage_blob_names = {b.blob_name for b in (stats.blobs or []) if getattr(b, "blob_name", None)}

        # --- indexed side (vector) ---
        metadatas = self._get_metadatas_for(vector_collection=vc, corpus_id=corpus_id)

        indexed_blob_names = {
            (m.get("blob_name") or m.get("doc_id"))
            for m in (metadatas or [])
            if isinstance(m, dict) and (m.get("blob_name") or m.get("doc_id"))
        }

        blobs_not_indexed = sorted(storage_blob_names - indexed_blob_names)
        indexed_not_in_storage = sorted(indexed_blob_names - storage_blob_names)
        indexed_in_storage = sorted(indexed_blob_names & storage_blob_names)

        return {
            "vector_collection": vc,
            "blob_container": blob_container,
            "corpus_id": corpus_id,
            "total_blobs": len(storage_blob_names),
            "total_documents": len(indexed_blob_names),
            "blobs_not_indexed_count": len(blobs_not_indexed),
            "indexed_not_in_storage_count": len(indexed_not_in_storage),
            "indexed_in_storage_count": len(indexed_in_storage),
            "blobs_not_indexed": blobs_not_indexed,
            "indexed_not_in_storage": indexed_not_in_storage,
            "indexed_in_storage": indexed_in_storage[:50],
        }

