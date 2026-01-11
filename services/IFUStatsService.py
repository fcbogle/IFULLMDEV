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
    def __init__(self, *, vector_store, document_loader, collection_name: str):
        self.vector_store = vector_store
        self.document_loader = document_loader
        self.collection_name = collection_name

    def _get_metadatas_for(self, *, blob_container: str, corpus_id: str) -> List[Dict[str, Any]]:
        where = {"$and": [{"corpus_id": corpus_id}, {"container": blob_container}]}

        try:
            res = self.vector_store.collection.get(where=where, include=["metadatas"])
            return res.get("metadatas") or []
        except (TypeError, ValueError):
            # Fallback: fetch all metadatas and filter locally
            res = self.vector_store.collection.get(include=["metadatas"])
            metadatas = res.get("metadatas") or []
            return [
                m for m in metadatas
                if (m.get("corpus_id") == corpus_id and m.get("container") == blob_container)
            ]

    def get_stats(self, *, blob_container: str, corpus_id: Optional[str] = None) -> IFUStatsResponse:
        corpus = corpus_id or ACTIVE_CORPUS_ID

        # --- 1) Chroma stats (corpus-aware) ---
        metadatas = self._get_metadatas_for(blob_container=blob_container, corpus_id=corpus)
        total_chunks = len(metadatas)

        # Group by doc_id
        doc_chunk_counts: Dict[str, int] = defaultdict(int)
        doc_page_count: Dict[str, Optional[int]] = {}
        doc_last_modified: Dict[str, Optional[datetime]] = {}
        doc_type: Dict[str, Optional[str]] = {}
        doc_name: Dict[str, Optional[str]] = {}
        lang_counts_by_doc = defaultdict(Counter)

        for m in metadatas:
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

        # --- 2) Blob stats (from Azure) ---
        blob_rows = self.document_loader.get_blob_details(container=blob_container)

        blobs: List[BlobStats] = []
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
            collection_name=self.collection_name,
            corpus_id=corpus,
            total_chunks=total_chunks,
            total_documents=len(documents),
            documents=documents,
            blob_container=blob_container,
            total_blobs=total_blobs,
            blobs=blobs,
        )

    def get_indexed_doc_samples(
            self,
            *,
            blob_container: str,
            corpus_id: Optional[str] = None,
            lang: Optional[str] = None,
            max_docs: int = 10,
            chunks_per_doc: int = 3,
    ) -> List[Dict[str, Any]]:
        corpus = corpus_id or ACTIVE_CORPUS_ID
        lang_norm = (lang or "").strip().lower() or None

        stats = self.get_stats(blob_container=blob_container, corpus_id=corpus)
        docs = (stats.documents or [])[:max_docs]

        results: List[Dict[str, Any]] = []
        for d in docs:
            samples = self.vector_store.get_doc_sample_chunks(
                doc_id=d.doc_id,
                corpus_id=corpus,
                container=blob_container,
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
