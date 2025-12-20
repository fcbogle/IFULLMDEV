# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-20
# Description: IFUStatsService.py
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Dict, List

from api.schemas.ifu_stats import IFUStatsResponse, DocumentStats, BlobStats


@dataclass
class IFUStatsService:
    loader: Any

    def get_stats(self, blob_container: str) -> IFUStatsResponse:
        stats: Dict[str, Any] = self.loader.get_stats(blob_container=blob_container)

        docs_raw: List[Dict[str, Any]] = stats.get("documents", []) or []
        documents = [
            DocumentStats(
                doc_id=d.get("doc_id"),
                chunk_count=d.get("chunk_count", 0),
                page_count=d.get("page_count"),
                last_modified=d.get("last_modified"),
                document_type=d.get("document_type"),
            )
            for d in docs_raw
            if isinstance(d, dict)
        ]

        blob_entries = self.loader.get_blob_details(blob_container) or []
        blobs = [
            BlobStats(
                blob_name=b.get("blob_name") or b.get("name"),
                size=b.get("size"),
                content_type=b.get("content_type"),
                last_modified=b.get("last_modified"),
            )
            for b in blob_entries
            if isinstance(b, dict)
        ]

        return IFUStatsResponse(
            collection_name=stats.get("collection_name"),
            total_chunks=int(stats.get("total_chunks", 0) or 0),
            total_documents=len(documents),
            documents=documents,
            blob_container=blob_container,
            total_blobs=len(blobs),
            blobs=blobs,
        )

