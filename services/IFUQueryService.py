# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-20
# Description: IFUQueryService
# -----------------------------------------------------------------------------
from typing import Any, Dict, List, Optional


class IFUQueryService:
    def __init__(self, *, store, collection_name: str):
        self.store = store
        self.collection_name = (collection_name or "").strip()

    def query(
        self,
        *,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query vector store.

        `collection` selects the vector collection at runtime.
        If not provided, defaults to `self.collection_name` for backward compatibility.
        """
        effective_collection = (collection or self.collection_name or "").strip()
        if not effective_collection:
            raise ValueError("collection resolved to empty value")

        return self.store.query(
            collection=effective_collection,
            query_text=query_text,
            n_results=int(n_results),
            where=where,
        )

    @staticmethod
    def to_hits(
        raw: Dict[str, Any],
        include_text: bool = True,
        include_scores: bool = True,
        include_metadata: bool = True,
        include_distances: bool = True,
    ) -> List[Dict[str, Any]]:
        ids = (raw.get("ids") or [[]])
        docs = (raw.get("documents") or [[]])
        metas = (raw.get("metadatas") or [[]])
        dists = (raw.get("distances") or [[]])

        ids0 = ids[0] if ids and isinstance(ids[0], list) else []
        docs0 = docs[0] if docs and isinstance(docs[0], list) else []
        metas0 = metas[0] if metas and isinstance(metas[0], list) else []
        dists0 = dists[0] if dists and isinstance(dists[0], list) else []

        hits: List[Dict[str, Any]] = []

        def _pick_page(md: Dict[str, Any]) -> Optional[int]:
            for k in ("page", "page_start", "page_end"):
                v = md.get(k)
                if isinstance(v, int):
                    return v
                if isinstance(v, str) and v.isdigit():
                    return int(v)
            return None

        n = max(len(ids0), len(docs0), len(metas0), len(dists0))
        for i in range(n):
            md = metas0[i] if i < len(metas0) else None
            text = docs0[i] if i < len(docs0) else None
            chunk_id = ids0[i] if i < len(ids0) else None
            dist = dists0[i] if i < len(dists0) else None

            doc_id = md.get("doc_id") if isinstance(md, dict) else None
            page = _pick_page(md) if isinstance(md, dict) else None

            hit: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page": page,
            }

            if include_text:
                hit["text"] = text

            if include_metadata:
                hit["metadata"] = md if isinstance(md, dict) else None

            if include_distances:
                hit["distance"] = float(dist) if dist is not None else None

            if include_scores:
                # higher-is-better similarity score (monotonic transform of distance)
                hit["score"] = (1.0 / (1.0 + float(dist))) if dist is not None else None

            hits.append(hit)

        return hits

