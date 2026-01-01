# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-20
# Description: IFUQueryService
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class IFUQueryService:
    store: Any  # your IFUVectorStore / ChromaIFUVectorStore
    collection_name: str = "ifu-docs-test"

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Delegate to a vector store; this should embed a query and call Chroma
        return self.store.query_text(query_text=query_text, n_results=n_results, where=where)

    @staticmethod
    def to_hits(
            raw: Dict[str, Any],
            include_text: bool = True,
            include_scores: bool = True,
            include_metadata: bool = True,
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
            # prefer an explicit "page" if present, else fall back
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
            if include_scores:
                hit["score"] = float(dist) if dist is not None else None
            if include_metadata:
                hit["metadata"] = md if isinstance(md, dict) else None

            hits.append(hit)

        return hits

