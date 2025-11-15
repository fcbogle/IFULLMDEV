# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-14
# Description: IFUEmbedder
# -----------------------------------------------------------------------------
import time
from typing import Optional, List, Iterable, Any

import numpy as np
from openai import AzureOpenAI, OpenAI

from config.Config import Config
from embedding.EmbeddingRecord import EmbeddingRecord
from utility.logging_utils import get_class_logger


class IFUEmbedder:
    def __init__(
            self,
            cfg: Config,
            *,
            batch_size: int = 64,
            normalize: bool = True,
            out_dtype: str = "float32",
            filter_lang: Optional[set[str]] = None,  # e.g., {"en"} to embed only English
            logger=None,
    ):
        self.cfg = cfg
        self.batch_size = batch_size
        self.normalize = normalize
        self.out_dtype = out_dtype
        self.filter_lang = filter_lang
        self.logger = logger or get_class_logger(self.__class__)

        # Azure OpenAI client setup
        self.client = AzureOpenAI(
            api_key=cfg.openai_azure_api_key,
            azure_endpoint=cfg.openai_azure_endpoint,
            api_version="2024-10-21"
        )
        self.model = cfg.openai_azure_embed_deployment or "text-embedding-3-large"
        self.logger.info("OpenAI Azure Embedder initialized '{self.model}', dtype={self.out_dtype}")

    def _init_client(self) -> None:
        """
        Tries classic AzureOpenAI(...) first; if the installed SDK signature
        is incompatible, falls back to OpenAI(base_url=.../deployments/<model>).
        Sets self._use_deployment_param accordingly.
        """
        endpoint = self.cfg.openai_azure_endpoint.rstrip("/")
        key = self.cfg.openai_azure_api_key
        api_version = getattr(self.cfg, "openai_api_version", "2024-10-21")

        # Classic style
        if AzureOpenAI is not None:
            try:
                self.client = AzureOpenAI(
                    api_key=key,
                    azure_endpoint=endpoint,
                    api_version=api_version,
                )
                self._use_deployment_param = True
                return
            except TypeError as e:
                # Newer SDKs may alter signature (e.g., posthog_client). Fall through.
                self.logger.debug(f"AzureOpenAI init fell through to base_url mode: {e}")

        # Fallback: deployment encoded in base_url; do not pass model= on each call
        self.client = OpenAI(
            api_key=key,
            base_url=f"{endpoint}/openai/deployments/{self.model}",
        )
        self._use_deployment_param = False

    def _embed_batch(self, texts: List[str], *, max_retries: int = 5) -> np.ndarray:
        delay = 0.8
        for attempt in range(1, max_retries + 1):
            try:
                if self._use_deployment_param:
                    resp = self.client.embeddings.create(model=self.model, input=texts)
                else:
                    resp = self.client.embeddings.create(input=texts)

                arr = np.asarray([d.embedding for d in resp.data], dtype=np.float32)

                # Normalize vectors (cosine-friendly)
                if self.normalize:
                    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                    arr = arr / norms

                # Convert dtype if needed
                if self.out_dtype == "float16":
                    arr = arr.astype(np.float16)
                elif self.out_dtype != "float32":
                    self.logger.warning(f"Unsupported out_dtype '{self.out_dtype}', defaulting to float32")
                return arr

            except Exception as e:
                self.logger.warning(f"Embedding batch failed (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    raise
                time.sleep(delay)
                delay *= 1.7  # backoff

        # Unreachable and include for type checkers
        return np.empty((0, 0), dtype=np.float32)

    def embed_chunks(self, chunks: Iterable[Any]) -> List[EmbeddingRecord]:
        """
        Filter (lang/empties) → batch → _embed_batch → EmbeddingRecord list.
        Expects each chunk to have: .text, .chunk_id; optional metadata (doc_id, page_start, etc.).
        """
        items: List[Any] = []
        for c in chunks:
            if self.filter_lang and getattr(c, "lang", None) not in self.filter_lang:
                continue
            text = getattr(c, "text", "")
            if not text or not text.strip():
                continue
            items.append(c)

        total = len(items)
        self.logger.info(f"Embedding {total} chunks (batch={self.batch_size})")
        out: List[EmbeddingRecord] = []
        if total == 0:
            return out

        for i in range(0, total, self.batch_size):
            batch = items[i:i+self.batch_size]
            texts = [c.text for c in batch]
            try:
                arr = self._embed_batch(texts)
                for c, v in zip(batch, arr):
                    meta = {
                        "doc_id": getattr(c, "doc_id", None),
                        "doc_name": getattr(c, "doc_name", None),
                        "page_start": getattr(c, "page_start", None),
                        "page_end": getattr(c, "page_end", None),
                        "char_start": getattr(c, "char_start", None),
                        "char_end": getattr(c, "char_end", None),
                        "lang": getattr(c, "lang", None),
                        "lang_confidence": float(getattr(c, "lang_confidence", 0.0)),
                        "version": getattr(c, "version", None),
                        "region": getattr(c, "region", None),
                    }
                    extra = getattr(c, "metadata", None)
                    if isinstance(extra, dict):
                        meta = {**extra, **meta}

                        out.append(EmbeddingRecord(
                            chunk_id=getattr(c, "chunk_id"),
                            vector=v,
                            text=c.text,
                            metadata=meta,
                        ))
            except Exception as e:
                self.logger.error(f"Embedding batch at offset {i} failed: {e}")

        self.logger.info(f"Completed embeddings for {len(out)} chunks.")
        return out
