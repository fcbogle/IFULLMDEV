# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-14
# Description: IFUEmbedder
# -----------------------------------------------------------------------------
import time
from typing import Optional, List, Iterable, Any, Set

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
            filter_lang: Optional[Set[str]] = None,  # e.g., {"en"} to embed only English
            logger=None,
    ):
        self.cfg = cfg
        self.batch_size = batch_size
        self.normalize = normalize
        self.out_dtype = out_dtype
        self.filter_lang = filter_lang
        self.logger = logger or get_class_logger(self.__class__)

        # This is the “model” argument we will pass to embeddings.create(...)
        # For Azure, this is the *deployment name*
        self.model = cfg.openai_azure_embed_deployment

        # ------------------------------------------------------------------
        # Client setup
        # Priority: Azure OpenAI for embeddings; fallback to direct OpenAI.
        # ------------------------------------------------------------------
        if cfg.openai_azure_api_key and cfg.openai_azure_endpoint:
            # Azure OpenAI client (modern SDK)
            self.client = AzureOpenAI(
                api_key=cfg.openai_azure_api_key,
                azure_endpoint=cfg.openai_azure_endpoint,
                api_version="2024-10-21",  # keep in sync with your Azure config
            )
            self.logger.info(
                "IFUEmbedder using AzureOpenAI client "
                f"(endpoint={cfg.openai_azure_endpoint}, deployment={self.model})"
            )
        else:
            # Fallback: direct OpenAI (for local/testing, if you ever want that)
            base_url = cfg.openai_base_url or "https://api.openai.com/v1"
            self.client = OpenAI(
                api_key=cfg.openai_api_key,
                base_url=base_url,
            )
            self.logger.info(
                "IFUEmbedder using OpenAI client "
                f"(base_url={base_url}, model={self.model})"
            )

        self.model = cfg.openai_azure_embed_deployment or "text-embedding-3-large"
        self.logger.info("OpenAI Azure Embedder initialized '{self.model}', dtype={self.out_dtype}")

    def test_connection(self) -> bool:
        """
        Verifies that Azure OpenAI embedding service is reachable and the
        configured deployment responds correctly.

        Returns:
            True if a minimal embedding request succeeds, otherwise False.
        """
        try:
            # --------------------------
            # Sanity check required config
            # --------------------------
            if not self.cfg.openai_azure_api_key:
                self.logger.error("Azure OpenAI API key missing (AZURE_OPENAI_API_KEY)")
                return False

            if not self.cfg.openai_azure_endpoint:
                self.logger.error("Azure OpenAI endpoint missing (AZURE_OPENAI_ENDPOINT)")
                return False

            if not self.cfg.openai_azure_embed_deployment:
                self.logger.error("Azure OpenAI embedding deployment missing (AZURE_OPENAI_EMBED_DEPLOYMENT)")
                return False

            self.logger.debug(
                f"Testing Azure OpenAI embedding connection: "
                f"endpoint={self.cfg.openai_azure_endpoint}, "
                f"deployment={self.cfg.openai_azure_embed_deployment}"
            )

            # --------------------------
            # Issue minimal test request
            # --------------------------
            response = self.client.embeddings.create(
                model=self.cfg.openai_azure_embed_deployment,
                input=["connection test"]
            )

            if not hasattr(response, "data") or not response.data:
                self.logger.error("Azure OpenAI returned an empty embedding response.")
                return False

            vec = response.data[0].embedding
            if not vec or not isinstance(vec, list):
                self.logger.error("Azure OpenAI embedding response contains no valid vector.")
                return False

            # --------------------------
            # Optional: log dimensionality
            # --------------------------
            self.logger.info(
                f"Azure OpenAI embedding connection OK "
                f"(received vector dim={len(vec)})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Azure OpenAI embedding connection failed: {e}")
            return False

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
        """
        Embed a batch of texts using the configured embedding model.

        Returns:
            np.ndarray of shape (batch_size, embedding_dim), dtype float32/float16.
        """
        delay = 0.8
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )

                arr = np.asarray([d.embedding for d in resp.data], dtype=np.float32)

                # Normalize vectors (cosine-friendly)
                if self.normalize:
                    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                    arr = arr / norms

                # Convert dtype if needed
                if self.out_dtype == "float16":
                    arr = arr.astype(np.float16)
                elif self.out_dtype != "float32":
                    self.logger.warning(
                        "Unsupported out_dtype '%s', defaulting to float32",
                        self.out_dtype,
                    )

                return arr

            except Exception as e:
                self.logger.warning(
                    "Embedding batch failed (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    e,
                )
                if attempt == max_retries:
                    raise
                time.sleep(delay)
                delay *= 1.7  # backoff

        # Unreachable and just here for type checkers
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
            batch = items[i:i + self.batch_size]
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
