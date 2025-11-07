# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-06
# Description: EmbeddingHealth
# -----------------------------------------------------------------------------
import time
import logging
from typing import Optional
from openai import AzureOpenAI
from AzureConfig import AzureConfig
from utility.logging_utils import get_logger  # adjust import path if needed


class EmbeddingHealth:
    """
    Smoke test for Azure OpenAI Embeddings.

    Verifies:
      - The embedding API call completes successfully
      - The response contains a valid vector
      - The vector dimension matches the expected dimension (if provided)
    """

    def __init__(
        self,
        cfg: AzureConfig,
        expected_dim: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.expected_dim = expected_dim
        self.logger = logger or get_logger(__name__)

        self.logger.info(
            "Initialising EmbeddingHealth with model deployment: %s",
            getattr(cfg, "openai_azure_embed_deployment", "text-embedding-3-large"),
        )

        # Azure OpenAI Client
        self.client = AzureOpenAI(
            api_key=cfg.openai_azure_api_key,
            azure_endpoint=cfg.openai_azure_endpoint,
            api_version="2024-10-21",
        )

        self.model = getattr(
            cfg, "openai_azure_embed_deployment", "text-embedding-3-large"
        )

    def run(self) -> bool:
        """
        Run the embedding smoke test.

        Returns:
            True if the embedding call succeeds and (optionally) the dimension matches.
        """
        test_text = "Azure OpenAI embedding healthcheck"
        self.logger.info("Running embedding healthcheck using model: %s", self.model)

        try:
            start = time.time()
            resp = self.client.embeddings.create(input=test_text, model=self.model)
            elapsed_ms = (time.time() - start) * 1000.0

            if not resp.data or not resp.data[0].embedding:
                self.logger.error("No embedding data returned in response.")
                return False

            embedding = resp.data[0].embedding
            dim = len(embedding)

            self.logger.info(
                "Embedding call succeeded in %.1f ms. Returned dimension: %d",
                elapsed_ms,
                dim,
            )

            # Optional dimension validation
            if self.expected_dim is not None:
                if dim != self.expected_dim:
                    self.logger.warning(
                        "Dimension mismatch: expected %d, got %d.",
                        self.expected_dim,
                        dim,
                    )
                    return False
                else:
                    self.logger.info(
                        "Dimension matches expected value: %d", self.expected_dim
                    )

            self.logger.info("Embedding healthcheck PASSED.")
            return True

        except Exception as e:
            self.logger.exception("Embedding healthcheck FAILED: %s", e)
            return False


if __name__ == "__main__":
    import logging

    # Configure root logging (colour and line numbers handled by get_logger)
    logging.basicConfig(level=logging.INFO)

    cfg = AzureConfig.from_env()

    # Expected dimensions for models:
    # - text-embedding-3-small → 1536
    # - text-embedding-3-large → 3072
    expected_dim = 3072

    eh = EmbeddingHealth(cfg, expected_dim=expected_dim)
    ok = eh.run()

    eh.logger.info("EmbeddingHealth result: %s", "PASS" if ok else "FAIL")