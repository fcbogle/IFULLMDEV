# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-06
# Description: OpenAIHealth
# -----------------------------------------------------------------------------

import time
import logging
from typing import Optional

from openai import OpenAI
from AzureConfig import AzureConfig
from utility.logging_utils import get_logger  # adjust if this lives in another package


class OpenAIHealth:
    """
    Smoke tests for OpenAI API connectivity and basic chat completion.
    """

    def __init__(self, cfg: AzureConfig, model_override: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or get_logger(__name__)

        base_url = getattr(cfg, "openai_base_url", None)
        if base_url:
            self.client = OpenAI(
                api_key=cfg.openai_api_key,
                base_url=base_url,
                organization="org-H2zMljc8F9wEWQ9XxPsuVDlq",
            )
            self.logger.info("Initialised OpenAI client with base_url: %s", base_url)
        else:
            self.client = OpenAI(api_key=cfg.openai_api_key)
            self.logger.info("Initialised OpenAI client for default endpoint: https://api.openai.com/v1")

        self.model = (
            model_override
            or getattr(cfg, "openai_chat_model", None)
            or "gpt-4o"
        )
        self.logger.info("Configured model: %s", self.model)

    def get_service_info(self) -> dict:
        """
        Returns metadata about the OpenAI chat configuration.
        """
        masked_key = f"{self.cfg.openai_api_key[:4]}..." if self.cfg.openai_api_key else None
        info = {
            "provider": "OpenAI",
            "endpoint": getattr(self.cfg, "openai_base_url", "https://api.openai.com/v1"),
            "model": self.model,
            "api_version": "N/A (OpenAI direct)",
            "api_key_prefix": masked_key,
        }
        self.logger.info("Service info: %s", info)
        return info

    def run(self) -> bool:
        """
        Run a standard OpenAI Chat smoke test.
        """
        self.logger.info("Starting OpenAI Chat healthcheck with model: %s", self.model)
        start = time.time()

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a model probe. Reply briefly to confirm connectivity."},
                    {"role": "user", "content": "Say OK if you can read this."},
                ],
                max_tokens=10,
            )
            elapsed_ms = (time.time() - start) * 1000.0
            self.logger.info("Chat call succeeded in %.1f ms.", elapsed_ms)

            org_id = getattr(self.client, "organization", None)
            if org_id:
                self.logger.debug("Organization configured: %s", org_id)
            else:
                self.logger.debug("No organization explicitly set; using default for API key.")

            if not resp.choices:
                self.logger.error("Chat response contained no choices.")
                return False

            message = resp.choices[0].message
            content = (message.content or "").strip() if message else ""
            if not content:
                self.logger.error("Chat response content is empty.")
                return False

            self.logger.info("Response content: %r", content)

            usage = getattr(resp, "usage", None)
            if usage:
                self.logger.info("Usage summary: %s", usage)
            else:
                self.logger.debug("No usage field present in response.")

            self.logger.info("OpenAI Chat healthcheck PASSED.")
            return True

        except Exception as e:
            self.logger.exception("OpenAI Chat healthcheck FAILED: %s", e)
            return False

    def run_heavy_test(self, paragraphs: int = 20, max_tokens: int = 512) -> bool:
        """
        Heavier test to consume more tokens and verify usage metering.
        """
        self.logger.info(
            "Starting heavy test with model=%s, paragraphs=%d, max_tokens=%d",
            self.model,
            paragraphs,
            max_tokens,
        )

        base_paragraph = (
            "This is a longer healthcheck paragraph intended to increase token usage "
            "for testing OpenAI billing and usage metrics. It should be semantically "
            "coherent but not necessarily meaningful.\n"
        )
        long_prompt = base_paragraph * paragraphs

        try:
            start = time.time()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an OpenAI probe verifying API usage and billing. "
                            "Respond concisely."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Read the following text and summarise it in one sentence:\n\n"
                            + long_prompt
                        ),
                    },
                ],
                max_tokens=max_tokens,
            )
            elapsed_ms = (time.time() - start) * 1000.0
            self.logger.info("Heavy test call completed in %.1f ms.", elapsed_ms)

            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.logger.info("Heavy test usage: %s", usage)
            else:
                self.logger.warning("No usage field on heavy test response.")

            if not resp.choices:
                self.logger.error("Heavy test response contained no choices.")
                return False

            message = resp.choices[0].message
            content = (message.content or "").strip() if message else ""
            if not content:
                self.logger.error("Heavy test response content is empty.")
                return False

            self.logger.info("Heavy test response preview: %.200r", content)
            self.logger.info("OpenAI heavy healthcheck PASSED.")
            return True

        except Exception as e:
            self.logger.exception("OpenAI heavy test FAILED: %s", e)
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = AzureConfig.from_env()
    ch = OpenAIHealth(cfg)

    info = ch.get_service_info()
    for k, v in info.items():
        print(f"{k}: {v}")

    print("\n--- Running STANDARD usage test ---")
    ok = ch.run()
    print(f"OpenAI Chat healthcheck: {'PASS' if ok else 'FAIL'}")

    print("\n--- Running HEAVY usage test ---")
    ok_heavy = ch.run_heavy_test(paragraphs=30, max_tokens=512)
    print(f"OpenAI Chat heavy healthcheck: {'PASS' if ok_heavy else 'FAIL'}")

