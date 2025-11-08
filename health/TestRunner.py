# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-07
# Description: TestRunner
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import Dict, Optional

from AzureConfig import AzureConfig
from utility.logging_utils import get_class_logger

from health.BlobHealth import BlobHealth
from health.ChromaHealth import ChromaHealth
from health.EmbeddingHealth import EmbeddingHealth
from health.OpenAIHealth import OpenAIHealth


class TestRunner:
    """
    Orchestrates all smoke tests and reports a consolidated result.

    Tests included:
      - BlobHealth   (Blob Storage round-trip)
      - ChromaHealth (Chroma Cloud R/W)
      - EmbeddingHealth (Azure OpenAI embeddings)
      - OpenAIHealth (OpenAI chat)
    """

    def __init__(self, cfg: AzureConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.logger = logger or get_class_logger(self.__class__)

        self.logger.info("Initialising SmokeTestRunner")

        # Instantiate individual health check classes
        self.blob_health = BlobHealth(cfg)
        self.chroma_health = ChromaHealth(cfg)
        self.embedding_health = EmbeddingHealth(
            cfg,
            expected_dim=3072,  # adjust if your embedding model differs
        )
        self.openai_health = OpenAIHealth(cfg)

    # -------------------------------------------------------------------------
    def run_all(self, run_heavy_openai: bool = False) -> Dict[str, bool]:
        """
        Run all configured smoke tests.

        :param run_heavy_openai: If True, runs the heavy OpenAI test as well.
        :return: Dict mapping test names to True/False.
        """
        self.logger.info(
            "Starting smoke test suite (run_heavy_openai=%s)", run_heavy_openai
        )

        results: Dict[str, bool] = {}

        # Blob storage test
        try:
            self.logger.info("Running BlobHealth.check_blob()")
            ok_blob = self.blob_health.check_blob()
            results["blob_health"] = ok_blob
            self._log_result("BlobHealth", ok_blob)
        except Exception as e:
            self.logger.exception("BlobHealth.check_blob() raised an exception: %s", e)
            results["blob_health"] = False

        # Chroma R/W test (using a temporary healthcheck collection)
        test_index = "healthcheck-index"
        try:
            self.logger.info("Running ChromaHealth.add_index('%s')", test_index)
            self.chroma_health.add_index(test_index)
            # If add_index completes without logging a failure or raising, we treat as success
            # (You can tighten this later by returning a bool from add_index.)
            results["chroma_health"] = True
            self._log_result("ChromaHealth", True)
        except Exception as e:
            self.logger.exception("ChromaHealth.add_index('%s') failed: %s", test_index, e)
            results["chroma_health"] = False

        # Clean up healthcheck collection (best-effort)
        try:
            self.logger.info("Running ChromaHealth.remove_index('%s')", test_index)
            self.chroma_health.remove_index(test_index)
        except Exception as e:
            self.logger.warning(
                "ChromaHealth.remove_index('%s') raised a warning: %s", test_index, e
            )

        # Embedding test (Azure OpenAI)
        try:
            self.logger.info("Running EmbeddingHealth.run()")
            ok_embed = self.embedding_health.run()
            results["embedding_health"] = ok_embed
            self._log_result("EmbeddingHealth", ok_embed)
        except Exception as e:
            self.logger.exception("EmbeddingHealth.run() raised an exception: %s", e)
            results["embedding_health"] = False

        # OpenAI chat test (standard)
        try:
            self.logger.info("Running OpenAIHealth.run()")
            ok_openai = self.openai_health.run()
            results["openai_health"] = ok_openai
            self._log_result("OpenAIHealth", ok_openai)
        except Exception as e:
            self.logger.exception("OpenAIHealth.run() raised an exception: %s", e)
            results["openai_health"] = False

        # Optional heavy test
        if run_heavy_openai:
            try:
                self.logger.info("Running OpenAIHealth.run_heavy_test()")
                ok_heavy = self.openai_health.run_heavy_test(
                    paragraphs=30,
                    max_tokens=512,
                )
                results["openai_heavy_health"] = ok_heavy
                self._log_result("OpenAIHealth (heavy)", ok_heavy)
            except Exception as e:
                self.logger.exception(
                    "OpenAIHealth.run_heavy_test() raised an exception: %s", e
                )
                results["openai_heavy_health"] = False

        # Summary
        self._log_summary(results)
        return results

    # -------------------------------------------------------------------------
    def _log_result(self, name: str, ok: bool) -> None:
        if ok:
            self.logger.info("%s: PASS", name)
        else:
            self.logger.error("%s: FAIL", name)

    def _log_summary(self, results: Dict[str, bool]) -> None:
        total = len(results)
        passed = sum(1 for v in results.values() if v)
        failed = total - passed

        self.logger.info("Smoke test summary: %d total, %d passed, %d failed", total, passed, failed)

        for name, ok in results.items():
            status = "PASS" if ok else "FAIL"
            self.logger.info("  %s: %s", name, status)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    cfg = AzureConfig.from_env()
    runner = TestRunner(cfg)

    results = runner.run_all(run_heavy_openai=False)

    # Optional simple console summary (separate from logger)
    print("\n=== Smoke Test Results ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")

    overall_ok = all(results.values())
    print(f"\nOverall smoke test result: {'PASS' if overall_ok else 'FAIL'}")