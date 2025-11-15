# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-15
# Description: test_embedder_from_sample_pdf
# -----------------------------------------------------------------------------
import os
from pathlib import Path

from sympy.testing import pytest

from config import Config
from embedding.IFUEmbedder import IFUEmbedder
from extractor.IFUTextExtractor import IFUTextExtractor

cfg = Config.from_env()
embedder = IFUEmbedder(cfg)
extractor = IFUTextExtractor()

REQUIRED_AZURE_OPENAI = ["OPENAI_AZURE_API_KEY", "OPENAI_AZURE_ENDPOINT"]
MISSING_AZURE_OPENAI = [k for k in REQUIRED_AZURE_OPENAI if not os.getenv(k)]

# Default to your known path; allow override with IFU_LOCAL_TEST_PDF
DEFAULT_PDF = "/Users/frankbogle/Documents/ifu/BMK2IFU.pdf"
TEST_PDF_PATH = Path(os.getenv("IFU_LOCAL_TEST_PDF", DEFAULT_PDF))


def _skip_if_missing_prereqs():
    if not TEST_PDF_PATH.is_file():
        pytest.skip(f"Local IFU PDF not found: {TEST_PDF_PATH}")
    if MISSING_AZURE_OPENAI:
        pytest.skip(f"Missing env vars for Azure OpenAI: {', '.join(MISSING_AZURE_OPENAI)}")
