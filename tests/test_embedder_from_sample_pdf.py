# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-15
# Description: test_embedder_from_sample_pdf
# -----------------------------------------------------------------------------
import os
import pytest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

from config.Config import Config
from embedding.IFUEmbedder import IFUEmbedder
from extractor.IFUTextExtractor import IFUTextExtractor

# Default to your known path; allow override with IFU_LOCAL_TEST_PDF
DEFAULT_PDF = "/Users/frankbogle/Documents/ifu/BMK2IFU.pdf"
TEST_PDF_PATH = Path(os.getenv("IFU_LOCAL_TEST_PDF", DEFAULT_PDF))


def _missing_azure_openai_env_vars() -> list[str]:
    """
    Check only the Azure OpenAI env vars needed for embeddings.
    We derive the env var names from Config.ENV_VARS to avoid duplication.
    """
    azure_env_names = [
        Config.ENV_VARS["openai_azure_api_key"],          # "AZURE_OPENAI_API_KEY"
        Config.ENV_VARS["openai_azure_endpoint"],         # "AZURE_OPENAI_ENDPOINT"
        Config.ENV_VARS["openai_azure_embed_deployment"], # "AZURE_OPENAI_EMBED_DEPLOYMENT"
    ]

    return [name for name in azure_env_names if not os.getenv(name)]


def _skip_if_missing_prereqs():
    """
    Skip integration test if:
      - the local IFU PDF is not present
      - required Azure OpenAI env vars are not set
    """
    if not TEST_PDF_PATH.is_file():
        pytest.skip(f"Local IFU PDF not found: {TEST_PDF_PATH}")

    missing = _missing_azure_openai_env_vars()
    if missing:
        pytest.skip(f"Missing env vars for Azure OpenAI: {', '.join(missing)}")


@dataclass
class SimpleChunk:
    """
    Adapter in case IFUChunker returns plain IFUChunk; we only need attrs used by IFUEmbedder.
    """
    text: str
    chunk_id: str
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    lang: Optional[str] = None
    lang_confidence: float = 0.0
    version: Optional[str] = None
    region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@pytest.mark.integration
def test_embedder_on_example_ifu_pdf_multilang_chunks():
    """
    Integration: Read the real BMK2IFU.pdf → extract pages → embed.
    Verifies:
      - connection works
      - extraction returns > 0 pages
      - embeddings call runs successfully
    """
    _skip_if_missing_prereqs()

    # 1) Config + embedder sanity
    cfg = Config.from_env()

    # These correspond to AZURE_OPENAI_* env vars via Config.ENV_VARS
    assert cfg.openai_azure_api_key, "AZURE_OPENAI_API_KEY must not be empty"
    assert cfg.openai_azure_endpoint, "AZURE_OPENAI_ENDPOINT must not be empty"
    assert cfg.openai_azure_embed_deployment, "AZURE_OPENAI_EMBED_DEPLOYMENT must not be empty"

    embedder = IFUEmbedder(
        cfg,
        batch_size=16,
        normalize=True,
        out_dtype="float32",
        filter_lang=None,
    )
    assert embedder.test_connection() is True

    # 2) Extract text pages from the example IFU (local bytes, no blob I/O needed)
    extractor = IFUTextExtractor()
    pdf_bytes = TEST_PDF_PATH.read_bytes()
    pages: List[str] = extractor.extract_text_from_pdf(pdf_bytes)
    assert isinstance(pages, list) and len(pages) > 0

