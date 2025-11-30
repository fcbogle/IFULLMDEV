# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-30
# Description: test_ifu_multi_doc_loader.py
# -----------------------------------------------------------------------------
import os

import pytest

from config.Config import Config
from loader.IFUDocumentLoader import IFUDocumentLoader
from ingestion.IFUFileLoader import IFUFileLoader
from chunking.IFUChunker import IFUChunker
from embedding.IFUEmbedder import IFUEmbedder
from vectorstore.ChromaIFUVectorStore import ChromaIFUVectorStore
from LangDetectDetector import LangDetectDetector

pytestmark = pytest.mark.integration


def _build_cfg_or_skip() -> Config:
    """
    Build a Config instance from environment variables.
    If required fields are missing and Config raises, skip the test
    instead of failing the whole suite.
    """
    try:
        cfg = Config(
            ifu_sample_pdf=os.getenv("IFU_SAMPLE_PDF"),
            storage_account=os.getenv("AZURE_STORAGE_ACCOUNT"),
            storage_key=os.getenv("AZURE_STORAGE_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL"),
            openai_azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_azure_embed_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
            chroma_endpoint=os.getenv("CHROMA_ENDPOINT"),
            chroma_api_key=os.getenv("CHROMA_API_KEY"),
            chroma_tenant=os.getenv("CHROMA_TENANT"),
            chroma_database=os.getenv("CHROMA_DATABASE"),
        )
    except (TypeError, ValueError) as e:
        pytest.skip(f"Config could not be initialised for IFUDocumentLoader test: {e}")

    return cfg


@pytest.mark.integration
def test_ifu_document_loader_initialises():
    """
    Simple integration test:

      - Config can be built
      - IFUDocumentLoader can be instantiated
      - Core components are initialised and of expected types
    """
    cfg = _build_cfg_or_skip()

    loader = IFUDocumentLoader(
        cfg=cfg,
        collection_name="ifu_chunks_init_test",
    )

    # Top-level attributes
    assert loader.cfg is cfg
    assert loader.collection_name == "ifu_chunks_init_test"

    # Internals should be initialised
    assert isinstance(loader.loader, IFUFileLoader)
    assert isinstance(loader.chunker, IFUChunker)
    assert isinstance(loader.embedder, IFUEmbedder)
    assert isinstance(loader.store, ChromaIFUVectorStore)

    # Language detector should be present and of correct type
    assert loader.lang_detector is not None
    assert isinstance(loader.lang_detector, LangDetectDetector)

    # Logger should exist
    assert hasattr(loader, "logger")
    assert loader.logger is not None
