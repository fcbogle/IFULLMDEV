# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-30
# Description: test_ifu_multi_doc_loader.py
# -----------------------------------------------------------------------------
import os
from pathlib import Path
from typing import List

import pytest
import logging

from extractor.IFUTextExtractor import IFUTextExtractor
from chunking.IFUChunker import IFUChunker
from chunking.LangDetectDetector import LangDetectDetector
from config.Config import Config
from embedding.IFUEmbedder import IFUEmbedder
from ingestion.IFUFileLoader import IFUFileLoader
from loader.IFUDocumentLoader import IFUDocumentLoader
from vectorstore.ChromaIFUVectorStore import ChromaIFUVectorStore

pytestmark = pytest.mark.integration

test_logger = logging.getLogger("test_logger")


def _build_cfg_or_skip() -> Config:
    test_logger.info("Initialising Config for integration tests...")

    try:
        cfg = Config(
            ifu_sample_pdf=os.getenv("IFU_SAMPLE_PDF"),
            ifu_sample_folder=os.getenv("IFU_SAMPLE_FOLDER"),
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
        test_logger.info("Config initialised successfully")

    except (TypeError, ValueError) as e:
        pytest.skip(f"Config could not be initialised for IFUDocumentLoader test: {e}")

    return cfg


def _get_sample_pdf_files(cfg: Config) -> list[Path]:
    if not cfg.ifu_sample_folder:
        pytest.skip("IFU_SAMPLE_FOLDER not set – cannot locate sample PDFs")

    path = Path(cfg.ifu_sample_folder)
    if path.is_file():
        return [path]

    if path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No PDFs found in IFU_SAMPLE_FOLDER")
        return pdfs

    pytest.skip(f"IFU_SAMPLE_PDF path is neither file nor directory: {path}")


@pytest.mark.integration
def test_ifu_document_loader_initialises():
    cfg = _build_cfg_or_skip()

    loader = IFUDocumentLoader(cfg=cfg, collection_name="ifu-docs-test")

    assert loader.cfg is cfg
    assert loader.collection_name == "ifu-docs-test"

    assert isinstance(loader.loader, IFUFileLoader)
    assert isinstance(loader.chunker, IFUChunker)
    assert isinstance(loader.embedder, IFUEmbedder)
    assert isinstance(loader.store, ChromaIFUVectorStore)
    assert isinstance(loader.extractor, IFUTextExtractor)

    assert loader.lang_detector is not None
    assert isinstance(loader.lang_detector, LangDetectDetector)

    assert hasattr(loader, "logger")
    assert loader.logger is not None


@pytest.mark.integration
def test_multi_doc_loader_upload_then_ingest_from_blobs():
    """
    Integration test that verifies:

      - Sample PDFs on disk can be accessed
      - upload_multiple_pdfs uploads them to Azure Blob Storage
      - ingest_blob_pdfs processes the uploaded blobs via _process_single_pdf
        (download -> extract -> chunk -> embed -> upsert)
    """
    cfg = _build_cfg_or_skip()
    sample_pdfs = _get_sample_pdf_files(cfg)

    loader = IFUDocumentLoader(cfg=cfg, collection_name="ifu-docs-test")

    container = "ifu-docs-test"  # dedicated test container
    blob_prefix = ""

    # ---- Act: upload local PDFs ----
    results = loader.upload_multiple_pdfs(
        pdf_paths=sample_pdfs,
        container=container,
        blob_prefix=blob_prefix,
    )

    # ---- Assert: local paths were all processed ----
    assert set(results.keys()) == set(sample_pdfs)

    # ---- Assert: blob names look correct ----
    for local_path, blob_name in results.items():
        assert blob_name.startswith(blob_prefix)
        assert blob_name.endswith(local_path.name)

    # ---- Assert: blobs exist and have content (quick check) ----
    container_client = loader.loader.blob_service.get_container_client(container)
    for local_path, blob_name in results.items():
        blob_client = container_client.get_blob_client(blob_name)
        props = blob_client.get_blob_properties()
        assert props.size > 0, f"Blob '{blob_name}' has zero size"

    # ---- Act: ingest the uploaded blobs using the real loader path ----
    blob_names = list(results.values())
    ingested = loader.ingest_blob_pdfs(blob_names, container=container)

    # ---- Assert: all blobs were ingested successfully ----
    assert ingested == len(blob_names)

    # ---- Optional: sanity check that Chroma has metadata (sample) ----
    docs = loader.store.list_documents(limit=600)  # adjust if needed
    assert isinstance(docs, list)
    assert docs, "Expected documents summary from Chroma after ingestion"

    # Light check: at least one doc includes the expected keys
    sample = docs[0]
    assert "doc_id" in sample
    assert "chunk_count" in sample
    assert "page_count" in sample
    # These will be None until you extend list_documents() to aggregate them:
    # assert "document_type" in sample
    # assert "last_modified" in sample

    # ---- Optional: semantic query sanity check ----
    try:
        result = loader.store.query_text("Blatchford", n_results=5)
    except Exception as e:
        pytest.skip(f"Chroma query_text failed with {e!r}; skipping semantic query assertion")

    assert isinstance(result, dict)
    assert "ids" in result
    assert isinstance(result["ids"], list)

