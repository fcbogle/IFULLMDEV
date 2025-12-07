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
    """
    Build a Config instance from environment variables.
    If required fields are missing and Config raises, skip the test
    instead of failing the whole suite.
    """

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
        collection_name="ifu-docs-test",
    )

    # Top-level attributes
    assert loader.cfg is cfg
    assert loader.collection_name == "ifu-docs-test"

    # Internals should be initialised
    assert isinstance(loader.loader, IFUFileLoader)
    assert isinstance(loader.chunker, IFUChunker)
    assert isinstance(loader.embedder, IFUEmbedder)
    assert isinstance(loader.store, ChromaIFUVectorStore)
    assert isinstance(loader.extractor, IFUTextExtractor)

    # Language detector should be present and of correct type
    assert loader.lang_detector is not None
    assert isinstance(loader.lang_detector, LangDetectDetector)

    # Logger should exist
    assert hasattr(loader, "logger")
    assert loader.logger is not None


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
def test_multi_doc_loader_download_chunk_embed_query():
    """
    Integration test that verifies:

      - Sample PDFs on disk can be accessed
      - upload_multiple_pdfs uploads them to Azure Blob Storage
      - The resulting blobs exist and contain data
    """
    cfg = _build_cfg_or_skip()
    sample_pdfs = _get_sample_pdf_files(cfg)

    floader = IFUFileLoader(cfg)

    loader = IFUDocumentLoader(
        cfg=cfg,
        collection_name="ifu-docs-test",
    )

    container = "ifu-docs-test"  # use a dedicated test container
    blob_prefix = "integration_test/"

    # ---- Act: upload local PDFs ----
    results = loader.upload_multiple_pdfs(
        pdf_paths=sample_pdfs,
        container=container,
        blob_prefix=blob_prefix,
    )

    # ---- Assert: local paths were all processed ----
    assert set(results.keys()) == set(sample_pdfs)
    assert all(isinstance(p, Path) for p in results.keys())

    # ---- Assert: blob names look correct ----
    for local_path, blob_name in results.items():
        assert blob_name.startswith(blob_prefix)
        assert blob_name.endswith(local_path.name)

    # ---- Assert: blobs really exist in Azure and contain data ----
    container_client = loader.loader.blob_service.get_container_client(container)

    try:
        for local_path, blob_name in results.items():
            # Verify blob exists and has content
            blob_client = container_client.get_blob_client(blob_name)
            props = blob_client.get_blob_properties()
            assert props.size > 0, f"Blob '{blob_name}' has zero size"

            # Download bytes from blob
            pdf_list = floader.list_documents(container="ifu-docs-test")
            assert isinstance(pdf_list, list)

            pdf_bytes = floader.load_document(blob_name, container)
            assert isinstance(pdf_bytes, (bytes, bytearray))
            assert len(pdf_bytes) > 0

            # Extract text from PDF bytes
            raw = loader.extractor.extract_text_from_pdf(pdf_bytes)

            # raw is List[str] so normalise pages
            pages: List[str]
            if isinstance(raw, list) and all(isinstance(p, str) for p in raw):
                pages = raw

            elif isinstance(raw, list) and all(isinstance(p, list) for p in raw):
                # List[List[str]] – e.g., paragraphs per page -> join inner lists
                pages = [" ".join(part for part in page if isinstance(part, str)) for page in raw]

            elif isinstance(raw, str):
                # Single string – whole doc as one page
                pages = [raw]

            else:
                raise TypeError(
                    f"Unexpected type from extract_text_from_pdf: {type(raw)}; "
                    f"expected str | List[str] | List[List[str]]"
                )

            assert pages, f"No pages extracted for blob '{blob_name}'"
            assert all(isinstance(p, str) for p in pages)

            # Chunk document into IFUChunks



    finally:
        _ = None
