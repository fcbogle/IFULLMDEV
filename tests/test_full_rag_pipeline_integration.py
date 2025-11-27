# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-22
# Description: test_full_rag_pipeline_integration.py
# -----------------------------------------------------------------------------

import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from IFUTextExtractor import IFUTextExtractor
from chunking.LangDetectDetector import LangDetectDetector  # or your class name

from chunking.IFUChunker import IFUChunker
from embedding.IFUEmbedder import IFUEmbedder
from ingestion.IFUFileLoader import IFUFileLoader
from config.Config import Config
from vectorstore.ChromaIFUVectorStore import ChromaIFUVectorStore
from vectorstore.IFUVectorStore import IFUVectorStore


def _build_cfg_or_skip(pdf_path: Path):
    """
    Build strict Config and skil cleanly oif missing variables
    :param pdf_path:
    :return:
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
        pytest.skip(f"Config could not be initialised for blob test: {e}")

        # minimal required fields for blob upload/download
    missing = []
    for f in ["storage_account", "storage_key"]:
        if not getattr(cfg, f, None):
            missing.append(f)

    if missing:
        pytest.skip(f"Missing storage config for blob test: {missing}")

    return cfg

def _openai_chat_cfg_from_env():
    """
    Minimal cfg object for OpenAIChat (OpenAI-only).
    Avoids strict Config requirements for chat.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_CHAT_MODEL") or os.getenv("OPENAI_MODEL")
    org = os.getenv("OPENAI_ORG")  # optional

    if not api_key or not model:
        return None

    return SimpleNamespace(
        openai_api_key=api_key,
        openai_chat_model=model,
        openai_org=org,
    )

@pytest.mark.integration
def test_full_pipeline_blob_to_rag_to_chat_roundtrip():
    """
    Full integration test:

      A) Blob stage:
         1) Upload local PDF to Azure Blob
         2) Download same PDF from Blob
         3) Extract text from downloaded bytes

      B) RAG stage:
         4) Chunk extracted text
         5) Embed chunks
         6) Upsert embeddings into Chroma
         7) Embed question
         8) Query Chroma for top-k chunks

      C) Chat stage:
         9) Build chat prompt from retrieved chunks
        10) Call OpenAIChat
        11) Validate response

      D) Cleanup:
        12) Delete uploaded blob
        13) (optional) delete test collection
    :return:
    """
    # Resolve local PDF path
    local_pdf = os.getenv("IFU_SAMPLE_PDF")
    if local_pdf:
        local_pdf = Path(local_pdf)
    else:
        local_pdf = Path("/Users/frankbogle/Documents/ifu/BMK2IFU.pdf")

    if not local_pdf.exists():
        pytest.skip(f"Local PDF not found: {local_pdf}")

    cfg = _build_cfg_or_skip(local_pdf)

    tokenizer = lambda text: text.split()

    lang_detector = LangDetectDetector()

    # Instantiate core pipeline classes

    loader = IFUFileLoader(cfg=cfg)

    extractor = IFUTextExtractor()

    chunker = IFUChunker(
        tokenizer=tokenizer,
        lang_detector=lang_detector,
        chunk_size_tokens=300,
        overlap_tokens=100,
    )

    embedder = IFUEmbedder(
        cfg,
        batch_size=64,
        normalize=True,
        out_dtype="float32",
        filter_lang=None,
    )

    # Use dedicated collection to avoid test data pollution
    collection_name = os.getenv("IFU_TEST_COLLECTION", "ifu_chunks_e2e_test")
    store: IFUVectorStore = ChromaIFUVectorStore(
        cfg=cfg,
        embedder=embedder,
        collection_name=collection_name,
    )

    # Upload sample file to blob storage
    container = os.getenv("IFU_BLOB_CONTAINER", "ifudocs")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_name = f"ifu_pipeline_tests/{local_pdf.stem}_{ts}.pdf"

    uploaded_blob_name = loader.upload_document_from_path(
        local_path=str(local_pdf),
        container=container,
        blob_name=blob_name,
    )  #

    assert uploaded_blob_name == blob_name

    # Download PDF bytes from blob
    downloaded_bytes = loader.load_document(
        container=container,
        blob_name=blob_name,
    )
    assert isinstance(downloaded_bytes, (bytes, bytearray))
    assert len(downloaded_bytes) > 0

    # Extract text from downloaded bytes
    text = extractor.extract_text_from_pdf(downloaded_bytes)
    assert isinstance(text, list), f"Expected extracted text to be str, got {type(text)}"

    # Chunk extracted text
    assert len(text) > 1, "Number of pages extracted less than 1"
    doc_id = f"{local_pdf.stem}_{ts}"  # e.g. BMK2IFU_9f84c7a2
    doc_name = local_pdf.name

    doc_metadata = {
        "version": "Unknown",
        "region": "Unknown",
        "is_primary_language": True,
    }

    chunks = chunker.chunk_document(
        doc_id=doc_id,
        doc_name=doc_name,
        pages=text,
        doc_metadata=doc_metadata,
    )

    # Type and length checks
    assert isinstance(chunks, list) and len(chunks) > 0, "No chunks produced from IFU pages"



