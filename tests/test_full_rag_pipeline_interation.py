# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-22
# Description: test_full_rag_pipeline_interation.py
# -----------------------------------------------------------------------------
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.Config import Config


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


