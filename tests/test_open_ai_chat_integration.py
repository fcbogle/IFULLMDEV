# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-22
# Description: test_open_ai_chat_integration.py
# -----------------------------------------------------------------------------
import os

import pytest

from chat.OpenAIChat import OpenAIChat
from config.Config import Config


def _missing_openai_chat_env_vars() -> list[str]:
    """
    Check only the Azure OpenAI env vars needed for embeddings.
    We derive the env var names from Config.ENV_VARS to avoid duplication.
    """
    openai_env_names = [
        Config.ENV_VARS["openai_api_key"],          # "OPENAI_API_KEY"
        Config.ENV_VARS["openai_base_url"],         # "OPENAI_ENDPOINT"
        Config.ENV_VARS["openai_chat_model"],       # "OPENAI_CHAT_MODEL"
    ]

    return [name for name in openai_env_names if not os.getenv(name)]

def _skip_if_missing_prereqs():
    missing = _missing_openai_chat_env_vars()
    if missing:
        pytest.skip(f"Missing env vars for Azure OpenAI: {', '.join(missing)}")

def _chat_is_configured(cfg: Config) -> bool:
    # Simple test to ensure there is a chat client
    return bool(
        getattr(cfg, "openai_api_key", None)
        and getattr(cfg, "openai_chat_model", None)
    )

@pytest.mark.integration
def test_openai_chat_simple_roundtrip():
    """
    Integration test:
      - instantiate OpenAIChat (OpenAI only)
      - send a tiny prompt
      - verify response is returned
    """
    _skip_if_missing_prereqs()

    cfg = Config.from_env()

    # These correspond to OPENAI_* env vars via Config.ENV_VARS
    assert cfg.openai_api_key, "OPENAI_API_KEY must not be empty"
    assert cfg.openai_base_url, "OPENAI_BASE_URL must not be empty"
    assert cfg.openai_chat_model, "OPENAI_CHAT_MODEL must not be empty"

    if not _chat_is_configured(cfg):
        pytest.skip("OpenAI chat not configured; skipping integration test.")

    chat = OpenAIChat(cfg=cfg)

    resp = chat.simple_chat(
        user_text="Reply with a single word: OK",
        system_text="You are a test assistant.",
        temperature=0.0,
        max_tokens=5,
    )

    assert isinstance(resp, dict)
    assert "answer" in resp
    assert resp["answer"].strip().upper() == "OK"

@pytest.mark.integration
def test_openai_chat_healthcheck():
    """
    Light healthcheck integration test.
    """
    cfg = Config.from_env()
    if cfg is None:
        pytest.skip("OPENAI_API_KEY / OPENAI_CHAT_MODEL not set; skipping integration test.")

    chat = OpenAIChat(cfg=cfg)
    assert chat.healthcheck() is True

