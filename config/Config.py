# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-08
# Description: Config
# -----------------------------------------------------------------------------

import os
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv

# Load .env once globally
load_dotenv(find_dotenv(usecwd=True), override=True)


@dataclass(frozen=True)
class Config:
    # IFU Sample PDF
    ifu_sample_pdf: str
    ifu_sample_folder: str

    # Azure Storage
    storage_account: str
    storage_key: str

    # OpenAI (direct, for chat)
    openai_base_url: str
    openai_api_key: str
    openai_chat_model: str

    # Azure OpenAI (for embeddings, and possibly chat if you add it later)
    openai_azure_api_key: str
    openai_azure_endpoint: str
    openai_azure_embed_deployment: str

    # Chroma Vector Database
    chroma_endpoint: str
    chroma_api_key: str
    chroma_tenant: str
    chroma_database: str

    # Chunking
    ifu_chunk_size_tokens: int
    ifu_overlap_tokens: int

    # ---- Single source of truth: field_name -> ENV VAR NAME ----
    ENV_VARS = {
        # IFU Sample PDF
        "ifu_sample_pdf": "IFU_SAMPLE_PDF",
        "ifu_sample_folder": "IFU_SAMPLE_FOLDER",

        # Storage
        "storage_account": "AZURE_STORAGE_ACCOUNT",
        "storage_key": "AZURE_STORAGE_KEY",

        # OpenAI direct
        "openai_base_url": "OPENAI_BASE_URL",      # e.g. https://api.openai.com/v1
        "openai_api_key": "OPENAI_API_KEY",
        "openai_chat_model": "OPENAI_CHAT_MODEL",

        # Azure OpenAI
        "openai_azure_api_key": "AZURE_OPENAI_API_KEY",
        "openai_azure_endpoint": "AZURE_OPENAI_ENDPOINT",
        "openai_azure_embed_deployment": "AZURE_OPENAI_EMBED_DEPLOYMENT",

        # Chroma
        "chroma_endpoint": "CHROMA_ENDPOINT",
        "chroma_api_key": "CHROMA_API_KEY",
        "chroma_tenant": "CHROMA_TENANT",
        "chroma_database": "CHROMA_DATABASE",

        # Chunking
        "ifu_chunk_size_tokens": "IFU_CHUNK_SIZE_TOKENS",
        "ifu_overlap_tokens": "IFU_OVERLAP_TOKENS",


    }

    # Convenient *groups* for use in tests / health checks
    AZURE_OPENAI_ENV_VARS = (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_EMBED_DEPLOYMENT",
    )

    OPENAI_DIRECT_ENV_VARS = (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    )

    @staticmethod
    def from_env() -> "Config":
        """Build Config object from environment variables with proper typing."""
        kwargs = {}

        for field_name, env_name in Config.ENV_VARS.items():
            raw = os.getenv(env_name)

            # ---- typed fields ----
            if field_name == "ifu_chunk_size_tokens":
                kwargs[field_name] = int(raw) if raw else 1200

            elif field_name == "ifu_overlap_tokens":
                kwargs[field_name] = int(raw) if raw else 100

            # ---- default: string fields ----
            else:
                kwargs[field_name] = raw or ""

        return Config(**kwargs)

    def __post_init__(self):
        """
        Fail fast if any required config is missing.

        If you ever want different strictness for different contexts
        (e.g. embed-only tests), you can:
          - remove this, and
          - add an explicit validate() method instead.
        """
        missing_fields = [k for k, v in self.__dict__.items() if not v]

        if missing_fields:
            missing_env_vars = [self.ENV_VARS[f] for f in missing_fields]
            raise ValueError(f"Missing required environment variables: {missing_env_vars}")

    def summary(self) -> dict:
        """Return a safe, non-sensitive summary for logging."""
        return {
            "storage_account": self.storage_account,
            "openai_base_url": self.openai_base_url,
            "openai_azure_endpoint": self.openai_azure_endpoint,
            "openai_azure_embed_deployment": self.openai_azure_embed_deployment,
            "chroma_endpoint": self.chroma_endpoint,
            "chroma_tenant": self.chroma_tenant,
            "chroma_database": self.chroma_database,
        }
