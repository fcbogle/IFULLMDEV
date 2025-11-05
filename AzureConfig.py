# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-01
# Description: AzureConfig
# -----------------------------------------------------------------------------
import os
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv

@dataclass(frozen=True)
class AzureConfig:
    # Azure Storage
    storage_account: str
    storage_key: str
    # Azure Search
    search_endpoint: str
    search_key: str
    # OpenAI
    openai_endpoint: str
    openai_api_key: str
    openai_embed_deployment: str
    openai_chat_deployment: str | None = None  # optional

    @staticmethod
    def from_env() -> "AzureConfig":
        load_dotenv(find_dotenv(usecwd=True), override=True)
        return AzureConfig(
            storage_account=os.getenv("AZURE_STORAGE_ACCOUNT", ""),
            storage_key=os.getenv("AZURE_STORAGE_KEY", ""),
            search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
            search_key=os.getenv("AZURE_SEARCH_KEY", ""),
            openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            openai_chat_deployment=os.getenv("OPENAI_CHAT_MODEL"),
            openai_embed_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", ""),
        )
