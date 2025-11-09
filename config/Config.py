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
    # Azure Storage
    storage_account: str
    storage_key: str

    # OpenAI
    openai_base_url: str
    openai_azure_api_key: str
    openai_azure_embed_deployment: str
    openai_api_key: str
    openai_azure_endpoint: str

    # Chroma Vector Database
    chroma_endpoint: str
    chroma_api_key: str
    chroma_tenant: str
    chroma_database: str

    @staticmethod
    def from_env() -> "Config":
        """Build Config object from environment variables."""
        return Config(
            storage_account=os.getenv("AZURE_STORAGE_ACCOUNT", ""),
            storage_key=os.getenv("AZURE_STORAGE_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_azure_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            openai_azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            openai_azure_embed_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", ""),
            chroma_endpoint=os.getenv("CHROMA_ENDPOINT", ""),
            chroma_api_key=os.getenv("CHROMA_API_KEY", ""),
            chroma_tenant=os.getenv("CHROMA_TENANT", ""),
            chroma_database=os.getenv("CHROMA_DATABASE", ""),
        )

    def __post_init__(self):
        """Fail fast if any required config is missing."""
        missing = [k for k, v in self.__dict__.items() if not v]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

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
