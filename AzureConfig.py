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
    storage_account: str
    storage_key: str
    search_endpoint: str
    search_key: str

    @staticmethod
    def from_env() -> "AzureConfig":
        load_dotenv(find_dotenv(usecwd=True), override=True)
        return AzureConfig(
            storage_account=os.getenv("AZURE_STORAGE_ACCOUNT", ""),
            storage_key=os.getenv("AZURE_STORAGE_KEY", ""),
            search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
            search_key=os.getenv("AZURE_SEARCH_KEY", ""),
        )
