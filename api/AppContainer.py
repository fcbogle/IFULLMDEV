# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-20
# Description: AppContainer.py
# -----------------------------------------------------------------------------
from health.TestRunner import TestRunner
from config.Config import Config
from embedding.IFUEmbedder import IFUEmbedder
from loader.IFUDocumentLoader import IFUDocumentLoader
from services.IFUHealthService import IFUHealthService
from services.IFUQueryService import IFUQueryService
from vectorstore.ChromaIFUVectorStore import ChromaIFUVectorStore
from services.IFUStatsService import IFUStatsService


class AppContainer:
    """
    Owns heavy object instantiation and application wiring.

    This container is instantiated once at startup and provides
    singleton instances to FastAPI dependencies.
    """

    def __init__(self) -> None:
        # Configuration
        self.cfg = Config.from_env()

        # Smoke Test TestRunner
        self.test_runner = TestRunner(cfg=self.cfg)
        self.health_service = IFUHealthService(test_runner=self.test_runner)

        # Core infrastructure
        self.embedder = IFUEmbedder(cfg=self.cfg)
        self.store = ChromaIFUVectorStore(cfg=self.cfg, embedder=self.embedder)

        # Provides a singleton IFUDocumentLoader instance
        self.multi_doc_loader = IFUDocumentLoader(
            cfg=self.cfg,
            collection_name="ifu-docs-test",
        )

        # Provides a singleton IFUStatsService instance
        self.stats_service = IFUStatsService(
            loader=self.multi_doc_loader
        )

        # Provides a singleton IFUQueryService instance
        self.query_service = IFUQueryService(
            store=self.store,
            collection_name="ifu-docs-test",
        )

        # Provides a singleton IFUHealthService instance
        self.get_health_service = IFUHealthService(
            test_runner=self.test_runner
        )

# Singleton container instance
app_container = AppContainer()
