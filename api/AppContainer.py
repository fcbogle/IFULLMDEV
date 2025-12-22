# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-20
# Description: AppContainer.py
# -----------------------------------------------------------------------------

from config.Config import Config

from chunking.LangDetectDetector import LangDetectDetector
from embedding.IFUEmbedder import IFUEmbedder
from extractor.IFUTextExtractor import IFUTextExtractor
from services.IFUHealthService import IFUHealthService
from services.IFUIngestService import IFUIngestService
from services.IFUQueryService import IFUQueryService
from services.IFUStatsService import IFUStatsService
from health.TestRunner import TestRunner
from loader.IFUDocumentLoader import IFUDocumentLoader
from vectorstore.ChromaIFUVectorStore import ChromaIFUVectorStore


class AppContainer:
    """
    Owns heavy object instantiation and application wiring.
    Singleton instances are provided via FastAPI dependencies.
    """

    def __init__(self) -> None:
        # Configuration
        self.cfg = Config.from_env()

        # Smoke tests / health
        self.test_runner = TestRunner(cfg=self.cfg)

        # Core infrastructure
        self.embedder = IFUEmbedder(cfg=self.cfg)
        self.store = ChromaIFUVectorStore(cfg=self.cfg, embedder=self.embedder)

        # Blob access (thin loader)
        self.document_loader = IFUDocumentLoader(cfg=self.cfg)

        # Infrastructure for ingestion pipeline
        self.lang_detector = LangDetectDetector()
        self.chunker = IFUIngestService.build_default_chunker(lang_detector=self.lang_detector)
        self.extractor = IFUTextExtractor()

        # Return a singleton IFUHealthService instance
        self.health_service = IFUHealthService(test_runner=self.test_runner)

        # Return a singleton IFUIngestService instance
        self.ingest_service = IFUIngestService(
            document_loader=self.document_loader,
            store=self.store,
            embedder=self.embedder,
            chunker=self.chunker,
            extractor=self.extractor,
            collection_name="ifu-docs-test",
        )

        # Return a singleton IFUStatsService instance
        self.stats_service = IFUStatsService(
            document_loader=self.document_loader,
            store=self.store,
            collection_name="ifu-docs-test",
        )

        # Return a singleton IFUQueryService instance
        self.query_service = IFUQueryService(
            store=self.store,
            collection_name="ifu-docs-test",
        )

# Singleton container instance
app_container = AppContainer()

