# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: dependencies.py
# -----------------------------------------------------------------------------
from functools import lru_cache

from api.AppContainer import app_container
from config.Config import Config
from services.IFUHealthService import IFUHealthService
from services.IFUQueryService import IFUQueryService
from services.IFUStatsService import IFUStatsService
from loader.IFUDocumentLoader import IFUDocumentLoader

@lru_cache
def get_cfg() -> Config:
    return Config.from_env()
def get_health_service() -> IFUHealthService:
    # use the singleton service from the container
    return app_container.health_service
def get_stats_service() -> IFUStatsService:
    # use the singleton service from the container
    return app_container.stats_service

def get_multi_doc_loader() -> IFUDocumentLoader:
    # already a singleton in app_container; caching is optional
    return app_container.multi_doc_loader

def get_query_service() -> IFUQueryService:
    # use the singleton service from the container
    return app_container.query_service
