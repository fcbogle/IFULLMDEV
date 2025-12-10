# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: dependencies.py
# -----------------------------------------------------------------------------
from functools import lru_cache

from config.Config import Config
from loader.IFUDocumentLoader import IFUDocumentLoader


@lru_cache
def get_cfg() -> Config:
    return Config.from_env()

@lru_cache
def get_multi_doc_loader() -> IFUDocumentLoader:
    return IFUDocumentLoader(cfg=get_cfg())