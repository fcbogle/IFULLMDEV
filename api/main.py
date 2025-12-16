# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: main.py
# -----------------------------------------------------------------------------
from fastapi import FastAPI
from api.routers import health, ifu_stats

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",)
app = FastAPI(title="IFULLMDEV API")
app.include_router(health.router)
app.include_router(ifu_stats.router)
# app.include_router(ifu_upload.router)
# app.include_router(ifu_ingest.router)
# app.include_router(ifu_docs.router)
# app.include_router(ifu_query.router)