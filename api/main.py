# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-12-07
# Description: main.py
# -----------------------------------------------------------------------------
import os

from fastapi import FastAPI
from api.routers import health, ifu_stats, query, documents, chat, blobs
import gradio as gr
import logging

from ui.gradio_app import build_gradio_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",)
app = FastAPI(title="IFULLMDEV API")
app.include_router(health.router)
app.include_router(ifu_stats.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(blobs.router)

# Mount Gradio (served by the SAME uvicorn process/port)
API_BASE_URL = os.getenv("IFU_API_BASE_URL", "http://127.0.0.1:8000")
gradio_blocks = build_gradio_app(api_base_url=API_BASE_URL)
app = gr.mount_gradio_app(app, gradio_blocks, path="/ui")