# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2026-01-02
# Description: assets.py
# -----------------------------------------------------------------------------
import base64
from pathlib import Path

def img_to_data_uri(path: str) -> str:
    """
    Load a local image file and return a data URI suitable for embedding in HTML.
    Works well for logos/icons in Gradio.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    ext = p.suffix.lower().lstrip(".") or "png"
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "svg": "image/svg+xml",
    }.get(ext, "application/octet-stream")

    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"