# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2026-01-05
# Description: sitecustomize.py
# -----------------------------------------------------------------------------
# sitecustomize.py
"""
Ensure Chroma sees a new-enough sqlite3 on Azure App Service.

This file is auto-imported by Python at startup if it's on sys.path.
"""
import sys

try:
    import pysqlite3  # from pysqlite3-binary
    sys.modules["sqlite3"] = pysqlite3
except Exception as e:
    # If this fails, Chroma will likely fail later too.
    # Keep it simple: don't crash here; allow logs to show the real failure.
    pass

# Optional debug (safe): prove which sqlite module/version is in use
try:
    import sqlite3
    ver = getattr(sqlite3, "sqlite_version", None)
    mod = getattr(sqlite3, "__file__", None)
    print(f"[sitecustomize] sqlite_version={ver} sqlite_module={mod}")
except Exception:
    pass
