# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-12
# Description: conftest.py
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

# add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))