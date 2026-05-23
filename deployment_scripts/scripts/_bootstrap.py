"""Make ``src`` importable when scripts are run as ``python scripts/NN_*.py``.

Without this, sys.path doesn't include the project root, so
``from src.modeling import train`` would fail. Each entry script
imports this module first.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
