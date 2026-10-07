"""Ensure example scripts can import the in-repo ``diffsynth`` package.

Python imports ``sitecustomize`` automatically during startup when it is
available on ``sys.path``. Because scripts under ``examples/`` run with that
directory on ``sys.path[0]``, this file makes the repository root importable
without requiring callers to export ``PYTHONPATH`` manually.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
