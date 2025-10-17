# scripts/sip3_teachers/__init__.py
"""
sip3_teachers: cleaning pipeline for the teacher dataset (sip3-barcharts).
"""

from pathlib import Path

# Repo root (sip3-barcharts/)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Re-export the main entrypoint so you can `from sip3_teachers import run`
from .pipeline import run

__all__ = ["run", "REPO_ROOT"]
__version__ = "0.1.0"
