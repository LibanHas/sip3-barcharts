# scripts/sip3_teachers/config.py
from pathlib import Path

# repo root = sip3-barcharts/
REPO_ROOT = Path(__file__).resolve().parents[2]

BASE = REPO_ROOT
RAW_CSV   = REPO_ROOT / "data/teachers_raw.csv"
OUT_CSV   = REPO_ROOT / "data/teachers_clean.csv"
OUT_MULTI = REPO_ROOT / "data/teachers_multi_long.csv"

# Optional alias file for multi-select choice normalization
CHOICE_ALIASES = REPO_ROOT / "config/alias_map_choices_teachers.csv"

# Sanity cap for months parsing (avoid extreme outliers)
MAX_MONTHS = 300
