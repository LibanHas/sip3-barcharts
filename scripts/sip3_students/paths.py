# paths.py
from pathlib import Path

BASE = Path(".").resolve()
DATA_DIR = BASE / "data"
CONFIG_DIR = BASE / "config"

RAW_CSV   = DATA_DIR / "students_raw.csv"
ALIASES   = CONFIG_DIR / "alias_map_students.csv"
OUT_CSV   = DATA_DIR / "students_clean.csv"
OUT_MULTI = DATA_DIR / "students_multi_long.csv"

def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    CONFIG_DIR.mkdir(exist_ok=True, parents=True)
