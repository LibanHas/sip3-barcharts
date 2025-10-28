# io_students.py
import re
import pandas as pd
from typing import Iterable
from .paths import RAW_CSV, ALIASES, OUT_CSV, OUT_MULTI, ensure_dirs


NOISE_COL_PATTERNS = [r"^Unnamed", r"^index$", r"^Timestamp$"]
KNOWN_NULL_TOKENS = {"", " ", "-", ".", "無回答", "未記入", "N/A", "na", "NaN", "nan"}

def _drop_noise_cols(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for c in df.columns:
        if any(re.search(p, str(c)) for p in NOISE_COL_PATTERNS):
            continue
        keep.append(c)
    return df[keep]

def _clean_headers(cols: Iterable[str]) -> list[str]:
    # Trim, collapse spaces, remove common suffixes like （複数回答可）
    out = []
    for c in cols:
        nc = re.sub(r"\s+", " ", str(c)).strip()
        nc = re.sub(r"（複数回答可）|\(複数回答可\)", "", nc)
        out.append(nc)
    return out

def load_raw() -> pd.DataFrame:
    ensure_dirs()
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = _clean_headers(df.columns)
    df = _drop_noise_cols(df)
    # Convert known null tokens to pandas NA
    for c in df.columns:
        df[c] = df[c].map(lambda x: pd.NA if str(x).strip() in KNOWN_NULL_TOKENS else x)
    return df

def load_alias_map() -> pd.DataFrame:
    ensure_dirs()
    # expected columns: column, raw, canon
    return pd.read_csv(ALIASES, encoding="utf-8-sig", dtype=str, keep_default_na=False)

def save_clean(df: pd.DataFrame):
    ensure_dirs()
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

def save_multi(df: pd.DataFrame):
    ensure_dirs()
    df.to_csv(OUT_MULTI, index=False, encoding="utf-8")
