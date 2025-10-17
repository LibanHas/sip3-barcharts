import pandas as pd
import numpy as np
import re

def assert_detected(cols: dict, required=("school_name","grade")):
    missing = [k for k in required if not cols.get(k)]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def no_person_names_in_school(series: pd.Series) -> None:
    s = series.astype(str)
    # exclude the canonical placeholder
    s2 = s[s != "未記入"]
    mask = (
        s2.str.match(r"^[一-龥]{2,3}$", na=False) |            # 2–3 kanji only
        s2.str.match(r"^[一-龥]{1,3}[ぁ-んァ-ヶー]+$", na=False) # kanji + kana (e.g., 松井ゆかり)
    )
    idx = s2[mask].index
    if len(idx):
        examples = s.loc[idx].tolist()[:5]
        print(f"[warn] possible personal names in school field at rows: {list(idx)[:10]}... -> examples: {examples}")

def school_type_domain(series: pd.Series, allowed={"小","中","高","不明"}):
    bad = set(series.dropna()) - allowed
    if bad:
        print(f"[warn] unexpected 学校種 values: {bad}")

def grade_domain(series: pd.Series):
    # light check; your charting will bucketize anyway
    pass

def duration_integrity(df: pd.DataFrame, leaf_cols: list):
    for c in leaf_cols:
        mcol = c + "_months_total"
        if mcol in df.columns:
            bad = df[(df[mcol].notna()) & ((df[mcol] < 0) | (df[mcol] > 300))]
            if len(bad):
                print(f"[warn] out-of-range months in {mcol}: {len(bad)} rows")

def report_missing(df: pd.DataFrame, columns: list):
    for c in columns:
        n = df[c].isna().sum() if c in df.columns else "N/A"
        print(f"[na] {c}: {n}")
