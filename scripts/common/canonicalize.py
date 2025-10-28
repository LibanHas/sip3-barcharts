# scripts/common/canonicalize.py
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
import pandas as pd
import numpy as np
from typing import Iterable, Optional, List

# ---------- Generic alias applier (your original helper) ----------
def apply_aliases(df: pd.DataFrame, alias_df: pd.DataFrame) -> pd.DataFrame:
    """
    alias_df schema: column, raw, canon
    For each (column), map df[column] values from raw -> canon into a new column f"{column}_canon".
    If a value does not match, leave original as-is (or NA) and copy to *_canon.
    """
    out = df.copy()
    grouped = alias_df.groupby("column")
    for col, sub in grouped:
        if col not in out.columns:
            continue
        mapping = dict(zip(sub["raw"].astype(str), sub["canon"].astype(str)))
        canon_col = f"{col}_canon"
        out[canon_col] = out[col].astype("string")
        out[canon_col] = out[canon_col].map(mapping).fillna(out[canon_col])
        out[canon_col] = out[canon_col].replace({"": pd.NA, " ": pd.NA})
    return out


# ---------- School canonicalization (single source of truth) ----------
class SchoolCanonicalizer:
    """
    Single source of truth for school-name canonicalization.
    - Prefers existing canonical columns if present
    - Else derives df['school_canon'] with minimal fixes + overrides
    """

    # Keep in sync across the project
    ALLOWED_SCHOOLS: List[str] = [
        "不明",
        "北海道天塩高等学校",
        "岩沼小学校",
        "洗足学園中学校",
        "西京高等学校",
        "西京高等学校付属中学校",
        "西賀茂中学校",
        "北海道寿都高等学校",  # explicitly included
    ]

    CANONICAL_COLS: List[str] = ["学校名_canon", "school_name_canon", "school_canon"]

    # Central override registry (merged from your latest working set)
    OVERRIDES = {
        # Base set
        "洗足学園中学校高校": "洗足学園中学校",
        "立岩沼小学校": "岩沼小学校",
        "天塩高等学校": "北海道天塩高等学校",
        "西賀茂中学校校": "西賀茂中学校",
        "西京": "西京高等学校",
        "西京中学校付属高等学校": "西京高等学校付属中学校",
        "西京中学校": "西京高等学校付属中学校",
        "西京中学": "西京高等学校付属中学校",

        # Common variants / short forms
        "西京高校": "西京高等学校",
        "洗足中学校": "洗足学園中学校",
        "洗足中": "洗足学園中学校",
        "天塩高校": "北海道天塩高等学校",
        "西賀茂": "西賀茂中学校",
        "洗足学園中学校・高等学校": "洗足学園中学校",
        "洗足学園中学高等学校": "洗足学園中学校",

        # Diagnostics-driven (from your logs)
        "さいきょう": "西京高等学校",
        "西賀茂小学校": "西賀茂中学校",
        "加茂中": "西賀茂中学校",
        "寿都高等学校": "北海道寿都高等学校",
        "北海道寿都高等学校": "北海道寿都高等学校",
        "洗足学園中学校高等学校": "洗足学園中学校",
        "西京付属中": "西京高等学校付属中学校",
        "西京高等学校付属": "西京高等学校付属中学校",
        "西京中学校学校付属高等学校": "西京高等学校付属中学校",
        "岩沼小学生": "岩沼小学校",
        "西賀も中学校": "西賀茂中学校",
        "nisigamo": "西賀茂中学校",
    }

    @staticmethod
    def _base_clean(s: str) -> str:
        """Minimal fixes only: NFKC width normalize, 附属→付属, collapse '学校校$' → '学校'."""
        t = ud.normalize("NFKC", str(s)).strip()
        if not t:
            return t
        t = t.replace("附属", "付属")
        t = re.sub(r"学校校$", "学校", t)
        return t

    @classmethod
    def normalize(cls, name: str) -> str:
        """Return canonical name if matchable; else '不明'."""
        if pd.isna(name):
            return "不明"
        t = cls._base_clean(str(name))
        if t in cls.OVERRIDES:
            t = cls.OVERRIDES[t]
        return t if t in cls.ALLOWED_SCHOOLS else "不明"

    @classmethod
    def find_or_make_school_canon(cls, df: pd.DataFrame, debug: bool = False) -> str:
        """
        Ensure df['school_canon'] exists (prefer a canonical source column if present).
        Returns the column name ('school_canon').
        """
        cols = list(df.columns)
        if debug:
            print("[DEBUG] columns:", cols)

        # Prefer existing canonical columns
        for c in cls.CANONICAL_COLS:
            if c in df.columns:
                if debug:
                    print("[DEBUG] using canonical column:", c)
                df["school_canon"] = (
                    df[c]
                    .astype(str)
                    .map(lambda x: x if x in cls.ALLOWED_SCHOOLS else cls.normalize(x))
                    .fillna("不明")
                )
                return "school_canon"

        # Otherwise derive from a raw-ish column
        raw = None
        for c in ("学校名", "school_name"):
            if c in df.columns:
                raw = c
                break
        if raw is None:
            for c in cols:
                s = str(c)
                if "学校" in s and "名" in s:
                    raw = c
                    break
        if raw is None:
            raise KeyError("Could not find a school-name column.")

        if debug:
            print("[DEBUG] deriving from column:", raw)

        df["school_canon"] = df[raw].astype(str).map(cls.normalize).fillna("不明")
        return "school_canon"

    @classmethod
    def assert_only_allowed(cls, df: pd.DataFrame):
        uniq = set(df["school_canon"].dropna().unique())
        extra = uniq - set(cls.ALLOWED_SCHOOLS)
        if extra:
            raise AssertionError(f"Unexpected school labels present: {extra}")

    @staticmethod
    def debug_fumei(
        df: pd.DataFrame,
        raw_cols: Optional[Iterable[str]] = None,
        topn: int = 20,
        q4_col: Optional[str] = None,
        q6_col: Optional[str] = None,
    ) -> None:
        """
        Print where 「不明」 is coming from and whether those rows have Q4/Q6 filled.
        """
        fumei = df[df["school_canon"] == "不明"].copy()
        print(f"[DEBUG] 不明 rows: {len(fumei)}")

        if raw_cols is None:
            raw_cols = []
            for c in df.columns:
                s = str(c)
                if ("学校名" in s) or (s in ("学校名", "school_name")) or (s.endswith("_canon") and "学校" in s):
                    raw_cols.append(c)

        print("[DEBUG] candidate raw school columns:", raw_cols)
        for col in raw_cols:
            vc = fumei[col].value_counts(dropna=False).head(topn)
            if len(vc):
                print(f"\n[DEBUG] Top values mapping to 不明 from {col}:")
                print(vc)

        if q4_col:
            ok_q4 = fumei[q4_col].notna().sum()
            print(f"\n[DEBUG] 不明 rows with non-null Q4: {ok_q4}/{len(fumei)}")
        if q6_col:
            ok_q6 = fumei[q6_col].notna().sum()
            print(f"[DEBUG] 不明 rows with non-null Q6: {ok_q6}/{len(fumei)}")

        out = Path("figs/debug_students_fumei.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        fumei.to_csv(out, index=False)
        print(f"[info] wrote {out}")
