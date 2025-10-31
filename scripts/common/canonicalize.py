# scripts/common/canonicalize.py
from __future__ import annotations

from pathlib import Path
import re
import unicodedata as ud
from typing import Iterable, Optional, List, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "apply_aliases",
    "SchoolCanonicalizer",
    "post_disambiguate_middle_vs_high",
]

# -------------------- Single alias source ------------------------------------
ALIAS_SCHOOLS = Path("config/alias_schools.csv")

# Column name variants we support in the alias CSV
_ALIAS_COL_COLUMN        = ("column", "カラム", "列", "対象列")
_ALIAS_COL_RAW           = ("raw", "RAW", "値", "入力", "variant", "入力値")
_ALIAS_COL_CANON         = ("canon", "canonical", "正規化", "canonical_name", "正規名")
_ALIAS_COL_OPTIONAL_NOTE = ("note", "notes", "備考", "メモ")


def _read_alias_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path or not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str).replace({"": np.nan})
    except Exception:
        return None


def _first_present(cols: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    pool = {c: True for c in cols if isinstance(c, str)}
    for cand in candidates:
        if cand in pool:
            return cand
    return None


def _coerce_alias_schema(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Normalize alias CSV to: columns = ["column","raw","canon","note"].

    Supports:
      A) 'column,raw,canon,(note)'
      B) Block form with first header literally '学校名' (then raw, canon, note)
    """
    cols = list(df.columns)

    # Case A: unified header
    col_column = _first_present(cols, _ALIAS_COL_COLUMN)
    col_raw    = _first_present(cols, _ALIAS_COL_RAW)
    col_canon  = _first_present(cols, _ALIAS_COL_CANON)
    col_note   = _first_present(cols, _ALIAS_COL_OPTIONAL_NOTE)

    if col_column and col_raw and col_canon:
        out = pd.DataFrame({
            "column": df[col_column].astype(str),
            "raw":    df[col_raw].astype(str),
            "canon":  df[col_canon].astype(str),
            "note":   (df[col_note].astype(str) if col_note else pd.Series([np.nan]*len(df), dtype="string")),
        })
        return out

    # Case B: block rows like: 学校名, raw, canon, note
    if len(cols) >= 3 and cols[0] in ("学校名", "学校名（入力）", "学校名_raw"):
        raw_col   = cols[1] if len(cols) > 1 else None
        canon_col = cols[2] if len(cols) > 2 else None
        note_col  = cols[3] if len(cols) > 3 else None
        if raw_col and canon_col:
            out = pd.DataFrame({
                "column": "学校名",
                "raw":    df[raw_col].astype(str),
                "canon":  df[canon_col].astype(str),
                "note":   (df[note_col].astype(str) if note_col in df.columns else pd.Series([np.nan]*len(df), dtype="string")),
            })
            return out

    return None


def _load_school_alias_table() -> Optional[pd.DataFrame]:
    raw = _read_alias_csv(ALIAS_SCHOOLS)
    if raw is None:
        return None
    ali = _coerce_alias_schema(raw)
    if ali is None:
        # Heuristic fallback: keep only rows where column contains 学校名
        if {"column", "raw", "canon"}.issubset(set(raw.columns)):
            sub = raw[raw["column"].astype(str).str.contains("学校名", na=False)]
            if len(sub):
                ali = sub[["column", "raw", "canon"]].copy()
                ali["note"] = np.nan
    if ali is not None:
        # Keep only school-name rows
        ali = ali[ali["column"].astype(str).str.contains("学校名", na=False)].copy()
        if len(ali) == 0:
            return None
    return ali


def _gather_allowed_schools() -> List[str]:
    ali = _load_school_alias_table()
    allowed = set(["不明"])
    if ali is not None and "canon" in ali.columns:
        canon = (
            ali["canon"]
            .dropna()
            .map(lambda s: ud.normalize("NFKC", str(s)).strip())
        )
        allowed.update([c for c in canon if c])
    return sorted(allowed)


# -------------------- Generic alias applier ----------------------------------
def apply_aliases(df: pd.DataFrame, alias_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply mappings from an alias table to dataframe columns.

    alias_df schema (normalized): ["column","raw","canon","note"]
    For each distinct 'column' in alias_df, map df[column] values raw->canon
    into a new column f"{column}_canon". Non-matches keep original value.
    """
    out = df.copy()
    if alias_df is None or alias_df.empty:
        return out
    for col, sub in alias_df.groupby("column"):
        if col not in out.columns:
            continue
        mapping  = dict(zip(sub["raw"].astype(str), sub["canon"].astype(str)))
        canoncol = f"{col}_canon"
        out[canoncol] = out[col].astype("string")
        out[canoncol] = out[canoncol].map(mapping).fillna(out[canoncol])
        out[canoncol] = out[canoncol].replace({"": pd.NA, " ": pd.NA})
    return out


# -------------------- School canonicalization (dynamic) ----------------------
class SchoolCanonicalizer:
    """
    Single source of truth for school-name canonicalization (teachers + students).
    Uses a single alias file: config/alias_schools.csv
    """

    _ALLOWED_CACHE: Optional[List[str]] = None
    CANONICAL_COLS: List[str] = ["学校名_canon", "school_name_canon", "school_canon"]

    # Hand-tuned fixes
    OVERRIDES = {
        # Senzoku Gakuen variants -> 中学校 (JHS)
        "洗足学園中学校高校": "洗足学園中学校",
        "洗足学園中学校・高等学校": "洗足学園中学校",
        "洗足学園中学高等学校": "洗足学園中学校",
        "洗足学園中学校高等学校": "洗足学園中学校",
        "洗足中学校": "洗足学園中学校",
        "洗足中": "洗足学園中学校",
        "私立洗足学園中学校": "洗足学園中学校",
        "私立洗足学園中学高等学校": "洗足学園中学校",

        # Hokkaido schools
        "天塩高等学校": "北海道天塩高等学校",
        "天塩高校": "北海道天塩高等学校",
        "北海道天塩高校": "北海道天塩高等学校",
        "寿都高等学校": "北海道寿都高等学校",

        # Nishikyo HS & JHS
        "西京": "西京高等学校",
        "西京高校": "西京高等学校",
        "西京付属中": "西京高等学校付属中学校",
        "西京中学校付属高等学校": "西京高等学校付属中学校",
        "西京中学校": "西京高等学校付属中学校",
        "西京中学校学校付属高等学校": "西京高等学校付属中学校",
        "京都市立西京高等学校附属中学校": "西京高等学校付属中学校",
        "京都市立西京附属中学校": "西京高等学校付属中学校",
        "京都市立西京高校附属中学校": "西京高等学校付属中学校",
        "京都市西京高等学校附属中学校": "西京高等学校付属中学校",

        # Nishigamo JHS
        "西賀茂": "西賀茂中学校",
        "西賀茂中学校校": "西賀茂中学校",
        "西賀も中学校": "西賀茂中学校",
        "nisigamo": "西賀茂中学校",
        "京都市立西賀茂中学校": "西賀茂中学校",

        # Other fixes
        "岩沼小学生": "岩沼小学校",
        "立岩沼小学校": "岩沼小学校",
        "岩沼市立岩沼小学校": "岩沼小学校",
        "京都市立明徳小学校": "明徳小学校",
        "京都市立西京付属中学校": "西京高等学校付属中学校",
        "京都市立西京高校付属中学校": "西京高等学校付属中学校",
    }

    @classmethod
    def _allowed_schools(cls) -> List[str]:
        if cls._ALLOWED_CACHE is not None:
            return cls._ALLOWED_CACHE
        allowed = set(_gather_allowed_schools())
        # Ensure both '不明' and the explicit HS label are always accepted
        allowed.update({"不明", "洗足学園高等学校"})
        cls._ALLOWED_CACHE = sorted(allowed)
        return cls._ALLOWED_CACHE

    @staticmethod
    def _base_clean(s: str) -> str:
        t = ud.normalize("NFKC", str(s)).strip()
        if not t:
            return t
        t = t.replace("附属", "付属")
        t = re.sub(r"\s+", "", t)         # remove all whitespace inside
        t = re.sub(r"学校校$", "学校", t)   # collapse duplicated suffix
        return t

    @classmethod
    def normalize(cls, name: str) -> str:
        if pd.isna(name):
            return "不明"
        t = cls._base_clean(str(name))
        if t in cls._allowed_schools():
            return t
        if t in cls.OVERRIDES:
            t2 = cls.OVERRIDES[t]
            return t2 if t2 in cls._allowed_schools() else "不明"
        # Strip owner prefixes commonly seen
        t3 = re.sub(r"^(京都市立|岩沼市立|私立|北海道立)", "", t)
        if t3 in cls.OVERRIDES:
            t3 = cls.OVERRIDES[t3]
        return t3 if t3 in cls._allowed_schools() else "不明"

    @classmethod
    def find_or_make_school_canon(cls, df: pd.DataFrame, debug: bool = False) -> str:
        if debug:
            print("[DEBUG] allowed schools:", cls._allowed_schools())
            print("[DEBUG] columns:", list(df.columns))

        # Prefer an existing canonical column; re-normalize into our allowlist
        for c in cls.CANONICAL_COLS:
            if c in df.columns:
                df["school_canon"] = (
                    df[c].astype(str)
                      .map(lambda x: x if x in cls._allowed_schools() else cls.normalize(x))
                      .fillna("不明")
                )
                return "school_canon"

        # Otherwise derive from the best raw school column we can find
        raw = None
        for c in ("学校名", "school_name"):
            if c in df.columns:
                raw = c
                break
        if raw is None:
            # heuristic search
            for c in df.columns:
                s = str(c)
                if ("学校" in s and "名" in s) or s.endswith("_school") or s.endswith("_学校名"):
                    raw = c
                    break
        if raw is None:
            raise KeyError("Could not find a school-name column.")

        df["school_canon"] = df[raw].astype(str).map(cls.normalize).fillna("不明")
        return "school_canon"

    @classmethod
    def assert_only_allowed(cls, df: pd.DataFrame):
        uniq = set(df["school_canon"].dropna().unique())
        extra = uniq - set(cls._allowed_schools())
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
        Print diagnostics for rows whose school canonicalization is '不明'.
        Optionally, show whether those rows answered Q4/Q6.
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


# -------------------- Post-canonicalization disambiguation -------------------
def post_disambiguate_middle_vs_high(df: pd.DataFrame) -> None:
    """
    If school_canon == '洗足学園中学校' but 学校種 indicates a high school,
    promote the canonical label to '洗足学園高等学校'.
    """
    if "school_canon" not in df.columns or "学校種" not in df.columns:
        return
    typ = df["学校種"].astype(str)
    is_senzoku = df["school_canon"].eq("洗足学園中学校")
    to_hs = is_senzoku & typ.str.contains(r"^高", na=False)
    df.loc[to_hs, "school_canon"] = "洗足学園高等学校"
