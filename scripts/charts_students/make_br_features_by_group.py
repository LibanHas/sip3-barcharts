# Usage:
#   python3 -m scripts.charts_students.make_br_features_by_group
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
import pandas as pd

# Project helpers
try:
    from scripts.common.plotkit import setup, grouped_hbar  # seaborn + JP font + helper
except Exception:
    def setup(): return None
    raise

# Centralized canonicalizer
from scripts.common.canonicalize import SchoolCanonicalizer as SC

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/students_clean.csv")
DATA_LONG  = Path("data/students_multi_long.csv")  # optional
OUT_BASE   = Path("figs/students/br_features")
OUT_SCHOOL = OUT_BASE / "by_school"
OUT_GRADE  = OUT_BASE / "by_grade"
OUT_SCHOOL.mkdir(parents=True, exist_ok=True)
OUT_GRADE.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_RESP_PER_SCHOOL = 2
MIN_RESP_PER_GRADE  = 2
TITLE_SCHOOL = "BookRollのよく使う機能（学生） — 学校別"
TITLE_GRADE  = "BookRollのよく使う機能（学生） — 学年別"

# Known column names / detection
BR_MULTI_COL_CANDIDATES = [
    "BookRollでよく使う機能を選んでください(複数選択(せんたく)可)",
    "BookRollでよく使う機能を選んでください（複数選択可）",
    "BookRollでよく使う機能",
]

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()【】\[\]「」『』,:;.-]", "", s)
    return s.lower()

def _find_br_multiselect_col(df: pd.DataFrame) -> str | None:
    # 1) exact/known headers
    for c in BR_MULTI_COL_CANDIDATES:
        if c in df.columns:
            return c
    # 2) fuzzy search
    for c in df.columns:
        nc = _norm(c)
        if "bookroll" in nc and ("機能" in nc or "よく使う" in nc or "選んで" in nc):
            return c
    return None

_SPLIT_RE = re.compile(r"[;,/｜|]|[、]")

def _explode_multiselect_series(s: pd.Series) -> pd.DataFrame:
    """Explode a multiselect column into long form (one row per choice)."""
    # Normalize empties to NaN
    s = s.astype(str).map(lambda x: x if x.strip() else pd.NA)

    # Split tolerant of various delimiters: comma/semicolon/slash/pipe/JP comma
    exploded = (
        s.dropna()
         .map(lambda x: [ud.normalize("NFKC", t).strip() for t in _SPLIT_RE.split(x) if t.strip()])
         .explode()
         .rename("choice")
         .to_frame()
    )
    return exploded

def _load_long_from_multi(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Create a long-form dataframe with columns:
      respondent_id, 学年_canon, 学校名_canon (or school_canon), choice
    """
    # pick BR multiselect
    br_col = _find_br_multiselect_col(df_clean)
    if not br_col:
        raise KeyError("Could not find the BookRoll multiselect column in cleaned data.")

    # Ensure we have respondent_id for per-person de-duplication
    if "respondent_id" not in df_clean.columns:
        # if not present, fabricate a stable row index
        df_clean = df_clean.copy()
        df_clean["respondent_id"] = df_clean.index.astype(str)

    # Columns we want to carry forward
    carry = ["respondent_id", "学校名_canon", "学年_canon"]
    present = [c for c in carry if c in df_clean.columns]
    base = df_clean[present + [br_col]].copy()

    # explode multiselect to long
    exploded = _explode_multiselect_series(base[br_col])
    out = base.join(exploded, how="inner")
    out = out.drop(columns=[br_col])
    return out

def _load_long_from_multi_file(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Use students_multi_long.csv (if present) and filter to BookRoll feature rows.
    Expect columns: respondent_id, column_key, choice, [学校名_canon], [学年_canon]
    """
    if "respondent_id" not in df_long.columns or "choice" not in df_long.columns:
        raise KeyError("students_multi_long.csv is missing 'respondent_id' or 'choice' columns.")

    # heuristic: pick rows that correspond to BookRoll features
    key_col = "column_key" if "column_key" in df_long.columns else None
    if key_col:
        mask = df_long[key_col].isin(["Q_br_features", "BookRoll_features"])
        if not mask.any():
            # fallback: fuzzy check column_key text
            mask = df_long[key_col].astype(str).str.contains("BookRoll|BR|br_features", case=False, na=False)
        use = df_long[mask].copy()
    else:
        # If column_key absent, assume the file was pre-filtered to BookRoll
        use = df_long.copy()

    need_cols = ["respondent_id", "choice"]
    carry = [c for c in ["学校名_canon", "学年_canon"] if c in use.columns]
    use = use[need_cols + carry].copy()
    return use

def load_bookroll_long() -> pd.DataFrame:
    """
    Prefer students_multi_long.csv if present; otherwise explode from students_clean.csv.
    Always return: respondent_id, choice, and (学校名_canon, 学年_canon) if available.
    """
    if DATA_LONG.exists():
        df_long = pd.read_csv(DATA_LONG, dtype=str, keep_default_na=False)
        return _load_long_from_multi_file(df_long)
    else:
        df_clean = pd.read_csv(DATA_CLEAN, dtype=str, keep_default_na=False)
        return _load_long_from_multi(df_clean)

def main():
    setup()

    # --- load long-form choices ---------------------------------------------
    long = load_bookroll_long()

    # Normalize empties -> NaN then drop obviously empty choices
    long = long.replace({"": pd.NA, " ": pd.NA})
    long = long.dropna(subset=["choice"]).copy()

    # Ensure respondent_id is string
    long["respondent_id"] = long["respondent_id"].astype(str)

    # --- Canonical school & grade -------------------------------------------
    # If centralized school canon not present, create it on the merged clean df.
    # We only have long here, so we’ll add/ensure '学校名_canon' via SC.
    # SC.find_or_make_school_canon expects the raw school name to exist somewhere
    # in your cleaned data pipeline; but students_clean already created 学校名_canon.
    # If missing here, we still call SC to make a derived 'school_canon' from 学校名_canon.
    # 1) Start with school
    if "学校名_canon" in long.columns:
        # convert to SC.school_canon
        tmp = long.rename(columns={"学校名_canon": "school_canon"}).copy()
    else:
        # If not present (rare), try to build from DATA_CLEAN by respondent_id
        tmp = long.copy()
        if DATA_CLEAN.exists():
            base = pd.read_csv(DATA_CLEAN, dtype=str, keep_default_na=False)[["respondent_id", "学校名_canon"]]
            tmp = tmp.merge(base.rename(columns={"学校名_canon": "school_canon"}),
                            on="respondent_id", how="left")
        else:
            tmp["school_canon"] = pd.NA

    # Force canonical domain using SC (makes and validates 'school_canon')
    SC.find_or_make_school_canon(tmp, debug=False)   # ensures tmp['school_canon']
    SC.assert_only_allowed(tmp)

    # 2) Grade canon (optional but helpful for labels)
    if "学年_canon" not in tmp.columns:
        if DATA_CLEAN.exists():
            base_g = pd.read_csv(DATA_CLEAN, dtype=str, keep_default_na=False)[["respondent_id", "学年_canon"]]
            tmp = tmp.merge(base_g, on="respondent_id", how="left")
        else:
            tmp["学年_canon"] = "不明"

    # Deduplicate (respondent may have selected same choice multiple times)
    tmp = tmp.drop_duplicates(subset=["respondent_id", "school_canon", "学年_canon", "choice"])

    # ---- BY SCHOOL ----------------------------------------------------------
    resp_per_school = tmp.groupby("school_canon")["respondent_id"].nunique()
    ok_schools = set(resp_per_school[resp_per_school >= MIN_RESP_PER_SCHOOL].index)
    df_school = tmp[tmp["school_canon"].isin(ok_schools)].copy()
    if not df_school.empty:
        # Title & output
        out_png = OUT_SCHOOL / "bookroll_features_by_school.png"
        grouped_hbar(
            df_school[["school_canon", "choice"]],
            group_col="school_canon",
            title=TITLE_SCHOOL,
            outpath=out_png,
            row_height=1.15,
            bar_width=0.92,
            edgecolor="white",
            linewidth=1.3,
            show_values=True,
            label_min=12,
            write_counts_csv=True,  # writes alongside PNG with *_counts.csv
        )
        print(f"[info] wrote {out_png}")
    else:
        print("[warn] No schools met MIN_RESP_PER_SCHOOL; skipping by-school plot.")

    # ---- BY GRADE -----------------------------------------------------------
    resp_per_grade = tmp.groupby("学年_canon")["respondent_id"].nunique()
    ok_grades = set(resp_per_grade[resp_per_grade >= MIN_RESP_PER_GRADE].index)
    df_grade = tmp[tmp["学年_canon"].isin(ok_grades)].copy()
    if not df_grade.empty:
        out_png = OUT_GRADE / "bookroll_features_by_grade.png"
        grouped_hbar(
            df_grade[["学年_canon", "choice"]],
            group_col="学年_canon",
            title=TITLE_GRADE,
            outpath=out_png,
            row_height=1.15,
            bar_width=0.92,
            edgecolor="white",
            linewidth=1.3,
            show_values=True,
            label_min=12,
            write_counts_csv=True,
        )
        print(f"[info] wrote {out_png}")
    else:
        print("[warn] No grades met MIN_RESP_PER_GRADE; skipping by-grade plot.")

if __name__ == "__main__":
    main()
