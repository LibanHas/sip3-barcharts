# Usage:
#   python3 -m scripts.charts_students.make_multiselect_by_school_and_grade
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

# Project helpers
from scripts.common.plotkit import setup, grouped_hbar  # seaborn + font + helper
from scripts.common.canonicalize import SchoolCanonicalizer as SC

# ---- IO ---------------------------------------------------------------------
DATA_LONG   = Path("data/students_multi_long.csv")
CLEAN_CSV   = Path("data/students_clean.csv")
CLEAN_PQ    = Path("data/students_clean.parquet")
OUT_SCHOOL  = Path("figs/students/by_school")
OUT_GRADE   = Path("figs/students/by_grade")
OUT_SCHOOL.mkdir(parents=True, exist_ok=True)
OUT_GRADE.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_RESP_PER_SCHOOL = 2
MIN_RESP_PER_GRADE  = 2

# Only plot these multi-select questions (keys from the long file)
PLOT_KEYS = ["Q_subjects", "Q_br_features", "Q_lp_features"]

# ---- Utils ------------------------------------------------------------------
def _slug(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s))
    s = re.sub(r"[^\w\-]+", "", s)
    return s.strip("_")[:80] or "plot"

def _read_clean_df() -> pd.DataFrame | None:
    if CLEAN_PQ.exists():
        return pd.read_parquet(CLEAN_PQ)
    if CLEAN_CSV.exists():
        return pd.read_csv(CLEAN_CSV, dtype=str, keep_default_na=False)
    return None

def _ensure_columns_present(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = "不明"
    return out

# ---- Load & join ------------------------------------------------------------
def load_data() -> pd.DataFrame:
    # Long-form multi-select rows
    df = pd.read_csv(DATA_LONG, dtype=str, keep_default_na=False)

    # Standardize empties in core columns
    for c in ("respondent_id", "学校名_canon", "学校種_canon", "学年_canon",
              "column_key", "choice"):
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"": "不明", "nan": "不明"}).fillna("不明")

    # Ensure respondent_id exists and is string (for joins)
    if "respondent_id" not in df.columns:
        raise KeyError("students_multi_long.csv must include 'respondent_id'")
    df["respondent_id"] = df["respondent_id"].astype(str)

    # Backfill canon columns from the cleaned file if missing
    need_school = "学校名_canon" not in df.columns
    need_level  = "学校種_canon" not in df.columns
    need_grade  = "学年_canon"   not in df.columns

    clean = _read_clean_df()
    if (need_school or need_level or need_grade) and clean is not None:
        clean = clean.copy()
        if "respondent_id" in clean.columns:
            clean["respondent_id"] = clean["respondent_id"].astype(str)

        # Prefer a canonical school column if present, else raw-ish (we’ll canonicalize anyway)
        school_source = None
        for c in ("学校名_canon", "school_name_canon", "学校名", "school_name"):
            if c in clean.columns:
                school_source = c
                break

        join_cols = ["respondent_id"]
        if school_source is not None:
            join_cols.append(school_source)
        if "学校種_canon" in clean.columns:
            join_cols.append("学校種_canon")
        if "学年_canon" in clean.columns:
            join_cols.append("学年_canon")
        join_cols = list(dict.fromkeys(join_cols))  # de-dupe

        have_all = all(c in clean.columns for c in join_cols)
        if have_all:
            add = clean[join_cols].copy()
            # Standardize the school column name for the merge result
            if school_source is not None and school_source != "学校名_canon":
                add = add.rename(columns={school_source: "学校名_canon"})
            df = df.merge(add, on="respondent_id", how="left", suffixes=("", "_from_clean"))

    # Ensure columns exist even if we couldn’t join
    df = _ensure_columns_present(df, ["学校名_canon", "学校種_canon", "学年_canon"])

    # Create a single source of truth: df['school_canon']
    # If 学校名_canon already exists, SC will still bucket to ALLOWED and map out typos
    df["学校名_canon"] = df["学校名_canon"].astype(str)
    _ = SC.find_or_make_school_canon(df, debug=False)   # -> sets df['school_canon']
    SC.assert_only_allowed(df)

    return df

# ---- Main -------------------------------------------------------------------
def main():
    setup()  # seaborn theme + JP fonts

    df = load_data()

    # ---------------------- 学校別 (by school) ----------------------
    # Count unique respondents per school using the canonical column
    resp_per_school = df.groupby("school_canon")["respondent_id"].nunique()
    ok_schools = set(resp_per_school[resp_per_school >= MIN_RESP_PER_SCHOOL].index)

    for key, g in df[df["column_key"].isin(PLOT_KEYS)].groupby("column_key"):
        g_s = g[g["school_canon"].isin(ok_schools)].copy()
        if g_s.empty:
            continue

        # Titles per key
        title = {
            "Q_subjects":    "使用教科（学生） — 学校別",
            "Q_br_features": "BookRollのよく使う機能（学生） — 学校別",
            "Q_lp_features": "ログパレのよく使う機能（学生） — 学校別",
        }.get(key, f"{key} — 学校別")

        out = OUT_SCHOOL / f"{_slug(key)}_by_school.png"
        # grouped_hbar expects a frame with [group_col, choice]
        grouped_hbar(
            g_s[["school_canon", "choice"]],
            group_col="school_canon",
            title=title,
            outpath=out,
            row_height=1.15,
            bar_width=0.92,
            edgecolor="white",
            linewidth=1.3,
            show_values=True,
            label_min=12,
            write_counts_csv=True,  # writes alongside the PNG
        )
        print(f"[info] wrote {out}")

    # ---------------------- 学年別 (by grade) ----------------------
    # Ensure 学年_canon exists (load_data already tries; this is belt & braces)
    if "学年_canon" not in df.columns:
        clean = _read_clean_df()
        if clean is not None and {"respondent_id", "学年_canon"} <= set(clean.columns):
            clean = clean[["respondent_id", "学年_canon"]].copy()
            clean["respondent_id"] = clean["respondent_id"].astype(str)
            df = df.merge(clean, on="respondent_id", how="left")
        else:
            df["学年_canon"] = "不明"

    resp_per_grade = df.groupby("学年_canon")["respondent_id"].nunique()
    ok_grades = set(resp_per_grade[resp_per_grade >= MIN_RESP_PER_GRADE].index)

    for key, g in df[df["column_key"].isin(PLOT_KEYS)].groupby("column_key"):
        g_g = g[g["学年_canon"].isin(ok_grades)].copy()
        if g_g.empty:
            continue

        title = {
            "Q_subjects":    "使用教科（学生） — 学年別",
            "Q_br_features": "BookRollのよく使う機能（学生） — 学年別",
            "Q_lp_features": "ログパレのよく使う機能（学生） — 学年別",
        }.get(key, f"{key} — 学年別")

        out = OUT_GRADE / f"{_slug(key)}_by_grade.png"
        grouped_hbar(
            g_g[["学年_canon", "choice"]],
            group_col="学年_canon",
            title=title,
            outpath=out,
            row_height=1.15,
            bar_width=0.92,
            edgecolor="white",
            linewidth=1.3,
            show_values=True,
            label_min=12,
            write_counts_csv=True,
        )
        print(f"[info] wrote {out}")

if __name__ == "__main__":
    main()
