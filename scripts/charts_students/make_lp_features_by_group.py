# Usage:
#   python3 -m scripts.charts_students.make_lp_features_by_group
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
DATA_LONG  = Path("data/students_multi_long.csv")  # optional, preferred if present
OUT_BASE   = Path("figs/students/lp_features")
OUT_SCHOOL = OUT_BASE / "by_school"
OUT_GRADE  = OUT_BASE / "by_grade"
OUT_SCHOOL.mkdir(parents=True, exist_ok=True)
OUT_GRADE.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_RESP_PER_SCHOOL = 2
MIN_RESP_PER_GRADE  = 2
TITLE_SCHOOL = "ログパレのよく使う機能（学生） — 学校別"
TITLE_GRADE  = "ログパレのよく使う機能（学生） — 学年別"

# ---- Column detection -------------------------------------------------------
LP_MULTI_COL_CANDIDATES = [
    "分析ツール(ログパレ)でよく使う機能を選んでください(複数選択(せんたく)可)",
    "分析ツール(ログパレ)でよく使う機能を選んでください（複数選択可）",
    "分析ツール（ログパレ）でよく使う機能を選んでください",
    "分析ツールでよく使う機能を選んでください",
    "ログパレでよく使う機能",
]

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()【】\[\]「」『』,:;.-]", "", s)
    return s.lower()

def _find_lp_multiselect_col(df: pd.DataFrame) -> str | None:
    # 1) exact/known headers
    for c in LP_MULTI_COL_CANDIDATES:
        if c in df.columns:
            return c
    # 2) fuzzy search
    for c in df.columns:
        nc = _norm(c)
        if ("分析" in nc or "ログパレ" in nc or "logpalette" in nc or "lp" in nc) and ("機能" in nc or "選んで" in nc or "よく使う" in nc):
            return c
    return None

# ---- Multiselect explode ----------------------------------------------------
_SPLIT_RE = re.compile(r"[;,/｜|]|[、]")

def _explode_multiselect_series(s: pd.Series) -> pd.DataFrame:
    """Explode a multiselect column into long form (one row per choice)."""
    s = s.astype(str).map(lambda x: x if x.strip() else pd.NA)
    exploded = (
        s.dropna()
         .map(lambda x: [ud.normalize("NFKC", t).strip() for t in _SPLIT_RE.split(x) if t.strip()])
         .explode()
         .rename("choice")
         .to_frame()
    )
    return exploded

# ---- Loaders ----------------------------------------------------------------
def _load_long_from_multi(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Create a long-form dataframe with columns:
      respondent_id, 学年_canon, 学校名_canon (or school_canon), choice
    using the LP multiselect column from students_clean.csv.
    """
    lp_col = _find_lp_multiselect_col(df_clean)
    if not lp_col:
        raise KeyError("Could not find the LogPalette multiselect column in cleaned data.")

    if "respondent_id" not in df_clean.columns:
        df_clean = df_clean.copy()
        df_clean["respondent_id"] = df_clean.index.astype(str)

    carry = ["respondent_id", "学校名_canon", "学年_canon"]
    present = [c for c in carry if c in df_clean.columns]
    base = df_clean[present + [lp_col]].copy()

    exploded = _explode_multiselect_series(base[lp_col])
    out = base.join(exploded, how="inner").drop(columns=[lp_col])
    return out

def _load_long_from_multi_file(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Use students_multi_long.csv (if present) and filter to LogPalette feature rows.
    Expect columns: respondent_id, column_key, choice, [学校名_canon], [学年_canon]
    """
    if "respondent_id" not in df_long.columns or "choice" not in df_long.columns:
        raise KeyError("students_multi_long.csv is missing 'respondent_id' or 'choice' columns.")

    key_col = "column_key" if "column_key" in df_long.columns else None
    if key_col:
        mask = df_long[key_col].isin(["Q_lp_features", "LogPalette_features", "LP_features"])
        if not mask.any():
            mask = df_long[key_col].astype(str).str.contains("LogPalette|ログパレ|lp_features", case=False, na=False)
        use = df_long[mask].copy()
    else:
        use = df_long.copy()

    need_cols = ["respondent_id", "choice"]
    carry = [c for c in ["学校名_canon", "学年_canon"] if c in use.columns]
    return use[need_cols + carry].copy()

def load_lp_long() -> pd.DataFrame:
    """
    Prefer students_multi_long.csv if present; otherwise explode from students_clean.csv.
    """
    if DATA_LONG.exists():
        df_long = pd.read_csv(DATA_LONG, dtype=str, keep_default_na=False)
        return _load_long_from_multi_file(df_long)
    else:
        df_clean = pd.read_csv(DATA_CLEAN, dtype=str, keep_default_na=False)
        return _load_long_from_multi(df_clean)

# ---- Optional: choice aliasing (merge near-duplicates) ----------------------
# If you later want to canonicalize LP feature labels, populate this dict
# (or load from a CSV similar to alias_map_students.csv).
CHOICE_ALIAS = {
    # e.g. "ヒートマップ": "ヒートマップ（閲覧）",
    #      "Heatmap": "ヒートマップ（閲覧）",
}
def canonicalize_choice(s: str) -> str:
    return CHOICE_ALIAS.get(s, s)

# ---- Main -------------------------------------------------------------------
def main():
    setup()

    # Load long-form LP choices
    long = load_lp_long().replace({"": pd.NA, " ": pd.NA})
    long = long.dropna(subset=["choice"]).copy()
    long["respondent_id"] = long["respondent_id"].astype(str)

    # Canonical school (via central canonicalizer)
    if "学校名_canon" in long.columns:
        tmp = long.rename(columns={"学校名_canon": "school_canon"}).copy()
    else:
        tmp = long.copy()
        if DATA_CLEAN.exists():
            base = pd.read_csv(DATA_CLEAN, dtype=str, keep_default_na=False)[["respondent_id", "学校名_canon"]]
            tmp = tmp.merge(base.rename(columns={"学校名_canon": "school_canon"}),
                            on="respondent_id", how="left")
        else:
            tmp["school_canon"] = pd.NA

    SC.find_or_make_school_canon(tmp, debug=False)  # ensures tmp['school_canon']
    SC.assert_only_allowed(tmp)

    # Grade canon
    if "学年_canon" not in tmp.columns:
        if DATA_CLEAN.exists():
            base_g = pd.read_csv(DATA_CLEAN, dtype=str, keep_default_na=False)[["respondent_id", "学年_canon"]]
            tmp = tmp.merge(base_g, on="respondent_id", how="left")
        else:
            tmp["学年_canon"] = "不明"

    # Optional: canonicalize feature labels
    tmp["choice"] = tmp["choice"].map(canonicalize_choice)

    # Deduplicate respondent-choice within grouping keys
    tmp = tmp.drop_duplicates(subset=["respondent_id", "school_canon", "学年_canon", "choice"])

    # ---- BY SCHOOL ----------------------------------------------------------
    resp_per_school = tmp.groupby("school_canon")["respondent_id"].nunique()
    ok_schools = set(resp_per_school[resp_per_school >= MIN_RESP_PER_SCHOOL].index)
    df_school = tmp[tmp["school_canon"].isin(ok_schools)].copy()

    if not df_school.empty:
        out_png = OUT_SCHOOL / "logpalette_features_by_school.png"
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
            write_counts_csv=True,  # also writes *_counts.csv next to PNG
        )
        print(f"[info] wrote {out_png}")
    else:
        print("[warn] No schools met MIN_RESP_PER_SCHOOL; skipping by-school plot.")

    # ---- BY GRADE -----------------------------------------------------------
    resp_per_grade = tmp.groupby("学年_canon")["respondent_id"].nunique()
    ok_grades = set(resp_per_grade[resp_per_grade >= MIN_RESP_PER_GRADE].index)
    df_grade = tmp[tmp["学年_canon"].isin(ok_grades)].copy()

    if not df_grade.empty:
        out_png = OUT_GRADE / "logpalette_features_by_grade.png"
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
