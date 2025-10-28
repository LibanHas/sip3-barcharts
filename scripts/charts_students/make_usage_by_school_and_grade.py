# Usage:
#   python3 -m scripts.charts_students.make_usage_by_school_and_grade
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

# Project helpers
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# Centralized canonicalizer
from scripts.common.canonicalize import SchoolCanonicalizer as SC

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/students_clean.csv")
OUT_DIR    = Path("figs/students/usage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_RESP_PER_SCHOOL = 2
MIN_RESP_PER_GRADE  = 2

TITLE_SCHOOL = "LEAFの平均利用期間（月） — 学校別（学生）"
TITLE_GRADE  = "LEAFの平均利用期間（月） — 学年別（学生）"

OUT_SCHOOL_PNG = OUT_DIR / "avg_duration_by_school.png"
OUT_SCHOOL_CSV = OUT_DIR / "avg_duration_by_school.csv"
OUT_GRADE_PNG  = OUT_DIR / "avg_duration_by_grade.png"
OUT_GRADE_CSV  = OUT_DIR / "avg_duration_by_grade.csv"

# ---- Plot style -------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

# ---- Column finders ---------------------------------------------------------
def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()【】\[\]「」『』,:;.-]", "", s)
    return s.lower()

def _find_col(df, exact_jp: str = None, must_contain=(), contains=None):
    cols = list(df.columns)
    norm_cols = {c: _norm(c) for c in cols}
    if exact_jp:
        target = _norm(exact_jp)
        for c, nc in norm_cols.items():
            if nc == target:
                return c
    if must_contain:
        tokens = [_norm(t) for t in must_contain]
        for c, nc in norm_cols.items():
            if all(t in nc for t in tokens):
                return c
    if contains:
        token = _norm(contains)
        for c, nc in norm_cols.items():
            if token in nc:
                return c
    return None

def _find_usage_col(df):
    # Q5 usage frequency (for label overlay)
    candidates = [
        "LEAFシステム(BookRoll,分析(ぶんせき)ツール)を授業・授業外(宿題など)でどれくらい利用していますか",
        "LEAFシステムの利用頻度について教えてください",
        "Q5_利用頻度",
        "利用頻度",
        "システム利用頻度",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        nc = _norm(col)
        if ("利用頻度" in nc) or ("利用していますか" in nc):
            return col
    return None

# ---- Q4 parse: 'X/Y' -> total months ---------------------------------------
_YM_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")

def parse_year_month_to_months(x):
    if pd.isna(x):
        return np.nan
    s = ud.normalize("NFKC", str(x)).strip()
    m = _YM_RE.match(s)
    if m:
        years = int(m.group(1)); months = int(m.group(2))
        return years * 12 + months
    # allow bare numbers as months (fallback)
    if re.fullmatch(r"\d+", s):
        return float(s)
    return np.nan

# ---- Q5 map: frequency -> ordinal score ------------------------------------
FREQ_MAP = {
    "ほぼ毎時間": 4,
    "1週間に数回程度": 3,
    "1ヶ月に数回程度": 2,
    "ほとんど使用していない": 1,
}
def map_freq(x):
    if pd.isna(x):
        return np.nan
    s = ud.normalize("NFKC", str(x)).strip()
    return FREQ_MAP.get(s, np.nan)

# ---- Grade helpers ----------------------------------------------------------
def _find_grade_canon(df: pd.DataFrame) -> str:
    """
    Prefer a canonical grade column if present.
    Otherwise try to find a '学年' column heuristically.
    """
    # If your canonicalizer has a grade function, try it first
    if hasattr(SC, "find_or_make_grade_canon"):
        col = SC.find_or_make_grade_canon(df, debug=False)  # may set df['grade_canon']
        if isinstance(col, str):
            return col
        if "grade_canon" in df.columns:
            return "grade_canon"

    # Common canonical column
    if "学年_canon" in df.columns:
        return "学年_canon"

    # Raw fallback
    likely = [
        "あなたの学年を教えてください",
        "学年",
    ]
    for c in likely:
        if c in df.columns:
            return c

    # heuristic search
    for c in df.columns:
        nc = _norm(c)
        if "学年" in nc:
            return c

    # If nothing, create a dummy (all 不明)
    df["学年_canon"] = "不明"
    return "学年_canon"

# ---- Plotting (shared) ------------------------------------------------------
def _plot_by_group(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    out_png: Path,
    out_csv: Path,
    min_n: int,
    annotate_with_freq: bool = True,
    deemphasize_unknown_label: str = "不明",
):
    # Count respondents with valid duration
    resp_per_group = df.groupby(group_col)["use_months_total"].apply(lambda s: s.notna().sum())
    ok_groups = resp_per_group[resp_per_group >= min_n].index

    # Aggregate
    g = (
        df[df[group_col].isin(ok_groups)]
        .groupby(group_col)
        .agg(
            respondents=("use_months_total", lambda s: int(s.notna().sum())),
            avg_months=("use_months_total", "mean"),
            avg_freq=("freq_score", "mean"),
        )
        .sort_values("avg_months", ascending=False)
    )

    # Save CSV
    g.to_csv(out_csv, encoding="utf-8", index=True)

    # Plot prep
    labels = g.index.tolist()
    y_pos  = np.arange(len(labels))
    x_vals = g["avg_months"].fillna(0.0).values

    fig_h = max(3.8, 0.7 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    # Style
    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    bars = ax.barh(y_pos, x_vals, height=0.6, zorder=2, edgecolor="none")

    # Axes
    ax.set_yticks(y_pos, labels=labels, fontsize=TICK_FONTSIZE)

    xmax = float(np.nanmax(x_vals)) if len(x_vals) else 0.0
    pad  = 0.12 if xmax > 0 else 0.2
    ax.set_xlim(0, xmax * (1 + pad) + (0.5 if xmax < 6 else 0))
    ax.xaxis.set_major_locator(MultipleLocator(5) if xmax >= 15 else MaxNLocator(6))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel("平均利用期間（月）", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    # Annotations
    for i, (val, f) in enumerate(zip(x_vals, g["avg_freq"].values)):
        if np.isnan(val):
            continue
        freq_lbl = ""
        if annotate_with_freq:
            if (not np.isnan(f)) and f >= 3.5:
                freq_lbl = "毎時間"
            elif (not np.isnan(f)) and f >= 2.5:
                freq_lbl = "週数回"
            elif (not np.isnan(f)) and f >= 1.5:
                freq_lbl = "月数回"
            elif not np.isnan(f):
                freq_lbl = "ほとんど使用せず"

        if val <= 0 or np.isclose(val, 0.0):
            if freq_lbl:
                ax.text(0.3, i, freq_lbl, va="center", ha="left",
                        color="black", fontsize=BASE_FONTSIZE)
            continue

        text = f"{val:.1f}ヶ月" + (f"｜{freq_lbl}" if freq_lbl else "")
        inside_threshold = 0.18 * ax.get_xlim()[1]
        if val >= inside_threshold:
            ax.text(val - 0.3, i, text, va="center", ha="right",
                    color="white", fontsize=BASE_FONTSIZE, fontweight="bold")
        else:
            ax.text(val + 0.3, i, text, va="center", ha="left",
                    color="black", fontsize=BASE_FONTSIZE)

    # De-emphasize unknown
    if deemphasize_unknown_label:
        for bar, lab in zip(bars, labels):
            if lab == deemphasize_unknown_label:
                bar.set_alpha(0.45)

    plt.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}")
    print(f"[info] wrote {out_csv}")

# ---- Main -------------------------------------------------------------------
def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the cleaner first.")

    # Load
    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    # Canonical school column (centralized)
    _ = SC.find_or_make_school_canon(df, debug=False)   # -> sets df['school_canon']
    SC.assert_only_allowed(df)

    # Q4 column (duration)
    col_q4 = _find_col(
        df,
        exact_jp="LEAFシステム（BookRoll，分析（ぶんせき）ツール）を授業・授業外（宿題など）で何か月くらい利用していますか。※利用期間がX年Yか月だった場合、「X/Y」というようにスラッシュで区切って、全て半角で入力してください（例：1年の場合→1/0、2年3か月の場合→2/3、未使用の場合→0/0）",
        must_contain=["利用", "何か月", "X", "Y"],
    )
    if col_q4 is None:
        raise KeyError("Could not find Q4 (usage duration) column in cleaned data.")

    # Q5 usage frequency column (for contextual labels)
    usage_col = _find_usage_col(df)
    if not usage_col:
        raise KeyError("Could not find usage frequency column")

    # Derive metrics
    df["use_months_total"] = df[col_q4].map(parse_year_month_to_months)
    df["freq_score"]       = df[usage_col].map(map_freq)

    # ---- (1) By SCHOOL ------------------------------------------------------
    _plot_by_group(
        df=df,
        group_col="school_canon",
        title=TITLE_SCHOOL,
        out_png=OUT_SCHOOL_PNG,
        out_csv=OUT_SCHOOL_CSV,
        min_n=MIN_RESP_PER_SCHOOL,
        annotate_with_freq=True,
        deemphasize_unknown_label="不明",
    )

    # ---- (2) By GRADE -------------------------------------------------------
    grade_col = _find_grade_canon(df)  # prefers 学年_canon if present
    _plot_by_group(
        df=df,
        group_col=grade_col,
        title=TITLE_GRADE,
        out_png=OUT_GRADE_PNG,
        out_csv=OUT_GRADE_CSV,
        min_n=MIN_RESP_PER_GRADE,
        annotate_with_freq=True,
        deemphasize_unknown_label="不明",
    )

if __name__ == "__main__":
    main()
