# Usage:
#   python3 -m scripts.charts_students.make_proficiency
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Optional: seaborn/font setup (JP font etc.)
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# Centralized canonicalizer (schools)
from scripts.common.canonicalize import SchoolCanonicalizer as SC

# ----------------------------- Paths / Config --------------------------------
DATA_CLEAN = Path("data/students_clean.csv")

OUT_BASE     = Path("figs/students/proficiency")
OUT_SCHOOL   = OUT_BASE / "by_school"
OUT_GRADE    = OUT_BASE / "by_grade"
OUT_SCHOOL.mkdir(parents=True, exist_ok=True)
OUT_GRADE.mkdir(parents=True, exist_ok=True)

MIN_RESP_PER_SCHOOL = 2
MIN_RESP_PER_GRADE  = 2

TITLE_SCHOOL = "LEAFの使いこなし度（0–4） — 学校別（学生）"
TITLE_GRADE  = "LEAFの使いこなし度（0–4） — 学年別（学生）"

PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

# ------------------------------ Helpers --------------------------------------
def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()【】\[\]「」『』,:;.-]", "", s)
    return s.lower()

# Q6 (proficiency) finder
PROFICIENCY_COLUMNS = [
    "今年度、 LEAFシステム(BookRoll,分析(ぶんせき)ツール)をどの程度使いこなすことができましたか",
    "今年度、LEAFシステム(BookRoll,分析(ぶんせき)ツール)をどの程度使いこなすことができましたか",
    "今年度、 LEAFシステム(BookRoll, 分析ツール)をどの程度使いこなすことができましたか",
    "使いこなすことができましたか",
    "使いこなし",
]
def find_prof_col(df: pd.DataFrame) -> str | None:
    for col in PROFICIENCY_COLUMNS:
        if col in df.columns:
            return col
    for col in df.columns:
        nc = _norm(col)
        if ("使いこな" in nc) or ("使いこなし" in nc):
            return col
    return None

# Map labels -> numeric (0–4)
PROF_MAP = {
    "システムそのものがよくわからない": 0,
    "全く使いこなすことができなかった": 1,
    "あまり使いこなすことができなかった": 2,
    "使いこなすことができた": 3,
    "とても使いこなすことができた": 4,
}
def map_proficiency(x):
    if pd.isna(x):
        return np.nan
    s = ud.normalize("NFKC", str(x)).strip()
    if s in PROF_MAP:
        return float(PROF_MAP[s])
    # tolerant fallbacks
    if ("システムそのもの" in s) and ("わからない" in s):
        return 0.0
    if ("全く" in s) or ("まったく" in s):
        return 1.0
    if "あまり" in s:
        return 2.0
    if "とても" in s:
        return 4.0
    if ("使いこなすことができた" in s) or ("使いこなせた" in s):
        return 3.0
    return np.nan

def add_scale_legend(ax, fontsize=10):
    by_score = sorted([(v, k) for k, v in PROF_MAP.items()], key=lambda x: x[0])
    lines = [f"{s}：{lab}" for s, lab in by_score]
    txt = "スコア対応（0–4）\n" + "\n".join(lines)
    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=fontsize,
        linespacing=1.3,
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.35", edgecolor="none"),
        zorder=10,
    )

def plot_grouped_barh(
    df_summary: pd.DataFrame,
    title: str,
    out_png: Path,
    x_label: str = "平均使いこなし度（0–4）",
    de_emphasize_label: str | None = "不明",
):
    g = df_summary.copy()
    g["avg_prof_plot"] = g["avg_prof"].fillna(0.0)
    g = g.sort_values("avg_prof_plot", ascending=False)

    labels = g.index.tolist()
    y_pos  = np.arange(len(labels))
    x_vals = g["avg_prof_plot"].values

    fig_h = max(3.8, 0.7 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    # Grid & frame
    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    bars = ax.barh(y_pos, x_vals, height=0.6, zorder=2, edgecolor="none")

    # Axes
    ax.set_yticks(y_pos, labels=labels, fontsize=TICK_FONTSIZE)
    ax.set_xlim(0, 4.1)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    # Value labels
    for i, (val, n) in enumerate(zip(x_vals, g["respondents"].values)):
        if np.isnan(val):
            continue
        txt = (f"n={int(n)}" if (val == 0 and int(n) > 0) else f"{val:.1f}｜n={int(n)}")
        inside_threshold = 2.2
        if val >= inside_threshold:
            ax.text(val - 0.05, i, txt, va="center", ha="right",
                    color="white", fontsize=BASE_FONTSIZE, fontweight="bold")
        else:
            ax.text(val + 0.05, i, txt, va="center", ha="left",
                    color="black", fontsize=BASE_FONTSIZE)

    # Optional de-emphasis
    if de_emphasize_label is not None:
        for bar, lab in zip(bars, labels):
            if lab == de_emphasize_label:
                bar.set_alpha(0.45)

    add_scale_legend(ax, fontsize=10)
    plt.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}")

# ------------------------------- Main ----------------------------------------
def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the cleaner first.")

    # Load
    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    # Canonicalize school
    _ = SC.find_or_make_school_canon(df, debug=False)   # creates df['school_canon']
    SC.assert_only_allowed(df)
    allowed_order = getattr(
        SC, "ALLOWED_SCHOOLS",
        ["不明", "北海道天塩高等学校", "岩沼小学校", "洗足学園中学校",
         "西京高等学校", "西京高等学校付属中学校", "西賀茂中学校", "北海道寿都高等学校"]
    )

    # Proficiency column
    col_prof = find_prof_col(df)
    if col_prof is None:
        raise KeyError("Could not find Q6 (proficiency) column in cleaned data.")
    print(f"[INFO] Using proficiency column: {col_prof}")

    # Map to numeric 0–4
    df["prof_score"] = df[col_prof].map(map_proficiency)

    # -------------------- BY SCHOOL --------------------
    resp_per_school = df.groupby("school_canon")["prof_score"].apply(lambda s: s.notna().sum())
    ok_schools = set(resp_per_school[resp_per_school >= MIN_RESP_PER_SCHOOL].index)

    g_school = (
        df[df["school_canon"].isin(ok_schools)]
        .groupby("school_canon")
        .agg(
            respondents=("prof_score", lambda s: int(s.notna().sum())),
            avg_prof=("prof_score", "mean"),
        )
        .reindex(allowed_order)
    )
    g_school.to_csv(OUT_SCHOOL / "avg_proficiency_by_school.csv", encoding="utf-8")
    plot_grouped_barh(
        g_school,
        title=TITLE_SCHOOL,
        out_png=OUT_SCHOOL / "avg_proficiency_by_school.png",
    )

    # --------------------- BY GRADE --------------------
    # Expect a cleaned canonical column; if not, try to find something sensible.
    if "学年_canon" not in df.columns:
        # heuristic fallback
        grade_col = None
        for c in df.columns:
            if "学年" in str(c):
                grade_col = c
                break
        if grade_col is None:
            # If truly missing, synthesize a single bucket to avoid crash
            df["学年_canon"] = "不明"
        else:
            df["学年_canon"] = df[grade_col].fillna("不明").astype(str)

    resp_per_grade = df.groupby("学年_canon")["prof_score"].apply(lambda s: s.notna().sum())
    ok_grades = set(resp_per_grade[resp_per_grade >= MIN_RESP_PER_GRADE].index)

    g_grade = (
        df[df["学年_canon"].isin(ok_grades)]
        .groupby("学年_canon")
        .agg(
            respondents=("prof_score", lambda s: int(s.notna().sum())),
            avg_prof=("prof_score", "mean"),
        )
        .sort_values("avg_prof", ascending=False)
    )
    g_grade.to_csv(OUT_GRADE / "avg_proficiency_by_grade.csv", encoding="utf-8")
    plot_grouped_barh(
        g_grade,
        title=TITLE_GRADE,
        out_png=OUT_GRADE / "avg_proficiency_by_grade.png",
        # For grades we usually don't have a "不明" bucket, so don’t dim anything:
        de_emphasize_label=None,
    )

if __name__ == "__main__":
    main()
