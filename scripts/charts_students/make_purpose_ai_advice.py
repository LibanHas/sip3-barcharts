# Usage:
#   python3 -m scripts.charts_students.make_purpose_ai_advice
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Project helpers (fonts, JP rendering)
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# Centralized canonicalizer (sets df["school_canon"], asserts allowed)
from scripts.common.canonicalize import SchoolCanonicalizer as SC

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/students_clean.csv")
OUT_DIR    = Path("figs/students/purpose_ai_advice")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_RESP = 2
TITLE_SCHOOL = "目的：AIによる問題推薦やアドバイスを受けるため（0–3）— 学校別（学生）"
TITLE_GRADE  = "目的：AIによる問題推薦やアドバイスを受けるため（0–3）— 学年別（学生）"
OUT_SCHOOL_PNG = OUT_DIR / "purpose_ai_advice_by_school.png"
OUT_SCHOOL_CSV = OUT_DIR / "purpose_ai_advice_by_school.csv"
OUT_GRADE_PNG  = OUT_DIR / "purpose_ai_advice_by_grade.png"
OUT_GRADE_CSV  = OUT_DIR / "purpose_ai_advice_by_grade.csv"

# ---- Plot style -------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

# ---- Column helpers ---------------------------------------------------------
CAND_COLS = [
    "AIによる問題推薦(すいせん)やアドバイスを受けるため",
    "AIによる問題推薦（すいせん）やアドバイスを受けるため",
    "AIによる問題推薦やアドバイスを受けるため",
    # tolerate light spacing/marks variations
    "AI による問題推薦やアドバイスを受けるため",
]

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()【】\[\]「」『』,:;.-]", "", s)
    return s.lower()

def find_question_col(df: pd.DataFrame) -> str | None:
    for c in CAND_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        nc = _norm(c)
        if ("ai" in nc) and ("推薦" in nc or "すいせん" in nc) and ("アドバイス" in nc):
            return c
    return None

def find_grade_col(df: pd.DataFrame) -> str | None:
    if "学年_canon" in df.columns:
        return "学年_canon"
    for cand in ["あなたの学年を教えてください", "学年", "学年（整理前）"]:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "学年" in str(c):
            return c
    return None

# ---- Scale mapping (4-point Likert) -----------------------------------------
# 0: 全く使用しない
# 1: あまり使用しない
# 2: 使用することがある
# 3: 頻繁（ひんぱん）に使用している
SCALE_MAP = {
    "全く使用しない": 0,
    "あまり使用しない": 1,
    "使用することがある": 2,
    "頻繁（ひんぱん）に使用している": 3,
    # tolerate slight variant
    "頻繁に使用している": 3,
}

def map_scale(x):
    if pd.isna(x):
        return np.nan
    s = ud.normalize("NFKC", str(x)).strip()
    return float(SCALE_MAP.get(s, np.nan))

def add_scale_legend(ax, fontsize=10):
    txt = (
        "スコア対応（0–3）\n"
        "0：全く使用しない\n"
        "1：あまり使用しない\n"
        "2：使用することがある\n"
        "3：頻繁（ひんぱん）に使用している"
    )
    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=fontsize, linespacing=1.3,
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.35", edgecolor="none"),
        zorder=10,
    )

# ---- Plot helper ------------------------------------------------------------
def plot_barh_avg(
    df_sum: pd.DataFrame,
    title: str,
    out_png: Path,
    xlabel: str = "平均スコア（0–3）",
    headroom: float = 0.08,
):
    g_plot = df_sum.copy()
    g_plot["avg_plot"] = g_plot["avg_score"].fillna(0.0)
    g_plot = g_plot.sort_values("avg_plot", ascending=False)

    labels = g_plot.index.tolist()
    y_pos  = np.arange(len(labels))
    x_vals = g_plot["avg_plot"].values

    fig_h = max(3.8, 0.7 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    bars = ax.barh(y_pos, x_vals, height=0.6, zorder=2, edgecolor="none")

    ax.set_yticks(y_pos, labels=labels, fontsize=TICK_FONTSIZE)
    ax.set_xlim(0, 3.0 + headroom)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    for i, (val, n) in enumerate(zip(x_vals, g_plot["respondents"].values)):
        if np.isnan(val) or val < 0:
            continue
        txt = f"{val:.1f}｜n={int(n)}"
        if val >= 1.6:
            ax.text(val - 0.05, i, txt, va="center", ha="right",
                    color="white", fontsize=BASE_FONTSIZE, fontweight="bold")
        else:
            ax.text(val + 0.05, i, txt, va="center", ha="left",
                    color="black", fontsize=BASE_FONTSIZE)

    add_scale_legend(ax, fontsize=10)

    plt.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}")

# ---- Main -------------------------------------------------------------------
def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the cleaner first.")

    # Load
    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    # Canonical school (centralized)
    _ = SC.find_or_make_school_canon(df, debug=False)
    SC.assert_only_allowed(df)

    # Find columns
    col_q = find_question_col(df)
    if not col_q:
        raise KeyError("質問列（AIによる問題推薦やアドバイスを受けるため）が見つかりません。")
    col_grade = find_grade_col(df)
    if not col_grade:
        raise KeyError("学年列が見つかりません（'学年_canon' など）。")

    # Map to numeric scale
    df["purpose_score"] = df[col_q].map(map_scale)

    # ------- by SCHOOL -------
    resp_by_school = df.groupby("school_canon")["purpose_score"].apply(lambda s: s.notna().sum())
    ok_schools = resp_by_school[resp_by_school >= MIN_RESP].index

    g_school = (
        df[df["school_canon"].isin(ok_schools)]
        .groupby("school_canon")
        .agg(
            respondents=("purpose_score", lambda s: int(s.notna().sum())),
            avg_score=("purpose_score", "mean"),
        )
    )

    allowed_order = getattr(
        SC, "ALLOWED_SCHOOLS",
        ["不明", "北海道天塩高等学校", "岩沼小学校", "洗足学園中学校",
         "西京高等学校", "西京高等学校付属中学校", "西賀茂中学校", "北海道寿都高等学校"]
    )
    g_school = g_school.reindex([idx for idx in allowed_order if idx in g_school.index])

    # Save / Plot (school)
    g_school.to_csv(OUT_SCHOOL_CSV, encoding="utf-8", index=True)
    print(f"[info] wrote {OUT_SCHOOL_CSV}")
    plot_barh_avg(g_school, TITLE_SCHOOL, OUT_SCHOOL_PNG, xlabel="平均スコア（0–3）")

    # ------- by GRADE -------
    resp_by_grade = df.groupby(col_grade)["purpose_score"].apply(lambda s: s.notna().sum())
    ok_grades = resp_by_grade[resp_by_grade >= MIN_RESP].index

    g_grade = (
        df[df[col_grade].isin(ok_grades)]
        .groupby(col_grade)
        .agg(
            respondents=("purpose_score", lambda s: int(s.notna().sum())),
            avg_score=("purpose_score", "mean"),
        )
    ).sort_values("avg_score", ascending=False)

    # Save / Plot (grade)
    g_grade.to_csv(OUT_GRADE_CSV, encoding="utf-8", index=True)
    print(f"[info] wrote {OUT_GRADE_CSV}")
    plot_barh_avg(g_grade, TITLE_GRADE, OUT_GRADE_PNG, xlabel="平均スコア（0–3）")

if __name__ == "__main__":
    main()
