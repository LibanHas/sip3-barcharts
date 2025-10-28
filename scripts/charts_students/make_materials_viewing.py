# Usage:
#   python3 -m scripts.charts_students.make_materials_viewing
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

# Centralized canonicalizer (school + grade if available)
from scripts.common.canonicalize import SchoolCanonicalizer as SC

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/students_clean.csv")
OUT_DIR    = Path("figs/students/materials_viewing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_RESP_PER_GROUP = 2  # minimum valid responses needed to include a bar
TITLE_SCHOOL = "教材や問題を見るため — 平均使用頻度（0–3）／学校別（学生）"
TITLE_GRADE  = "教材や問題を見るため — 平均使用頻度（0–3）／学年別（学生）"

OUT_SCHOOL_PNG = OUT_DIR / "avg_materials_viewing_by_school.png"
OUT_SCHOOL_CSV = OUT_DIR / "avg_materials_viewing_by_school.csv"
OUT_GRADE_PNG  = OUT_DIR / "avg_materials_viewing_by_grade.png"
OUT_GRADE_CSV  = OUT_DIR / "avg_materials_viewing_by_grade.csv"

# ---- Plot style -------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

# ---- Column finders ---------------------------------------------------------
TARGET_COL_CANDIDATES = [
    "教材や問題を見るため",
    "教材や問題を見るため（頻度）",
    # safety variants (normalize will handle parens/spacing/punct)
]

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()【】\[\]「」『』,:;.-]", "", s)
    return s.lower()

def find_target_col(df: pd.DataFrame) -> str | None:
    for c in TARGET_COL_CANDIDATES:
        if c in df.columns:
            return c
    # fuzzy
    want = _norm("教材や問題を見るため")
    for c in df.columns:
        if want in _norm(c):
            return c
    return None

# ---- Mapping (0–3) ----------------------------------------------------------
# 0: 全く使用しない
# 1: あまり使用しない
# 2: 使用することがある
# 3: 頻繁（ひんぱん）に使用している
# NaN: 質問の意味がわからない / blank
FREQ_MAP = {
    "全く使用しない": 0,
    "あまり使用しない": 1,
    "使用することがある": 2,
    "頻繁（ひんぱん）に使用している": 3,
}

INVALID_SET = {"質問の意味がわからない"}

def map_freq_to_score(x) -> float:
    if pd.isna(x):
        return np.nan
    s = ud.normalize("NFKC", str(x)).strip()
    if s in INVALID_SET:
        return np.nan
    if s in FREQ_MAP:
        return float(FREQ_MAP[s])
    # tolerant fallbacks
    if "全く" in s:
        return 0.0
    if "あまり" in s:
        return 1.0
    if "ある" in s or "使用することがある" in s:
        return 2.0
    if "頻繁" in s:
        return 3.0
    return np.nan

def legend_box_text() -> str:
    lines = [
        "スコア対応（0–3）",
        "0：全く使用しない",
        "1：あまり使用しない",
        "2：使用することがある",
        "3：頻繁に使用している",
    ]
    return "\n".join(lines)

def add_scale_legend(ax, fontsize=10):
    ax.text(
        0.98, 0.98, legend_box_text(),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=fontsize,
        linespacing=1.3,
        bbox=dict(facecolor="white", alpha=0.88, boxstyle="round,pad=0.4", edgecolor="none"),
        zorder=10,
    )

# ---- Grade helpers ----------------------------------------------------------
# Prefer canonical grade if available
GRADE_CANON_COL = "学年_canon"
GRADE_RAW_CANDIDATES = ["あなたの学年を教えてください", "学年"]

# Standard order if we detect canonical labels like 小6,中1…高3
CANON_GRADE_ORDER = ["小6", "中1", "中2", "中3", "高1", "高2", "高3"]

def find_grade_col(df: pd.DataFrame) -> str | None:
    if GRADE_CANON_COL in df.columns:
        return GRADE_CANON_COL
    for c in GRADE_RAW_CANDIDATES:
        if c in df.columns:
            return c
    # fuzzy
    for c in df.columns:
        if "学年" in str(c):
            return c
    return None

def order_for_grades(series: pd.Series) -> list[str]:
    vals = [v for v in series.dropna().unique()]
    # if looks like canonical short labels, use fixed order
    if all(v in CANON_GRADE_ORDER for v in vals):
        return [v for v in CANON_GRADE_ORDER if v in vals]
    # else try to sort “小/中/高 + number” roughly
    def _key(v: str):
        if isinstance(v, str):
            if v.startswith("小"):
                try:
                    return (0, int(re.sub(r"\D", "", v) or 0))
                except Exception:
                    return (0, 0)
            if v.startswith("中"):
                try:
                    return (1, int(re.sub(r"\D", "", v) or 0))
                except Exception:
                    return (1, 0)
            if v.startswith("高"):
                try:
                    return (2, int(re.sub(r"\D", "", v) or 0))
                except Exception:
                    return (2, 0)
        return (9, 0)
    return sorted(vals, key=_key)

# ---- Plotting core ----------------------------------------------------------
def barh_avg(
    df_sum: pd.DataFrame,
    title: str,
    out_png: Path,
    x_max: float = 3.0,
    de_emphasize_label: str | None = "不明",
    show_minor=False,
):
    g = df_sum.copy()
    g["avg_plot"] = g["avg_score"].fillna(0.0)
    g = g.sort_values("avg_plot", ascending=False)

    labels = g.index.tolist()
    y_pos  = np.arange(len(labels))
    x_vals = g["avg_plot"].values
    ns     = g["respondents"].values

    fig_h = max(3.8, 0.7 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    # grid/frame
    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    bars = ax.barh(y_pos, x_vals, height=0.6, zorder=2, edgecolor="none")

    # axes
    ax.set_yticks(y_pos, labels=labels, fontsize=TICK_FONTSIZE)
    ax.set_xlim(0, x_max + 0.1)
    ax.xaxis.set_major_locator(MultipleLocator(0.5 if x_max <= 4 else 1))
    if show_minor:
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel("平均使用頻度（0–3）", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    # annotations
    for i, (val, n) in enumerate(zip(x_vals, ns)):
        if np.isnan(val) or val < 0:
            continue
        text = f"{val:.1f}｜n={int(n)}"
        inside_threshold = x_max * 0.55
        if val >= inside_threshold:
            ax.text(val - 0.05, i, text, va="center", ha="right",
                    color="white", fontsize=BASE_FONTSIZE, fontweight="bold")
        else:
            ax.text(val + 0.05, i, text, va="center", ha="left",
                    color="black", fontsize=BASE_FONTSIZE)

    # Optional de-emphasis (e.g., 不明)
    if de_emphasize_label is not None:
        for bar, lab in zip(bars, labels):
            if lab == de_emphasize_label:
                bar.set_alpha(0.45)

    # legend box (score mapping)
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

    # Canonical school column
    _ = SC.find_or_make_school_canon(df, debug=False)   # -> sets df['school_canon']
    SC.assert_only_allowed(df)
    school_order = getattr(
        SC, "ALLOWED_SCHOOLS",
        ["不明", "北海道天塩高等学校", "岩沼小学校", "洗足学園中学校",
         "西京高等学校", "西京高等学校付属中学校", "西賀茂中学校", "北海道寿都高等学校"]
    )

    # Find target column
    col = find_target_col(df)
    if col is None:
        raise KeyError("Could not find the column for『教材や問題を見るため』.")

    # Map to numeric score (0–3)
    df["materials_viewing_score"] = df[col].map(map_freq_to_score)

    # -------------------------
    # A) By SCHOOL
    # -------------------------
    resp_per = df.groupby("school_canon")["materials_viewing_score"].apply(lambda s: s.notna().sum())
    ok_groups = resp_per[resp_per >= MIN_RESP_PER_GROUP].index

    g_school = (
        df[df["school_canon"].isin(ok_groups)]
        .groupby("school_canon")
        .agg(
            respondents=("materials_viewing_score", lambda s: int(s.notna().sum())),
            avg_score=("materials_viewing_score", "mean"),
        )
        .reindex(school_order)
    )

    # Save CSV
    g_school.to_csv(OUT_SCHOOL_CSV, encoding="utf-8", index=True)
    print(f"[info] wrote {OUT_SCHOOL_CSV}")

    # Plot
    barh_avg(
        g_school.dropna(subset=["respondents"]),  # keep only groups present after reindex
        title=TITLE_SCHOOL,
        out_png=OUT_SCHOOL_PNG,
        x_max=3.0,
        de_emphasize_label="不明",
        show_minor=True,
    )

    # -------------------------
    # B) By GRADE
    # -------------------------
    grade_col = find_grade_col(df)
    if grade_col is None:
        print("[WARN] Could not find a grade column; skipping grade-level chart.")
        return

    # build grade order
    grade_order = order_for_grades(df[grade_col])

    resp_per_g = df.groupby(grade_col)["materials_viewing_score"].apply(lambda s: s.notna().sum())
    ok_grades = resp_per_g[resp_per_g >= MIN_RESP_PER_GROUP].index

    g_grade = (
        df[df[grade_col].isin(ok_grades)]
        .groupby(grade_col)
        .agg(
            respondents=("materials_viewing_score", lambda s: int(s.notna().sum())),
            avg_score=("materials_viewing_score", "mean"),
        )
        .reindex([g for g in grade_order if g in ok_grades])
    )

    g_grade.to_csv(OUT_GRADE_CSV, encoding="utf-8", index=True)
    print(f"[info] wrote {OUT_GRADE_CSV}")

    # grade plot (no de-emphasis)
    barh_avg(
        g_grade.dropna(subset=["respondents"]),
        title=TITLE_GRADE,
        out_png=OUT_GRADE_PNG,
        x_max=3.0,
        de_emphasize_label=None,
        show_minor=True,
    )

if __name__ == "__main__":
    main()
