# Usage:
#   python3 -m scripts.charts_students.make_purpose_reflection
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Fonts / JP settings (optional)
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# Centralized canonicalizer
from scripts.common.canonicalize import SchoolCanonicalizer as SC

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/students_clean.csv")
OUT_DIR    = Path("figs/students/purpose")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_RESP_PER_SCHOOL = 2
MIN_RESP_PER_GRADE  = 2

QUESTION_TITLE = "目的別：自分の解答や学び方を振り返るため（0–3）"
# Output base names (we’ll append _by_school / _by_grade)
OUTBASE_PNG = OUT_DIR / "purpose_reflection"
OUTBASE_CSV = OUT_DIR / "purpose_reflection"

# ---- Plot style -------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

# ---- Column finder ----------------------------------------------------------
CANDIDATE_COLS = [
    "自分の解答や学び方を振(ふ)り返るため",
    "自分の解答や学び方を振り返るため",
    "自分の解答や学び方をふり返るため",
]
def find_purpose_col(df: pd.DataFrame) -> str | None:
    for c in CANDIDATE_COLS:
        if c in df.columns:
            return c
    # best-effort fuzzy match
    norm = lambda s: re.sub(r"\s+", "", ud.normalize("NFKC", str(s)))
    for c in df.columns:
        nc = norm(c)
        if ("自分の解答" in nc or "自分ノ解答" in nc) and ("学び方" in nc or "学び" in nc) and ("振" in nc):
            return c
    return None

# ---- Mapping (Likert → 0–3) -------------------------------------------------
SCORE_MAP = {
    "頻繁（ひんぱん）に使用している": 3,
    "使用することがある": 2,
    "あまり使用しない": 1,
    "全く使用しない": 0,
    "質問の意味がわからない": np.nan,
}
def map_to_score(x):
    if pd.isna(x):
        return np.nan
    s = ud.normalize("NFKC", str(x)).strip()
    if s in SCORE_MAP:
        return SCORE_MAP[s]
    # tolerant heuristics
    if "頻繁" in s:
        return 3
    if "全く" in s:
        return 0
    if "あまり" in s:
        return 1
    if "意味がわからない" in s:
        return np.nan
    if "使用することがある" in s:
        return 2
    return np.nan

def add_scale_legend(ax, fontsize=10):
    lines = [
        "スコア対応（0–3）",
        "3：頻繁（ひんぱん）に使用している",
        "2：使用することがある",
        "1：あまり使用しない",
        "0：全く使用しない",
    ]
    ax.text(
        0.98, 0.98, "\n".join(lines),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=fontsize,
        linespacing=1.25,
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.35", edgecolor="none"),
        zorder=10,
    )

# ---- Grade column finder ----------------------------------------------------
def find_grade_col(df: pd.DataFrame) -> str | None:
    # prefer canonical if present
    if "学年_canon" in df.columns:
        return "学年_canon"
    # fallbacks
    for c in df.columns:
        if "学年" in str(c):
            return c
    return None

# ---- Chart helpers ----------------------------------------------------------
def plot_barh_avg(df_sum: pd.DataFrame, title: str, xlim: tuple[float,float], out_png: Path):
    """df_sum must have index as labels and columns ['respondents','avg_score']"""
    # sort by avg descending for readability
    g = df_sum.copy()
    g["avg_plot"] = g["avg_score"].fillna(0.0)
    g = g.sort_values("avg_plot", ascending=False)

    labels = g.index.tolist()
    y_pos  = np.arange(len(labels))
    x_vals = g["avg_plot"].values

    fig_h = max(3.8, 0.7 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    # style
    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    bars = ax.barh(y_pos, x_vals, height=0.6, zorder=2, edgecolor="none")

    # axes
    ax.set_yticks(y_pos, labels=labels, fontsize=TICK_FONTSIZE)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel("平均スコア（0–3）", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    # annotations
    for i, (val, n) in enumerate(zip(x_vals, g["respondents"].values)):
        if np.isnan(val) or n == 0:
            continue
        text = f"{val:.2f}｜n={int(n)}"
        inside_threshold = 1.6  # show inside if bar is long enough
        if val >= inside_threshold:
            ax.text(val - 0.05, i, text, va="center", ha="right",
                    color="white", fontsize=BASE_FONTSIZE, fontweight="bold")
        else:
            ax.text(val + 0.05, i, text, va="center", ha="left",
                    color="black", fontsize=BASE_FONTSIZE)

    # de-emphasize 不明 if present
    for bar, lab in zip(bars, labels):
        if str(lab) == "不明":
            bar.set_alpha(0.45)

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
    _ = SC.find_or_make_school_canon(df, debug=False)   # sets df['school_canon']
    SC.assert_only_allowed(df)

    # Preferred display order for schools (fallback to SC list)
    school_order = getattr(
        SC, "ALLOWED_SCHOOLS",
        ["不明", "北海道天塩高等学校", "岩沼小学校", "洗足学園中学校",
         "西京高等学校", "西京高等学校付属中学校", "西賀茂中学校", "北海道寿都高等学校"]
    )

    # Find the target column
    col = find_purpose_col(df)
    if col is None:
        raise KeyError("Could not find column for『自分の解答や学び方を振(ふ)り返るため』")

    # Map to 0–3
    df["purpose_score"] = df[col].map(map_to_score)

    # ---------------- By school ----------------
    resp_per_school = df.groupby("school_canon")["purpose_score"].apply(lambda s: s.notna().sum())
    ok_schools = resp_per_school[resp_per_school >= MIN_RESP_PER_SCHOOL].index

    g_school = (
        df[df["school_canon"].isin(ok_schools)]
        .groupby("school_canon")
        .agg(
            respondents=("purpose_score", lambda s: int(s.notna().sum())),
            avg_score=("purpose_score", "mean"),
        )
        .reindex(school_order)
    )
    # save CSV
    out_csv_school = OUTBASE_CSV.with_name(OUTBASE_CSV.stem + "_by_school.csv")
    g_school.to_csv(out_csv_school, encoding="utf-8", index=True)
    print(f"[info] wrote {out_csv_school}")

    # plot (0–3 scale)
    out_png_school = OUTBASE_PNG.with_name(OUTBASE_PNG.stem + "_by_school.png")
    plot_barh_avg(
        g_school,
        title=f"{QUESTION_TITLE} — 学校別（学生）",
        xlim=(0, 3.05),
        out_png=out_png_school,
    )

    # ---------------- By grade -----------------
    grade_col = find_grade_col(df)
    if grade_col is None:
        print("[WARN] 学年列が見つからないため、学年別チャートはスキップします。")
        return

    resp_per_grade = df.groupby(grade_col)["purpose_score"].apply(lambda s: s.notna().sum())
    ok_grades = resp_per_grade[resp_per_grade >= MIN_RESP_PER_GRADE].index

    g_grade = (
        df[df[grade_col].isin(ok_grades)]
        .groupby(grade_col)
        .agg(
            respondents=("purpose_score", lambda s: int(s.notna().sum())),
            avg_score=("purpose_score", "mean"),
        )
    )
    # save CSV
    out_csv_grade = OUTBASE_CSV.with_name(OUTBASE_CSV.stem + "_by_grade.csv")
    g_grade.to_csv(out_csv_grade, encoding="utf-8", index=True)
    print(f"[info] wrote {out_csv_grade}")

    # plot
    out_png_grade = OUTBASE_PNG.with_name(OUTBASE_PNG.stem + "_by_grade.png")
    plot_barh_avg(
        g_grade,
        title=f"{QUESTION_TITLE} — 学年別（学生）",
        xlim=(0, 3.05),
        out_png=out_png_grade,
    )

if __name__ == "__main__":
    main()
