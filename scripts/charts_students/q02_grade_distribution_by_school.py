#   python3 -m scripts.charts_students.q02_grade_distribution_by_school
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import unicodedata as ud
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Project style (JP fonts, etc.)
try:
    from scripts.common.plotkit import setup
except Exception:
    setup = lambda: None

# Canonicalize schools exactly like teacher scripts
from scripts.common.canonicalize import (
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
    post_disambiguate_students_by_grade,   # ← NEW
)

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/students_clean.csv")
OUT_DIR    = Path("figs/students/demographics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_N        = 1
TOP_N        = None
LABEL_GT     = 10.0
PLOT_DPI     = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 11
LABEL_FONTSIZE = 12
TITLE = "学年別構成（学校別・生徒）"

GRADES = [
    "小学校5年生", "小学校6年生",
    "中学校1年生", "中学校2年生", "中学校3年生",
    "高校1年生", "高校2年生", "高校3年生",
]

# ---- Helpers ----------------------------------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\\[\\]「」『』,:;－ー―–—\\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def find_grade_col(df: pd.DataFrame) -> Optional[str]:
    cands = [c for c in df.columns if "学年" in str(c)]
    if cands:
        return cands[0]
    want = [_norm("学年")]
    for c in df.columns:
        if all(t in _norm(c) for t in want):
            return c
    return None

def ensure_school(df: pd.DataFrame) -> pd.DataFrame:
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)
    return df

# ---- Plotting ---------------------------------------------------------------
def plot_stacked_pct(ct: pd.DataFrame, pct: pd.DataFrame, out_png: Path):
    n = ct.sum(axis=1).astype(int)
    order = n.sort_values(ascending=False).index
    ct, pct, n = ct.loc[order], pct.loc[order], n.loc[order]

    if TOP_N:
        keep = order[:TOP_N]
        ct, pct, n = ct.loc[keep], pct.loc[keep], n.loc[keep]

    labels = [f"{idx}（n={int(n[idx])}）" for idx in pct.index]
    y = np.arange(len(labels))

    cats = [g for g in GRADES if g in pct.columns]
    if "不明" in pct.columns and "不明" not in cats:
        cats.append("不明")

    base = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    while len(base) < len(cats):
        base = base + base
    colors = {c: base[i] for i, c in enumerate(cats)}

    fig_h = max(4.0, 0.55 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(12, fig_h), dpi=PLOT_DPI)

    left = np.zeros(len(pct), dtype=float)
    for c in cats:
        vals = pct.get(c, pd.Series(0.0, index=pct.index)).values
        ax.barh(y, vals, left=left, height=0.6, edgecolor="none", color=colors[c], label=c)
        if LABEL_GT is not None:
            for i, (v, lft) in enumerate(zip(vals, left)):
                if v >= LABEL_GT:
                    ax.text(lft + v/2, i, f"{v:.0f}%", va="center", ha="center",
                            fontsize=BASE_FONTSIZE-1, color="white", fontweight="bold")
        left += vals

    ax.set_yticks(y, labels=labels, fontsize=TICK_FONTSIZE)
    ax.set_xlim(0, 100)
    ax.set_xlabel("割合(%)", fontsize=LABEL_FONTSIZE)
    ax.set_title(TITLE, fontsize=TITLE_FONTSIZE, pad=10)
    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=0)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.invert_yaxis()

    handles = [mpatches.Patch(color=colors[c], label=c) for c in cats]
    ax.legend(handles=handles, ncols=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

# ---- Main -------------------------------------------------------------------
def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"{DATA_CLEAN} が見つかりません。")

    df = pd.read_csv(DATA_CLEAN, dtype=str).replace({"": np.nan}).infer_objects(copy=False)

    # Canonicalize schools
    df = ensure_school(df)

    # Locate grade column
    q2 = find_grade_col(df)
    if not q2:
        raise KeyError("学年の設問列（『あなたの学年を教えてください』）が見つかりません。")

    # ← NEW: fix HS/JHS mix-ups using the grade answer
    post_disambiguate_students_by_grade(df, grade_col=q2)

    # Keep only recognized grade categories; bucket others to '不明'
    df = df.dropna(subset=[q2])
    df[q2] = df[q2].apply(lambda s: s if s in GRADES else "不明")
    grades_cols = GRADES + (["不明"] if "不明" in df[q2].unique() and "不明" not in GRADES else [])

    # Count table and respondent filter
    ct = df.groupby(["school_canon", q2]).size().unstack(fill_value=0)
    ct = ct.reindex(columns=grades_cols, fill_value=0)
    n_by_school = ct.sum(axis=1).astype(int)
    ct = ct.loc[n_by_school[n_by_school >= MIN_N].index]

    # Percentages per school
    pct = (ct.div(ct.sum(axis=1), axis=0) * 100.0).round(1)

    # Save CSVs
    (OUT_DIR / "Q02_学年別構成__学校別_counts.csv").write_text(
        ct.assign(n_total=ct.sum(axis=1)).to_csv(encoding="utf-8"), encoding="utf-8"
    )
    pct.to_csv(OUT_DIR / "Q02_学年別構成__学校別_pct.csv", encoding="utf-8")

    # Plot
    out_png = OUT_DIR / "Q02_学年別構成__学校別.png"
    plot_stacked_pct(ct, pct, out_png)
    print(f"[info] wrote {out_png}")

if __name__ == "__main__":
    main()
