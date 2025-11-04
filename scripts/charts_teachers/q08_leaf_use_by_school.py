# scripts/charts_teachers/q08_leaf_use_by_school.py
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Project helpers
try:
    from scripts.common.plotkit import setup  # sets JP font/theme
except Exception:
    setup = lambda: None

from scripts.common.canonicalize import SchoolCanonicalizer as SC, post_disambiguate_middle_vs_high

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/likert_by_school")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Q08_LEAF頻度__学校別.png"
OUT_CSV = OUT_DIR / "Q08_LEAF頻度__学校別.csv"
OUT_CSV_COUNTS = OUT_DIR / "Q08_LEAF頻度__学校別_counts.csv"
OUT_DEBUG = OUT_DIR / "Q08_debug_rows.csv"

# ---- Config -----------------------------------------------------------------
MIN_RESP_PER_GROUP = 1     # exclude schools with < this many answered responses to Q08
TOP_N = None               # e.g., 12 to show only top-12 by respondents; None = all

TITLE        = "LEAFシステム利用頻度（学校別・100%積み上げ・教員）"
X_LABEL      = "割合（%）"
Y_LABEL      = "学校名"
LEGEND_TITLE = "頻度"

# Canonical order (left→right) & legend order
CATEGORY_ORDER = [
    "ほぼ毎時間",
    "1週間に数回程度",
    "1ヶ月に数回程度",
    "ほとんど使用していない",
]

# Keep the same school order as your other charts (edit as needed)
ALLOWED_SCHOOL_ORDER: List[str] = [
    "西京高等学校付属中学校",
    "洗足学園高等学校",
    "西賀茂中学校",
    "洗足学園中学校",
    "岩沼小学校",
    "明徳小学校",
    "不明",
]

# ---- Normalization & column finders -----------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def _find_col(df: pd.DataFrame, exact_jp_list: List[str], contains_any: List[str]) -> Optional[str]:
    norm_cols = {c: _norm(c) for c in df.columns}
    for exact in exact_jp_list or []:
        tgt = _norm(exact)
        for c, nc in norm_cols.items():
            if nc == tgt:
                return c
    tokens = [_norm(t) for t in (contains_any or [])]
    for c, nc in norm_cols.items():
        if all(t in nc for t in tokens):
            return c
    for c, nc in norm_cols.items():
        if any(t in nc for t in tokens):
            return c
    return None

def find_col_leaf_freq(df: pd.DataFrame) -> Optional[str]:
    exact = ["LEAFシステム（BookRoll，分析ツール）を授業・授業外（宿題など）でどれくらい利用していますか",
             "LEAFシステム(BookRoll,分析ツール)を授業・授業外(宿題など)でどれくらい利用していますか"]
    contains = ["leaf", "bookroll", "分析", "授業", "利用", "頻度"]
    return _find_col(df, exact, contains)

# ---- Mapping raw answers to canonical categories ----------------------------
def map_leaf_freq(val: str) -> str:
    if pd.isna(val):
        return "不明"
    s = ud.normalize("NFKC", str(val)).strip()

    # direct matches first
    if s in CATEGORY_ORDER:
        return s

    # tolerant heuristics
    t = s.replace("程度", "")
    if "毎" in t and ("時" in t or "回" in t):  # 毎時間 / 毎回→授業ごと
        return "ほぼ毎時間"
    if "週" in t:
        return "1週間に数回程度"
    if "月" in t:
        return "1ヶ月に数回程度"
    if "ほとんど" in t or "使用していない" in t or "未使用" in t:
        return "ほとんど使用していない"

    return "その他"

# ---- Tables -----------------------------------------------------------------
def make_tables(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    cats: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns:
      pct_df: rows=groups, cols=cats, values=percentage [0..100]
      cnt_df: rows=groups, cols=cats, values=counts (int)
      n_per_group: total respondents per group (int)
    """
    cnt = (
        df.groupby([group_col, value_col])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=cats, fill_value=0)
          .astype(int)
    )
    n = cnt.sum(axis=1).astype(int)
    pct = cnt.div(n.replace(0, np.nan), axis=0) * 100.0
    pct = pct.fillna(0.0)
    return pct, cnt, n

# ---- Plotting ---------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

def _style_axes(ax):
    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def plot_bar_100pct_h(
    pct_df: pd.DataFrame,
    n_per_group: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    legend_title: str,
    out_png: Path,
):
    # Build y labels with n
    base_labels = pct_df.index.tolist()
    y_labels = [f"{lab}（n={int(n_per_group.loc[lab])}）" for lab in base_labels]

    # Figure size scales with number of schools
    y_count = len(base_labels)
    fig_h = max(4.0, 0.65 * y_count + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    # Stacked bars
    y_pos = np.arange(y_count)
    left = np.zeros(y_count, dtype=float)
    for col in pct_df.columns:  # already in desired left→right order
        vals = pct_df[col].values
        ax.barh(y_pos, vals, left=left, height=0.6, label=col,
                zorder=2, edgecolor="white", linewidth=0.5)
        left += vals

    _style_axes(ax)
    ax.set_yticks(y_pos, labels=y_labels, fontsize=TICK_FONTSIZE)
    ax.invert_yaxis()  # first school at the top
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    # Legend (outside, right) in the same category order
    handles, labels = ax.get_legend_handles_labels()
    order = [pct_df.columns.get_loc(c) for c in pct_df.columns]
    handles = [handles[i] for i in order]
    labels  = [labels[i] for i in order]
    leg = ax.legend(handles, labels, title=legend_title,
                    bbox_to_anchor=(1.02, 1.0), loc="upper left",
                    borderaxespad=0.0, frameon=False, fontsize=BASE_FONTSIZE)
    if leg and leg.get_title():
        leg.get_title().set_fontsize(BASE_FONTSIZE)

    plt.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}")

# ---- Main -------------------------------------------------------------------
def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the teacher cleaner first.")

    # Load
    df = pd.read_csv(DATA_CLEAN, dtype=str).replace({"": np.nan}).infer_objects(copy=False)

    # Canonicalize schools
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # Find Q08 column
    leaf_col = find_col_leaf_freq(df)
    if not leaf_col:
        raise KeyError("Q08 column (LEAF頻度) not found.")

    # Map raw→canonical categories + write debug
    df["_leaf_raw"]  = df[leaf_col]
    df["_leaf_norm"] = df["_leaf_raw"].map(map_leaf_freq)

    dbg_cols = ["school_canon", "_leaf_raw", "_leaf_norm"]
    df[dbg_cols].to_csv(OUT_DEBUG, index=False, encoding="utf-8")
    print(f"[info] wrote {OUT_DEBUG}")

    # Keep only rows with an assigned school (still keep 「不明」 group if present)
    # Filter by answered (i.e., exclude 「不明」/「その他」 for the MIN_RESP threshold)
    answered = (
        df.assign(_ans=df["_leaf_norm"].isin(CATEGORY_ORDER))
          .groupby("school_canon")["_ans"].sum()
    )
    keep = answered[answered >= MIN_RESP_PER_GROUP].index
    if "不明" in df["school_canon"].unique():
        keep = keep.union(pd.Index(["不明"]))
    df = df[df["school_canon"].isin(keep)]

    # Apply fixed school order
    present = [s for s in ALLOWED_SCHOOL_ORDER if s in df["school_canon"].unique()]
    extras  = [s for s in df["school_canon"].unique() if s not in present]
    school_order = present + extras

    # Build tables in deterministic category order
    cats = CATEGORY_ORDER.copy()
    # Only add "その他"/"不明" if present
    for extra_cat in ["不明", "その他"]:
        if extra_cat in df["_leaf_norm"].unique():
            cats.append(extra_cat)

    pct_df, cnt_df, n_per_group = make_tables(df, "school_canon", "_leaf_norm", cats)

    # Reindex rows to forced school order
    pct_df = pct_df.reindex(index=[i for i in school_order if i in pct_df.index])
    n_per_group = n_per_group.loc[pct_df.index]
    cnt_df = cnt_df.loc[pct_df.index, pct_df.columns]

    # Save CSVs (percentages + counts)
    out_pct = pct_df.copy()
    out_pct["n"] = n_per_group
    out_pct.to_csv(OUT_CSV, encoding="utf-8", index=True)
    cnt_df.to_csv(OUT_CSV_COUNTS, encoding="utf-8", index=True)
    print(f"[info] wrote {OUT_CSV}")
    print(f"[info] wrote {OUT_CSV_COUNTS}")

    # Plot
    # Only plot the canonical 4 categories; extras (不明/その他) are excluded from the bars
    cols_for_plot = [c for c in CATEGORY_ORDER if c in pct_df.columns]
    plot_bar_100pct_h(
        pct_df=pct_df[cols_for_plot],
        n_per_group=n_per_group,
        title=TITLE,
        xlabel=X_LABEL,
        ylabel=Y_LABEL,
        legend_title=LEGEND_TITLE,
        out_png=OUT_PNG,
    )

if __name__ == "__main__":
    main()
