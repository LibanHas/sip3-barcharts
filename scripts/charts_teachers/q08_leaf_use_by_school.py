# Usage:
#   python3 -m scripts.charts_teachers.q08_leaf_use_by_school
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Project helpers (students style)
from scripts.common.plotkit import setup  # sets JP font etc.
from scripts.common.canonicalize import SchoolCanonicalizer as SC, post_disambiguate_middle_vs_high

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/likert_by_school")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Q08_LEAF頻度__学校別.png"
OUT_CSV = OUT_DIR / "Q08_LEAF頻度__学校別.csv"

# ---- Config / Filters -------------------------------------------------------
MIN_RESP_PER_GROUP = 1   # <= you asked for 1 respondent minimum
TOP_N  = 12              # show top-N schools by respondent count (set None to disable)

TITLE        = "LEAFシステム利用頻度（学校別・100%積み上げ・教員）"
X_LABEL      = "割合(%)"
Y_LABEL      = "学校名"
LEGEND_TITLE = "頻度"

# Expected answer order (adjust if your survey differs)
LEAF_FREQ_ORDER = [
    "ほぼ毎時間",
    "1週間に数回程度",
    "1ヶ月に数回程度",
    "ほとんど使用していない",
]

# ---- Normalization & column finders -----------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\\[\\]「」『』,:;.-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def _find_col_exact_or_contains(df: pd.DataFrame, exact_jp_list: list[str], contains_any: list[str]) -> str | None:
    cols = list(df.columns)
    norm_cols = {c: _norm(c) for c in cols}

    # exact normalized match (priority order)
    for exact in (exact_jp_list or []):
        tgt = _norm(exact)
        for c, nc in norm_cols.items():
            if nc == tgt:
                return c

    # contains any token
    tokens = [_norm(t) for t in (contains_any or [])]
    for c, nc in norm_cols.items():
        if any(t in nc for t in tokens):
            return c
    return None

def find_col_respondent_id(df: pd.DataFrame):
    for c in ["respondent_id", "回答者ID", "回答ID", "id"]:
        if c in df.columns:
            return c
    return None

def find_col_leaf_freq(df: pd.DataFrame):
    exact = ["LEAFシステム(BookRoll,分析ツール)を授業・授業外(宿題など)でどれくらい利用していますか"]
    contains = ["leaf", "bookroll", "分析ツール", "利用頻度", "頻度"]
    return _find_col_exact_or_contains(df, exact, contains)

# ---- Canonicalize schools (same as q06) -------------------------------------
def ensure_school_canon(df: pd.DataFrame) -> str:
    """
    Use centralized SchoolCanonicalizer to create/ensure a canonical column.
    Returns the canonical column name; raises if unavailable.
    """
    col = SC.find_or_make_school_canon(df, debug=False)  # may add df['学校名_canon'] or df['school_canon']
    if isinstance(col, str):
        return col
    for name in ("school_canon", "学校名_canon"):
        if name in df.columns:
            return name
    raise KeyError("Canonical school column not found after SchoolCanonicalizer pass.")

# ---- Table maker ------------------------------------------------------------
def make_pct_table(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    order: list[str],
):
    """
    Returns (pct_df, n_per_group)
      pct_df: rows=groups, cols=order, values=percentage [0..100]
      n_per_group: respondent counts per group (int)
    """
    ct = (
        df.groupby([group_col, value_col])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=order, fill_value=0)
    )
    n = ct.sum(axis=1).astype(int)
    pct = ct.div(n.replace(0, np.nan), axis=0) * 100.0
    pct = pct.fillna(0.0)
    return pct, n

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
    out_csv: Path,
):
    # Save CSV (percentages + n)
    out = pct_df.copy()
    out["n"] = n_per_group
    out.to_csv(out_csv, encoding="utf-8", index=True)

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
    for col in pct_df.columns:
        vals = pct_df[col].values
        ax.barh(y_pos, vals, left=left, height=0.6, label=col,
                zorder=2, edgecolor="white", linewidth=0.5)
        left += vals

    _style_axes(ax)
    ax.set_yticks(y_pos, labels=y_labels, fontsize=TICK_FONTSIZE)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    # Legend (outside, right)
    leg = ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1.0), loc="upper left",
                    borderaxespad=0.0, frameon=False, fontsize=BASE_FONTSIZE)
    if leg and leg.get_title():
        leg.get_title().set_fontsize(BASE_FONTSIZE)

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
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the teacher cleaner first.")

    # Load (avoid FutureWarning by splitting replace/infer)
    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan})
    df = df.infer_objects(copy=False)

    # --- Canonicalize schools (same as your previous two scripts) ---
    school_col = ensure_school_canon(df)        # e.g., '学校名_canon'
    if school_col != "school_canon":            # mirror to expected name if needed
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # Respondent id (for TOP_N sorting later)
    rid_col = find_col_respondent_id(df)

    # Q08 column (LEAF usage)
    leaf_col = find_col_leaf_freq(df)
    if not leaf_col:
        raise KeyError("Column for Q08 'LEAFシステム利用頻度' not found (tried exact & contains match).")

    # Normalize frequency values and expose explicit '不明' for missing
    vals = df[leaf_col].astype(str)
    vals = vals.where(vals.str.strip().str.lower().ne("nan"), np.nan)  # handle literal "nan"
    df["_leaf"] = vals.fillna("不明").map(lambda s: ud.normalize("NFKC", s).strip())

    # Map into canonical category set (extend with 不明/その他 as needed)
    cats = list(LEAF_FREQ_ORDER)
    mapped = df["_leaf"].apply(lambda x: x if x in cats else ("不明" if x == "不明" else "その他"))
    if "不明" in mapped.unique() and "不明" not in cats:
        cats.append("不明")
    if "その他" in mapped.unique() and "その他" not in cats:
        cats.append("その他")
    df["_leaf_norm"] = mapped

    # ---- Filter by number of ANSWERED Q08 responses per school ---------------
    if MIN_RESP_PER_GROUP:
        answered = (
            df.assign(_has_ans=df["_leaf_norm"].ne("不明"))
              .groupby("school_canon")["_has_ans"]
              .sum()
        )
        keep = answered[answered >= MIN_RESP_PER_GROUP].index
        # Always keep 「不明」 school group if present
        if "不明" in df["school_canon"].unique():
            keep = keep.union(pd.Index(["不明"]))
        df = df[df["school_canon"].isin(keep)]

    # Optionally restrict to TOP_N by respondent count (after filtering)
    if TOP_N and rid_col:
        top = (
            df.groupby("school_canon")[rid_col]
              .nunique()
              .sort_values(ascending=False)
              .head(TOP_N)
              .index
        )
        df = df[df["school_canon"].isin(top)]

    # Make percentage table (rows=schools, cols=cats)
    pct_df, n_per_group = make_pct_table(df, "school_canon", "_leaf_norm", cats)

    # Sort schools by respondent count (desc) if we have respondent IDs
    if rid_col:
        order_idx = (
            df.groupby("school_canon")[rid_col]
              .nunique()
              .sort_values(ascending=False)
              .index
        )
        pct_df = pct_df.loc[order_idx.intersection(pct_df.index)]
        n_per_group = n_per_group.loc[pct_df.index]

    # Plot & write
    plot_bar_100pct_h(
        pct_df=pct_df,
        n_per_group=n_per_group,
        title=TITLE,
        xlabel=X_LABEL,
        ylabel=Y_LABEL,
        legend_title=LEGEND_TITLE,
        out_png=OUT_PNG,
        out_csv=OUT_CSV,
    )

if __name__ == "__main__":
    main()
