# Usage:
#   python3 -m scripts.charts_teachers.q06_ict_by_school
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

# Project helpers (students style)
from scripts.common.plotkit import setup  # sets JP font etc.
from scripts.common.canonicalize import SchoolCanonicalizer as SC, post_disambiguate_middle_vs_high

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/likert_by_school")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Q06_ICT頻度__学校別.png"
OUT_CSV = OUT_DIR / "Q06_ICT頻度__学校別.csv"

# ---- Config / Filters -------------------------------------------------------
MIN_N  = 1     # keep schools with >= MIN_N answers to Q6
TOP_N  = 12    # show top-N schools by respondent count (set None to disable)

TITLE        = "授業でのICT活用（学校別・100%積み上げ・教員）"
X_LABEL      = "割合(%)"
Y_LABEL      = "学校名"
LEGEND_TITLE = "頻度"

# Expected answer order (high -> low frequency)
ICT_FREQ_ORDER = [
    "ほぼ毎時間",
    "1週間に数回程度",
    "1ヶ月に数回程度",
    "ほとんど使用していない",
]

# ---- Visual policy ----------------------------------------------------------
SMALL_N_THRESHOLD = 3          # hatch + transparency when n < this
SORT_BY_CATEGORY  = "ほぼ毎時間"  # sort within n-blocks by this % (desc)

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

def find_col_ict_freq(df: pd.DataFrame):
    exact = ["授業でのICT活用"]  # your Q6 header
    contains = ["ict", "授業", "活用", "利用頻度", "頻度"]
    return _find_col_exact_or_contains(df, exact, contains)

# ---- Canonicalize schools ---------------------------------------------------
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

# ---- Colors -----------------------------------------------------------------
def _category_colors(categories: list[str]) -> dict[str, tuple]:
    """
    Darker = more frequent use (semantic mapping).
    We assign explicit shades by label so the meaning is stable regardless of order.
    """
    cmap = plt.cm.Blues
    shade = {
        "ほぼ毎時間":        cmap(0.90),  # darkest
        "1週間に数回程度":    cmap(0.70),
        "1ヶ月に数回程度":    cmap(0.50),
        "ほとんど使用していない": cmap(0.30),  # lightest
        "不明":              (0.70, 0.70, 0.70, 1.0),  # neutral grey if ever shown
        "その他":            (0.55, 0.55, 0.55, 1.0),
    }
    return {c: shade.get(c, cmap(0.40)) for c in categories}

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
    # Save CSV (percentages rounded + n)
    out = pct_df.round(1).copy()
    out["n"] = n_per_group
    out.to_csv(out_csv, encoding="utf-8", index=True)

    # Labels with n
    base_labels = pct_df.index.tolist()
    y_labels = [f"{lab}（n={int(n_per_group.loc[lab])}）" for lab in base_labels]

    # Colors (dark -> light)
    cat_colors = _category_colors(list(pct_df.columns))

    # Figure
    y_count = len(base_labels)
    fig_h = max(4.6, 0.65 * y_count + 1.8)
    fig, ax = plt.subplots(figsize=(12.2, fig_h), dpi=PLOT_DPI)

    # Stacked bars with small-N styling
    y_pos = np.arange(y_count)
    left = np.zeros(y_count, dtype=float)
    for col in pct_df.columns:
        vals = pct_df[col].values
        bars = ax.barh(
            y_pos, vals, left=left, height=0.6,
            label=col, color=cat_colors[col],
            edgecolor="white", linewidth=0.6, zorder=2
        )
        for i, b in enumerate(bars):
            if int(n_per_group.loc[base_labels[i]]) < SMALL_N_THRESHOLD:
                b.set_hatch("//")
                b.set_alpha(0.45)
        left += vals

    _style_axes(ax)
    ax.set_yticks(y_pos, labels=y_labels, fontsize=TICK_FONTSIZE)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_xlabel(f"{xlabel}　（{legend_title}）", fontsize=LABEL_FONTSIZE, labelpad=8)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)

    # Reserve a bottom band for legend (top line) + footnote (bottom line)
    # rect = [left, bottom, right, top] in figure coords
    plt.tight_layout(rect=[0.02, 0.125, 0.98, 0.90])   # ↓ even smaller bottom margin

    # Legend on its own line between xlabel and footnote
    handles = [Patch(facecolor=cat_colors[c], edgecolor="none", label=c) for c in pct_df.columns]
    fig.legend(
    handles=handles,
    loc="lower center",
    ncol=len(handles),
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.082),  # slightly closer to the x-axis
    borderaxespad=0.0,
    handlelength=1.2, handleheight=0.6,
    columnspacing=0.9, labelspacing=0.35,
    )

    # Footnote at the very bottom of the reserved band
    fig.text(
    0.5, 0.048,
    "※ ハッチの学校は少数サンプル（n=1）は参考値としてご覧ください。",
    ha="center", va="center", fontsize=11, color="#555"
    )

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
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    # Canonicalize schools
    school_col = ensure_school_canon(df)        # e.g., '学校名_canon'
    if school_col != "school_canon":            # mirror to expected name if needed
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # Respondent id (for TOP_N sorting later)
    rid_col = find_col_respondent_id(df)

    # Q6 column
    ict_col = find_col_ict_freq(df)
    if not ict_col:
        raise KeyError("Column for Q6 '授業でのICT活用' not found (tried exact & contains match).")

    # Normalize frequency values and expose explicit '不明' for missing
    vals = df[ict_col].astype(str)
    vals = vals.where(vals.str.strip().str.lower().ne("nan"), np.nan)  # guard literal "nan"
    df["_ict"] = vals.fillna("不明").map(lambda s: ud.normalize("NFKC", s).strip())

    # Filter by number of ANSWERED Q6 responses per school
    if MIN_N:
        answered = (
            df.assign(_has_ans=df["_ict"].ne("不明"))
              .groupby("school_canon")["_has_ans"]
              .sum()
        )
        keep = answered[answered >= MIN_N].index
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

    # Build plotting order for categories (add '不明' and 'その他' if present)
    cats = list(ICT_FREQ_ORDER)
    if "不明" in df["_ict"].unique() and "不明" not in cats:
        cats.append("不明")
    mapped = df["_ict"].apply(lambda x: x if x in cats else "その他")
    if "その他" in mapped.unique() and "その他" not in cats:
        cats.append("その他")
    df["_ict_norm"] = mapped

    # Make percentage table (rows=schools, cols=cats)
    pct_df, n_per_group = make_pct_table(df, "school_canon", "_ict_norm", cats)

    # ---- SORTING ----
    # 「不明」 first (if present), then ascending n (1, 3, 4, 8, ...).
    # Within each n-block, sort by % of SORT_BY_CATEGORY (desc).
    sort_col = SORT_BY_CATEGORY if SORT_BY_CATEGORY in pct_df.columns else None

    ordered_blocks: list[list[str]] = []
    if "不明" in pct_df.index:
        ordered_blocks.append(["不明"])

    rem_idx = [i for i in pct_df.index if i != "不明"]
    for n_val in sorted(n_per_group.loc[rem_idx].unique()):
        block = [i for i in rem_idx if n_per_group.loc[i] == n_val]
        if sort_col:
            block = list(pct_df.loc[block].sort_values(sort_col, ascending=False).index)
        ordered_blocks.append(block)

    final_idx = [i for block in ordered_blocks for i in block]
    pct_df = pct_df.loc[final_idx]
    n_per_group = n_per_group.loc[final_idx]

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
