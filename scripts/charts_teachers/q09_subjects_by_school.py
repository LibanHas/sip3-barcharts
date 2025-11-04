# scripts/charts_teachers/q09_subjects_by_school.py
# Usage:
#   python3 -m scripts.charts_teachers.q09_subjects_by_school
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import re
import unicodedata as ud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------- Project helpers / style ----------
try:
    from scripts.common.plotkit import setup  # sets JP fonts/theme
except Exception:
    setup = lambda: None

from scripts.common.canonicalize import (
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
)

# ---------- IO ----------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/multi_by_school")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV_PCT    = OUT_DIR / "Q09_使用教科__学校別.csv"
OUT_CSV_COUNTS = OUT_DIR / "Q09_使用教科__学校別_counts.csv"
OUT_DEBUG      = OUT_DIR / "Q09_debug_rows.csv"
OUT_PNG        = OUT_DIR / "Q09_使用教科__学校別.png"

# ---------- Config ----------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

TITLE   = "LEAF使用教科（学校別・100%積み上げ・教員）"
X_LABEL = "割合（%）"
Y_LABEL = "学校名"
LEGEND_TITLE = "教科"

# Canonical subject labels (fixed order for charts/CSV)
SUBJECTS_CANON = [
    "国語",
    "社会",
    "算数・数学",
    "理科",
    "外国語",
    "その他",
]

# Keep the same school order as your other charts
ALLOWED_SCHOOL_ORDER: List[str] = [
    "西京高等学校付属中学校",
    "洗足学園高等学校",
    "西賀茂中学校",
    "洗足学園中学校",
    "岩沼小学校",
    "明徳小学校",
    "不明",
]

# Minimum answered Q9 responses to include a school
MIN_RESP_PER_SCHOOL = 1
TOP_N_SCHOOLS: Optional[int] = None  # e.g., 12 to limit height; None = all

# ---------- Column finders / normalizers ----------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def find_col_q9_subjects(df: pd.DataFrame) -> Optional[str]:
    exact = "LEAFシステム（BookRoll，分析ツール）をどの教科で使用しますか（複数選択可）"
    tgt = _norm(exact)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    # Fallback: token search
    tokens = [_norm("どの教科"), _norm("使用"), _norm("複数選択")]
    for c in df.columns:
        nc = _norm(c)
        if all(t in nc for t in tokens):
            return c
    return None

# ---------- Multi-select parsing ----------
# Accept common separators (half/full width), pipes, whitespace
SEP_RE = re.compile(r"[;,；，、／/|\s]+")

# Expanded alias table
SUBJ_ALIASES = [
    (["国語", "国語科", "現国", "現代文"], "国語"),
    (["社会", "社会科", "地歴", "公民"], "社会"),
    (["数学", "算数", "算", "数", "理数"], "算数・数学"),
    (["理科", "サイエンス", "科学"], "理科"),
    (["英語", "英語科", "外国語", "英会話"], "外国語"),
]
NONUSER_TOKENS = {"使用していない", "未使用", "使っていない"}

def _canon_subject(raw: str) -> Optional[str]:
    s = ud.normalize("NFKC", raw).strip()
    if not s:
        return None
    if s in NONUSER_TOKENS:
        return "__NONUSER__"
    s_norm = _norm(s)
    for needles, canon in SUBJ_ALIASES:
        if any(_norm(n) in s_norm for n in needles):
            return canon
    if "その他" in s:
        return "その他"
    return "その他"

def parse_multiselect(series: pd.Series) -> pd.DataFrame:
    """
    Returns DataFrame with:
      - subjects: list of canonical subjects (deduped, SUBJECTS_CANON order)
      - is_nonuser: True if response indicated non-use and no valid subjects
    """
    rows = []
    for x in series.astype(str).tolist():
        raw = x.strip()
        if raw.lower() in ("", "nan", "none"):
            rows.append({"subjects": [], "is_nonuser": False})
            continue
        parts = [p for p in SEP_RE.split(raw) if p.strip()]
        mapped = [m for m in (_canon_subject(p) for p in parts) if m]
        is_nonuser = ("__NONUSER__" in mapped)
        mapped = [m for m in mapped if m != "__NONUSER__"]
        mapped = [m for m in SUBJECTS_CANON if m in set(mapped)]
        rows.append({"subjects": mapped, "is_nonuser": is_nonuser and len(mapped) == 0})
    return pd.DataFrame(rows, index=series.index)

# ---------- Tables ----------
def make_tables(
    df: pd.DataFrame,
    group_col: str,
    value_list_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Build counts/percentages by school (rows) × subject (cols).
    Denominator = respondents with ≥1 subject selected for Q9.
    Returns:
      pct_df, cnt_df, n_answered (per school), n_nonusers (per school)
    """
    # denominator: answered with >=1 subject
    mask_used = df[value_list_col].map(lambda v: isinstance(v, list) and len(v) > 0)
    n_answered = df.loc[mask_used].groupby(group_col).size().astype(int).rename("n_answered")

    # non-users info
    n_nonusers = df.loc[df["_is_nonuser"]].groupby(group_col).size().astype(int)
    n_nonusers = n_nonusers.reindex(n_answered.index.union(n_nonusers.index)).fillna(0).astype(int)
    n_nonusers.name = "n_nonusers"

    # long form for counts
    long = (
        df.loc[mask_used, [group_col, value_list_col]]
          .explode(value_list_col)
          .rename(columns={value_list_col: "subject"})
    )
    ct = (
        long.groupby([group_col, "subject"])
            .size()
            .unstack(fill_value=0)
    )
    for s in SUBJECTS_CANON:
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[SUBJECTS_CANON]

    pct = (ct.divide(n_answered, axis=0) * 100.0).fillna(0.0)

    return pct, ct, n_answered, n_nonusers

# ---------- Plotting ----------
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
    # y labels with n
    base_labels = pct_df.index.tolist()
    y_labels = [f"{lab}（n={int(n_per_group.loc[lab])}）" for lab in base_labels]

    y_count = len(base_labels)
    fig_h = max(4.0, 0.65 * y_count + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    y_pos = np.arange(y_count)
    left = np.zeros(y_count, dtype=float)
    for col in pct_df.columns:  # SUBJECTS_CANON order already
        vals = pct_df[col].values
        ax.barh(y_pos, vals, left=left, height=0.6, label=col,
                zorder=2, edgecolor="white", linewidth=0.5)
        left += vals

    _style_axes(ax)
    ax.set_yticks(y_pos, labels=y_labels, fontsize=TICK_FONTSIZE)
    ax.invert_yaxis()  # first school at top
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    # Legend, fixed order
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

# ---------- Main ----------
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

    # Find Q9 column
    q9_col = find_col_q9_subjects(df)
    if not q9_col:
        raise KeyError("Q9 column (どの教科で使用しますか) not found in teachers_clean.csv")

    # Parse multi-select
    parsed = parse_multiselect(df[q9_col])
    df["_subjects_list"] = parsed["subjects"]
    df["_is_nonuser"]    = parsed["is_nonuser"]

    # Filter: keep schools with at least MIN_RESP_PER_SCHOOL answered (>=1 subject)
    mask_used = df["_subjects_list"].map(lambda v: isinstance(v, list) and len(v) > 0)
    answered = df.loc[mask_used].groupby("school_canon").size().astype(int)
    keep = answered[answered >= MIN_RESP_PER_SCHOOL].index
    # Always keep 「不明」 if present
    if "不明" in df["school_canon"].unique():
        keep = keep.union(pd.Index(["不明"]))
    df = df[df["school_canon"].isin(keep)]

    # Optionally restrict to TOP_N by answered count
    if TOP_N_SCHOOLS is not None:
        top = answered.sort_values(ascending=False).head(TOP_N_SCHOOLS).index
        df = df[df["school_canon"].isin(top)]

    # Build tables
    pct_df, cnt_df, n_answered, n_nonusers = make_tables(df, "school_canon", "_subjects_list")

    # Apply forced school order (and add any extras after)
    present = [s for s in ALLOWED_SCHOOL_ORDER if s in pct_df.index]
    extras  = [s for s in pct_df.index if s not in present]
    order_index = present + extras
    pct_df = pct_df.loc[order_index]
    cnt_df = cnt_df.loc[order_index]
    n_answered = n_answered.loc[order_index]
    n_nonusers = n_nonusers.loc[order_index]

    # Save CSVs (percentages + counts + denominators)
    out_pct = pct_df.copy()
    out_pct["n_answered"] = n_answered
    out_pct["n_nonusers"] = n_nonusers
    out_pct.to_csv(OUT_CSV_PCT, encoding="utf-8")
    print(f"[info] wrote {OUT_CSV_PCT}")

    out_cnt = cnt_df.copy()
    out_cnt["n_answered"] = n_answered
    out_cnt["n_nonusers"] = n_nonusers
    out_cnt.to_csv(OUT_CSV_COUNTS, encoding="utf-8")
    print(f"[info] wrote {OUT_CSV_COUNTS}")

    # Debug mapping file
    df[["school_canon", q9_col, "_subjects_list", "_is_nonuser"]].to_csv(
        OUT_DEBUG, index=False, encoding="utf-8"
    )
    print(f"[info] wrote {OUT_DEBUG}")

    # Plot single stacked 100% chart (all schools)
    # Only the canonical subjects are plotted; nonusers are excluded from bars.
    plot_bar_100pct_h(
        pct_df=pct_df[SUBJECTS_CANON],
        n_per_group=n_answered,
        title=TITLE,
        xlabel=X_LABEL,
        ylabel=Y_LABEL,
        legend_title=LEGEND_TITLE,
        out_png=OUT_PNG,
    )

if __name__ == "__main__":
    main()
