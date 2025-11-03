# scripts/charts_teachers/q06_leaf_subjects_by_school.py
# Usage:
#   python -m scripts.charts_teachers.q06_leaf_subjects_by_school
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import re
import unicodedata as ud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# --- project style (fonts/theme) ---------------------------------------------
try:
    from scripts.common.plotkit import setup
except Exception:
    setup = lambda: None

# Canonicalize schools exactly like other teacher charts
from scripts.common.canonicalize import (
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
)

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/likert_by_school")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: row order to match your other charts if present
ORDER_CSV  = Path("config/school_order_q6.csv")  # must have 'school_canon' column

# ---- Config -----------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 11
LABEL_FONTSIZE = 12
ANNOTATE_GT    = 10.0   # annotate cells >= 10%
MIN_N          = 1      # show schools with >=1 valid responses for this question
TOP_N          = None   # or an int if you want only top-N by respondent count

# Subjects we care about (canonical labels + token variants)
SUBJECTS = ["国語", "社会", "算数・数学", "理科", "外国語", "その他"]

# Color map (scarlet gradient, like your other batch)
CMAP = LinearSegmentedColormap.from_list(
    "scarlet",
    ["#ffffff", "#ffe6e6", "#ffb3b3", "#ff6b6b", "#e31a1c", "#8b0000"]
)

FOOTNOTE = "分母＝各学校で当該設問（複数選択）に何らか回答した人数／ 小規模校も表示（n≥1）"

# ---- Helpers ----------------------------------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def find_col_subjects(df: pd.DataFrame) -> Optional[str]:
    exact = "LEAFシステム(BookRoll,分析ツール)をどの教科で使用しますか(複数選択可)"
    tgt = _norm(exact)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    # relaxed fallback
    tokens = ["leafシステム", "どの教科", "使用", "複数選択"]
    for c in df.columns:
        nc = _norm(c)
        if all(t in nc for t in tokens):
            return c
    return None

# split multi-select answers into individual trimmed items
SEP_RE = re.compile(r"[;；、,/]|(?:\s*\+\s*)")

CANON_MAP: Dict[str, str] = {
    "国語": "国語",
    "社会": "社会",
    "算数": "算数・数学",
    "数学": "算数・数学",
    "算数・数学": "算数・数学",
    "理科": "理科",
    "外国語": "外国語",
    "英語": "外国語",
    "使用していない": "その他",
    "使用していない;": "その他",
    "使用していない。": "その他",
    "その他": "その他",
}

def canon_subject(item: str) -> Optional[str]:
    if item is None:
        return None
    t = ud.normalize("NFKC", str(item)).strip()
    if not t:
        return None
    # remove trailing punctuation commonly seen
    t = t.rstrip("。．.")
    # exact/alias map
    if t in CANON_MAP:
        return CANON_MAP[t]
    # light contains guards
    if "算数" in t or "数学" in t:
        return "算数・数学"
    if "国語" in t:
        return "国語"
    if "社会" in t:
        return "社会"
    if "理科" in t:
        return "理科"
    if "外国語" in t or "英語" in t:
        return "外国語"
    if "使用" in t and "ない" in t:
        return "その他"
    if "その他" in t:
        return "その他"
    return None  # unknown tokens dropped

def parse_multiselect(series: pd.Series) -> List[List[str]]:
    out: List[List[str]] = []
    for val in series.fillna(""):
        parts = [p.strip() for p in SEP_RE.split(str(val)) if p.strip()]
        out.append([p for p in parts if p])
    return out

def _load_q6_order() -> Optional[List[str]]:
    if not ORDER_CSV.exists():
        return None
    try:
        order_df = pd.read_csv(ORDER_CSV, dtype=str)
        col = "school_canon" if "school_canon" in order_df.columns else order_df.columns[0]
        order = order_df[col].dropna().tolist()
        order = [ud.normalize("NFKC", s) for s in order]
        return order
    except Exception:
        return None

def _reindex_like_q6(g: pd.DataFrame, order: Optional[List[str]]) -> pd.DataFrame:
    if not order:
        return g
    in_order = [idx for idx in order if idx in g.index]
    extras   = [idx for idx in g.index if idx not in set(order)]
    return g.reindex(in_order + extras)

def plot_heatmap(pct_df: pd.DataFrame, n_per_school: pd.Series, title: str, out_png: Path):
    rows = pct_df.index.tolist()
    cols = SUBJECTS
    M = pct_df.reindex(columns=cols, fill_value=0.0).values

    fig_w = max(12.0, 0.90 * len(cols) + 3.6)
    fig_h = max(6.0, 0.46 * len(rows) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PLOT_DPI)

    im = ax.imshow(
        M, aspect="auto", origin="upper",
        vmin=0, vmax=100, cmap=CMAP, interpolation="bicubic"
    )

    ax.set_xticks(np.arange(len(cols)), labels=cols, fontsize=TICK_FONTSIZE)
    ylabels = [f"{r}（n={int(n_per_school.loc[r])}）" for r in rows]
    ax.set_yticks(np.arange(len(rows)), labels=ylabels, fontsize=TICK_FONTSIZE)

    ax.minorticks_off(); ax.grid(False)
    for side in ["top","right","left","bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color((0,0,0,0.35))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("割合(%)", fontsize=LABEL_FONTSIZE)

    # annotate cells >= threshold
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = float(M[i, j])
            if v >= ANNOTATE_GT:
                txt_color = "white" if v >= 60 else "black"
                outline = "black" if txt_color == "white" else "white"
                ax.text(
                    j, i, f"{v:.0f}%",
                    ha="center", va="center",
                    fontsize=BASE_FONTSIZE-1, color=txt_color,
                    weight="bold" if v >= 80 else None,
                    path_effects=[pe.withStroke(linewidth=2, foreground=outline, alpha=0.9)],
                )

    ax.set_xlabel("教科（複数選択）", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)
    fig.text(0.01, -0.02, FOOTNOTE, ha="left", va="top", fontsize=10)

    plt.tight_layout(rect=[0, 0.14, 1, 1])
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}")

def main():
    setup()

    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the teacher cleaner first.")

    # Load
    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    # Canonicalize school; split 中/高 if ambiguous
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # Find the multi-select subjects column
    col = find_col_subjects(df)
    if not col:
        print("[ERROR] Q6 (subjects) column not found.")
        return
    print(f"[OK] subjects column: {col}")

    # Parse multi-select into canonical subjects
    lists = parse_multiselect(df[col])
    canon_lists = [[canon_subject(p) for p in lst] for lst in lists]
    canon_lists = [[p for p in lst if p] for lst in canon_lists]

    # Build a long dataframe (one row per subject selection)
    rows = []
    for school, lst in zip(df["school_canon"], canon_lists):
        if not lst:
            continue
        for sub in lst:
            rows.append((school, sub))
    long = pd.DataFrame(rows, columns=["school_canon", "subject"])
    if long.empty:
        print("[WARN] No valid subject selections found.")
        return

    # Denominator: number of respondents who chose at least one subject (by school)
    responders = (
        pd.Series([bool(lst) for lst in canon_lists])
          .groupby(df["school_canon"])
          .sum()
          .astype(int)
    )
    # Keep schools with >= MIN_N responders; optionally top-N
    keep = responders[responders >= MIN_N].sort_values(ascending=False)
    if TOP_N:
        keep = keep.head(TOP_N)
    schools = keep.index
    long = long[long["school_canon"].isin(schools)]
    if long.empty:
        print("[WARN] No schools after filters → skip")
        return

    # Cross-tab and percentages
    ct = (
        long.groupby(["school_canon", "subject"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=SUBJECTS, fill_value=0)
    )
    pct = (ct.divide(keep, axis=0) * 100.0).fillna(0.0)

    # Optional: enforce external school order
    order = _load_q6_order()
    pct  = _reindex_like_q6(pct, order)
    keep = keep.reindex(pct.index)

    # Save CSV
    out_csv = OUT_DIR / "Q06_教科別__学校別.csv"
    pct_out = pct.copy().round(1)
    pct_out["n"] = keep
    pct_out.to_csv(out_csv, encoding="utf-8")
    print(f"[info] wrote {out_csv}")

    # Plot heatmap
    title = "LEAFシステムをどの教科で使用しますか（学校別・教員）"
    out_png = OUT_DIR / "Q06_教科別__学校別_heatmap.png"
    plot_heatmap(pct, keep, title, out_png)

if __name__ == "__main__":
    main()
