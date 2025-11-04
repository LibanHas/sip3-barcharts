# scripts/charts_teachers/q10_bookroll_features_heatmap_by_school.py
# Usage:
#   python3 -m scripts.charts_teachers.q10_bookroll_features_heatmap_by_school
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import re
import unicodedata as ud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

# Project helpers / style (uses your JP font if available)
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# Canonicalize schools exactly like other teacher scripts
from scripts.common.canonicalize import (
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
)

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/multi_by_school")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "Q10_BookRoll機能__学校別_heatmap.png"
OUT_CSV = OUT_DIR / "Q10_BookRoll機能__学校別_heatmap.csv"

# ---- Config -----------------------------------------------------------------
MIN_N       = 1      # minimum respondents in a school who ANSWERED Q10 to include row
TOP_N       = 20     # show top-N schools by n_answered (set None to disable)
ANNOTATE_GT = 10.0   # annotate cells >= this % with "xx%"

# NEW: toggle for weighted coloring (color ∝ % × n/n_max)
WEIGHT_BY_N = True

PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 11
LABEL_FONTSIZE = 12

TITLE    = "BookRollでよく使う機能（学校別・教員）"
FOOTNOTE = "色＝割合×人数の重み付け（各校の人数/最大人数）。文字ラベルは素の割合(%)。"

# --- Color map ---------------------------------------------------------------
CMAP = LinearSegmentedColormap.from_list(
    "scarlet",
    [
        "#ffffff",  # white (0%)
        "#ffe6e6",  # very light
        "#ffb3b3",  # light
        "#ff6b6b",  # mid
        "#e31a1c",  # scarlet / strong red
        "#8b0000",  # deep red (near 100%)
    ]
)

# Canonical feature set (column order for heatmap/CSV)
FEATURES_CANON: List[str] = [
    "メモ", "マーカー", "手書き", "クイズ",
    "URLリコメンド", "辞書", "タイマー",
    "ほとんど使ったことがない", "その他",
]

# Short display (x-axis only)
DISPLAY_ALIASES = {
    "URLリコメンド": "URL\nリコメンド",
    "ほとんど使ったことがない": "ほとんど\n使わない",
}

# ---- Parsing helpers ---------------------------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def find_col_q10(df: pd.DataFrame) -> Optional[str]:
    exact = "BookRollでよく使う機能を選んでください（複数選択可）"
    tgt = _norm(exact)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    tokens = [_norm("BookRoll"), _norm("機能"), _norm("複数選択")]
    for c in df.columns:
        if all(t in _norm(c) for t in tokens):
            return c
    return None

SEP_RE = re.compile(r"[;,／/、\s]+")

ALIASES = [
    (["メモ","memo"], "メモ"),
    (["マーカー","marker","ハイライト"], "マーカー"),
    (["手書き","ペン","pen","draw","手描き"], "手書き"),
    (["クイズ","quiz"], "クイズ"),
    (["urlリコメンド","url推薦","リンク推薦","レコメンド","recommend"], "URLリコメンド"),
    (["辞書","dictionary"], "辞書"),
    (["タイマー","timer"], "タイマー"),
    (["ほとんど使ったことがない","未使用","使わない"], "ほとんど使ったことがない"),
]

def _canon_feature(raw: str) -> Optional[str]:
    s = ud.normalize("NFKC", raw).strip()
    n = _norm(s)
    if not n:
        return None
    for needles, canon in ALIASES:
        if any(_norm(k) in n for k in needles):
            return canon
    if "その他" in s or "other" in n:
        return "その他"
    return "その他"

def explode_multiselect(series: pd.Series) -> pd.Series:
    """
    Convert multi-select column to list-of-features per respondent.
    Empty / 'nan' -> empty list.
    """
    out: List[List[str]] = []
    for x in series.astype(str).tolist():
        if x.strip().lower() in ("", "nan", "none"):
            out.append([])
            continue
        parts = [p for p in SEP_RE.split(x) if p.strip()]
        mapped = list({_canon_feature(p) for p in parts if _canon_feature(p)})
        mapped = [m for m in FEATURES_CANON if m in mapped]  # deterministic order
        out.append(mapped)
    return pd.Series(out, index=series.index)

def _wrap_jp(s: str, width: int = 6) -> str:
    return "\n".join([s[i:i+width] for i in range(0, len(s), width)])

# ---- Heatmap plotting --------------------------------------------------------
def plot_heatmap(pct: pd.DataFrame, n_answered: pd.Series, out_png: Path, weight_by_n: bool = True):
    rows = pct.index.tolist()
    cols = FEATURES_CANON

    # Matrix for text labels = raw percentages (0..100)
    M_pct = pct.reindex(columns=cols, fill_value=0.0).values

    # Matrix for color = optionally weighted by n / n_max
    if weight_by_n and len(n_answered) > 0:
        weights = (n_answered / n_answered.max()).values[:, None]  # (R,1) in [0..1]
        M_color = M_pct * weights
        cbar_label = "重み付き割合(%)"
    else:
        M_color = M_pct
        cbar_label = "割合(%)"

    fig_w = max(9.5, 0.65 * len(cols) + 2.5)
    fig_h = max(6.0, 0.46 * len(rows) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PLOT_DPI)

    im = ax.imshow(
        M_color,
        aspect="auto",
        origin="upper",
        vmin=0, vmax=100,
        cmap=CMAP,
        interpolation="bicubic",
    )

    # axis labels
    xlabels = [DISPLAY_ALIASES.get(c, c) for c in cols]
    xlabels = [_wrap_jp(s, width=6) for s in xlabels]
    ylabels = [f"{r}（n={int(n_answered.loc[r])}）" for r in rows]
    ax.set_xticks(np.arange(len(cols)), labels=xlabels, fontsize=TICK_FONTSIZE)
    ax.set_yticks(np.arange(len(rows)), labels=ylabels, fontsize=TICK_FONTSIZE)

    ax.minorticks_off()
    ax.grid(False)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color((0, 0, 0, 0.35))

    # colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=LABEL_FONTSIZE)

    # annotations show RAW % (not weighted)
    if ANNOTATE_GT is not None:
        for i in range(M_pct.shape[0]):
            for j in range(M_pct.shape[1]):
                v = float(M_pct[i, j])
                if v >= ANNOTATE_GT:
                    txt_color = "white" if (M_color[i, j] >= 60) else "black"
                    ax.text(
                        j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=BASE_FONTSIZE-1, color=txt_color,
                        weight="bold" if v >= 80 else None
                    )

    # title & footnote
    ax.set_title(TITLE, fontsize=TITLE_FONTSIZE, pad=10)
    if WEIGHT_BY_N:
        fig.text(0.01, -0.02, FOOTNOTE, ha="left", va="top", fontsize=10)
    else:
        fig.text(0.01, -0.02, "分母＝各学校でQ10に回答（1つ以上選択）した人数／ 小規模校も表示（n≥1）",
                 ha="left", va="top", fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
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
    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    # Canonicalize school; split 中/高 if ambiguous
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # Q10 multi-select column
    q10_col = find_col_q10(df)
    if not q10_col:
        raise KeyError("Q10 column (BookRollでよく使う機能…) not found in teachers_clean.csv")

    # Parse multi-select lists
    df = df.copy()
    df["_feat_list"] = explode_multiselect(df[q10_col])

    # Keep only respondents who answered Q10 (selected >=1 feature)
    answered_mask = df["_feat_list"].map(lambda v: isinstance(v, list) and len(v) > 0)
    df_ans = df.loc[answered_mask, ["school_canon", "_feat_list"]].copy()

    # Denominator per school
    n_by_school = df_ans.groupby("school_canon").size().sort_values(ascending=False).astype(int)

    # Apply MIN_N / TOP_N
    keep = n_by_school[n_by_school >= MIN_N]
    if TOP_N:
        keep = keep.head(TOP_N)
    keep_idx = keep.index

    df_ans = df_ans[df_ans["school_canon"].isin(keep_idx)]
    if df_ans.empty:
        raise RuntimeError("No schools passed MIN_N/TOP_N filters for Q10 heatmap.")

    # Long form: explode lists to one row per (school, feature)
    long = df_ans.explode("_feat_list").rename(columns={"_feat_list": "feature"})

    # Counts per (school, feature)
    ct = long.groupby(["school_canon", "feature"]).size().unstack(fill_value=0)

    # Ensure all canonical features as columns (ordered)
    for f in FEATURES_CANON:
        if f not in ct.columns:
            ct[f] = 0
    ct = ct[FEATURES_CANON]

    # Percentages within each school
    pct = (ct.divide(keep, axis=0) * 100.0).fillna(0.0).round(1)

    # Sort rows by n_answered desc, then by マーカー desc (arbitrary tie-breaker)
    order = (
        pd.DataFrame({"n": keep, "marker": pct.get("マーカー", pd.Series(0, index=pct.index))})
          .sort_values(["n", "marker"], ascending=[False, False])
          .index
    )
    pct = pct.loc[order]
    keep = keep.loc[order]

    # Save CSV (percentages + n_answered)
    out = pct.copy()
    out["n_answered"] = keep
    out.to_csv(OUT_CSV, encoding="utf-8")
    print(f"[info] wrote {OUT_CSV}")

    # Plot heatmap (weighted color, raw % labels)
    plot_heatmap(pct, keep, OUT_PNG, weight_by_n=WEIGHT_BY_N)

if __name__ == "__main__":
    main()
