# scripts/charts_teachers/q12to20_likert_heatmap_by_school.py
# Usage:
#   python3 -m scripts.charts_teachers.q12to20_likert_heatmap_by_school
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import re
import unicodedata as ud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------
# Project style helpers
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# Canonicalize schools (same as other teacher scripts)
from scripts.common.canonicalize import (
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
)

# ---- IO -------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/likert_by_school")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config ---------------------------------------------------------
MIN_N          = 1    # include schools with 1+ answers
TOP_N          = None # keep all; set to a number to cap rows
WEIGHT_BY_N    = True # <<< color intensity uses (% * n / max_n)
ANNOTATE_GT    = 10.0 # set to a number (e.g., 10.0) if you want to hide tiny labels

PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 11
LABEL_FONTSIZE = 12

# Likert, fixed order on X axis
LIKERT = [
    "頻繁に使用している",
    "使用することがある",
    "あまり使用しない",
    "全く使用しない",
    "当該機能を知らない",
]

# Scarlet soft gradient (white -> deep red)
CMAP = LinearSegmentedColormap.from_list(
    "scarlet",
    ["#ffffff", "#ffe6e6", "#ffb3b3", "#ff6b6b", "#e31a1c", "#8b0000"]
)

# --------------------------------------------------------------------
# Q12–Q20 stems (the exact Japanese in your questionnaire)
QUESTIONS: List[Tuple[str, str]] = [
    ("Q12", "教材を配布するため"),
    ("Q13", "児童・生徒に自分の解答を振り返らせるため"),
    ("Q14", "児童・生徒にクラスの人の解答を提示・参照させるため"),
    ("Q15", "児童・生徒にクラスの人の学び方を提示・参照させるため"),
    ("Q16", "児童・生徒にAIによる問題推薦やアドバイスを受けさせるため"),
    ("Q17", "データに基づいてグループを編成するため"),
    ("Q18", "クラス全体の傾向を見るため"),
    ("Q19", "個人の様子を見るため"),
    ("Q20", "取り組みのプロセスを見るため"),
]

# --------------------------------------------------------------------
# Text utils
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def _wrap_jp(s: str, width: int = 6) -> str:
    return "\n".join([s[i:i+width] for i in range(0, len(s), width)])

def _safe(s: str) -> str:
    s = ud.normalize("NFKC", s)
    s = re.sub(r"[\/\\:\*\?\"<>\|]", "_", s)
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    return s

# --------------------------------------------------------------------
# Column finder (robust: exact match first, fallback "contains" on stem)
def find_q_col(df: pd.DataFrame, stem: str) -> Optional[str]:
    tgt = _norm(stem)
    # exact (normalized) match
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    # contains: allow questionnaire prefixes/suffixes around the stem
    for c in df.columns:
        if tgt in _norm(c):
            return c
    return None

# --------------------------------------------------------------------
# Build 5-category percentage table per school
def pct_likert_by_school(df: pd.DataFrame, school_col: str, qcol: str) -> Tuple[pd.DataFrame, pd.Series]:
    vals = df[qcol].astype(str).map(lambda s: ud.normalize("NFKC", s).strip())
    vals = vals.where(vals.str.lower().ne("nan"), np.nan)

    sub = df.loc[vals.notna(), [school_col, qcol]].copy()
    if sub.empty:
        return pd.DataFrame(columns=LIKERT), pd.Series(dtype=int)

    ct = (
        sub.groupby([school_col, qcol]).size()
           .unstack(fill_value=0)
           .reindex(columns=LIKERT, fill_value=0)
    )
    n = ct.sum(axis=1).astype(int)
    pct = (ct.div(n.replace(0, np.nan), axis=0) * 100.0).fillna(0.0).round(1)
    return pct, n

# --------------------------------------------------------------------
# Heatmap plotting
def plot_heatmap_likert(title_left: str, pct_df: pd.DataFrame, n_by_school: pd.Series, out_png: Path):
    # order rows: larger n first, then higher “頻繁に使用している”
    order = (
        pd.DataFrame({
            "n": n_by_school,
            "top1": pct_df.get("頻繁に使用している", pd.Series(0, index=pct_df.index))
        }).sort_values(["n", "top1"], ascending=[False, False]).index
    )
    pct_df = pct_df.loc[order]
    n_by_school = n_by_school.loc[order]

    rows = pct_df.index.tolist()
    cols = LIKERT

    # Matrix for text (raw %)
    M_pct = pct_df.reindex(columns=cols, fill_value=0.0).values

    # Matrix for color (weighted or not)
    if WEIGHT_BY_N and len(n_by_school) > 0:
        weights = (n_by_school / n_by_school.max()).values[:, None]  # (schools x 1)
        M_color = M_pct * weights
        cbar_label = "重み付き割合(%)"
        footnote = "色＝割合×人数の重み付け（各校の人数/最大人数）。文字ラベルは素の割合(%)。"
    else:
        M_color = M_pct
        cbar_label = "割合(%)"
        footnote = "分母＝各学校で当該設問に回答（1つ選択）した人数／ 小規模校も表示（n≥1）"

    fig_w = max(11.5, 0.85 * len(cols) + 3.8)
    fig_h = max(6.0, 0.46 * len(rows) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PLOT_DPI)

    im = ax.imshow(
        M_color, aspect="auto", origin="upper",
        vmin=0, vmax=100, cmap=CMAP, interpolation="bicubic"
    )

    xlabels = [_wrap_jp(c, width=6) for c in cols]
    ylabels = [f"{r}（n={int(n_by_school.loc[r])}）" for r in rows]
    ax.set_xticks(np.arange(len(cols)), labels=xlabels, fontsize=TICK_FONTSIZE)
    ax.set_yticks(np.arange(len(rows)), labels=ylabels, fontsize=TICK_FONTSIZE)

    # Keep only outer frame
    ax.minorticks_off(); ax.grid(False)
    for side in ["top","right","left","bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color((0,0,0,0.35))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=LABEL_FONTSIZE)

    # Annotate cells with raw % (switchable threshold)
    for i in range(M_pct.shape[0]):
        for j in range(M_pct.shape[1]):
            v = float(M_pct[i, j])
            if (ANNOTATE_GT is None) or (v >= ANNOTATE_GT):
                # choose text color based on the WEIGHTED background for readability
                ink = "white" if (M_color[i, j] >= 60) else "black"
                outline = "black" if ink == "white" else "white"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=BASE_FONTSIZE-1, color=ink,
                        weight="bold" if v >= 80 else None,
                        path_effects=[pe.withStroke(linewidth=2, foreground=outline, alpha=0.9)])

    ax.set_title(f"{title_left}  — 学校別（5段階）", fontsize=TITLE_FONTSIZE, pad=10)
    fig.text(0.01, -0.02, footnote, ha="left", va="top", fontsize=10)

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # extra bottom room for wrapped x-labels
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}")

# --------------------------------------------------------------------
def main():
    setup()

    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the teacher cleaner first.")

    # Load
    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan})
    df = df.infer_objects(copy=False)

    # Canonicalize school; disambiguate 中/高 if needed
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # Iterate Q12–Q20
    for qid, stem in QUESTIONS:
        qcol = find_q_col(df, stem)
        if not qcol:
            print(f"[warn] {qid}: column not found for stem: {stem!r} — skipping.")
            continue

        # Build percentages
        pct_df, n_ans = pct_likert_by_school(df, "school_canon", qcol)
        if pct_df.empty:
            print(f"[warn] {qid}: no answers — skipping.")
            continue

        # Apply MIN_N/TOP_N across schools
        keep = n_ans[n_ans >= MIN_N]
        if TOP_N:
            keep = keep.sort_values(ascending=False).head(TOP_N)
        idx = keep.index
        pct_df = pct_df.loc[idx]
        n_ans  = keep

        # Save CSV
        out_csv = OUT_DIR / f"{qid}_{_safe(stem)}__学校別_likert.csv"
        csv_out = pct_df.copy()
        csv_out["n_answered"] = n_ans
        csv_out.to_csv(out_csv, encoding="utf-8")
        print(f"[info] wrote {out_csv}")

        # Plot
        out_png = OUT_DIR / f"{qid}_{_safe(stem)}__学校別_likert_heatmap.png"
        plot_heatmap_likert(stem, pct_df, n_ans, out_png)

if __name__ == "__main__":
    main()
