# Usage:
#   python3 -m scripts.charts_teachers.q11_logpalette_features_heatmap_by_school
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import re
import unicodedata as ud
import matplotlib.patheffects as pe
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

OUT_PNG = OUT_DIR / "Q11_ログパレ機能__学校別_heatmap.png"
OUT_CSV = OUT_DIR / "Q11_ログパレ機能__学校別_heatmap.csv"

# ---- Config -----------------------------------------------------------------
MIN_N       = 1      # minimum respondents in a school who ANSWERED Q11 to include row
TOP_N       = 20     # show top-N schools by n_answered (set None to disable)
ANNOTATE_GT = 10.0   # annotate cells >= this % with "xx%"

PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 11
LABEL_FONTSIZE = 12

TITLE    = "分析ツール（ログパレ）でよく使う機能（学校別・教員）"
FOOTNOTE = "分母＝各学校でQ11に回答（1つ以上選択）した人数／ MIN_Nフィルタ適用"

# ---- Color map (scarlet gradient) -------------------------------------------
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

# Canonical feature set for Q11 (column order for heatmap/CSV)
FEATURES_CANON: List[str] = [
    "閲覧時間集計",
    "マーカー分析",
    "メモ分析",
    "クイズ集計",
    "ワードクラウド",
    "ペンストローク分析",
    "グループ編成",
    "AI推薦",
    "理解度チェックダッシュボード",
    "AIエージェント（AI先生・Tammy等）",
    "ほとんど使ったことがない",
    "その他",
]

# Short display labels (x-axis only)
DISPLAY_ALIASES = {
    "ペンストローク分析": "ペンストローク\n分析",
    "理解度チェックダッシュボード": "理解度チェック\nダッシュボード",
    "AIエージェント（AI先生・Tammy等）": "AIエージェント\n（AI先生等）",
    "ほとんど使ったことがない": "ほとんど\n使わない",
}

# ---- Column finders / parsing -----------------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def find_col_q11(df: pd.DataFrame) -> Optional[str]:
    # Questionnaire wording (exact match first)
    exact = "分析ツール（ログパレ）でよく使う機能を選んでください（複数選択可）"
    tgt = _norm(exact)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    # Fallback: contains tokens
    tokens = [_norm("分析ツール"), _norm("ログパレ"), _norm("機能"), _norm("複数選択")]
    for c in df.columns:
        if all(t in _norm(c) for t in tokens):
            return c
    return None

# Split tokens on common separators
SEP_RE = re.compile(r"[;,／/、\s]+")

# Aliases (lowercased, tolerant)
ALIASES = [
    (["閲覧時間", "閲覧時間集計", "viewtime"], "閲覧時間集計"),
    (["マーカー分析", "marker分析", "ハイライト分析"], "マーカー分析"),
    (["メモ分析", "memo分析"], "メモ分析"),
    (["クイズ集計", "quiz集計", "クイズ分析"], "クイズ集計"),
    (["ワードクラウド", "wordcloud"], "ワードクラウド"),
    (["ペンストローク", "ストローク", "penstroke"], "ペンストローク分析"),
    (["グループ編成", "group編成"], "グループ編成"),
    (["ai推薦", "aiリコメンド", "推薦", "レコメンド"], "AI推薦"),
    (["理解度チェック", "ダッシュボード", "dashboard"], "理解度チェックダッシュボード"),
    (["aiエージェント", "ai先生", "tammy"], "AIエージェント（AI先生・Tammy等）"),
    (["ほとんど使ったことがない", "未使用", "使わない"], "ほとんど使ったことがない"),
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
def plot_heatmap(pct: pd.DataFrame, n_answered: pd.Series, out_png: Path):
    rows = pct.index.tolist()
    cols = FEATURES_CANON
    M = pct.reindex(columns=cols, fill_value=0.0).values

    # Wider canvas for many columns
    fig_w = max(12.0, 0.90 * len(cols) + 3.6)
    fig_h = max(6.0, 0.46 * len(rows) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PLOT_DPI)

    # Heatmap UNDER annotations
    im = ax.imshow(
        M, aspect="auto", origin="upper",
        vmin=0, vmax=100, cmap=CMAP, interpolation="bicubic",
    )
    im.set_zorder(1)

    # X/Y labels
    xlabels = [DISPLAY_ALIASES.get(c, c) for c in cols]
    xlabels = [_wrap_jp(s, width=4) for s in xlabels]
    ylabels = [f"{r}（n={int(n_answered.loc[r])}）" for r in rows]
    ax.set_xticks(np.arange(len(cols)), labels=xlabels, fontsize=TICK_FONTSIZE-1)
    ax.set_yticks(np.arange(len(rows)), labels=ylabels, fontsize=TICK_FONTSIZE)

    # Only outer border
    ax.minorticks_off(); ax.grid(False)
    for side in ["top","right","left","bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color((0,0,0,0.35))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("割合(%)", fontsize=LABEL_FONTSIZE)

    # --- Annotations: always on top, with halo for contrast ---
    thresh = ANNOTATE_GT if ANNOTATE_GT is not None else 0.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = float(M[i, j])
            if v >= thresh:
                ink = "white" if v >= 60 else "black"
                outline = "black" if ink == "white" else "white"
                ax.text(
                    j, i, f"{v:.0f}%",
                    ha="center", va="center",
                    fontsize=BASE_FONTSIZE-1,
                    color=ink,
                    weight="bold" if v >= 80 else None,
                    zorder=3,
                    path_effects=[pe.withStroke(linewidth=2, foreground=outline, alpha=0.9)],
                )

    ax.set_title(TITLE, fontsize=TITLE_FONTSIZE, pad=10)

    # Extra room at bottom for wrapped x-labels
    plt.tight_layout(rect=[0, 0.14, 1, 1])
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
    df = df.replace({"": np.nan})
    df = df.infer_objects(copy=False)

    # Canonicalize school; split 中/高 if ambiguous
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # Q11 multi-select column
    q11_col = find_col_q11(df)
    if not q11_col:
        raise KeyError("Q11列（分析ツール／ログパレの機能：複数選択）を teachers_clean.csv から検出できませんでした。")

    # Parse multi-select lists
    df = df.copy()
    df["_feat_list"] = explode_multiselect(df[q11_col])

    # Keep only respondents who answered Q11 (selected >=1 feature)
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
        raise RuntimeError("No schools passed MIN_N/TOP_N filters for Q11 heatmap.")

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

    # Sort rows by n_answered desc, then by 閲覧時間集計 desc (arbitrary stable secondary key)
    order = (
        pd.DataFrame({"n": keep, "view": pct.get("閲覧時間集計", pd.Series(0, index=pct.index))})
          .sort_values(["n", "view"], ascending=[False, False])
          .index
    )
    pct = pct.loc[order]
    keep = keep.loc[order]

    # Save CSV (percentages + n_answered)
    out = pct.copy()
    out["n_answered"] = keep
    out.to_csv(OUT_CSV, encoding="utf-8")
    print(f"[info] wrote {OUT_CSV}")

    # Plot heatmap
    plot_heatmap(pct, keep, OUT_PNG)

if __name__ == "__main__":
    main()
