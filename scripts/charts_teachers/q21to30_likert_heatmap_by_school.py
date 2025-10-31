# Usage:
#   python3 -m scripts.charts_teachers.q21to30_likert_heatmap_by_school
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re
import unicodedata as ud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# Project helpers / style (sets JP font, etc.)
try:
    from scripts.common.plotkit import setup  # optional
except Exception:
    setup = lambda: None

# Canonicalize schools exactly like other teacher scripts
from scripts.common.canonicalize import (
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
)

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/likert_by_school")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_N       = 1        # keep all schools that have >= MIN_N answers to the specific question
TOP_N       = 20       # show top-N schools by respondent count (None to disable)
ANNOTATE_GT = 10.0     # annotate cells >= this % with "xx%"

PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 11
LABEL_FONTSIZE = 12

# Likert order (left→right on the heatmap)
LIKERT_ORDER = [
    "あてはまる",
    "少しあてはまる",
    "あまりあてはまらない",
    "あてはまらない",
    "使っていない",
]

DISPLAY_ALIASES = {
    "少しあてはまる": "少し\nあてはまる",
    "あまりあてはまらない": "あまり\nあてはまらない",
    "使っていない": "使って\nいない",
}

FOOTNOTE = "分母＝各学校で当該設問に回答（1つ選択）した人数／ 小規模校も表示（n≥1）"

# ---- Color map (scarlet gradient) -------------------------------------------
CMAP = LinearSegmentedColormap.from_list(
    "scarlet",
    [
        "#ffffff",  # white (0%)
        "#ffe6e6",  # very light
        "#ffb3b3",  # light
        "#ff6b6b",  # mid
        "#e31a1c",  # scarlet
        "#8b0000",  # deep red
    ]
)

# ---- Q21–Q30 specs ----------------------------------------------------------
# exact Japanese prompts (for “exact match”); also provide token fallbacks
QUESTION_SPECS: List[Tuple[str, str, List[str]]] = [
    ("Q21", "児童生徒の理解度が高まったと感じる", ["理解度", "高まった"]),
    ("Q22", "授業準備時間が短縮した", ["授業準備", "短縮"]),
    ("Q23", "根拠に基づいた授業設計や教材改善を行えるようになった", ["授業設計", "教材改善", "根拠"]),
    ("Q24", "分析ツールを使用することで子供たちの学習方法に変化が生じた", ["学習方法", "変化"]),
    ("Q25", "個別最適な学びを実現する支援になったと感じる", ["個別最適"]),
    ("Q26", "主体的で対話的で深い学びを実現する支援になったと感じる", ["主体的", "対話的", "深い学び"]),
    ("Q27", "協働的な学びを実現する支援になったと感じる", ["協働的"]),
    ("Q28", "探究的な学びを実現する支援になったと感じる", ["探究的"]),
    ("Q29", "使い方・操作手順がわかりやすい", ["使い方", "操作手順"]),
    ("Q30", "LEAFシステムに満足している", ["満足"]),
]

# ---- Normalization helpers ---------------------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def find_col_likert(df: pd.DataFrame, exact_jp: str, tokens: List[str]) -> Optional[str]:
    # exact match first
    tgt = _norm(exact_jp)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    # fallback: all tokens contained
    toks = [_norm(t) for t in tokens]
    for c in df.columns:
        nc = _norm(c)
        if all(t in nc for t in toks):
            return c
    return None

# map variants to canonical labels in LIKERT_ORDER
def canon_likert(s: str) -> Optional[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = ud.normalize("NFKC", str(s)).strip()
    # tolerant contains checks
    if "使" in t and "ない" in t:
        return "使っていない"
    if "あてはまらない" in t:
        if "あまり" in t:
            return "あまりあてはまらない"
        return "あてはまらない"
    if "少し" in t and "あてはまる" in t:
        return "少しあてはまる"
    if "あてはまる" in t:
        return "あてはまる"
    # otherwise missing/other → None (excluded from pct denominator)
    return None

def wrap_jp(s: str, width: int = 6) -> str:
    return "\n".join([s[i:i+width] for i in range(0, len(s), width)])

# ---- Plotting ---------------------------------------------------------------
def plot_heatmap_full(
    pct_df: pd.DataFrame,
    n_per_school: pd.Series,
    title: str,
    x_label: str,
    out_png: Path,
    annotate_gt: float = ANNOTATE_GT,
):
    rows = pct_df.index.tolist()
    cols = LIKERT_ORDER
    M = pct_df.reindex(columns=cols, fill_value=0.0).values

    fig_w = max(12.0, 0.90 * len(cols) + 3.6)
    fig_h = max(6.0, 0.46 * len(rows) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PLOT_DPI)

    im = ax.imshow(
        M,
        aspect="auto",
        origin="upper",
        vmin=0, vmax=100,
        cmap=CMAP,
        interpolation="bicubic",
    )

    # x/y labels
    xlabels = [DISPLAY_ALIASES.get(c, c) for c in cols]
    xlabels = [wrap_jp(s, width=6) for s in xlabels]
    ylabels = [f"{r}（n={int(n_per_school.loc[r])}）" for r in rows]

    ax.set_xticks(np.arange(len(cols)), labels=xlabels, fontsize=TICK_FONTSIZE-1)
    ax.set_yticks(np.arange(len(rows)), labels=ylabels, fontsize=TICK_FONTSIZE)

    # frame only
    ax.minorticks_off(); ax.grid(False)
    for side in ["top","right","left","bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color((0,0,0,0.35))

    # colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("割合(%)", fontsize=LABEL_FONTSIZE)

    # annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = float(M[i, j])
            if v >= annotate_gt:
                txt_color = "white" if v >= 60 else "black"
                outline = "black" if txt_color == "white" else "white"
                ax.text(
                    j, i, f"{v:.0f}%",
                    ha="center", va="center",
                    fontsize=BASE_FONTSIZE-1, color=txt_color,
                    weight="bold" if v >= 80 else None,
                    path_effects=[pe.withStroke(linewidth=2, foreground=outline, alpha=0.9)],
                )

    ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)
    fig.text(0.01, -0.02, FOOTNOTE, ha="left", va="top", fontsize=10)

    plt.tight_layout(rect=[0, 0.14, 1, 1])
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}")

def plot_heatmap_top2(
    top2_pct: pd.Series,
    n_per_school: pd.Series,
    title: str,
    out_png: Path,
):
    # Single-column heatmap (Top-2%)
    rows = top2_pct.index.tolist()
    cols = ["Top2使用率(%)"]
    M = top2_pct.values.reshape(-1, 1)

    fig_w = 7.2
    fig_h = max(6.0, 0.46 * len(rows) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PLOT_DPI)

    im = ax.imshow(
        M,
        aspect="auto",
        origin="upper",
        vmin=0, vmax=100,
        cmap=CMAP,
        interpolation="bicubic",
    )

    ax.set_xticks([0], labels=["Top2使用\n率(%)"], fontsize=TICK_FONTSIZE)
    ylabels = [f"{r}（n={int(n_per_school.loc[r])}）" for r in rows]
    ax.set_yticks(np.arange(len(rows)), labels=ylabels, fontsize=TICK_FONTSIZE)

    ax.minorticks_off(); ax.grid(False)
    for side in ["top","right","left","bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color((0,0,0,0.35))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("割合(%)", fontsize=LABEL_FONTSIZE)

    # annotate (always show)
    for i, v in enumerate(M.ravel()):
        txt_color = "white" if v >= 60 else "black"
        outline = "black" if txt_color == "white" else "white"
        ax.text(
            0, i, f"{v:.0f}%",
            ha="center", va="center",
            fontsize=BASE_FONTSIZE-1, color=txt_color,
            weight="bold" if v >= 80 else None,
            path_effects=[pe.withStroke(linewidth=2, foreground=outline, alpha=0.9)],
        )

    ax.set_title(title + " — 学校別", fontsize=TITLE_FONTSIZE, pad=10)
    fig.text(0.01, -0.02, FOOTNOTE + "／ Top2＝『あてはまる＋少しあてはまる』", ha="left", va="top", fontsize=10)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
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

    for qcode, qtext, tokens in QUESTION_SPECS:
        col = find_col_likert(df, qtext, tokens)
        if not col:
            print(f"[WARN] {qcode}: column not found → skip")
            continue

        # Normalize answers to canonical Likert labels
        ans = df[col].map(canon_likert)

        # Keep only valid answers for the denominator
        mask_valid = ans.notna()
        sub = pd.DataFrame({"school_canon": df["school_canon"], "ans": ans})[mask_valid]

        if sub.empty:
            print(f"[WARN] {qcode}: no valid answers after normalization → skip")
            continue

        # Denominator per school (number who answered this question)
        n_per_school = sub.groupby("school_canon").size().astype(int)

        # Apply MIN_N/TOP_N
        keep = n_per_school[n_per_school >= MIN_N].sort_values(ascending=False)
        if TOP_N:
            keep = keep.head(TOP_N)
        schools = keep.index
        sub = sub[sub["school_canon"].isin(schools)]
        if sub.empty:
            print(f"[WARN] {qcode}: no schools after filters → skip")
            continue

        # Percentage table (rows=schools, cols=LIKERT_ORDER)
        ct = (
            sub.groupby(["school_canon", "ans"])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=LIKERT_ORDER, fill_value=0)
        )
        pct = (ct.divide(keep, axis=0) * 100.0).fillna(0.0)

        # Sort rows by n desc, then by positive share (あてはまる) desc
        pos_share = pct.get("あてはまる", pd.Series(0, index=pct.index))
        order = pd.DataFrame({"n": keep, "pos": pos_share}).sort_values(["n", "pos"], ascending=[False, False]).index
        pct = pct.loc[order]
        keep = keep.loc[order]

        # Save CSV (full distribution + n)
        out_csv = OUT_DIR / f"{qcode}_学校別_5分岐.csv"
        pct_out = pct.copy()
        pct_out["n"] = keep
        pct_out.to_csv(out_csv, encoding="utf-8")
        print(f"[info] wrote {out_csv}")

        # Plot full 5-option heatmap
        title = f"{qtext}（学校別・教員）"
        out_png = OUT_DIR / f"{qcode}_5分岐__学校別_heatmap.png"
        plot_heatmap_full(
            pct_df=pct,
            n_per_school=keep,
            title=title,
            x_label="回答（左：肯定／右：否定・未使用）",
            out_png=out_png,
        )

        # Top-2 (あてはまる＋少しあてはまる)
        top2 = (pct["あてはまる"].fillna(0.0) + pct["少しあてはまる"].fillna(0.0)).round(1)
        out_csv2 = OUT_DIR / f"{qcode}_Top2_学校別.csv"
        top2_out = pd.DataFrame({"Top2(%)": top2, "n": keep})
        top2_out.to_csv(out_csv2, encoding="utf-8")
        print(f"[info] wrote {out_csv2}")

        out_png2 = OUT_DIR / f"{qcode}_Top2__学校別_heatmap.png"
        plot_heatmap_top2(
            top2_pct=top2,
            n_per_school=keep,
            title=qtext,
            out_png=out_png2,
        )

if __name__ == "__main__":
    main()
