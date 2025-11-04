# Usage:
#   python3 -m scripts.charts_teachers.q08_bookroll_features_by_school
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import unicodedata as ud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator

# ---- Project style (fonts/theme) --------------------------------------------
try:
    from scripts.common.plotkit import setup  # sets JP fonts, theme
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

OUT_PNG = OUT_DIR / "Q08_機能__学校別_heatmap.png"
OUT_CSV = OUT_DIR / "Q08_機能__学校別.csv"
DBG_CSV = OUT_DIR / "Q08_debug_rows.csv"

# Optional: enforce same order as Q6/Q7 (if you saved it)
ORDER_CSV = Path("config/school_order_q6.csv")  # change to q7 if you prefer

# ---- Config -----------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 11
LABEL_FONTSIZE = 12
ANNOTATE_GT    = 10.0  # annotate cells >= 10%
MIN_N_LABEL    = 1     # show schools with >=1 total respondents (matches Q7 labels)
TOP_N          = None  # or an int to truncate by n label (desc)

TITLE  = "BookRollでよく使う機能（学校別・教員）"
X_LAB  = "割合(%)"
Y_LAB  = "学校名"
FOOTNOTE = "分母＝各学校で当該設問（複数選択）に何らか回答した人数／ 小規模校も表示（n≥1）"

# Heatmap color (same scarlet family as Q6)
CMAP = LinearSegmentedColormap.from_list(
    "scarlet",
    ["#ffffff", "#ffe6e6", "#ffb3b3", "#ff6b6b", "#e31a1c", "#8b0000"]
)

# ---- Q8: features list + canonicalization -----------------------------------
FEATURES = [
    "メモ", "マーカー", "手書き", "クイズ", "URLリコメンド",
    "辞書", "タイマー", "その他", "ほとんど使ったことがない",
]

FEATURE_LABELS: Dict[str, str] = {
    # if you want to shorten labels on x-axis, map here (else they fall back to key)
    "URLリコメンド": "URL\nリコメンド",
    "ほとんど使ったことがない": "ほとんど\n使ったことがない",
}

# relaxed canonical map / contains guards
CANON_FEATS: Dict[str, str] = {
    "メモ": "メモ",
    "マーカー": "マーカー",
    "手書き": "手書き",
    "クイズ": "クイズ",
    "URLリコメンド": "URLリコメンド",
    "辞書": "辞書",
    "タイマー": "タイマー",
    "その他": "その他",
    "ほとんど使ったことがない": "ほとんど使ったことがない",
    # common alternates
    "テスト": "クイズ",
    "小テスト": "クイズ",
    "リンク推薦": "URLリコメンド",
    "おすすめurl": "URLリコメンド",
    "URL リコメンド": "URLリコメンド",
}

# ---- Column finding ----------------------------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def find_col_features(df: pd.DataFrame) -> Optional[str]:
    exact = "BookRollでよく使う機能を選んでください（複数選択可）"
    tgt = _norm(exact)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    tokens = ["bookroll", "よく使う", "機能", "複数選択"]
    for c in df.columns:
        nc = _norm(c)
        if all(t in nc for t in tokens):
            return c
    return None

def find_col_respondent_id(df: pd.DataFrame) -> Optional[str]:
    for c in ["respondent_id", "回答者ID", "回答ID", "id"]:
        if c in df.columns:
            return c
    return None

# ---- Parsing (robust to trailing separators like 'マーカー;') ----------------
SEP_RE = re.compile(r"[;；、,／/・]|(?:\s+\+\s+)|\s+")

def _clean_token(t: str) -> str:
    t = ud.normalize("NFKC", str(t))
    for z in ("\ufeff", "\u200b", "\u2060", "\u180e", "\u200d"):
        t = t.replace(z, "")
    return t.strip(" 　;；。、.\n\r\t")

def parse_multiselect(series: pd.Series) -> List[List[str]]:
    out: List[List[str]] = []
    for raw in series.fillna(""):
        s = _clean_token(raw)
        if not s:
            out.append([])
            continue

        # NEW: capture "TOKEN;;;;" -> ["TOKEN"]
        m_one = re.match(r"^\s*([^\s;；、,/／・]+?)\s*[;；、,/／・]+\s*$", s)
        if m_one:
            parts = [_clean_token(m_one.group(1))]
        else:
            parts = [p for p in (_clean_token(p) for p in SEP_RE.split(s)) if p]

        if not parts:
            # keyword scan fallback
            kw = list(CANON_FEATS.keys())
            found = []
            for k in kw:
                if k in s and k not in found:
                    found.append(k)
            parts = found

        if not parts and _clean_token(s):
            parts = ["その他"]

        out.append(parts)
    return out

def canon_feature(token: str) -> Optional[str]:
    if token is None:
        return None
    t = ud.normalize("NFKC", str(token)).strip().rstrip("。．.")
    if not t:
        return None
    if t in CANON_FEATS:
        return CANON_FEATS[t]
    # contains guards
    if "URL" in t or "リコメンド" in t:
        return "URLリコメンド"
    if "テスト" in t:
        return "クイズ"
    if "使ったことがない" in t or ("使用" in t and "ない" in t):
        return "ほとんど使ったことがない"
    if "メモ" in t: return "メモ"
    if "マーカー" in t: return "マーカー"
    if "手書" in t: return "手書き"
    if "辞書" in t: return "辞書"
    if "タイマ" in t: return "タイマー"
    if "その他" in t: return "その他"
    return None

# ---- Helpers ----------------------------------------------------------------
def _load_order() -> Optional[List[str]]:
    if not ORDER_CSV.exists():
        return None
    try:
        df = pd.read_csv(ORDER_CSV, dtype=str)
        col = "school_canon" if "school_canon" in df.columns else df.columns[0]
        order = df[col].dropna().map(lambda s: ud.normalize("NFKC", str(s))).tolist()
        return order
    except Exception:
        return None

def _reindex_like(g: pd.DataFrame, order: Optional[List[str]]) -> pd.DataFrame:
    if not order:
        return g
    in_order = [idx for idx in order if idx in g.index]
    extras   = [idx for idx in g.index if idx not in set(order)]
    return g.reindex(in_order + extras)

def _find_or_make_rid(df: pd.DataFrame) -> str:
    rid = find_col_respondent_id(df)
    if rid:
        return rid
    # fallback synthetic id so counts still work
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={"index": "_row_id"}, inplace=True)
    return "_row_id"

# ---- Plotting ---------------------------------------------------------------
def plot_heatmap(pct_df: pd.DataFrame, n_label: pd.Series, title: str, out_png: Path):
    rows = pct_df.index.tolist()
    cols = FEATURES
    M = pct_df.reindex(columns=cols, fill_value=0.0).values

    fig_w = max(12.0, 0.90 * len(cols) + 3.6)
    fig_h = max(6.0, 0.46 * len(rows) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PLOT_DPI)

    im = ax.imshow(
        M, aspect="auto", origin="upper",
        vmin=0, vmax=100, cmap=CMAP, interpolation="bicubic"
    )

    # x labels (wrap certain labels)
    xlabels = [FEATURE_LABELS.get(c, c) for c in cols]
    ax.set_xticks(np.arange(len(cols)), labels=xlabels, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="x", pad=6)

    # y labels with n from ALL respondents (Q7-style)
    ylabels = [f"{r}（n={int(n_label.loc[r])}）" for r in rows]
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

    ax.set_xlabel("BookRoll機能（複数選択）", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)
    fig.text(0.01, -0.02, FOOTNOTE, ha="left", va="top", fontsize=10)

    plt.tight_layout(rect=[0, 0.20, 1, 1])
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

    # Canonicalize school; split 中/高 if ambiguous (like Q7)
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # Respondent id (for stable counts/order)
    rid_col = _find_or_make_rid(df)

    # Q8 column
    col = find_col_features(df)
    if not col:
        raise KeyError("Q8 column not found (BookRoll features).")

    # Parse Q8 (per respondent)
    lists_raw = parse_multiselect(df[col])
    lists_canon = [[canon_feature(x) for x in row] for row in lists_raw]
    lists_canon = [[x for x in row if x] for row in lists_canon]

    # Debug dump (what *this* script saw)
    dbg = pd.DataFrame({
        "school_canon": df["school_canon"],
        rid_col: df[rid_col],
        "_q8_raw": df[col].astype(str),
        "_q8_parsed": [";".join(x) for x in lists_canon],
        "_q8_has_ans": [bool(x) for x in lists_canon],
    })
    dbg.to_csv(DBG_CSV, index=False, encoding="utf-8")
    print(f"[DEBUG] wrote row-level debug: {DBG_CSV}")

    # 1) n label (Q7-style): DISTINCT respondents per school (regardless of Q8 answer)
    n_label = (
        df.groupby("school_canon")[rid_col]
          .nunique()
          .astype(int)
    )

    # 2) Q8 responders: responded to Q8 at least with one feature (for percentages)
    responders_q8 = (
        pd.Series([len(v) > 0 for v in lists_canon])
          .groupby(df["school_canon"])
          .sum()
          .astype(int)
    )

    # Keep at least MIN_N_LABEL respondents in labels
    n_label = n_label[n_label >= MIN_N_LABEL].sort_values(ascending=False)

    # Optional TOP_N
    if TOP_N:
        n_label = n_label.head(TOP_N)

    # School order harmonized with Q6/Q7 if provided
    order = _load_order()
    if order:
        n_label = n_label.reindex(order).dropna().astype(int)

        desired = [
        "西京高等学校付属中学校",
        "洗足学園高等学校",
        "西賀茂中学校",
        "洗足学園中学校",
        "岩沼小学校",
        "明徳小学校",
        "不明",
    ]
    # keep only those present; append any extras not listed at the end
    present = [s for s in desired if s in n_label.index]
    extras  = [s for s in n_label.index if s not in desired]
    n_label = n_label.reindex(present + extras)

    # Build long df (one row per selected feature per teacher)
    rows: List[Tuple[str, str]] = []
    for school, feats in zip(df["school_canon"], lists_canon):
        for f in feats:
            rows.append((school, f))
    long = pd.DataFrame(rows, columns=["school_canon", "feature"])

    # Crosstab over FEATURES; reindex by n_label order
    ct = (
        long.groupby(["school_canon", "feature"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=n_label.index, columns=FEATURES, fill_value=0)
    )

    # Percentages: divide by Q8 responders per school (≥1 answer)
    den = responders_q8.reindex(ct.index).replace(0, np.nan)
    pct = (ct.divide(den, axis=0) * 100.0).fillna(0.0)

    # Save CSV (percentages + both denominators)
    out = pct.copy().round(1)
    out["n_q8_responders"] = den.fillna(0).astype(int)
    out["n"] = n_label
    out.to_csv(OUT_CSV, encoding="utf-8")
    print(f"[info] wrote {OUT_CSV}")

    # Plot heatmap
    plot_heatmap(pct_df=pct, n_label=n_label, title=TITLE, out_png=OUT_PNG)

if __name__ == "__main__":
    main()
