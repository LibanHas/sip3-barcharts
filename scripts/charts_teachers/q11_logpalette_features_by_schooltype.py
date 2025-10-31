# Usage:
#   python3 -m scripts.charts_teachers.q11_logpalette_features_by_schooltype
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import re
import unicodedata as ud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Project helpers / style
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# Canonicalize schools exactly like q06/q09/q10
from scripts.common.canonicalize import SchoolCanonicalizer as SC, post_disambiguate_middle_vs_high

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/multi_by_schooltype")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Q11_ログパレ機能__学校種別.csv"

# ---- Config -----------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

TITLE_TPL = "分析ツール（ログパレ）でよく使う機能（学校種別・教員）— {schooltype}"
X_LABEL   = "機能"
Y_LABEL   = "割合(%)（その学校種の回答者のうち）"

# Canonical LogPalette feature labels (order for charts/CSV)
FEATURES_CANON = [
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

# ---- Column finders / normalizers -------------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def find_col_schooltype(df: pd.DataFrame) -> Optional[str]:
    for cand in ["学校種", "school_type", "学校段階"]:
        if cand in df.columns:
            return cand
    tgt = _norm("学校種")
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    return None

def find_col_q11_logpalette(df: pd.DataFrame) -> Optional[str]:
    # Questionnaire label (セクション3・Q11)
    exact = "分析ツール（ログパレ）でよく使う機能を選んでください（複数選択可）"
    tgt = _norm(exact)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    # Fallback: contains tokens
    tokens = [_norm("分析ツール"), _norm("ログパレ"), _norm("機能"), _norm("複数選択")]
    for c in df.columns:
        nc = _norm(c)
        if all(t in nc for t in tokens):
            return c
    return None

# ---- Multi-select parsing ----------------------------------------------------
SEP_RE = re.compile(r"[;,／/、\s]+")

# Alias rules (width/case tolerant, matched by containment on normalized strings)
FEAT_ALIASES = [
    (["閲覧時間", "閲覧時間集計", "閲覧時間の集計", "viewtime"], "閲覧時間集計"),
    (["マーカー分析", "ハイライト分析", "marker"], "マーカー分析"),
    (["メモ分析", "note"], "メモ分析"),
    (["クイズ集計", "quiz"], "クイズ集計"),
    (["ワードクラウド", "wordcloud", "ワクラ"], "ワードクラウド"),
    (["ペンストローク", "ペン", "stroke", "pen"], "ペンストローク分析"),
    (["グループ編成", "グループ", "group"], "グループ編成"),
    (["ai推薦", "aiレコメンド", "レコメンド", "recommend"], "AI推薦"),
    (["理解度チェック", "理解度ダッシュボード", "dashboard"], "理解度チェックダッシュボード"),
    (["aiエージェント", "ai先生", "tammy", "agent"], "AIエージェント（AI先生・Tammy等）"),
    (["ほとんど使ったことがない", "未使用", "使わない"], "ほとんど使ったことがない"),
]

def _canon_feature(raw: str) -> Optional[str]:
    s = ud.normalize("NFKC", raw).strip()
    s_norm = _norm(s)
    if not s_norm:
        return None
    for needles, canon in FEAT_ALIASES:
        if any(_norm(n) in s_norm for n in needles):
            return canon
    if "その他" in s or "other" in s_norm:
        return "その他"
    return "その他"

def explode_multiselect(series: pd.Series) -> pd.Series:
    """
    Convert a multi-select string column to a Series of canonical feature lists.
    Missing/empty -> empty list.
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

# ---- Plotting ---------------------------------------------------------------
def _style_axes(ax):
    ax.grid(axis="y", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def _wrap_jp(s: str, width: int = 6) -> str:
    # Insert a newline every `width` characters (works fine for JP where spaces are rare)
    return "\n".join([s[i:i+width] for i in range(0, len(s), width)])

def bar_pct_by_feature(pct_row: pd.Series, title: str, out_png: Path):
    idx = [f for f in FEATURES_CANON if f in pct_row.index]
    vals = pct_row.reindex(idx).fillna(0.0).values

    # Wider figure to reduce crowding
    fig_w = max(10.5, 1.15 * len(idx) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.6), dpi=PLOT_DPI)

    x = np.arange(len(idx))
    ax.bar(x, vals, width=0.7, zorder=2, edgecolor="white", linewidth=0.5)

    _style_axes(ax)

    # WRAPPED labels
    display_labels = [_wrap_jp(s, width=6) for s in idx]
    ax.set_xticks(x, labels=display_labels, fontsize=TICK_FONTSIZE-1, rotation=0)

    ax.set_ylim(0, max(100, float(np.nanmax(vals) if len(vals) else 0) * 1.15))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(Y_LABEL, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=BASE_FONTSIZE)

    plt.tight_layout()
    plt.gcf().patch.set_facecolor("white")
    plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"[info] wrote {out_png}")

# ---- Main -------------------------------------------------------------------
def main():
    setup()

    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the teacher cleaner first.")

    # Load clean data (split replace/infer to avoid FutureWarning)
    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan})
    df = df.infer_objects(copy=False)

    # --- Canonicalize schools (same flow as prior scripts) ---
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # --- Required columns: 学校種 + Q11 multi-select
    stype_col = find_col_schooltype(df)
    if not stype_col:
        raise KeyError("学校種 column not found in teachers_clean.csv")

    q11_col = find_col_q11_logpalette(df)
    if not q11_col:
        raise KeyError("Q11 column（分析ツールでよく使う機能…）not found in teachers_clean.csv")

    # --- Parse multi-select into lists
    df = df.copy()
    df["_feat_list"] = explode_multiselect(df[q11_col])

    # --- Build long-form via explode
    mask = df["_feat_list"].map(lambda v: isinstance(v, list) and len(v) > 0)
    cols = [stype_col, "_feat_list"]
    if "respondent_id" in df.columns:
        cols.append("respondent_id")

    long = (
        df.loc[mask, cols]
          .explode("_feat_list")
          .rename(columns={"_feat_list": "feature"})
    )
    if "respondent_id" in long.columns:
        long = long.rename(columns={"respondent_id": "rid"})
    else:
        long["rid"] = long.index.astype(str)

    # --- Denominator per 学校種 = respondents who answered Q11 (selected >=1 feature)
    denom = df.loc[mask].groupby(stype_col).size().astype(int).rename("denom")

    # --- Counts per (学校種, feature)
    ct = (
        long.groupby([stype_col, "feature"])
            .size()
            .unstack(fill_value=0)
    )
    # Ensure all canonical features exist as columns, in order
    for f in FEATURES_CANON:
        if f not in ct.columns:
            ct[f] = 0
    ct = ct[FEATURES_CANON]

    # --- Percentages within each 学校種
    pct = (ct.divide(denom, axis=0) * 100.0).fillna(0.0).round(1)

    # --- Save combined CSV (percentages + denominators)
    out = pct.copy()
    out["n_answered"] = denom
    out.to_csv(OUT_CSV, encoding="utf-8")
    print(f"[info] wrote {OUT_CSV}")

    # --- One PNG per 学校種
    for schooltype, row in pct.iterrows():
        title = TITLE_TPL.format(schooltype=schooltype)
        out_png = OUT_DIR / f"Q11_ログパレ機能__{schooltype}_学校種別.png"
        bar_pct_by_feature(row, title, out_png)

if __name__ == "__main__":
    main()
