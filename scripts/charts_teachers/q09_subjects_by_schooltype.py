# Usage:
#   python3 -m scripts.charts_teachers.q09_subjects_by_schooltype
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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

# Canonicalize schools exactly like q06
from scripts.common.canonicalize import SchoolCanonicalizer as SC, post_disambiguate_middle_vs_high

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/multi_by_schooltype")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Q09_使用教科__学校種別.csv"

# ---- Config -----------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

TITLE_TPL = "LEAF使用教科（学校種別・教員）— {schooltype}"
X_LABEL   = "教科"
Y_LABEL   = "割合(%)（その学校種の回答者のうち）"

# Canonical subject labels (order for charts/CSV)
SUBJECTS_CANON = [
    "国語",
    "社会",
    "算数・数学",
    "理科",
    "外国語",
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
    # fallback: fuzzy
    tgt = _norm("学校種")
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    return None

def find_col_q9_subjects(df: pd.DataFrame) -> Optional[str]:
    # Questionnaire label:
    exact = "LEAFシステム（BookRoll，分析ツール）をどの教科で使用しますか（複数選択可）"
    tgt = _norm(exact)
    for c in df.columns:
        if _norm(c) == tgt:
            return c
    # Fallback: contains tokens
    tokens = [_norm("どの教科"), _norm("使用"), _norm("複数選択")]
    for c in df.columns:
        nc = _norm(c)
        if all(t in nc for t in tokens):
            return c
    return None

# ---- Multi-select parsing ----------------------------------------------------
# Accept common separators and loose text; map to canonical 6 buckets
SEP_RE = re.compile(r"[;,／/、\s]+")

# Simple alias table (lowercased/normalized contains rules)
SUBJ_ALIASES = [
    (["国語"], "国語"),
    (["社会"], "社会"),
    (["数学", "算数"], "算数・数学"),
    (["理科", "サイエンス"], "理科"),
    (["英語", "外国語", "英会話"], "外国語"),
]

def _canon_subject(raw: str) -> Optional[str]:
    s = ud.normalize("NFKC", raw).strip()
    s_norm = _norm(s)
    if not s_norm:
        return None
    for needles, canon in SUBJ_ALIASES:
        if any(_norm(n) in s_norm for n in needles):
            return canon
    if "その他" in s or "other" in s_norm:
        return "その他"
    return "その他"

def explode_multiselect(series: pd.Series) -> pd.Series:
    """
    Turn a column of multi-select strings into a Series of lists of canonical subjects.
    Missing/empty -> empty list.
    """
    out = []
    for x in series.astype(str).tolist():
        if x.strip().lower() in ("", "nan", "none"):
            out.append([])
            continue
        parts = [p for p in SEP_RE.split(x) if p.strip()]
        mapped = list({_canon_subject(p) for p in parts if _canon_subject(p)})
        mapped = [m for m in SUBJECTS_CANON if m in mapped]  # deterministic order
        out.append(mapped)
    return pd.Series(out, index=series.index)

# ---- Plotting ---------------------------------------------------------------
def _style_axes(ax):
    ax.grid(axis="y", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def bar_pct_by_subject(subject_pct: pd.Series, title: str, out_png: Path):
    """
    subject_pct: index = SUBJECTS_CANON, values = percentage (0..100)
    """
    idx = [s for s in SUBJECTS_CANON if s in subject_pct.index]
    vals = subject_pct.reindex(idx).fillna(0.0).values

    fig_w = max(8.0, 1.0 * len(idx) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=PLOT_DPI)

    x = np.arange(len(idx))
    ax.bar(x, vals, width=0.7, zorder=2, edgecolor="white", linewidth=0.5)

    _style_axes(ax)
    ax.set_xticks(x, labels=idx, fontsize=TICK_FONTSIZE, rotation=0)
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

    # --- Canonicalize schools (same flow as q06) ---
    school_col = SC.find_or_make_school_canon(df, debug=False)  # returns created col name
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # --- Required columns: 学校種 + Q9 multi-select
    stype_col = find_col_schooltype(df)
    if not stype_col:
        raise KeyError("学校種 column not found in teachers_clean.csv")

    q9_col = find_col_q9_subjects(df)
    if not q9_col:
        raise KeyError("Q9 column (どの教科で使用しますか) not found in teachers_clean.csv")

    # --- Parse multi-select to lists of canonical subjects
    df = df.copy()
    df["_subjects_list"] = explode_multiselect(df[q9_col])

    # --- Build long-form using explode (avoid fillna([]))
    mask = df["_subjects_list"].map(lambda v: isinstance(v, list) and len(v) > 0)
    cols = [stype_col, "_subjects_list"]
    if "respondent_id" in df.columns:
        cols.append("respondent_id")

    long = (
        df.loc[mask, cols]
          .explode("_subjects_list")
          .rename(columns={"_subjects_list": "subject"})
    )
    if "respondent_id" in long.columns:
        long = long.rename(columns={"respondent_id": "rid"})
    else:
        long["rid"] = long.index.astype(str)

    # --- Denominators: respondents per 学校種 who answered Q9 (selected >=1 subject)
    denom = df.loc[mask].groupby(stype_col).size().astype(int).rename("denom")

    # --- Counts per (学校種, subject)
    ct = (
        long.groupby([stype_col, "subject"])
            .size()
            .unstack(fill_value=0)
    )
    # Ensure all canonical subjects exist as columns, in order
    for s in SUBJECTS_CANON:
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[SUBJECTS_CANON]

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
        out_png = OUT_DIR / f"Q09_使用教科__{schooltype}_学校種別.png"
        bar_pct_by_subject(row, title, out_png)

if __name__ == "__main__":
    main()
