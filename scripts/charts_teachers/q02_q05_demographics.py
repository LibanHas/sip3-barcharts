# Usage:
#   python3 -m scripts.charts_teachers.q02_q05_demographics
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Project helpers (match students style)
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# (Optional) centralized canonicalizer if you later want to de-dup school names, etc.
try:
    from scripts.common.canonicalize import SchoolCanonicalizer as SC
except Exception:
    SC = None

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/demographics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Plot style -------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

# ---- Titles / Filenames -----------------------------------------------------
T_TITLE_AGE         = "年齢（分布）— 教員"
T_TITLE_SCHOOLTYPE  = "学校種（分布）— 教員"
T_TITLE_EXPERIENCE  = "教職経験年数（分布）— 教員"
T_TITLE_ROLE        = "役職（分布）— 教員"

FN_AGE_PNG        = OUT_DIR / "Q02_年齢.png"
FN_AGE_CSV        = OUT_DIR / "Q02_年齢.csv"
FN_SCHOOLTYPE_PNG = OUT_DIR / "Q03_学校種.png"
FN_SCHOOLTYPE_CSV = OUT_DIR / "Q03_学校種.csv"
FN_EXPERIENCE_PNG = OUT_DIR / "Q04_経験年数.png"
FN_EXPERIENCE_CSV = OUT_DIR / "Q04_経験年数.csv"
FN_ROLE_PNG       = OUT_DIR / "Q05_役職.png"
FN_ROLE_CSV       = OUT_DIR / "Q05_役職.csv"

# ---- Column finders (match students style) ---------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;.\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def _find_col_exact_or_contains(df: pd.DataFrame, exact_jp_list: list[str], contains_any: list[str]) -> str | None:
    cols = list(df.columns)
    norm_cols = {c: _norm(c) for c in cols}

    # 1) exact normalized match (in order)
    for exact in exact_jp_list or []:
        tgt = _norm(exact)
        for c, nc in norm_cols.items():
            if nc == tgt:
                return c

    # 2) contains (any of tokens)
    tokens = [_norm(t) for t in (contains_any or [])]
    for c, nc in norm_cols.items():
        if any(t in nc for t in tokens):
            return c
    return None

def find_col_age(df: pd.DataFrame) -> str | None:
    exact = ["年齢"]
    contains = ["年齢", "age"]
    return _find_col_exact_or_contains(df, exact, contains)

def find_col_schooltype(df: pd.DataFrame) -> str | None:
    # 学校種, 学校の種類, 学校タイプ, etc.
    exact = ["学校種"]
    contains = ["学校種", "学校の種類", "学校タイプ", "schooltype"]
    return _find_col_exact_or_contains(df, exact, contains)

def find_col_experience(df: pd.DataFrame) -> str | None:
    # 教職経験年数, 経験年数, 勤務年数, etc.
    exact = ["教職経験年数"]
    contains = ["経験年数", "教職経験", "勤務年数"]
    return _find_col_exact_or_contains(df, exact, contains)

def find_col_role(df: pd.DataFrame) -> str | None:
    exact = ["役職(主業務に最も近いものを選択してください。)"]
    contains = ["役職", "主業務", "職位", "ポジション"]
    return _find_col_exact_or_contains(df, exact, contains)

# ---- Plot helpers (students-style matplotlib) -------------------------------
def _style_axes(ax):
    ax.grid(axis="y", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def _bar_vertical_from_series(s: pd.Series, title: str, xlabel: str, ylabel: str, out_png: Path, out_csv: Path):
    # Save CSV
    s.to_csv(out_csv, encoding="utf-8", header=["count"])
    # Plot
    idx = s.index.astype(str).tolist()
    vals = s.values.astype(float)
    fig_h = max(3.8, 0.55 * len(idx) + 1.4) if len(idx) > 8 else 4.5
    fig, ax = plt.subplots(figsize=(max(7.8, 0.75*len(idx)+3.0), fig_h), dpi=PLOT_DPI)

    bars = ax.bar(range(len(idx)), vals, zorder=2, edgecolor="white", linewidth=0.5)
    _style_axes(ax)

    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels(idx, fontsize=TICK_FONTSIZE, rotation=20, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

    # annotate
    for i, b in enumerate(bars):
        v = b.get_height()
        if np.isnan(v) or v <= 0:
            continue
        ax.text(b.get_x() + b.get_width()/2, v + max(0.1, 0.02*max(vals)), f"{int(v)}",
                va="bottom", ha="center", fontsize=BASE_FONTSIZE)

    plt.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}")
    print(f"[info] wrote {out_csv}")

def _bar_horizontal_from_series(s: pd.Series, title: str, xlabel: str, ylabel: str, out_png: Path, out_csv: Path):
    # Save CSV
    s.to_csv(out_csv, encoding="utf-8", header=["count"])
    # Plot
    idx = s.index.astype(str).tolist()
    vals = s.values.astype(float)
    y_pos = np.arange(len(idx))
    fig_h = max(3.8, 0.7 * len(idx) + 1.2)
    fig, ax = plt.subplots(figsize=(11.0, fig_h), dpi=PLOT_DPI)

    bars = ax.barh(y_pos, vals, height=0.6, zorder=2, edgecolor="white", linewidth=0.5)
    _style_axes(ax)

    ax.set_yticks(y_pos, labels=idx, fontsize=TICK_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)

    # annotate
    xmax = float(np.nanmax(vals)) if len(vals) else 0.0
    pad  = 0.18 if xmax > 0 else 0.25
    ax.set_xlim(0, xmax * (1 + pad) + (0.6 if xmax < 6 else 0.5))

    for i, b in enumerate(bars):
        v = b.get_width()
        if np.isnan(v) or v <= 0:
            continue
        ax.text(max(v + 0.45, 0.8), i, f"{int(v)}", va="center", ha="left",
                fontsize=BASE_FONTSIZE, clip_on=False, zorder=5)

    plt.tight_layout()
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

    # Load (match students’ dtype handling)
    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    # (Optional) central canonical checks (no-ops if SC absent)
    if SC is not None and hasattr(SC, "assert_only_allowed"):
        try:
            SC.assert_only_allowed(df)
        except Exception:
            pass

    # Column resolution
    col_age        = find_col_age(df)
    col_schooltype = find_col_schooltype(df)
    col_experience = find_col_experience(df)
    col_role       = find_col_role(df)

    # Q2 年齢 — vertical bar
    if col_age:
        s = df[col_age].value_counts(dropna=False).sort_values(ascending=False)
        _bar_vertical_from_series(
            s,
            title=T_TITLE_AGE,
            xlabel="年齢", ylabel="人数",
            out_png=FN_AGE_PNG, out_csv=FN_AGE_CSV,
        )
    else:
        print("[warn] Missing column for 年齢")

    # Q3 学校種 — vertical bar
    if col_schooltype:
        s = df[col_schooltype].value_counts(dropna=False).sort_values(ascending=False)
        _bar_vertical_from_series(
            s,
            title=T_TITLE_SCHOOLTYPE,
            xlabel="学校種", ylabel="人数",
            out_png=FN_SCHOOLTYPE_PNG, out_csv=FN_SCHOOLTYPE_CSV,
        )
    else:
        print("[warn] Missing column for 学校種")

    # Q4 教職経験年数 — horizontal bar
    if col_experience:
        s = df[col_experience].value_counts(dropna=False).sort_values(ascending=True)
        _bar_horizontal_from_series(
            s,
            title=T_TITLE_EXPERIENCE,
            xlabel="人数", ylabel="カテゴリ",
            out_png=FN_EXPERIENCE_PNG, out_csv=FN_EXPERIENCE_CSV,
        )
    else:
        print("[warn] Missing column for 教職経験年数")

    # Q5 役職 — vertical bar
    if col_role:
        s = df[col_role].value_counts(dropna=False).sort_values(ascending=False)
        _bar_vertical_from_series(
            s,
            title=T_TITLE_ROLE,
            xlabel="役職", ylabel="人数",
            out_png=FN_ROLE_PNG, out_csv=FN_ROLE_CSV,
        )
    else:
        print("[warn] Missing column for 役職")

if __name__ == "__main__":
    main()
