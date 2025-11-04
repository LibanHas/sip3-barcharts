# Usage:
#   python3 -m scripts.charts_students.q03_school_type_overview
from __future__ import annotations
from pathlib import Path
from typing import Optional
import re
import unicodedata as ud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from scripts.common.plotkit import setup  # JP fonts etc.
except Exception:
    setup = lambda: None

from scripts.common.canonicalize import (
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
)

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/students_clean.csv")
OUT_DIR    = Path("figs/students/demographics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
PLOT_DPI        = 300
BASE_FONTSIZE   = 12
TITLE_FONTSIZE  = 16
TICK_FONTSIZE   = 11
LABEL_FONTSIZE  = 12

VALID_TYPES = ["小学校", "中学校", "高等学校"]

# ---- Helpers ----------------------------------------------------------------
def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    return s

def find_type_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "学校種" in str(c):
            return c
    return None

def ensure_school(df: pd.DataFrame) -> pd.DataFrame:
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)
    return df

def infer_type_from_name(name: str) -> str:
    """Heuristic: infer 小/中/高 from the canonical school name itself."""
    if pd.isna(name):
        return "不明"
    s = str(name)
    if "小学校" in s:
        return "小学校"
    if "中学校" in s:
        return "中学校"
    if "高等学校" in s:
        return "高等学校"
    return "不明"

# ---- Plots ------------------------------------------------------------------
def plot_overall_type_counts(vc: pd.Series, out_png: Path):
    # stable order
    vc = vc.reindex(VALID_TYPES).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=PLOT_DPI)
    bars = ax.bar(vc.index, vc.values, zorder=2)
    ax.grid(axis="y", linestyle=(0, (2, 6)), alpha=0.25, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.set_title("学校種の構成（全体・生徒）", fontsize=TITLE_FONTSIZE, pad=10)
    ax.set_ylabel("回答数", fontsize=LABEL_FONTSIZE)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    # labels
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height()+0.5,
                f"{int(b.get_height())}", ha="center", va="bottom",
                fontsize=BASE_FONTSIZE)
    fig.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, bbox_inches="tight", dpi=PLOT_DPI)
    plt.close(fig)

def plot_per_school_counts(df: pd.DataFrame, out_png: Path):
    # show n by school, colored by its inferred type (so bars aren’t all 100% stacks)
    g = (
        df.groupby(["school_canon", "school_type_inferred"])
          .size()
          .unstack(fill_value=0)
    )
    n = g.sum(axis=1).sort_values(ascending=False)
    g = g.loc[n.index]
    labels = [f"{idx}（n={int(n[idx])}）" for idx in g.index]
    y = np.arange(len(labels))

    colors = {"小学校": "#6baed6", "中学校": "#74c476", "高等学校": "#fd8d3c"}
    fig_h = max(3.8, 0.55*len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    left = np.zeros(len(g), dtype=float)
    for t in VALID_TYPES:
        vals = g.get(t, pd.Series(0, index=g.index)).values
        ax.barh(y, vals, left=left, height=0.6, color=colors[t], edgecolor="none", label=t)
        left += vals

    ax.set_yticks(y, labels=labels, fontsize=TICK_FONTSIZE)
    ax.set_xlabel("回答数", fontsize=LABEL_FONTSIZE)
    ax.set_title("学校別の回答数（推定学校種で色分け）", fontsize=TITLE_FONTSIZE, pad=10)
    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=0)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.invert_yaxis()
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              frameon=False, fontsize=10, title="学校種", title_fontsize=11)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, bbox_inches="tight", dpi=PLOT_DPI)
    plt.close(fig)

# ---- Main -------------------------------------------------------------------
def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"{DATA_CLEAN} が見つかりません。")

    df = pd.read_csv(DATA_CLEAN, dtype=str).replace({"": np.nan}).infer_objects(copy=False)
    df = ensure_school(df)

    # locate Q3
    q3 = find_type_col(df)
    if not q3:
        raise KeyError("『あなたの学校種を教えてください』の列が見つかりません。")

    # sanitize types (others -> 不明)
    df["school_type_reported"] = df[q3].map(lambda s: s if s in VALID_TYPES else "不明")

    # infer from canonical name and build audit
    df["school_type_inferred"] = df["school_canon"].map(infer_type_from_name)

    audit = df.loc[
        (df["school_type_reported"].ne("不明")) &
        (df["school_type_inferred"].ne("不明")) &
        (df["school_type_reported"] != df["school_type_inferred"]),
        ["school_canon", q3, "school_type_reported", "school_type_inferred"]
    ].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = OUT_DIR / "Q03_学校種_不整合監査.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8")

    # overall composition (counts + pct)
    vc = df["school_type_reported"].value_counts().reindex(VALID_TYPES).fillna(0).astype(int)
    overall_counts_csv = OUT_DIR / "Q03_学校種__全体_counts.csv"
    overall_pct_csv    = OUT_DIR / "Q03_学校種__全体_pct.csv"
    vc.to_csv(overall_counts_csv, header=["count"], encoding="utf-8")
    (vc / vc.sum() * 100).round(1).to_csv(overall_pct_csv, header=["pct"], encoding="utf-8")

    # per-school counts with inferred type color
    per_school_csv = OUT_DIR / "Q03_学校種__学校別_counts.csv"
    (
        df.groupby(["school_canon", "school_type_inferred"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=VALID_TYPES, fill_value=0)
          .assign(n_total=lambda t: t.sum(axis=1))
          .sort_values("n_total", ascending=False)
          .to_csv(per_school_csv, encoding="utf-8")
    )

    # plots
    plot_overall_type_counts(vc, OUT_DIR / "Q03_学校種__全体_counts.png")
    plot_per_school_counts(df, OUT_DIR / "Q03_学校別_counts_colored_by_type.png")

    print(f"[info] wrote {overall_counts_csv}")
    print(f"[info] wrote {overall_pct_csv}")
    print(f"[info] wrote {per_school_csv}")
    print(f"[info] wrote {audit_path}")

if __name__ == "__main__":
    main()
