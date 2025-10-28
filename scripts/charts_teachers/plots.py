# scripts/charts_teachers/plots.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from .style import apply_style


# ---------- filename helpers ----------
def _slug(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-一-龥ぁ-んァ-ヶー]", "", s)
    return s[:80]

def _ensure_outdir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

# scripts/charts_teachers/plots.py
def _finalize_and_save(obj, outpath: Path):
    """
    Accepts either a matplotlib Figure or Axes, tight-layouts, saves, and closes.
    """
    if isinstance(obj, plt.Figure):
        fig = obj
    else:  # assume Axes
        fig = obj.figure
    _ensure_outdir(outpath)
    fig.tight_layout()
    fig.savefig(outpath, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)


# ---------- basic bars ----------
def bar_simple(series: pd.Series, title: str, xlabel: str, ylabel: str, outpath: Path):
    apply_style("paper")
    s = series.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    sns.barplot(x=s.index.astype(str), y=s.values, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar_label(ax.containers[0], padding=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    _finalize_and_save(fig, outpath)

def bar_horizontal(series: pd.Series, title: str, xlabel: str, ylabel: str, outpath: Path):
    apply_style("paper")
    s = series.sort_values(ascending=True)
    fig_h = max(config.FIGSIZE[1], 0.35 * len(s))
    fig, ax = plt.subplots(figsize=(config.FIGSIZE[0], fig_h))
    sns.barplot(x=s.values, y=s.index.astype(str), ax=ax, orient="h")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ax.containers:
        ax.bar_label(ax.containers[0], padding=3)
    _finalize_and_save(fig, outpath)

# ---------- 100% stacked (Likert / frequency) ----------
# scripts/charts_teachers/plots.py

# scripts/charts_teachers/plots.py
def bar_100pct(
    tidy: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    legend_title: str = "回答",
    orientation: str = "v",          # "v" or "h"
    show_group_n: bool = False,      # ← NEW: show (N=) next to group labels
):
    """
    Accepts tidy table from prepare.pct_table(...):
      columns = [group_col(s), value_col, n, denom, pct] where pct in 0–1.
    """
    apply_style("paper")
    cols = tidy.columns.tolist()
    value_col = [c for c in cols if c not in ("n", "denom", "pct")][-1]
    group_cols = [c for c in cols if c not in ("n", "denom", "pct", value_col)]
    assert len(group_cols) == 1, "bar_100pct expects exactly one grouping column"
    gcol = group_cols[0]

    # pivot to groups x categories
    pv = tidy.pivot(index=gcol, columns=value_col, values="pct").fillna(0.0)
    if hasattr(tidy[value_col], "cat"):
        pv = pv[tidy[value_col].cat.categories]

    # map of denominators (respondents per group)
    denom_map = (
        tidy[[gcol, "denom"]]
        .drop_duplicates()
        .set_index(gcol)["denom"]
        .astype("Int64")
    )

    # dynamic figure size
    if orientation == "h":
        fig_h = max(4.5, 0.55 * len(pv.index))
        fig_w = max(7.5, 7.0)
    else:
        fig_w = max(7.5, 0.9 * len(pv.index))
        fig_h = max(4.5, 4.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if orientation == "h":
        left = np.zeros(len(pv))
        y = np.arange(len(pv.index))
        for cat in pv.columns:
            vals = pv[cat].values
            ax.barh(y, vals, left=left, label=str(cat))
            left += vals

        # y tick labels (optionally with N)
        if show_group_n:
            labels = [
                f"{idx}（N={int(denom_map.get(idx, 0))}）"
                for idx in pv.index
            ]
        else:
            labels = [str(idx) for idx in pv.index]
        ax.set_yticks(y)
        ax.set_yticklabels(labels)

        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([f"{int(t*100)}%" for t in np.linspace(0, 1, 6)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    else:
        left = np.zeros(len(pv))
        x = np.arange(len(pv.index))
        for cat in pv.columns:
            vals = pv[cat].values
            ax.bar(x, vals, bottom=left, label=str(cat))
            left += vals

        # x tick labels (optionally with N)
        if show_group_n:
            labels = [
                f"{idx}（N={int(denom_map.get(idx, 0))}）"
                for idx in pv.index
            ]
        else:
            labels = [str(idx) for idx in pv.index]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")

        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([f"{int(t*100)}%" for t in np.linspace(0, 1, 6)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    _finalize_and_save(fig, outpath)


# ---------- multiselect: one PNG per facet (pct labels) ----------
# scripts/charts_teachers/plots.py
# scripts/charts_teachers/plots.py
def multi_bars_by_facet(
    tidy: pd.DataFrame,
    title_prefix: str,
    outdir: Path,
    facet_col: str = "学校種",
    sort_by: str = "pct",       # 'pct' or 'count'
    top_k: int | None = None,   # show top-K choices within each facet
):
    apply_style("paper")
    """
    Expects tidy columns: [facet_col, 'choice', 'count', 'pct'].
    Saves one horizontal bar chart per facet.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if tidy.empty:
        return

    # choose metric safely
    metric = "pct" if (sort_by == "pct" and "pct" in tidy.columns) else "count"

    for facet, sub in tidy.groupby(facet_col, dropna=False):
        sub = sub.sort_values(metric, ascending=True)
        if top_k:
            sub = sub.tail(top_k)

        fig_h = max(3.8, 0.35 * len(sub))
        fig, ax = plt.subplots(figsize=(7.5, fig_h))
        bars = ax.barh(sub["choice"], sub[metric])

        # axis formatting
        if metric == "pct":
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.set_xlabel("割合(%)")
        else:
            ax.set_xlabel("人数")

        ax.set_ylabel("選択肢")
        ax.set_title(f"{title_prefix} — {facet}")

        # value labels (no duplicate 'x' arg)
        for p in bars:
            w = p.get_width()
            if metric == "pct" and w < 0.03:
                continue
            label = f"{w*100:.0f}%" if metric == "pct" else f"{int(w)}"
            ax.text(w, p.get_y() + p.get_height()/2, label,
                    va="center", ha="left", fontsize=9)

        _finalize_and_save(ax, outdir / f"{_slug(title_prefix)}_{_slug(facet)}.png")

# ---------- numeric charts ----------
def histogram_numeric(
    df_tidy: pd.DataFrame,
    value_col: str,
    title: str,
    outpath: Path,
    by: str | None = None,
    bins: int = 10,
):
    """
    df_tidy[value_col] should already be numeric.
    If 'by' is provided, one figure per facet is saved.
    """
    apply_style("paper")
    if by:
        for facet, sub in df_tidy.groupby(by):
            fig, ax = plt.subplots(figsize=config.FIGSIZE)
            sns.histplot(sub[value_col], bins=bins, ax=ax)
            ax.set_title(f"{title} — {facet}")
            ax.set_xlabel("値")
            ax.set_ylabel("人数")
            fn = f"{_slug(outpath.stem)}_{_slug(facet)}.png"
            _finalize_and_save(fig, outpath.parent / fn)
    else:
        fig, ax = plt.subplots(figsize=config.FIGSIZE)
        sns.histplot(df_tidy[value_col], bins=bins, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("値")
        ax.set_ylabel("人数")
        _finalize_and_save(fig, outpath)

def box_by_facet(
    df_tidy: pd.DataFrame,
    value_col: str,
    title: str,
    outpath: Path,
    by: str = "学校種",
):
    """
    One box per facet on the x-axis.
    """
    apply_style("paper")
    fig_w = max(7, 0.9 * len(df_tidy[by].unique()))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    sns.boxplot(data=df_tidy, x=by, y=value_col, ax=ax)
    sns.stripplot(data=df_tidy, x=by, y=value_col, ax=ax, dodge=True, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(by)
    ax.set_ylabel(value_col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    _finalize_and_save(fig, outpath)


def bar_mean_by_facet(df, value_col, by, title, ylabel, outpath):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(7, max(3, len(df[by].unique()) * 0.4)))
    means = df.groupby(by)[value_col].mean().sort_values()
    ax = sns.barplot(x=means.values, y=means.index, palette="crest")
    ax.bar_label(ax.containers[0], fmt="%.1f", fontsize=9)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
