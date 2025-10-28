from pathlib import Path
import re
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

__all__ = ["setup", "grouped_hbar", "sns", "plt"]

_JP_FONT_CANDIDATES = [
    "Noto Sans CJK JP", "Noto Sans JP",
    "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Hiragino Maru Gothic ProN",
    "IPAexGothic", "IPAGothic",
]

def _pick_japanese_font(candidates=None):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in (candidates or _JP_FONT_CANDIDATES):
        if name in available:
            return name
    return None

def setup(theme: str = "whitegrid", context: str = "talk", font_family: Optional[str] = None):
    sns.set_theme(style=theme, context=context)
    chosen = font_family or _pick_japanese_font()
    if chosen:
        plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen

def _slug(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s))
    s = re.sub(r"[^\w\-]+", "", s)
    return s.strip("_")[:80] or "plot"

def grouped_hbar(
    df: pd.DataFrame,
    group_col: str,
    choice_col: str = "choice",
    count_col: Optional[str] = None,
    *,
    title: str = "",
    outpath: Optional[Path] = None,
    max_choices: int = 20,
    normalize: bool = False,
    legend_ncol: int = 1,
    row_height: float = 1.05,     # more vertical space per category
    bar_width: float = 0.92,      # thicker bars
    edgecolor: str = "white",     # visible separation between hue bars
    linewidth: float = 1.0,
    alpha: float = 0.98,
    show_values: bool = False,
    label_min: float = 8.0,
    write_counts_csv: bool = False,
):
    data = df.copy()
    if count_col is None:
        data = (
            data.groupby([group_col, choice_col])
                .size()
                .reset_index(name="count")
        )
        count_col = "count"
    if data.empty:
        return

    if normalize:
        totals = data.groupby(group_col)[count_col].transform("sum")
        data[count_col] = (data[count_col] / totals) * 100

    order = (
        data.groupby(choice_col)[count_col]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
    )[:max_choices]
    data = data[data[choice_col].isin(order)]
    cat = pd.Categorical(data[choice_col], categories=list(reversed(order)), ordered=True)
    data = data.assign(**{choice_col: cat})

    # figure height scales with number of categories
    h = max(4, row_height * len(order))
    plt.figure(figsize=(10, h))

    common = dict(
        data=data,
        y=choice_col, x=count_col,
        hue=group_col,
        width=bar_width,
        dodge=True,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
    )
    # seaborn 0.12+ uses errorbar; 0.11 uses ci
    try:
        ax = sns.barplot(**common, errorbar=None)
    except TypeError:
        ax = sns.barplot(**common, ci=None)

    ax.set_title(title)
    ax.set_xlabel("割合(%)" if normalize else "回答数")
    ax.set_ylabel("選択肢")
    leg = ax.legend(title=group_col, frameon=True, ncol=legend_ncol, loc="best")
    if leg:
        try:
            leg.set_draggable(True)
        except Exception:
            pass

    # value labels (only for sufficiently long bars)
    if show_values:
        # small nudge beyond bar tip
        offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
        for p in ax.patches:
            val = p.get_width()
            if pd.isna(val) or val < label_min:
                continue
            y = p.get_y() + p.get_height() / 2
            txt = f"{val:.0f}" if not normalize else f"{val:.1f}%"
            ax.text(val + offset, y, txt, va="center", ha="left", fontsize=9)

    plt.tight_layout()

    if write_counts_csv and outpath:
        csv_path = outpath.with_suffix(".csv")
        mat = data.pivot(index=choice_col, columns=group_col, values=count_col).fillna(0)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        mat.to_csv(csv_path, encoding="utf-8")
        print(f"[info] wrote {csv_path}")

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=220, bbox_inches="tight")
        plt.close()
    else:
        return ax.figure
