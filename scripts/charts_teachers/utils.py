# charts/utils.py
from __future__ import annotations
from pathlib import Path
import math
import matplotlib.pyplot as plt

# ---------- filesystem ----------
def _ensure_dir(path: Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def savefig(fig, outpath, facecolor="white", dpi=300, tight=True):
    """
    Save figure with sensible defaults (crisp + tight) and close it.
    """
    outpath = Path(outpath)
    _ensure_dir(outpath.parent)
    if tight:
        fig.tight_layout()
    fig.savefig(outpath, facecolor=facecolor, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- annotation helpers ----------
def _is_horizontal_bar(patch) -> bool:
    """
    Heuristic: horizontal bars (orient='h') have 'width' equal to the data value
    and a very small 'height'. Vertical bars invert that.
    """
    return patch.get_width() >= patch.get_height()

def _fmt_value(val: float, mode: str, decimals: int) -> str:
    if mode == "pct":
        return f"{val*100:.{decimals}f}%"
    # count
    # Use thousands separators, keep integers without decimals
    if abs(val - int(val)) < 1e-9:
        return f"{int(val):,}"
    return f"{val:,.{decimals}f}"

def annotate_bars(
    ax,
    mode: str = "auto",       # "auto" | "pct" | "count"
    decimals: int = 0,
    min_visible: float = 0.01, # for mode pct/auto: skip if <1 percentage point (0.01)
    pad: float = 0.01          # offset from bar end (in axis units)
):
    """
    Label bars with values. Works for horizontal or vertical bars (Seaborn/Matplotlib).
    - mode="auto": if all bar magnitudes <=1 → format as %; else as counts.
    - min_visible: for % (0–1), don’t annotate bars under this fraction.
    """
    patches = [p for p in ax.patches if p.get_height() != 0 or p.get_width() != 0]
    if not patches:
        return

    # Determine auto mode by inspecting bar magnitudes
    values = [p.get_width() if _is_horizontal_bar(p) else p.get_height() for p in patches]
    if mode == "auto":
        mode = "pct" if all((0 <= v <= 1) for v in values) else "count"

    # Axes limits for clamping text inside view
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for p in patches:
        horiz = _is_horizontal_bar(p)
        if horiz:
            val = p.get_width()
            if mode == "pct" and val < min_visible:
                continue
            x = p.get_x() + val + pad * (x_max - x_min)
            y = p.get_y() + p.get_height() / 2
            label = _fmt_value(val, mode, decimals)
            ax.text(x, y, label, va="center", ha="left", fontsize=9)
        else:
            val = p.get_height()
            if mode == "pct" and val < min_visible:
                continue
            x = p.get_x() + p.get_width() / 2
            y = p.get_y() + val + pad * (y_max - y_min)
            label = _fmt_value(val, mode, decimals)
            ax.text(x, y, label, va="bottom", ha="center", fontsize=9)

def format_axis_as_percent(ax, axis="x", ticks=6):
    """
    Turn an axis into % labels assuming data is 0–1. Leaves limits unchanged.
    """
    import numpy as np
    if axis == "x":
        lo, hi = ax.get_xlim()
        arr = np.linspace(lo, hi, ticks)
        ax.set_xticks(arr)
        ax.set_xticklabels([f"{int(100*t)}%" for t in arr])
    else:
        lo, hi = ax.get_ylim()
        arr = np.linspace(lo, hi, ticks)
        ax.set_yticks(arr)
        ax.set_yticklabels([f"{int(100*t)}%" for t in arr])
