# charts/style.py
import matplotlib as mpl
import seaborn as sns

def apply_style(context="paper"):
    # Good defaults for print/PDF and slides
    sns.set_theme(
        context=context,  # "paper" | "notebook" | "talk" | "poster"
        style="whitegrid", 
        palette="colorblind",  # readable + accessible
        font_scale=1.1,
    )
    # macOS-friendly Japanese fonts (fallbacks included)
    mpl.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans", "sans-serif"]
    # Clean axes & ticks
    mpl.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "figure.dpi": 144,      # looks crisp on web
        "savefig.dpi": 300,     # print-ready
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })
