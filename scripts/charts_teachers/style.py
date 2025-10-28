import matplotlib as mpl
from matplotlib import font_manager as fm
import seaborn as sns
from pathlib import Path

def apply_style(context: str = "paper"):
    # 1) Seaborn theme
    sns.set_theme(
        context=context,            # "paper" | "notebook" | "talk" | "poster"
        style="whitegrid",
        palette="colorblind",
        font_scale=1.1,
    )

    # 2) Try to add a bundled JP font (optional, if you ship one)
    #    e.g., put IPAexGothic.ttf at assets/fonts/IPAexGothic.ttf
    bundled = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "IPAexGothic.ttf"
    if bundled.exists():
        try:
            fm.fontManager.addfont(str(bundled))
        except Exception:
            pass

    # 3) JP-capable fallback stack (macOS first, then cross-platform)
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "IPAexGothic",              # bundled or system (if installed)
        "Hiragino Sans",            # macOS
        "Hiragino Kaku Gothic ProN",
        "Noto Sans CJK JP",         # Google Noto (if installed)
        "AppleGothic",              # macOS alt
        "Yu Gothic", "MS Gothic",   # Windows
        "DejaVu Sans", "Arial Unicode MS",
        "sans-serif",
    ]

    # 4) Rendering & export niceties
    mpl.rcParams["axes.unicode_minus"] = False  # proper minus sign with JP fonts
    mpl.rcParams["pdf.fonttype"] = 42           # embed TrueType (better Unicode)
    mpl.rcParams["ps.fonttype"]  = 42

    # 5) Clean axes & savefig defaults
    mpl.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "figure.dpi": 144,       # crisp on screens
        "savefig.dpi": 300,      # print-ready
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })