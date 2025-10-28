# scripts/charts_teachers/config.py
from pathlib import Path

# ---------- project roots/paths ----------
ROOT = Path(__file__).resolve().parents[2]   # repo root
DATA = ROOT / "data"
FIGS = ROOT / "figs" / "teachers"
(FIGS / "demographics").mkdir(parents=True, exist_ok=True)
(FIGS / "likert").mkdir(parents=True, exist_ok=True)
(FIGS / "multi").mkdir(parents=True, exist_ok=True)
(FIGS / "multi_by_school").mkdir(parents=True, exist_ok=True)
(FIGS / "numeric").mkdir(parents=True, exist_ok=True)

# input files produced by cleaning pipeline
TEACHERS_CLEAN = DATA / "teachers_clean.csv"
TEACHERS_MULTI = DATA / "teachers_multi_long.csv"

# ---------- facet orders ----------
SCHOOL_TYPES = ["小", "中", "高", "不明"]

# ---------- Japanese font fallback order (used by charts/style.py) ----------
# Keep Japanese-first fonts; add Noto/Yu Gothic for better availability
JP_FONT_FAMILIES = [
    "Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP",
    "IPAexGothic", "Arial Unicode MS", "DejaVu Sans", "sans-serif"
]

# ---------- scale orders ----------
# (A) Frequency scales like Q6/Q8 (“how often do you use …”)
ICT_FREQ_ORDER = ["ほとんど使用していない", "1ヶ月に数回程度", "1週間に数回程度", "ほぼ毎時間"]
LEAF_FREQ_ORDER = ICT_FREQ_ORDER  # same pattern; keep separate if it ever diverges

# (B) 4-point “usage/purpose” scale often seen in Q12–20 (“how much do you use this for …”)
USE_ORDER_4 = ["全く使用しない", "あまり使用しない", "使用することがある", "頻繁に使用している", "当該機能を知らない"]

# (C) Agreement scales Q21–30 (“効果/満足” etc.)
AGREE_ORDER = ["あてはまらない", "あまりあてはまらない", "少しあてはまる", "あてはまる", "使っていない"]

# ---------- figure defaults ----------
# Typically overridden by charts/style.apply_style()
DPI = 160
FIGSIZE = (7, 4)
