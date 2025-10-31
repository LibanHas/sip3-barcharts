# Usage:
#   python3 -m scripts.charts_students.make_timeuse_by_school_and_grade
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# Optional project font/theme
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# Centralized canonicalizer
from scripts.common.canonicalize import SchoolCanonicalizer as SC

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/students_clean.csv")
OUT_DIR    = Path("figs/students/timeuse")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_RESP_PER_GROUP = 2
PLOT_DPI           = 300
BASE_FONTSIZE      = 12
TITLE_FONTSIZE     = 16
TICK_FONTSIZE      = 12
LABEL_FONTSIZE     = 12

# Which questions (friendly label -> finder tokens)
# These strings come from your header list.
TIME_ITEMS: List[Tuple[str, Tuple[str, ...]]] = [
    ("宿題を行う時間（週）", ("宿題を行う時間",)),
    ("予習・復習を行う時間（週）", ("予習・復習を行う時間", "予習", "復習")),
    ("学校の勉強以外を行う時間（週）", ("学校の勉強以外を行う時間", "勉強以外")),
    ("自分の勉強の仕方を振り返る時間（週）", ("自分の勉強の仕方を振", "振り返る時間")),
    ("勉強の計画を立てる時間（週）", ("勉強の計画を立てる時間", "計画")),
]

# ---- Helpers ----------------------------------------------------------------
def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    # remove spaces and most punctuation to make matching tolerant
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]", "", s)
    return s

def find_col(df: pd.DataFrame, tokens: Tuple[str, ...]) -> Optional[str]:
    """Return the first column whose normalized header contains all token fragments."""
    want = [ _norm(t) for t in tokens ]
    for col in df.columns:
        nc = _norm(col)
        if all(t in nc for t in want):
            return col
    return None

# robust parser for "hours" answers
_HOURS_RANGE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[~〜-]\s*(\d+(?:\.\d+)?)\s*$")
_HOURS_NUM   = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:h|時間)?\s*$", flags=re.IGNORECASE)

KANJI_DIGITS = {
    "〇": "0", "零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"
}
def kanji_to_ascii(s: str) -> str:
    # very lightweight: map individual kanji digits; handle simple "十" cases like "十" (~10), "十一"(~11), "二十"(20)
    t = s
    # quick path: simple tens forms
    t = t.replace("～", "~")
    t = ud.normalize("NFKC", t)
    # Replace common phrases
    t = re.sub(r"(約|だいたい|およそ)", "", t)
    # Basic kanji numbers to arabic if they appear alone/simply
    # Try to convert patterns like 二十, 二十一, 十一
    if re.fullmatch(r"[一二三四五六七八九]?十[一二三四五六七八九]?", t):
        base = 10
        if t[0] in KANJI_DIGITS and t[0] != "十":
            base = int(KANJI_DIGITS[t[0]]) * 10
            tail = t[2:] if len(t) > 2 else ""
        else:
            tail = t[1:] if len(t) > 1 else ""
        if tail and tail in KANJI_DIGITS and KANJI_DIGITS[tail].isdigit():
            return str(base + int(KANJI_DIGITS[tail]))
        return str(base)
    # single kanji digit
    if t in KANJI_DIGITS:
        return KANJI_DIGITS[t]
    # per-char replace as a last resort (won't fix complex numbers like 百 etc.)
    for k, v in KANJI_DIGITS.items():
        t = t.replace(k, v)
    return t

def parse_hours(x) -> float:
    """Parse a single answer into hours (float). Return NaN if unparseable."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = ud.normalize("NFKC", s).strip()
    s = kanji_to_ascii(s)
    s = s.replace("時間/週", "").replace("時間", "").replace("h/週", "").replace("h", "")
    s = s.replace("／", "/")
    s = s.replace("〜", "~").replace("－", "-").replace("—", "-")
    s = re.sub(r"[^\d\.\-\~/~]", "", s)  # keep digits, ., -, ~, /
    s = s.strip()

    # handle ranges "1~2" or "1-2": take midpoint
    m = _HOURS_RANGE.match(s)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        return (a + b) / 2.0

    # bare number
    m = _HOURS_NUM.match(s)
    if m:
        return float(m.group(1))

    # things like "2/3" (unlikely here, but safeguard → treat as "2.3"? No → NaN)
    if re.fullmatch(r"\d+/\d+", s):
        return np.nan

    return np.nan

def sanitize_hours(series: pd.Series, clip_max: float = 100.0) -> pd.Series:
    """Map to float hours, drop negatives, clip ridiculous outliers (>clip_max)."""
    vals = series.map(parse_hours)
    vals = vals.where(vals >= 0, np.nan)
    vals = vals.where(vals <= clip_max, np.nan)
    return vals

def barh_mean_chart(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    allowed_order: Optional[List[str]],
    title: str,
    xlabel: str,
    out_png: Path,
    de_emphasize: Optional[str] = "不明",
):
    # filter groups with enough valid respondents
    counts = df.groupby(group_col)[value_col].apply(lambda s: s.notna().sum())
    ok = counts[counts >= MIN_RESP_PER_GROUP].index

    g = (
        df[df[group_col].isin(ok)]
        .groupby(group_col)
        .agg(
            respondents=(value_col, lambda s: int(s.notna().sum())),
            avg_hours=(value_col, "mean"),
            median_hours=(value_col, "median"),
        )
    )

    if allowed_order is not None:
        # reindex to desired school order; for grade, leave as-is
        if set(g.index) >= set(allowed_order):
            g = g.reindex(allowed_order)
    # for plotting, sort by mean desc
    gp = g.copy()
    gp["avg_plot"] = gp["avg_hours"].fillna(0.0)
    gp = gp.sort_values("avg_plot", ascending=False)

    labels = gp.index.tolist()
    y = np.arange(len(labels))
    x = gp["avg_plot"].values

    fig_h = max(3.8, 0.7 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    # style
    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)

    bars = ax.barh(y, x, height=0.6, zorder=2, edgecolor="none")

    # axes
    ax.set_yticks(y, labels=labels, fontsize=TICK_FONTSIZE)
    xmax = float(np.nanmax(x)) if len(x) else 0.0
    pad  = 0.12 if xmax > 0 else 0.2
    ax.set_xlim(0, xmax * (1 + pad) + (0.5 if xmax < 6 else 0))
    # nice ticks
    ax.xaxis.set_major_locator(MaxNLocator(6) if xmax < 20 else MultipleLocator(5))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    # labels: mean (1 decimal) | n
    for i, (val, n) in enumerate(zip(x, gp["respondents"].values)):
        if np.isnan(val):
            continue
        text = f"{val:.1f}h｜n={int(n)}"
        inside_threshold = 0.18 * ax.get_xlim()[1]
        if val >= inside_threshold:
            ax.text(val - 0.3, i, text, va="center", ha="right",
                    color="white", fontsize=BASE_FONTSIZE, fontweight="bold")
        else:
            ax.text(val + 0.3, i, text, va="center", ha="left",
                    color="black", fontsize=BASE_FONTSIZE)

    # de-emphasize unknown
    if de_emphasize:
        for bar, lab in zip(bars, labels):
            if lab == de_emphasize:
                bar.set_alpha(0.45)

    plt.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    return g  # return the table for CSV write

def ensure_school_and_grade(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Ensure school_canon exists and is valid
    _ = SC.find_or_make_school_canon(df, debug=False)
    SC.assert_only_allowed(df)
    allowed_order = getattr(
        SC, "ALLOWED_SCHOOLS",
        ["不明", "北海道天塩高等学校", "岩沼小学校", "洗足学園中学校",
         "西京高等学校", "西京高等学校付属中学校", "西賀茂中学校", "北海道寿都高等学校"]
    )
    # grade canon (already in your cleaned data as 学年_canon)
    grade_col = "学年_canon" if "学年_canon" in df.columns else None
    return df, allowed_order

# ---- Main -------------------------------------------------------------------
def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the cleaner first.")

    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    # Make/validate canonical group columns
    df, allowed_order = ensure_school_and_grade(df)

    # Locate columns and parse to numeric hours
    found: Dict[str, str] = {}
    for label, tokens in TIME_ITEMS:
        col = find_col(df, tokens)
        if not col:
            print(f"[WARN] Column not found, skipping: {label} (tokens={tokens})")
            continue
        found[label] = col

    if not found:
        print("[ERROR] No Section 7 time-use columns found.")
        return

    # Parse
    for label, col in found.items():
        df[label] = sanitize_hours(df[col])

    # For each item, make by-school + by-grade charts & CSVs
    grade_col = "学年_canon" if "学年_canon" in df.columns else None

    for label in found.keys():
        # --- by school
        title = f"{label}（平均） — 学校別（学生）"
        out_png = OUT_DIR / f"{_safe(label)}_by_school.png"
        out_csv = OUT_DIR / f"{_safe(label)}_by_school.csv"
        g_school = barh_mean_chart(
            df=df,
            group_col="school_canon",
            value_col=label,
            allowed_order=allowed_order,
            title=title,
            xlabel="平均時間（時間/週）",
            out_png=out_png,
        )
        g_school.to_csv(out_csv, encoding="utf-8")
        print(f"[info] wrote {out_png}")
        print(f"[info] wrote {out_csv}")

        # --- by grade
        if grade_col:
            title_g = f"{label}（平均） — 学年別（学生）"
            out_png_g = OUT_DIR / f"{_safe(label)}_by_grade.png"
            out_csv_g = OUT_DIR / f"{_safe(label)}_by_grade.csv"
            g_grade = barh_mean_chart(
                df=df.rename(columns={grade_col: "grade_canon"}),
                group_col="grade_canon",
                value_col=label,
                allowed_order=None,  # keep natural grade order present in data
                title=title_g,
                xlabel="平均時間（時間/週）",
                out_png=out_png_g,
                de_emphasize=None,
            )
            g_grade.to_csv(out_csv_g, encoding="utf-8")
            print(f"[info] wrote {out_png_g}")
            print(f"[info] wrote {out_csv_g}")

def _safe(s: str) -> str:
    s = ud.normalize("NFKC", s)
    s = s.replace("（", "_").replace("）", "")
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("・", "_").replace("/", "_").replace(" ", "")
    s = s.replace("，", "").replace("、", "")
    s = s.replace("【", "").replace("】", "")
    s = s.replace("［", "").replace("］", "")
    # other punctuation
    s = re.sub(r"[^\w\-\u3040-\u30FF\u4E00-\u9FFF]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

if __name__ == "__main__":
    main()
