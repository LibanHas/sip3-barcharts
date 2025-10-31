# Usage:
#   python3 -m scripts.charts_teachers.q07_time_allocation
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# Project helpers (students style)
try:
    from scripts.common.plotkit import setup  # sets JP fonts, theme
except Exception:
    setup = lambda: None

# Canonicalize schools exactly like q06
from scripts.common.canonicalize import SchoolCanonicalizer as SC, post_disambiguate_middle_vs_high

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/timeuse")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_RESP_PER_GROUP = 1
PLOT_DPI           = 300
BASE_FONTSIZE      = 12
TITLE_FONTSIZE     = 16
TICK_FONTSIZE      = 12
LABEL_FONTSIZE     = 12

# Which questions (friendly label -> finder tokens)
# Use tolerant tokens so header variations still match; missing items are skipped.
TIME_ITEMS: List[Tuple[str, Tuple[str, ...]]] = [
    ("学力把握時間（週）", ("児童生徒の学力を把握する時間", "1週間あたり")),
    ("プリント作成時間（週）", ("授業に関するプリント作成", "1週間あたり")),
    ("提出物確認時間（週）", ("提出物を回収・確認する時間", "1週間あたり")),
    ("個別指導時間（週）", ("個別指導", "1週間あたり")),
    ("会議・打合せ時間（週）", ("会議", "1週間あたり")),           # if present
    ("授業準備時間（週）", ("授業準備", "1週間あたり")),           # if present
    ("研修・自己研鑽時間（週）", ("研修", "1週間あたり")),         # if present
    ("残業時間（週）", ("残業時間", "1週間あたり")),
]

# ---- Helpers ----------------------------------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s

def find_col(df: pd.DataFrame, tokens: Tuple[str, ...]) -> Optional[str]:
    """Return the first column whose normalized header contains all token fragments."""
    want = [_norm(t) for t in tokens]
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
    t = s.replace("～", "~")
    t = ud.normalize("NFKC", t)
    t = re.sub(r"(約|だいたい|およそ)", "", t)
    # tens forms: 二十, 二十一, 十一
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
    if t in KANJI_DIGITS:
        return KANJI_DIGITS[t]
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
    s = (s.replace("時間/週", "")
           .replace("時間", "")
           .replace("h/週", "")
           .replace("h", "")
           .replace("／", "/")
           .replace("〜", "~")
           .replace("－", "-")
           .replace("—", "-"))
    s = re.sub(r"[^\d\.\-\~/~]", "", s).strip()

    m = _HOURS_RANGE.match(s)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        return (a + b) / 2.0

    m = _HOURS_NUM.match(s)
    if m:
        return float(m.group(1))

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
    # filter groups with enough valid respondents (for THIS item)
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

    if allowed_order is not None and set(g.index) >= set(allowed_order):
        g = g.reindex(allowed_order)

    gp = g.copy()
    gp["avg_plot"] = gp["avg_hours"].fillna(0.0)
    gp = gp.sort_values("avg_plot", ascending=False)

    labels = gp.index.tolist()
    y = np.arange(len(labels))
    x = gp["avg_plot"].values

    fig_h = max(3.8, 0.7 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)

    bars = ax.barh(y, x, height=0.6, zorder=2, edgecolor="none")

    ax.set_yticks(y, labels=labels, fontsize=TICK_FONTSIZE)
    xmax = float(np.nanmax(x)) if len(x) else 0.0
    pad  = 0.12 if xmax > 0 else 0.2
    ax.set_xlim(0, xmax * (1 + pad) + (0.5 if xmax < 6 else 0))
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

    if de_emphasize:
        for bar, lab in zip(bars, labels):
            if lab == de_emphasize:
                bar.set_alpha(0.45)

    plt.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    return g  # return the table for CSV write

def _safe(s: str) -> str:
    s = ud.normalize("NFKC", s)
    s = s.replace("（", "_").replace("）", "")
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("・", "_").replace("/", "_").replace(" ", "")
    s = s.replace("，", "").replace("、", "")
    s = s.replace("【", "").replace("】", "")
    s = s.replace("［", "").replace("］", "")
    s = re.sub(r"[^\w\-\u3040-\u30FF\u4E00-\u9FFF]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# ---- Main -------------------------------------------------------------------
def main():
    setup()

    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the teacher cleaner first.")

    # Load (avoid FutureWarning by splitting replace/infer)
    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan})
    df = df.infer_objects(copy=False)

    # --- Canonicalize schools (same as q06) ---
    school_col = SC.find_or_make_school_canon(df, debug=False)  # e.g., '学校名_canon' or 'school_canon'
    if school_col != "school_canon":  # mirror to expected name (do NOT rename to avoid 2-D columns later)
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # Locate columns and parse to numeric hours
    found: Dict[str, str] = {}
    for label, tokens in TIME_ITEMS:
        col = find_col(df, tokens)
        if not col:
            print(f"[WARN] Column not found, skipping: {label} (tokens={tokens})")
            continue
        found[label] = col

    if not found:
        print("[ERROR] No teacher time-use columns found.")
        return

    # Parse to numeric hours
    for label, col in found.items():
        df[label] = sanitize_hours(df[col])

    # For each item, make by-school charts & CSVs
    for label in found.keys():
        title   = f"{label}（平均） — 学校別（教員）"
        out_png = OUT_DIR / f"{_safe(label)}_by_school.png"
        out_csv = OUT_DIR / f"{_safe(label)}_by_school.csv"

        g_school = barh_mean_chart(
            df=df,
            group_col="school_canon",
            value_col=label,
            allowed_order=None,  # keep natural order; chart sorts by mean desc
            title=title,
            xlabel="平均時間（時間/週）",
            out_png=out_png,
        )
        g_school.to_csv(out_csv, encoding="utf-8")
        print(f"[info] wrote {out_png}")
        print(f"[info] wrote {out_csv}")

if __name__ == "__main__":
    main()
