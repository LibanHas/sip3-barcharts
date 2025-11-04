# scripts/charts_teachers/q07_leaf_duration_by_school.py
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

# Optional fonts/theme
try:
    from scripts.common.plotkit import setup
except Exception:
    setup = lambda: None

# Canonicalize schools exactly like q06
from scripts.common.canonicalize import (
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
)

# ---------------- IO ----------------
DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/leaf_duration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------- Config ---------------
MIN_RESP_PER_GROUP = 1
PLOT_DPI           = 300
BASE_FONTSIZE      = 12
TITLE_FONTSIZE     = 16
TICK_FONTSIZE      = 12
LABEL_FONTSIZE     = 12

# Keep the same school order as your reference chart
ALLOWED_ORDER: List[str] = [
    "西京高等学校付属中学校",
    "洗足学園高等学校",
    "西賀茂中学校",
    "洗足学園中学校",
    "岩沼小学校",
    "明徳小学校",
    "不明",
]

# ---- column finder (tolerant tokens) ----
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s

Q7_TOKENS = tuple(_norm(t) for t in ["LEAFシステム","BookRoll","分析ツール","授業","授業外","何か月"])

def find_q7_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        nc = _norm(col)
        if all(t in nc for t in Q7_TOKENS):
            return col
    return None

# ---- parsing "X/Y" / "2年3か月" → total months ----
KANJI = {"〇":"0","零":"0","一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7","八":"8","九":"9","十":"10"}

def _to_ascii_digits(s: str) -> str:
    t = ud.normalize("NFKC", s)
    for k, v in KANJI.items():
        t = t.replace(k, v)
    return t

RE_SLASH = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")                      # 2/3
RE_YM    = re.compile(r"^\s*(\d+)\s*年\s*(\d+)?\s*(?:か|ヶ|ケ)?月?\s*$")  # 2年3か月 / 2年
RE_M     = re.compile(r"^\s*(\d+)\s*(?:か|ヶ|ケ)?月\s*$")                # 6か月
RE_NUM   = re.compile(r"^\s*\d+\s*$")                                    # 2 → years

def parse_duration_to_months(x) -> float:
    if pd.isna(x):
        return np.nan
    s = _to_ascii_digits(str(x)).strip()
    s = (s.replace("／", "/")
           .replace("～", "~")
           .replace("—", "-")
           .replace("未使用", "0/0")
           .replace("ｍ", "m")
           .replace("ｄｆ", ""))

    m = RE_SLASH.match(s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return y * 12 + mo

    m = RE_YM.match(s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2)) if m.group(2) is not None else 0
        return y * 12 + mo

    m = RE_M.match(s)
    if m:
        return float(int(m.group(1)))

    if RE_NUM.match(s):
        return float(int(s) * 12)  # plain number → years

    return np.nan

def sanitize_months(series: pd.Series, clip_max: int = 240) -> pd.Series:
    vals = series.map(parse_duration_to_months)
    vals = vals.where((vals >= 0) & (vals <= clip_max))  # clamp to <= 20y
    return vals

# ------------- plotting --------------
def barh_months_by_school(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    out_png: Path,
    allowed_order: Optional[List[str]] = None,
    de_emphasize: Optional[str] = "不明",
):
    counts = df.groupby(group_col)[value_col].apply(lambda s: s.notna().sum())
    ok = counts[counts >= MIN_RESP_PER_GROUP].index

    g = (
        df[df[group_col].isin(ok)]
        .groupby(group_col)
        .agg(
            respondents=(value_col, lambda s: int(s.notna().sum())),
            avg_months=(value_col, "mean"),
            median_months=(value_col, "median"),
        )
    )

    # Apply requested order; keep any extra schools after in their current order
    if allowed_order:
        present = [s for s in allowed_order if s in g.index]
        extras  = [s for s in g.index if s not in present]
        g = g.reindex(present + extras)

    # Labels with n beside the school name
    g["label"] = [f"{school}（n={n}）" for school, n in zip(g.index, g["respondents"])]

    labels = g["label"].tolist()
    y = np.arange(len(labels))
    x = g["avg_months"].fillna(0).values

    fig_h = max(3.8, 0.7 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)

    bars = ax.barh(y, x, height=0.6, zorder=2, edgecolor="none")
    ax.set_yticks(y, labels=labels, fontsize=TICK_FONTSIZE)
    ax.invert_yaxis() 
    xmax = float(np.nanmax(x)) if len(x) else 0.0
    pad  = 0.12 if xmax > 0 else 0.2
    ax.set_xlim(0, xmax * (1 + pad) + (0.5 if xmax < 6 else 0))
    ax.xaxis.set_major_locator(MaxNLocator(6) if xmax < 60 else MultipleLocator(12))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    # Show mean months on bars
    for i, val in enumerate(x):
        if np.isnan(val):
            continue
        text = f"{val:.1f}m"
        inside_threshold = 0.18 * ax.get_xlim()[1]
        if val >= inside_threshold:
            ax.text(val - 0.3, i, text, va="center", ha="right",
                    color="white", fontsize=BASE_FONTSIZE, fontweight="bold")
        else:
            ax.text(val + 0.3, i, text, va="center", ha="left",
                    color="black", fontsize=BASE_FONTSIZE)

    if de_emphasize:
        for bar, lab in zip(bars, g.index):
            if lab == de_emphasize:
                bar.set_alpha(0.45)

    plt.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    return g

def _safe(s: str) -> str:
    s = ud.normalize("NFKC", s)
    s = s.replace("（", "_").replace("）", "")
    s = s.replace("(", "_").replace(")", "")
    s = s.replace("・", "_").replace("/", "_").replace(" ", "")
    s = s.replace("，", "").replace("、", "")
    s = s.replace("【", "").replace("】", "")
    s = re.sub(r"[^\w\-\u3040-\u30FF\u4E00-\u9FFF]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# --------------- main ----------------
def main():
    setup()

    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the teacher cleaner first.")

    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    # Canonicalize schools like q06
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon":
        df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df)
    post_disambiguate_middle_vs_high(df)

    # Find & parse Q7
    q7_col = find_q7_col(df)
    if not q7_col:
        print("[ERROR] Q7 column not found.")
        return

    df["LEAF利用月数"] = sanitize_months(df[q7_col])

    title   = "LEAF利用月数（平均） — 学校別（教員）"
    out_png = OUT_DIR / f"{_safe('LEAF利用月数_学校別_教員')}.png"
    out_csv = OUT_DIR / f"{_safe('LEAF利用月数_学校別_教員')}.csv"

    g = barh_months_by_school(
        df=df,
        group_col="school_canon",
        value_col="LEAF利用月数",
        title=title,
        xlabel="平均利用月数（月）",
        out_png=out_png,
        allowed_order=ALLOWED_ORDER,
    )
    g.to_csv(out_csv, encoding="utf-8")
    print(f"[info] wrote {out_png}")
    print(f"[info] wrote {out_csv}")

if __name__ == "__main__":
    main()
