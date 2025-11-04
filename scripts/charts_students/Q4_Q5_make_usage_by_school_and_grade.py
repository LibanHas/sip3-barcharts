# Usage:
#   python3 -m scripts.charts_students.make_usage_by_school_and_grade
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

# Project helpers
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# Centralized canonicalizer
from scripts.common.canonicalize import (
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
)

# ---- IO ---------------------------------------------------------------------
DATA_CLEAN = Path("data/students_clean.csv")
OUT_DIR    = Path("figs/students/usage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Config -----------------------------------------------------------------
MIN_RESP_PER_SCHOOL = 2
MIN_RESP_PER_GRADE  = 2

TITLE_SCHOOL = "LEAFの平均利用期間（月） — 学校別（学生）"
TITLE_GRADE  = "LEAFの平均利用期間（月） — 学年別（学生）"

OUT_SCHOOL_PNG = OUT_DIR / "avg_duration_by_school.png"
OUT_SCHOOL_CSV = OUT_DIR / "avg_duration_by_school.csv"
OUT_GRADE_PNG  = OUT_DIR / "avg_duration_by_grade.png"
OUT_GRADE_CSV  = OUT_DIR / "avg_duration_by_grade.csv"

# Audits
AUDIT_PARSE_ERR  = OUT_DIR / "Q04_usage_parse_errors.csv"
AUDIT_BARE_NUM   = OUT_DIR / "Q04_usage_bare_numbers.csv"
AUDIT_OUTLIERS   = OUT_DIR / "Q04_usage_outliers.csv"

# Parsing behavior
BARE_NUMBER_POLICY = "exclude"   # 'exclude' | 'assume_years' | 'assume_months'
OUTLIER_MAX_MONTHS = 120         # flag anything > 10 years as an outlier (kept, just flagged)

# ---- Plot style -------------------------------------------------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

# ---- Column finders ---------------------------------------------------------
_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;.\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = _PUNCT_RE.sub("", s)
    return s.lower()

def _find_col(df, exact_jp: str = None, must_contain=(), contains=None):
    cols = list(df.columns)
    norm_cols = {c: _norm(c) for c in cols}
    if exact_jp:
        target = _norm(exact_jp)
        for c, nc in norm_cols.items():
            if nc == target:
                return c
    if must_contain:
        tokens = [_norm(t) for t in must_contain]
        for c, nc in norm_cols.items():
            if all(t in nc for t in tokens):
                return c
    if contains:
        token = _norm(contains)
        for c, nc in norm_cols.items():
            if token in nc:
                return c
    return None

def _find_usage_col(df):
    candidates = [
        "LEAFシステム（BookRoll，分析（ぶんせき）ツール）を授業・授業外（宿題など）でどれくらい利用していますか",
        "LEAFシステム(BookRoll,分析(ぶんせき)ツール)を授業・授業外(宿題など)でどれくらい利用していますか",
        "LEAFシステムの利用頻度について教えてください",
        "Q5_利用頻度", "利用頻度", "システム利用頻度",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        nc = _norm(col)
        if ("利用頻度" in nc) or ("利用していますか" in nc):
            return col
    return None

# ---- Q4 parse: robust -------------------------------------------------------
YM_RE      = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")                # X/Y
YMOJI_RE   = re.compile(r"^\s*約?\s*(\d+)\s*年\s*(\d+)?\s*か?月?\s*$") # 約2年3か月 / 2年 / 2年0ヶ月
ONLY_Y_RE  = re.compile(r"^\s*約?\s*(\d+)\s*年\s*$")
ONLY_M_RE  = re.compile(r"^\s*(\d+)\s*か?月\s*$")
INT_RE     = re.compile(r"^\s*\d+\s*$")

NEGATIVE_RE = re.compile(r"-")

NA_TOKENS = {"", "nan", "なし", "未使用", "してない", "使用していない", "未", "—", "-", "ー"}

def parse_year_month_to_months(x):
    """
    Return (months, status) where status in {'ok','bare','na','err'}
    """
    if pd.isna(x):
        return np.nan, "na"

    s = ud.normalize("NFKC", str(x)).strip()
    s = s.replace("／", "/")  # full-width slash -> ASCII
    if s.lower() in NA_TOKENS:
        return 0.0, "ok"   # treat explicit '未使用/してない' as 0

    if NEGATIVE_RE.search(s):
        return np.nan, "err"

    m = YM_RE.match(s)
    if m:
        years  = int(m.group(1))
        months = int(m.group(2))
        total  = years * 12 + months
        return float(total), "ok"

    m = YMOJI_RE.match(s)
    if m:
        y = int(m.group(1))
        mm = int(m.group(2)) if m.group(2) is not None else 0
        return float(y*12 + mm), "ok"

    m = ONLY_Y_RE.match(s)
    if m:
        return float(int(m.group(1)) * 12), "ok"

    m = ONLY_M_RE.match(s)
    if m:
        return float(int(m.group(1))), "ok"

    if INT_RE.match(s):
        # Ambiguous bare integer
        if BARE_NUMBER_POLICY == "assume_years":
            return float(int(s) * 12), "ok"
        if BARE_NUMBER_POLICY == "assume_months":
            return float(int(s)), "ok"
        return np.nan, "bare"

    return np.nan, "err"

# ---- Q5 map: frequency -> ordinal score ------------------------------------
FREQ_MAP = {
    "ほぼ毎時間": 4,
    "1週間に数回程度": 3,
    "1ヶ月に数回程度": 2,
    "ほとんど使用していない": 1,
}
def map_freq(x):
    if pd.isna(x):
        return np.nan
    s = ud.normalize("NFKC", str(x)).strip()
    return FREQ_MAP.get(s, np.nan)

# ---- Grade helpers ----------------------------------------------------------
def _find_grade_canon(df: pd.DataFrame) -> str:
    if hasattr(SC, "find_or_make_grade_canon"):
        col = SC.find_or_make_grade_canon(df, debug=False)
        if isinstance(col, str):
            return col
        if "grade_canon" in df.columns:
            return "grade_canon"
    if "学年_canon" in df.columns:
        return "学年_canon"
    likely = ["あなたの学年を教えてください", "学年"]
    for c in likely:
        if c in df.columns:
            return c
    for c in df.columns:
        if "学年" in _norm(c):
            return c
    df["学年_canon"] = "不明"
    return "学年_canon"

# ---- Plotting (unchanged from your version) ---------------------------------
def _plot_by_group(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    out_png: Path,
    out_csv: Path,
    min_n: int,
    annotate_with_freq: bool = True,
    deemphasize_unknown_label: str = "不明",
):
    resp_per_group = df.groupby(group_col)["use_months_total"].apply(lambda s: s.notna().sum())
    ok_groups = resp_per_group[resp_per_group >= min_n].index

    g = (
        df[df[group_col].isin(ok_groups)]
        .groupby(group_col)
        .agg(
            respondents=("use_months_total", lambda s: int(s.notna().sum())),
            avg_months=("use_months_total", "mean"),
            avg_freq=("freq_score", "mean"),
        )
        .sort_values("avg_months", ascending=False)
    )

    g.to_csv(out_csv, encoding="utf-8", index=True)

    respondents = g["respondents"].astype(int).values
    base_labels = g.index.tolist()
    labels = [f"{lab}（n={n}）" for lab, n in zip(base_labels, respondents)]
    y_pos  = np.arange(len(labels))
    x_vals = g["avg_months"].fillna(0.0).values

    fig_h = max(3.8, 0.7 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    bars = ax.barh(y_pos, x_vals, height=0.6, zorder=2, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos, labels=labels, fontsize=TICK_FONTSIZE)
    xmax = float(np.nanmax(x_vals)) if len(x_vals) else 0.0
    pad  = 0.18 if xmax > 0 else 0.25
    ax.set_xlim(0, xmax * (1 + pad) + (0.6 if xmax < 6 else 0.5))
    ax.xaxis.set_major_locator(MultipleLocator(5) if xmax >= 15 else MaxNLocator(6))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel("平均利用期間（月）", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)
    fig.canvas.draw()

    def text_width_in_data(ax, text, fontsize=12, fontweight=None):
        t = plt.Text(0, 0, text, fontsize=fontsize, fontweight=fontweight)
        t.set_figure(ax.figure)
        renderer = ax.figure.canvas.get_renderer()
        bbox = t.get_window_extent(renderer=renderer)
        x0_px = ax.transData.transform((0, 0))[0]
        x1_px = ax.transData.transform((1, 0))[0]
        px_per_data_x = (x1_px - x0_px) if (x1_px - x0_px) != 0 else 1.0
        return bbox.width / px_per_data_x

    for i, bar in enumerate(bars):
        val = bar.get_width()
        f   = g["avg_freq"].iloc[i]
        if np.isnan(val):
            continue
        if annotate_with_freq and not np.isnan(f):
            if f >= 3.5:   freq_lbl = "毎時間"
            elif f >= 2.5: freq_lbl = "週数回"
            elif f >= 1.5: freq_lbl = "月数回"
            else:          freq_lbl = "ほとんど使用せず"
        else:
            freq_lbl = ""
        text = f"{val:.1f}ヶ月" + (f"｜{freq_lbl}" if freq_lbl else "")
        tw_data  = text_width_in_data(ax, text, fontsize=BASE_FONTSIZE, fontweight="bold")
        headroom = 0.8
        min_keep_inside = 0.16 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        safety = 1.12
        fits_inside = (val >= tw_data * safety + headroom) and (val >= min_keep_inside)
        if val <= 0 or np.isclose(val, 0.0):
            ax.text(0.8, i, text, va="center", ha="left", color="black",
                    fontsize=BASE_FONTSIZE, clip_on=False, zorder=5)
        elif fits_inside:
            ax.text(val - 0.35, i, text, va="center", ha="right", color="white",
                    fontsize=BASE_FONTSIZE, fontweight="bold", clip_on=False, zorder=5)
        else:
            ax.text(max(val + 0.45, 0.8), i, text, va="center", ha="left",
                    color="black", fontsize=BASE_FONTSIZE, clip_on=False, zorder=5)

    if deemphasize_unknown_label:
        for bar, base_lab in zip(bars, base_labels):
            if base_lab == deemphasize_unknown_label:
                bar.set_alpha(0.45)

    plt.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}")
    print(f"[info] wrote {out_csv}")

# ---- Main -------------------------------------------------------------------
def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the cleaner first.")

    df = pd.read_csv(DATA_CLEAN, dtype=str).replace({"": np.nan}).infer_objects(copy=False)

    # Canonical school column + disambiguation (same as other student charts)
    _ = SC.find_or_make_school_canon(df, debug=False)
    post_disambiguate_middle_vs_high(df)
    SC.assert_only_allowed(df)

    # Q4 column (duration)
    col_q4 = _find_col(
        df,
        exact_jp="LEAFシステム（BookRoll，分析（ぶんせき）ツール）を授業・授業外（宿題など）で何か月くらい利用していますか。※利用期間がX年Yか月だった場合、「X/Y」というようにスラッシュで区切って、全て半角で入力してください（例：1年の場合→1/0、2年3か月の場合→2/3、未使用の場合→0/0）",
        must_contain=["利用","何か月","X","Y"],
    )
    if col_q4 is None:
        raise KeyError("Could not find Q4 (usage duration) column in cleaned data.")

    # Q5 usage frequency (for contextual labels)
    usage_col = _find_usage_col(df)
    if not usage_col:
        raise KeyError("Could not find usage frequency column (Q5).")

    # Parse
    parsed_months, statuses = zip(*df[col_q4].map(parse_year_month_to_months))
    df["use_months_total"] = parsed_months
    df["q4_parse_status"]  = statuses
    df["freq_score"]       = df[usage_col].map(map_freq)

    # ---- audits --------------------------------------------------------------
    # 1) parse errors
    parse_err = df[df["q4_parse_status"] == "err"][["school_canon", col_q4]]
    parse_err.to_csv(AUDIT_PARSE_ERR, index=False, encoding="utf-8")

    # 2) bare numbers (ambiguous)
    bare_num = df[df["q4_parse_status"] == "bare"][["school_canon", col_q4]]
    bare_num.to_csv(AUDIT_BARE_NUM, index=False, encoding="utf-8")

    # Zero out/exclude ambiguous depending on policy
    if BARE_NUMBER_POLICY == "exclude":
        df.loc[df["q4_parse_status"] == "bare", "use_months_total"] = np.nan
    elif BARE_NUMBER_POLICY == "assume_years":
        df.loc[df["q4_parse_status"] == "bare", "use_months_total"] = (
            df.loc[df["q4_parse_status"] == "bare", col_q4].astype(float) * 12
        )
        df.loc[df["q4_parse_status"] == "bare", "q4_parse_status"] = "ok"
    elif BARE_NUMBER_POLICY == "assume_months":
        df.loc[df["q4_parse_status"] == "bare", "use_months_total"] = (
            df.loc[df["q4_parse_status"] == "bare", col_q4].astype(float)
        )
        df.loc[df["q4_parse_status"] == "bare", "q4_parse_status"] = "ok"

    # 3) outliers
    outliers = df[df["use_months_total"].astype(float) > OUTLIER_MAX_MONTHS][
        ["school_canon", col_q4, "use_months_total"]
    ]
    outliers.to_csv(AUDIT_OUTLIERS, index=False, encoding="utf-8")

    # ---- (1) By SCHOOL ------------------------------------------------------
    _plot_by_group(
        df=df,
        group_col="school_canon",
        title=TITLE_SCHOOL,
        out_png=OUT_SCHOOL_PNG,
        out_csv=OUT_SCHOOL_CSV,
        min_n=MIN_RESP_PER_SCHOOL,
        annotate_with_freq=True,
        deemphasize_unknown_label="不明",
    )

    # ---- (2) By GRADE -------------------------------------------------------
    grade_col = _find_grade_canon(df)
    _plot_by_group(
        df=df,
        group_col=grade_col,
        title=TITLE_GRADE,
        out_png=OUT_GRADE_PNG,
        out_csv=OUT_GRADE_CSV,
        min_n=MIN_RESP_PER_GRADE,
        annotate_with_freq=True,
        deemphasize_unknown_label="不明",
    )

    print(f"[audit] parse errors: {len(parse_err)} -> {AUDIT_PARSE_ERR}")
    print(f"[audit] bare numbers: {len(bare_num)} -> {AUDIT_BARE_NUM}")
    print(f"[audit] outliers: {len(outliers)} -> {AUDIT_OUTLIERS}")

if __name__ == "__main__":
    main()
