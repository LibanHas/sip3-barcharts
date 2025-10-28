# Usage:
#   python3 -m scripts.charts_students.make_likert_attitudes_singletons
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Project helpers
try:
    from scripts.common.plotkit import setup  # sets fonts etc.
except Exception:
    setup = lambda: None

# Centralized school canonicalizer
from scripts.common.canonicalize import SchoolCanonicalizer as SC

# -------------------- IO --------------------
DATA_CLEAN = Path("data/students_clean.csv")
OUT_DIR = Path("figs/students/attitudes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Likert setup --------------------
LIKERT_ORDER = [
    "そう思わない",
    "あまりそう思わない",
    "どちらともいえない",
    "ややそう思う",
    "そう思う",
]
LIKERT_DROP = {"質問の意味がわからない"}  # treated as NaN

LIKERT_COLORS = [
    "#d73027",  # strong negative
    "#fc8d59",  # negative
    "#dddddd",  # neutral
    "#91bfdb",  # positive
    "#4575b4",  # strong positive
]

# -------------------- Statements (exact column headers) --------------------
# -------------------- Statements (with acceptable variants) --------------------
STATEMENTS_MAP = {
    "理解度が高まったと感じる": ["理解度が高まったと感じる"],
    "勉強を効率よく短時間で行えるようになった": ["勉強を効率よく短時間で行えるようになった"],
    "自分に合った学びを実現する助けになったと感じる": ["自分に合った学びを実現する助けになったと感じる"],
    "自分で計画を立てたり自分の考えを広めたり深めたりして学ぶ助けになったと感じる": ["自分で計画を立てたり自分の考えを広めたり深めたりして学ぶ助けになったと感じる"],
    "他の人と協力しながら学ぶ助けになったと感じる": ["他の人と協力しながら学ぶ助けになったと感じる"],
    "自分で課題を設定して追究していく学びの助けになったと感じる": ["自分で課題を設定して追究していく学びの助けになったと感じる"],
    # ← the problematic one: accept both full-width and ASCII parentheses
    "使い方・操作（そうさ）手順がわかりやすい": [
        "使い方・操作（そうさ）手順がわかりやすい",
        "使い方・操作(そうさ)手順がわかりやすい",
    ],
    "LEAFシステムに満足している": ["LEAFシステムに満足している"],
}


# -------------------- Styling --------------------
PLOT_DPI       = 300
BASE_FONTSIZE  = 11
TITLE_FONTSIZE = 15
TICK_FONTSIZE  = 10
LABEL_FONTSIZE = 11
LEGEND_FONTSIZE= 10

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()【】\\[\\]「」『』,:;.-]", "", s)
    return s.lower()

def _mean_likert(series: pd.Series) -> float:
    """Map to 1..5 and return mean, ignoring NaNs and '質問の意味がわからない'."""
    m = {
        "そう思わない": 1,
        "あまりそう思わない": 2,
        "どちらともいえない": 3,
        "ややそう思う": 4,
        "そう思う": 5,
    }
    x = series.map(lambda v: m.get(str(v).strip()) if pd.notna(v) and str(v).strip() not in LIKERT_DROP else np.nan)
    return float(x.mean()) if len(x) else np.nan

def resolve_statement_column(df: pd.DataFrame, variants: list[str]) -> str | None:
    target_norms = [_norm(v) for v in variants]
    for col in df.columns:
        nc = _norm(col)
        if nc in target_norms:
            return col
    return None

def _pct_counts(series: pd.Series) -> pd.Series:
    """Return percentage distribution across LIKERT_ORDER (0..100), excluding drops/NaN."""
    s = series.dropna()
    s = s[~s.isin(LIKERT_DROP)]
    if len(s) == 0:
        return pd.Series([0]*len(LIKERT_ORDER), index=LIKERT_ORDER, dtype=float)
    counts = s.value_counts().reindex(LIKERT_ORDER, fill_value=0)
    return counts / counts.sum() * 100.0

def _slug(txt: str) -> str:
    t = ud.normalize("NFKC", txt)
    t = re.sub(r"[^\w一-龥ぁ-んァ-ンー]+", "_", t)
    return re.sub(r"_+", "_", t).strip("_")

def _ensure_school(df: pd.DataFrame) -> None:
    _ = SC.find_or_make_school_canon(df, debug=False)
    SC.assert_only_allowed(df)

def _grade_order(df: pd.DataFrame) -> list[str]:
    # Prefer your canonical grade column if present
    if "学年_canon" in df.columns:
        vals = [v for v in df["学年_canon"].dropna().unique().tolist()]
        # Try a sensible order if matches common labels
        preferred = [
            "小学校", "中学校",
            "高校1年", "高校2年", "高校3年",
            "その他"
        ]
        # Keep preferred order first, then append any remaining in data order
        ordered = [g for g in preferred if g in vals] + [g for g in vals if g not in preferred]
        return ordered
    # Fallback: try to infer from any column containing '学年'
    for c in df.columns:
        if "学年" in str(c):
            vals = df[c].dropna().unique().tolist()
            return vals
    return []

def _barh_100pct(ax, pct_df: pd.DataFrame, title: str, right_labels: pd.Series|None, show_legend=False):
    """
    pct_df: index = categories (schools or grades), columns = LIKERT_ORDER, values in %
    right_labels: optional Series (index-aligned) with e.g. '平均=3.4｜n=198'
    """
    cats = pct_df.index.tolist()
    y = np.arange(len(cats))
    left = np.zeros(len(cats), dtype=float)

    for k, color in zip(LIKERT_ORDER, LIKERT_COLORS):
        vals = pct_df[k].values if k in pct_df.columns else np.zeros(len(cats))
        ax.barh(y, vals, left=left, color=color, edgecolor="none", height=0.62, label=k)
        left += vals

    ax.set_yticks(y, labels=cats, fontsize=TICK_FONTSIZE)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for spine in ["top","right","left","bottom"]:
        ax.spines[spine].set_visible(False)

    ax.set_xlabel("割合（%）", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)

    if right_labels is not None and len(right_labels):
        for yi, cat in enumerate(cats):
            txt = right_labels.get(cat, "")
            if txt:
                ax.text(101, yi, txt, va="center", ha="left", fontsize=BASE_FONTSIZE, color="#444")

    if show_legend:
        leg = ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02),
            ncol=5, frameon=False, fontsize=LEGEND_FONTSIZE,
            title="回答（左→右：否定 ～ 肯定）"
        )
        if leg and leg.get_title():
            leg.get_title().set_fontsize(LEGEND_FONTSIZE)

def _make_single(df: pd.DataFrame, col: str, by: str, out_png: Path, out_csv: Path, title: str):
    """
    by: "school" or "grade"
    """
    if by == "school":
        _ensure_school(df)
        groups = df.groupby("school_canon")
        order = getattr(
            SC, "ALLOWED_SCHOOLS",
            ["不明","北海道天塩高等学校","岩沼小学校","洗足学園中学校",
             "西京高等学校","西京高等学校付属中学校","西賀茂中学校","北海道寿都高等学校"]
        )
        cats = [c for c in order if c in groups.groups]
        title_suffix = "— 学校別（学生）"
    elif by == "grade":
        grade_col = "学年_canon" if "学年_canon" in df.columns else None
        if not grade_col:
            # Try to find any '学年' column
            for c in df.columns:
                if "学年" in str(c):
                    grade_col = c; break
        if not grade_col:
            raise KeyError("学年カラムが見つかりません（'学年_canon' 推奨）")
        groups = df.groupby(grade_col)
        cats = _grade_order(df)
        title_suffix = "— 学年別（学生）"
    else:
        raise ValueError("by must be 'school' or 'grade'")

    # Build percentage table and right-edge labels (mean & n)
    rows = []
    right = {}
    for cat in cats:
        if cat not in groups.groups:
            continue
        ser = groups.get_group(cat)[col]
        # distribution
        pct = _pct_counts(ser)
        # mean + n
        mean = _mean_likert(ser)
        valid = ser.dropna()
        valid = valid[~valid.isin(LIKERT_DROP)]
        n = int(valid.shape[0])
        right[cat] = f"平均={mean:.1f}｜n={n}" if n > 0 and not np.isnan(mean) else ""
        rows.append(pct.rename(cat))

    if not rows:
        print(f"[WARN] No data for {col} by {by}")
        return

    pct_df = pd.DataFrame(rows)
    pct_df = pct_df.reindex(cats)  # keep desired order

    # save CSV (counts/percents/mean)
    # Add mean and n columns to CSV
    meta = pd.Series(right).rename("mean_n")
    mean_values = meta.map(lambda s: float(s.split("｜")[0].replace("平均=","")) if s else np.nan)
    n_values    = meta.map(lambda s: int(s.split("｜")[1].replace("n=","")) if s and "n=" in s else 0)
    out_df = pct_df.copy()
    out_df["mean"] = mean_values
    out_df["n"] = n_values
    out_df.to_csv(out_csv, encoding="utf-8")

    # plot
    fig_h = max(3.8, 0.55 * len(pct_df) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)
    setup()

    _barh_100pct(
        ax,
        pct_df=pct_df,
        title=f"{col} {title_suffix}",
        right_labels=pd.Series(right),
        show_legend=True
    )

    plt.tight_layout(rect=[0, 0, 0.98, 1])
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}\n[info] wrote {out_csv}")

def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the cleaner first.")

    df = pd.read_csv(DATA_CLEAN, dtype=str)
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    for display_label, variants in STATEMENTS_MAP.items():
        actual_col = resolve_statement_column(df, variants)
        if actual_col is None:
            print(f"[WARN] Column not found, skipping: {display_label} (tried {variants})")
            continue

        slug = _slug(display_label)

        # By school
        _make_single(
            df=df,
            col=actual_col,
            by="school",
            out_png=OUT_DIR / f"{slug}_by_school.png",
            out_csv=OUT_DIR / f"{slug}_by_school.csv",
            title=display_label,
        )

        # By grade
        _make_single(
            df=df,
            col=actual_col,
            by="grade",
            out_png=OUT_DIR / f"{slug}_by_grade.png",
            out_csv=OUT_DIR / f"{slug}_by_grade.csv",
            title=display_label,
        )


if __name__ == "__main__":
    main()
