# Usage:
#   python3 -m scripts.charts_students.make_likert_attitudes_singletons
from __future__ import annotations
from pathlib import Path
import re
import unicodedata as ud
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------- Project helpers (fonts etc.) ----------
try:
    from scripts.common.plotkit import setup  # sets JP font etc.
except Exception:
    setup = lambda: None

# Centralized canonicalizer
from scripts.common.canonicalize import SchoolCanonicalizer as SC

# ---------- IO ----------
DATA_CLEAN = Path("data/students_clean.csv")
OUT_DIR    = Path("figs/students/attitudes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
MIN_RESP_PER_GROUP = 2
PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 12
LABEL_FONTSIZE = 12

# Items (exact text expected; fuzzy finder will back you up)
ATTITUDE_ITEMS: List[str] = [
    "理解度が高まったと感じる",                                  # Q15
    "勉強を効率よく短時間で行えるようになった",                  # Q16
    "根拠（こんきょ）に基づいた勉強計画や復習を行えるようになった",  # Q17
    "分析（ぶんせき）ツールを使用することで自分の学習方法に変化が生じた", # Q18
    "自分に合った学びを実現する助けになったと感じる",              # Q19
    "自分で計画を立てたり自分の考えを広めたり深めたりして学ぶ助けになったと感じる", # Q20
    "他の人と協力しながら学ぶ助けになったと感じる",                 # Q21
    "自分で課題を設定して追究していく学びの助けになったと感じる",     # Q22
    "使い方・操作（そうさ）手順がわかりやすい",                    # Q23
    "LEAFシステムに満足している",                                # Q24
]


# ---------- Likert mapping (robust) ----------
LIKERT_ORDER_STD = ["そう思わない","あまりそう思わない","どちらともいえない","ややそう思う","そう思う"]
LIKERT_ORDER_ATEMAHAMARU = ["あてはまらない","あまりあてはまらない","どちらともいえない","少しあてはまる","あてはまる"]

LIKERT_MAP_STD = {
    "そう思わない": 1, "あまりそう思わない": 2, "どちらともいえない": 3, "ややそう思う": 4, "そう思う": 5,
}
LIKERT_MAP_ATEMAHAMARU = {
    "あてはまらない": 1, "あまりあてはまらない": 2, "どちらともいえない": 3, "少しあてはまる": 4, "あてはまる": 5,
}

IGNORE_AS_NA = {
    "使っていない", "未使用", "無回答",
    "質問の意味がわからない", "質問の意味が分からない",
    "", None
}

_PUNCT_RE = re.compile(r"[ \t\r\n　、。，．・/（）()【】\[\]「」『』,:;・｡･。．\-]+")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s)).strip()
    s = _PUNCT_RE.sub("", s)
    return s

def map_likert_any(x):
    """Map both wording families to 1..5; ignored labels -> NaN."""
    if pd.isna(x):
        return np.nan
    s = ud.normalize("NFKC", str(x)).strip()
    if s in IGNORE_AS_NA:
        return np.nan
    # exact
    if s in LIKERT_MAP_ATEMAHAMARU: return float(LIKERT_MAP_ATEMAHAMARU[s])
    if s in LIKERT_MAP_STD:         return float(LIKERT_MAP_STD[s])
    # fuzzy
    t = _norm(s)
    # atemahamaru family
    if "あまりあてはまらない" in t: return 2.0
    if "少しあてはまる"     in t: return 4.0
    if "どちらともいえない" in t: return 3.0
    if "あてはまらない"     in t: return 1.0
    if "あてはまる"         in t: return 5.0
    # std family
    if "あまりそう思わない" in t: return 2.0
    if "ややそう思う"       in t: return 4.0
    if "どちらともいえない" in t: return 3.0
    if "そう思わない"       in t: return 1.0
    if "そう思う"           in t: return 5.0
    return np.nan

# ---------- Column finder ----------
def _find_col(df: pd.DataFrame, target_jp: str) -> Optional[str]:
    # 1) exact
    if target_jp in df.columns:
        return target_jp
    # 2) normalized exact
    target_norm = _norm(target_jp)
    for c in df.columns:
        if _norm(c) == target_norm:
            return c
    # 3) contains main tokens (split by spaces if provided)
    tokens = [t for t in re.split(r"\s+", target_jp) if t]
    tokens_norm = [_norm(t) for t in tokens]
    for c in df.columns:
        nc = _norm(c)
        if all(tn in nc for tn in tokens_norm):
            return c
    print(f"[WARN] Column not found, skipping: {target_jp}")
    return None

# ---------- Grade ordering ----------
def _grade_order_from_data(df: pd.DataFrame) -> List[str]:
    # If you already have a canonical order elsewhere, use it.
    # Here we try a sensible order from observed labels.
    order_hint = [
        "小1","小2","小3","小4","小5","小6",
        "中1","中2","中3",
        "高1","高2","高3",
        "小学生","中学生","高校1年","高校2年","高校3年","高校生",
        "その他"
    ]
    present = list(pd.Index(df["grade_canon"].dropna().unique()))
    # sort by first index of hint; unseen go last alphabetically
    def _key(x):
        for i, h in enumerate(order_hint):
            if h in x:
                return (0, i)
        return (1, x)
    return sorted(present, key=_key)

# ---------- Plot helpers ----------
def add_scale_legend(ax, fontsize=10):
    lines = [
        "スコア対応（1–5）",
        "1：あてはまらない / そう思わない",
        "2：あまりあてはまらない / あまりそう思わない",
        "3：どちらともいえない",
        "4：少しあてはまる / ややそう思う",
        "5：あてはまる / そう思う",
        "※「使っていない」等は平均から除外"
    ]
    txt = "\n".join(lines)
    ax.text(
        0.98, 0.98, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=fontsize,
        linespacing=1.25,
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.35", edgecolor="none"),
        zorder=10,
    )

def plot_group_bar(
    df: pd.DataFrame, item_col: str, group_col: str,
    title: str, out_png: Path, out_csv: Path, order: Optional[List[str]] = None
):
    # map
    scores = df[item_col].map(map_likert_any)
    df2 = df.copy()
    df2["score"] = scores

    # compute denominators for unused%
    denom = df2.groupby(group_col)[item_col].size()
    used  = df2.loc[df2["score"].notna()].groupby(group_col)["score"].count()
    avg   = df2.loc[df2["score"].notna()].groupby(group_col)["score"].mean()

    g = pd.DataFrame({
        "respondents_used": used.astype("Int64"),
        "avg_score": avg,
        "denom_total": denom.astype("Int64"),
    })
    g["unused_pct"] = ((g["denom_total"] - g["respondents_used"]) / g["denom_total"] * 100.0).round(1)

    # drop groups with too few respondents (based on used, not denom)
    g = g[g["respondents_used"].fillna(0) >= MIN_RESP_PER_GROUP]

    if g.empty:
        print(f"[WARN] No groups pass MIN_RESP_PER_GROUP for {item_col} ({group_col}). Skipping.")
        return

    # reindex order
    if order:
        # keep only present
        present_order = [o for o in order if o in g.index]
        g = g.reindex(present_order)
    else:
        # sort by avg descending
        g = g.sort_values("avg_score", ascending=False)

    # save CSV
    g.to_csv(out_csv, encoding="utf-8", index=True)
    print(f"[info] wrote {out_csv}")

    # plot
    labels = g.index.tolist()
    y = np.arange(len(labels))
    x = g["avg_score"].values

    fig_h = max(3.8, 0.7 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=PLOT_DPI)

    ax.grid(axis="x", linestyle=(0, (2, 6)), alpha=0.25, zorder=1)
    for s in ["top","right","left","bottom"]:
        ax.spines[s].set_visible(False)

    bars = ax.barh(y, x, height=0.6, zorder=2, edgecolor="none")

    ax.set_yticks(y, labels=labels, fontsize=TICK_FONTSIZE)
    ax.set_xlim(1.0, 5.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.set_xlabel("平均スコア（1–5）", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)

    # annotate avg + n + 未使用%
    for i, (val, n, up) in enumerate(zip(g["avg_score"].values, g["respondents_used"].values, g["unused_pct"].values)):
        if np.isnan(val):
            continue
        text = f"{val:.2f}｜n={int(n)}｜未使用{up:.1f}%"
        inside_threshold = 3.1  # place inside if bar long enough
        if val >= inside_threshold:
            ax.text(val - 0.05, i, text, va="center", ha="right",
                    color="white", fontsize=BASE_FONTSIZE, fontweight="bold")
        else:
            ax.text(val + 0.05, i, text, va="center", ha="left",
                    color="black", fontsize=BASE_FONTSIZE)

    # de-emphasize 不明 (for school)
    if group_col == "school_canon":
        for bar, lab in zip(bars, labels):
            if lab == "不明":
                bar.set_alpha(0.45)

    add_scale_legend(ax, fontsize=10)
    plt.tight_layout()
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_png}")

# ---------- Main ----------
def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}. Run the cleaner first.")

    df = pd.read_csv(DATA_CLEAN, dtype=str)
    # normalize empties
    df = df.replace({"": np.nan}).infer_objects(copy=False)

    # Ensure school canon exists + validate allowed
    _ = SC.find_or_make_school_canon(df, debug=False)  # sets df["school_canon"]
    SC.assert_only_allowed(df)
    school_order = getattr(
        SC, "ALLOWED_SCHOOLS",
        ["不明","北海道天塩高等学校","岩沼小学校","洗足学園中学校",
         "西京高等学校","西京高等学校付属中学校","西賀茂中学校","北海道寿都高等学校"]
    )

    # Ensure grade_canon exists (your cleaner should have it; if not, fallback)
    if "grade_canon" not in df.columns:
        # naive fallback: use the raw column if present
        if "あなたの学年を教えてください" in df.columns:
            df["grade_canon"] = df["あなたの学年を教えてください"]
        else:
            df["grade_canon"] = np.nan
    grade_order = _grade_order_from_data(df)

    for item in ATTITUDE_ITEMS:
        col = _find_col(df, item)
        if not col:
            continue

        base = _norm(item)
        safe = base.replace("/", "_").replace("／", "_").replace("・","_")
        # Per school
        title_school = f"{item} — 学校別（学生）"
        out_png_school = OUT_DIR / f"{safe}_by_school.png"
        out_csv_school = OUT_DIR / f"{safe}_by_school.csv"
        plot_group_bar(
            df=df, item_col=col, group_col="school_canon",
            title=title_school, out_png=out_png_school, out_csv=out_csv_school,
            order=school_order
        )

        # Per grade
        title_grade = f"{item} — 学年別（学生）"
        out_png_grade = OUT_DIR / f"{safe}_by_grade.png"
        out_csv_grade = OUT_DIR / f"{safe}_by_grade.csv"
        plot_group_bar(
            df=df, item_col=col, group_col="grade_canon",
            title=title_grade, out_png=out_png_grade, out_csv=out_csv_grade,
            order=grade_order
        )

if __name__ == "__main__":
    main()
