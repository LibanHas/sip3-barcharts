# scripts/make_students_charts.py
# ------------------------------------------------------------
# Generates student survey charts:
# - Likert: 100% stacked (overall, by school, by school×grade)
# - Multi-select: frequency bars with UNIFORM option lists
#                  (overall, by school, by school×grade)
#
# Inputs:
#   data/students_clean.csv
#   data/students_multi_long.csv  (if present)
#
# Outputs:
#   outputs/students_viz/*.png
#   outputs/students_viz/school_grade/*.png
#   outputs/students_viz/school_grade_multi/*.png
# ------------------------------------------------------------

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm

# ---------- Paths ----------
BASE = Path(".")
IN_SINGLE = BASE / "data/students_clean.csv"
IN_MULTI  = BASE / "data/students_multi_long.csv"
OUT_DIR   = BASE / "outputs/students_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_SG = OUT_DIR / "school_grade"
OUT_DIR_SG.mkdir(parents=True, exist_ok=True)
OUT_DIR_SG_MULTI = OUT_DIR / "school_grade_multi"
OUT_DIR_SG_MULTI.mkdir(parents=True, exist_ok=True)

# -----------------------------
# JP-safe font (robust; bundled → system fallbacks)
# -----------------------------
def set_jp_font():
    """
    Try project-bundled IPAex first, then system JP fonts.
    Returns (family_name, FontProperties) and sets rcParams.
    """
    # 1) Project-bundled IPAex (repo-local)
    candidates = [
        Path("fonts/ipaexg.ttf"),
        Path("fonts/ipaexm.ttf"),
        Path(__file__).resolve().parents[1] / "fonts" / "ipaexg.ttf",
        Path(__file__).resolve().parents[1] / "fonts" / "ipaexm.ttf",
    ]
    added = []
    for p in candidates:
        if p.exists():
            try:
                fm.fontManager.addfont(str(p))
                added.append(p)
            except Exception:
                pass
    if added:
        fp = fm.FontProperties(fname=str(added[0]), weight="regular")
        fam = fp.get_name() or "IPAexGothic"
        mpl.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": [fam, "Hiragino Sans", "AppleGothic", "Yu Gothic UI", "Meiryo", "Noto Sans CJK JP"],
            "axes.unicode_minus": False,
            "font.weight": "regular",
            "axes.titleweight": "regular",
            "axes.labelweight": "regular",
        })
        resolved = fm.findfont(fp, fallback_to_default=False)
        print(f"[JP FONT] Using bundled: {fam} | {resolved}")
        return fam, fp

    # 2) System fallbacks (macOS / Windows / Linux)
    fallbacks = ["Hiragino Sans", "AppleGothic", "Yu Gothic UI", "Meiryo", "Noto Sans CJK JP", "IPAGothic", "TakaoGothic"]
    for fam in fallbacks:
        try:
            fp = fm.FontProperties(family=fam, weight="regular")
            _ = fm.findfont(fp, fallback_to_default=False)
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": [fam],
                "axes.unicode_minus": False,
                "font.weight": "regular",
                "axes.titleweight": "regular",
                "axes.labelweight": "regular",
            })
            print(f"[JP FONT] Using system: {fam}")
            return fam, fp
        except Exception:
            continue

    # 3) Last resort (may show tofu; keeps script running)
    print("[JP FONT] WARNING: No JP font found; text may show tofu.")
    fp = fm.FontProperties()
    mpl.rcParams.update({"axes.unicode_minus": False})
    return "default", fp

FAMILY_NAME, JP_FP = set_jp_font()

def set_textprops_regular(ax, fp=JP_FP):
    if ax.title: ax.title.set_fontproperties(fp)
    if ax.xaxis.label: ax.xaxis.label.set_fontproperties(fp)
    if ax.yaxis.label: ax.yaxis.label.set_fontproperties(fp)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontproperties(fp)

def legend_with_fp(ax, *args, **kwargs):
    kwargs.setdefault("prop", JP_FP)
    lg = ax.legend(*args, **kwargs)
    if lg:
        for t in lg.get_texts():
            t.set_fontproperties(JP_FP)
    return lg

# Seaborn theme (don’t pass font=… so we keep our rcParams)
sns.set_theme(style="whitegrid", rc={"axes.spines.right": False, "axes.spines.top": False, "figure.dpi": 120})

# Tiny font smoke test (writes a one-liner PNG so you can check glyphs)
try:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3.8, 1.2))
    plt.text(0.03, 0.6, "日本語フォント OK？（テスト）", fontsize=14, fontproperties=JP_FP)
    plt.axis("off")
    plt.savefig(OUT_DIR / "_font_smoketest.png", dpi=180, bbox_inches="tight")
    plt.close()
except Exception:
    pass

# ---------- Load ----------
if not IN_SINGLE.exists():
    raise FileNotFoundError(f"Missing {IN_SINGLE}")
df = pd.read_csv(IN_SINGLE)

df_multi = pd.read_csv(IN_MULTI) if IN_MULTI.exists() else pd.DataFrame()
print(f"Loaded: {len(df)} rows x {len(df.columns)} cols (students_clean.csv)")
if not df_multi.empty:
    print(f"Loaded: {len(df_multi)} rows (students_multi_long.csv)")

# ---------- Key columns ----------
col_school = next((c for c in df.columns if ("学校名" in c) or ("学校" in c and "名" in c)), None)
if col_school is None:
    col_school = next((c for c in df.columns if "学校" in c), None)
col_grade  = next((c for c in df.columns if "学年" in c), None)
col_schooltype = next((c for c in df.columns if "学校種" in c), None)
print(f"Detected columns → school: {col_school}, grade: {col_grade}, schooltype: {col_schooltype}")

# ---------- Likert detection ----------
LIKERT_LABELS = {
    "agree_5to1": ["とてもそう思う", "そう思う", "どちらともいえない", "あまりそう思わない", "全くそう思わない"],
    "agree_4to1": ["とてもそう思う", "そう思う", "あまりそう思わない", "全くそう思わない"],
    "freq":       ["ほぼ毎時間", "1週間に数回程度", "1ヶ月に数回程度", "ほとんど使用していない"],
    "agree_alt":  ["あてはまる", "少しあてはまる", "あまりあてはまらない", "あてはまらない", "使っていない"],
    "freq_alt":   ["頻繁（ひんぱん）に使用している","使用することがある","あまり使用しない","全く使用しない","質問の意味がわからない"],
}
LIKERT_SETS = [set(v) for v in LIKERT_LABELS.values()]

def is_likert(col: str) -> bool:
    vals = set(pd.Series(df[col].dropna().astype(str).unique(), dtype=str))
    return any(vals.issubset(s) or s.issubset(vals) for s in LIKERT_SETS)

def choose_order(col: str):
    vals = set(pd.Series(df[col].dropna().astype(str).unique(), dtype=str))
    for seq in LIKERT_LABELS.values():
        s = set(seq)
        if vals.issubset(s) or s.issubset(vals):
            return [x for x in seq if x in vals]
    # fallback: frequent first
    return df[col].value_counts().index.tolist()

likert_cols = [c for c in df.columns if is_likert(c)]
print(f"Detected {len(likert_cols)} Likert-like columns.")

# ---------- Helpers ----------
def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9一-龥ぁ-んァ-ヴー]+", "_", str(s)).strip("_")

def pct_table(series, categories):
    vc = series.value_counts(dropna=False)
    total = float(vc.sum()) if vc.sum() else 1.0
    arr = [vc.get(cat, 0) for cat in categories]
    return np.array(arr, dtype=float) / total * 100.0

def stacked_bar_100(ax, distribution, categories, title=None, xlabel="割合（%）"):
    left = 0.0
    for cat, val in zip(categories, distribution):
        ax.barh([0], [val], left=left, label=cat)
        left += val
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontproperties=JP_FP)
    legend_with_fp(ax, categories, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    set_textprops_regular(ax, JP_FP)

def stacked_by_group(df_g, col, categories, group_col, top_n=8, filename_prefix=""):
    counts = df_g[group_col].value_counts().head(top_n).index.tolist()
    if not counts:
        return
    data = []
    for g in counts:
        s = df_g.loc[df_g[group_col] == g, col]
        data.append(pct_table(s, categories))
    data = np.vstack(data)

    fig_h = max(2.5, 0.5 * len(counts) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    left = np.zeros(len(counts))
    for cat, row in zip(categories, data.T):
        ax.barh(counts, row, left=left, label=cat)
        left += row
    ax.set_xlim(0, 100)
    ax.set_xlabel("割合（%）", fontproperties=JP_FP)
    ax.set_title(f"{col}（上位{len(counts)} {group_col}）", fontproperties=JP_FP)
    legend_with_fp(ax, categories, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    set_textprops_regular(ax, JP_FP)
    fig.tight_layout()
    out = OUT_DIR / f"{filename_prefix}{sanitize(col)}__by_{sanitize(group_col)}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Grade ordering helpers
GRADE_ORDER = [
    "小学校5年生","小学校6年生",
    "中学校1年生","中学校2年生","中学校3年生",
    "高校1年生","高校2年生","高校3年生",
]
_grade_pos = {g:i for i,g in enumerate(GRADE_ORDER)}
def sort_grades_present(labels):
    labels = [str(x) for x in labels if pd.notna(x)]
    known  = [g for g in GRADE_ORDER if g in labels]
    other  = sorted([g for g in labels if g not in _grade_pos])
    return known + other

def stacked_by_grade_within_school(df_all, col, categories, school, group_school, group_grade, min_n=5):
    sub = df_all[(df_all[group_school] == school) & df_all[col].notna()]
    if sub.empty or group_grade not in sub.columns:
        return
    grades = sort_grades_present(sub[group_grade].dropna().unique().tolist())
    if not grades:
        return

    dists, ns = [], []
    for g in grades:
        s = sub.loc[sub[group_grade] == g, col]
        n = s.notna().sum()
        dists.append(pct_table(s, categories))
        ns.append(int(n))
    data = np.vstack(dists)

    fig_h = max(2.8, 0.55 * len(grades) + 1.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    left = np.zeros(len(grades))
    for j, cat in enumerate(categories):
        vals = data[:, j]
        ax.barh(grades, vals, left=left, label=cat)
        left += vals

    # annotate n per bar (right-hand side)
    for i, n in enumerate(ns):
        ax.text(102, i, f"n={n}", va="center", fontsize=9, fontproperties=JP_FP)

    ax.set_xlim(0, 110)
    ax.set_xlabel("割合（%）", fontproperties=JP_FP)
    ax.set_title(f"{col}｜{school}（学年別）", fontproperties=JP_FP)
    legend_with_fp(ax, categories, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    set_textprops_regular(ax, JP_FP)
    fig.tight_layout()
    out = OUT_DIR_SG / f"{sanitize(col)}__{sanitize(school)}__by_grade.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------- 1) Likert charts ----------
TOP_SCHOOLS_FOR_SG = 6  # limit to avoid too many files

for col in likert_cols:
    cats = choose_order(col)

    # Overall
    dist = pct_table(df[col], cats)
    fig, ax = plt.subplots(figsize=(9, 2.2))
    stacked_bar_100(ax, dist, cats, title=col)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{sanitize(col)}__overall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # By school (top-N)
    if col_school and df[col_school].notna().any():
        stacked_by_group(
            df.dropna(subset=[col_school]),
            col,
            cats,
            group_col=col_school,
            top_n=10,
            filename_prefix="school_",
        )

    # By school × grade (top schools)
    if col_school and col_grade and df[col_school].notna().any():
        top_schools = df[col_school].value_counts().head(TOP_SCHOOLS_FOR_SG).index.tolist()
        for sch in top_schools:
            stacked_by_grade_within_school(
                df, col, cats, sch, group_school=col_school, group_grade=col_grade, min_n=5
            )

# ---------- 2) Multi-select charts with UNIFORM option lists ----------
if not df_multi.empty:
    # Build a dictionary of ALL options per multi-select question (uniform lists)
    all_opts_by_col = (
        df_multi.groupby("column")["choice"]
        .apply(lambda s: sorted(pd.Series(s.dropna().unique()).astype(str)))
        .to_dict()
    )

    def freq_with_zeros(series, all_opts):
        vc = series.value_counts()
        # fill zeros for options not present in this subset
        for opt in all_opts:
            if opt not in vc.index:
                vc.loc[opt] = 0
        return vc.reindex(all_opts)

    # Overall frequency per question (uniform options)
    for col_name, sub in df_multi.groupby("column"):
        all_opts = all_opts_by_col.get(col_name, [])
        if not all_opts:
            continue
        freq = freq_with_zeros(sub["choice"], all_opts)
        fig_h = max(2.8, 0.4 * len(freq) + 1)
        fig, ax = plt.subplots(figsize=(9, fig_h))
        sns.barplot(x=freq.values, y=list(freq.index), ax=ax)
        ax.set_xlabel("回答数", fontproperties=JP_FP); ax.set_ylabel("", fontproperties=JP_FP)
        ax.set_title(f"{col_name}（全選択肢）", fontproperties=JP_FP)
        set_textprops_regular(ax, JP_FP)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{sanitize(col_name)}__multi_overall.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Per school (uniform options)
        if col_school and col_school in df.columns:
            top_schools = df[col_school].value_counts().head(8).index.tolist()
            for sch in top_schools:
                idx = df[df[col_school] == sch].index
                choices = sub[sub["id"].isin(idx)]["choice"]
                freq = freq_with_zeros(choices, all_opts)
                if freq.sum() == 0:
                    continue
                fig_h = max(2.6, 0.4 * len(freq) + 1)
                fig, ax = plt.subplots(figsize=(9, fig_h))
                sns.barplot(x=freq.values, y=list(freq.index), ax=ax)
                ax.set_xlabel("回答数", fontproperties=JP_FP); ax.set_ylabel("", fontproperties=JP_FP)
                ax.set_title(f"{col_name}｜{sch}（全選択肢）", fontproperties=JP_FP)
                set_textprops_regular(ax, JP_FP)
                fn = OUT_DIR / f"{sanitize(col_name)}__multi_{sanitize(sch)}.png"
                fig.tight_layout(); fig.savefig(fn, dpi=200, bbox_inches="tight")
                plt.close(fig)

        # Per school × grade (uniform options; top schools)
        if col_school and col_grade and col_school in df.columns and col_grade in df.columns:
            top_schools = df[col_school].value_counts().head(6).index.tolist()
            for sch in top_schools:
                idx = df[df[col_school] == sch][[col_grade]].copy()
                grades = idx[col_grade].dropna().unique().tolist()
                ORDER = [
                    "小学校5年生","小学校6年生",
                    "中学校1年生","中学校2年生","中学校3年生",
                    "高校1年生","高校2年生","高校3年生",
                ]
                grades = [g for g in ORDER if g in grades] + [g for g in grades if g not in ORDER]

                for g in grades:
                    ixs = idx[idx[col_grade] == g].index
                    choices = sub[sub["id"].isin(ixs)]["choice"]
                    freq = freq_with_zeros(choices, all_opts)
                    if freq.sum() == 0:
                        continue
                    fig_h = max(2.6, 0.4 * len(freq) + 1)
                    fig, ax = plt.subplots(figsize=(9, fig_h))
                    sns.barplot(x=freq.values, y=list(freq.index), ax=ax)
                    ax.set_xlabel("回答数", fontproperties=JP_FP); ax.set_ylabel("", fontproperties=JP_FP)
                    ax.set_title(f"{col_name}｜{sch}｜{g}（全選択肢）", fontproperties=JP_FP)
                    set_textprops_regular(ax, JP_FP)
                    fn = OUT_DIR_SG_MULTI / f"{sanitize(col_name)}__multi_{sanitize(sch)}__{sanitize(g)}.png"
                    fig.tight_layout(); fig.savefig(fn, dpi=200, bbox_inches="tight")
                    plt.close(fig)

print(f"✅ Charts written to: {OUT_DIR.resolve()}")
print(f"   School×Grade Likert charts → {OUT_DIR_SG.resolve()}")
print(f"   School×Grade Multi-select charts → {OUT_DIR_SG_MULTI.resolve()}")
