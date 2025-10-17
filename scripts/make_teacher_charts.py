# -*- coding: utf-8 -*-
"""
make_students_charts.py

Students: per-question charts, one chart per 学校種, bars split by 学年.
- Likert: 100% stacked (grades as bars)
- Multi-select: 100% stacked by options (uniform option list per question, grades as bars)

Inputs:
  data/students_clean.csv
  data/students_multi_long.csv  (optional; columns: id/row_id, column, choice)

Outputs:
  outputs/students_by_schooltype_grade/<schooltype>/<question>.png
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

# =====================#
# Font handling (same pattern as teachers)
# =====================#
def force_ipaex():
    candidates = [
        Path.home() / "Library/Fonts" / "ipaexg.ttf",
        Path.home() / "Library/Fonts" / "ipaexm.ttf",
    ]
    for p in candidates:
        if p.exists():
            try:
                fm.fontManager.addfont(str(p))
                name = fm.FontProperties(fname=str(p)).get_name()
                fp = fm.FontProperties(fname=str(p), weight="regular")
                mpl.rcParams.update({
                    "font.family": "sans-serif",
                    "font.sans-serif": [name, "Hiragino Sans", "AppleGothic"],
                    "axes.unicode_minus": False,
                    "font.weight": "regular",
                    "axes.titleweight": "regular",
                    "axes.labelweight": "regular",
                })
                return name, fp
            except Exception:
                pass
    for fallback in ("Hiragino Sans", "AppleGothic"):
        try:
            fp = fm.FontProperties(family=fallback, weight="regular")
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": [fallback],
                "axes.unicode_minus": False,
                "font.weight": "regular",
                "axes.titleweight": "regular",
                "axes.labelweight": "regular",
            })
            return fallback, fp
        except Exception:
            pass
    fp = fm.FontProperties()
    mpl.rcParams.update({"axes.unicode_minus": False})
    return "default", fp

FAMILY_NAME, JP_FP = force_ipaex()

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

# =====================#
# Paths
# =====================#
BASE = Path(__file__).resolve().parents[1]
IN_SINGLE = BASE / "data/students_clean.csv"
IN_MULTI  = BASE / "data/students_multi_long.csv"
OUT_DIR   = BASE / "outputs/students_by_schooltype_grade"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================#
# Theme
# =====================#
sns.set_theme(style="whitegrid")
mpl.rcParams["axes.unicode_minus"] = False

# =====================#
# Helpers
# =====================#
def safe_name(s):
    return re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff-]+", "_", str(s)).strip("_")[:140]

def set_percent_axis(ax):
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlim(0, 100)
    ax.set_xlabel("割合（%）", fontproperties=JP_FP)

# canonical Likert orders
ORDER_LIKERT_AGREE = ["あてはまらない","あまりあてはまらない","どちらともいえない","少しあてはまる","あてはまる","使っていない"]
ORDER_LIKERT_FREQ  = ["全く使用しない","あまり使用しない","使用することがある","頻繁（ひんぱん）に使用している","質問の意味がわからない"]
LIKERT_SETS = [
    set(ORDER_LIKERT_AGREE),
    set(["あてはまる","少しあてはまる","あまりあてはまらない","あてはまらない","使っていない"]),
    set(["ほぼ毎時間","1週間に数回程度","1ヶ月に数回程度","ほとんど使用していない"]),  # alt freq
    set(ORDER_LIKERT_FREQ),
]

def is_likert(series: pd.Series) -> bool:
    vals = set(pd.Series(series.dropna().astype(str).unique(), dtype=str))
    return any(vals.issubset(s) or s.issubset(vals) for s in LIKERT_SETS)

def choose_likert_order(series: pd.Series):
    vals = set(pd.Series(series.dropna().astype(str).unique(), dtype=str))
    for seq in (ORDER_LIKERT_AGREE, ORDER_LIKERT_FREQ,
                ["ほぼ毎時間","1週間に数回程度","1ヶ月に数回程度","ほとんど使用していない"]):
        s = set(seq)
        if vals.issubset(s) or s.issubset(vals):
            return [x for x in seq if x in vals]
    # fallback: most frequent → least
    return series.value_counts().index.tolist()

GRADE_ORDER_BY_TYPE = {
    "小学校": ["小学校5年生","小学校6年生"],
    "中学校": ["中学校1年生","中学校2年生","中学校3年生"],
    "高等学校": ["高校1年生","高校2年生","高校3年生"],
}
ALL_GRADES_ORDER = GRADE_ORDER_BY_TYPE["小学校"] + GRADE_ORDER_BY_TYPE["中学校"] + GRADE_ORDER_BY_TYPE["高等学校"]

def grades_for_type(df: pd.DataFrame, schooltype_col: str, grade_col: str, stype: str):
    present = df.loc[df[schooltype_col]==stype, grade_col].dropna().astype(str).unique().tolist()
    order = [g for g in GRADE_ORDER_BY_TYPE.get(stype, ALL_GRADES_ORDER) if g in present]
    # any weird extras go at the end
    order += [g for g in present if g not in order]
    return order

def pct_row(series: pd.Series, cats: list) -> np.ndarray:
    vc = series.value_counts(dropna=False)
    denom = float(series.notna().sum()) or 1.0
    return np.array([100.0 * vc.get(c, 0) / denom for c in cats], dtype=float)

# =====================#
# Load
# =====================#
if not IN_SINGLE.exists():
    raise FileNotFoundError(f"Missing {IN_SINGLE}")
df = pd.read_csv(IN_SINGLE)

df_multi = pd.read_csv(IN_MULTI) if IN_MULTI.exists() else pd.DataFrame()

# Detect key columns
col_schooltype = next((c for c in df.columns if "学校種" in c), "学校種")
col_grade      = next((c for c in df.columns if "学年"   in c), "学年")
if col_schooltype not in df.columns or col_grade not in df.columns:
    raise RuntimeError(f"Could not find 学校種/学年 columns. Found: {list(df.columns)}")

# =====================#
# Find question columns
# =====================#
question_cols = [c for c in df.columns if c not in (col_schooltype, col_grade)]
# very common metadata names to skip if present
SKIP = {"ID","開始時刻","完了時刻","メール","名前"}
question_cols = [c for c in question_cols if c not in SKIP]

likert_cols = [c for c in question_cols if is_likert(df[c])]
non_likert_cols = [c for c in question_cols if c not in likert_cols]

# =====================#
# Output directories by school type
# =====================#
schooltypes = [t for t in ["小学校","中学校","高等学校"] if t in df[col_schooltype].dropna().unique().tolist()]
if not schooltypes:
    schooltypes = sorted(df[col_schooltype].dropna().unique().tolist())

# =====================#
# 1) Likert charts — per school type, bars = grades
# =====================#
for q in likert_cols:
    cats = choose_likert_order(df[q])
    for stype in schooltypes:
        sub = df[(df[col_schooltype]==stype) & df[q].notna()]
        if sub.empty: 
            continue
        grades = grades_for_type(df, col_schooltype, col_grade, stype)
        if not grades:
            continue

        mat = []
        ns = []
        for g in grades:
            s = sub.loc[sub[col_grade]==g, q]
            if s.empty:
                mat.append(np.zeros(len(cats))); ns.append(0)
            else:
                mat.append(pct_row(s, cats)); ns.append(int(s.notna().sum()))
        M = np.vstack(mat)

        # plot
        fig_h = max(2.8, 0.55*len(grades) + 1.2)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        left = np.zeros(len(grades))
        for j, cat in enumerate(cats):
            ax.barh(grades, M[:, j], left=left, label=cat)
            left += M[:, j]

        # n labels
        for i, (g, n) in enumerate(zip(grades, ns)):
            ax.text(102, i, f"n={n}", va="center", fontsize=9)

        ax.set_title(f"{q}｜{stype}（学年別）", fontproperties=JP_FP)
        set_percent_axis(ax); set_textprops_regular(ax, JP_FP)
        ax.set_xlim(0, 110)
        legend_with_fp(ax, bbox_to_anchor=(1.02,1), loc="upper left", frameon=False)
        fig.tight_layout()

        out_dir = OUT_DIR / safe_name(stype)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{safe_name(q)}__{safe_name(stype)}__by_grade.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

# =====================#
# 2) Multi-select charts — per school type, bars = grades
# Expect df_multi: columns ["column","choice","id" or "row_id"]
# =====================#
if not df_multi.empty:
    id_col = "id" if "id" in df_multi.columns else ("row_id" if "row_id" in df_multi.columns else None)
    if id_col is not None and {"column","choice",id_col}.issubset(df_multi.columns):
        # Build uniform option lists per question
        all_opts_by_col = (
            df_multi.groupby("column")["choice"]
            .apply(lambda s: sorted(pd.Series(s.dropna().unique()).astype(str)))
            .to_dict()
        )

        for col_name, sub_m in df_multi.groupby("column"):
            all_opts = all_opts_by_col.get(col_name, [])
            if not all_opts:
                continue

            for stype in schooltypes:
                # grades present for this type
                grades = grades_for_type(df, col_schooltype, col_grade, stype)
                if not grades:
                    continue

                # Build matrix: rows = grades, cols = options → % of respondents in that grade who selected option
                mat = []
                ns = []
                for g in grades:
                    idxs = df[(df[col_schooltype]==stype) & (df[col_grade]==g)].index
                    denom = len(idxs)
                    ns.append(int(denom))
                    if denom == 0:
                        mat.append(np.zeros(len(all_opts))); continue
                    choices = sub_m[sub_m[id_col].isin(idxs)]["choice"]
                    vc = choices.value_counts()
                    row = np.array([100.0 * vc.get(opt, 0) / float(denom) for opt in all_opts], dtype=float)
                    mat.append(row)
                M = np.vstack(mat)

                # 100% stacked by options across each grade
                fig_h = max(2.8, 0.55*len(grades) + 1.2)
                fig, ax = plt.subplots(figsize=(10, fig_h))
                left = np.zeros(len(grades))
                for j, opt in enumerate(all_opts):
                    ax.barh(grades, M[:, j], left=left, label=opt)
                    left += M[:, j]

                for i, (g, n) in enumerate(zip(grades, ns)):
                    ax.text(102, i, f"n={n}", va="center", fontsize=9)

                ax.set_title(f"{col_name}｜{stype}（学年別・複数選択 100%積み上げ）", fontproperties=JP_FP)
                set_percent_axis(ax); set_textprops_regular(ax, JP_FP)
                ax.set_xlim(0, 110)
                legend_with_fp(ax, bbox_to_anchor=(1.02,1), loc="upper left", frameon=False)
                fig.tight_layout()

                out_dir = OUT_DIR / safe_name(stype)
                out_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_dir / f"{safe_name(col_name)}__multi__{safe_name(stype)}__by_grade.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

print(f"✅ Charts written to: {OUT_DIR.resolve()}")
