# scripts/charts_teachers/runners.py
"""
Chart runners for teacher survey.

Each runner:
- loads the needed data
- prepares/aggregates for a question set
- calls plotting primitives
- saves PNGs under figs/teachers/...
"""

from __future__ import annotations
from pathlib import Path
import re

from . import config, io, prepare, plots


# ---------- small utils ----------
def _require_columns(df, columns: list[str], where: str = ""):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        loc = f" in {where}" if where else ""
        raise KeyError(f"Missing columns{loc}: {missing}")

def _safe_fn(s: str) -> str:
    """Make a safe filename fragment from any string (handles JP + punctuation)."""
    s = re.sub(r"\s+", "_", str(s))
    s = re.sub(r'[\\/:"*?<>|]+', "_", s)
    return s[:120]


# ---------- 1) Demographics ----------
def run_bars_demographics():
    df = io.load_clean()
    out = config.FIGS / "demographics"

    col_age = "年齢"
    col_school_type = "学校種"
    col_exp = "教職経験年数"
    col_role = "役職(主業務に最も近いものを選択してください。)"

    _require_columns(df, [col_age, col_school_type, col_exp, col_role], "demographics")

    plots.bar_simple(
        df[col_age].value_counts(),
        title="年齢（分布）",
        xlabel="年齢",
        ylabel="人数",
        outpath=out / "Q02_年齢.png",
    )

    plots.bar_simple(
        df[col_school_type].value_counts(),
        title="学校種（分布）",
        xlabel="学校種",
        ylabel="人数",
        outpath=out / "Q03_学校種.png",
    )

    plots.bar_horizontal(
        df[col_exp].value_counts(),
        title="教職経験年数（分布）",
        xlabel="人数",
        ylabel="カテゴリ",
        outpath=out / "Q04_経験年数.png",
    )

    plots.bar_simple(
        df[col_role].value_counts(),
        title="役職（分布）",
        xlabel="役職",
        ylabel="人数",
        outpath=out / "Q05_役職.png",
    )


# ---------- 2) Frequency + Likert stacks (by 学校種, vertical) ----------
def run_frequency_and_likert():
    df = io.load_clean()
    out = config.FIGS / "likert"

    # Q6 授業でのICT活用
    col_ict_freq = "授業でのICT活用"
    _require_columns(df, ["学校種", col_ict_freq], "Q6")
    pct = prepare.pct_table(df, ["学校種"], col_ict_freq, order=config.ICT_FREQ_ORDER)
    plots.bar_100pct(
        pct,
        title="授業でのICT活用（100%積み上げ）",
        xlabel="学校種",
        ylabel="割合(%)",
        outpath=out / "Q06_ICT頻度.png",
        legend_title="頻度",
        show_group_n=True,
    )

    # Q8 LEAF利用頻度
    col_leaf_freq = "LEAFシステム(BookRoll,分析ツール)を授業・授業外(宿題など)でどれくらい利用していますか"
    if col_leaf_freq in df.columns:
        pct = prepare.pct_table(df, ["学校種"], col_leaf_freq, order=config.LEAF_FREQ_ORDER)
        plots.bar_100pct(
            pct,
            title="LEAF利用頻度（100%積み上げ）",
            xlabel="学校種",
            ylabel="割合(%)",
            outpath=out / "Q08_LEAF頻度.png",
            legend_title="頻度",
            show_group_n=True,
        )

    # Q12–20 使用目的（5段階）
    items_12_20 = [
        "教材を配布するため",
        "児童生徒に自分の解答を振り返らせるため",
        "児童生徒にクラスの人の解答を提示・参照させるため",
        "児童生徒にクラスの人の学び方を提示・参照させるため",
        "児童生徒にAIによる問題推薦やアドバイスを受けさせるため",
        "データに基づいてグループを編成するため",
        "クラス全体の傾向を見るため",
        "個人の様子を見るため",
        "取り組みのプロセスを見るため",
    ]
    for q in items_12_20:
        if q in df.columns:
            pct = prepare.pct_table(df, ["学校種"], q, order=config.USE_ORDER_4)
            plots.bar_100pct(
                pct,
                title=f"{q}（100%積み上げ）",
                xlabel="学校種",
                ylabel="割合(%)",
                outpath=out / f"Q_使用目的_{_safe_fn(q)}.png",
                legend_title="頻度",
                show_group_n=True,
            )

    # Q21–30 効果/満足（Likert）
    items_21_30 = [
        "児童生徒の理解度が高まったと感じる",
        "授業準備時間が短縮した",
        "根拠に基づいた授業設計や教材改善を行えるようになった",
        "分析ツールを使用することで子供たちの学習方法に変化が生じた",
        "個別最適な学びを実現する支援になったと感じる",
        "主体的で対話的で深い学びを実現する支援になったと感じる",
        "協働的な学びを実現する支援になったと感じる",
        "探究的な学びを実現する支援になったと感じる",
        "使い方・操作手順がわかりやすい",
        "LEAFシステムに満足している",
    ]
    for q in items_21_30:
        if q in df.columns:
            pct = prepare.pct_table(df, ["学校種"], q, order=config.AGREE_ORDER)
            plots.bar_100pct(
                pct,
                title=f"{q}（100%積み上げ）",
                xlabel="学校種",
                ylabel="割合(%)",
                outpath=out / f"Q_効果_{_safe_fn(q)}.png",
                legend_title="回答",
                show_group_n=True,
            )


# ---------- 2b) Frequency + Likert stacks（学校別 = 学校名_canon, horizontal） ----------
def run_frequency_and_likert_by_school(min_n: int = 2, top_n: int | None = 12):
    df = io.load_clean()
    out = config.FIGS / "likert_by_school"

    _require_columns(df, ["学校名_canon"], "likert_by_school")

    # Filter to schools with >= min_n respondents and (optionally) top_n
    if min_n and "respondent_id" in df.columns:
        n = df.groupby("学校名_canon")["respondent_id"].nunique()
        keep = n[n >= min_n].index
        df = df[df["学校名_canon"].isin(keep)]
        if top_n:
            keep_top = (
                df.groupby("学校名_canon")["respondent_id"]
                  .nunique().sort_values(ascending=False)
                  .head(top_n).index
            )
            df = df[df["学校名_canon"].isin(keep_top)]

    # Q6 授業でのICT活用（学校別）
    col_ict_freq = "授業でのICT活用"
    if col_ict_freq in df.columns:
        pct = prepare.pct_table(df, ["学校名_canon"], col_ict_freq, order=config.ICT_FREQ_ORDER)
        plots.bar_100pct(
            pct,
            title="授業でのICT活用（学校別・100%積み上げ）",
            xlabel="割合(%)",
            ylabel="学校名",
            outpath=out / "Q06_ICT頻度__学校別.png",
            legend_title="頻度",
            orientation="h",
            show_group_n=True,
        )

    # Q8 LEAF利用頻度（学校別）
    col_leaf_freq = "LEAFシステム(BookRoll,分析ツール)を授業・授業外(宿題など)でどれくらい利用していますか"
    if col_leaf_freq in df.columns:
        pct = prepare.pct_table(df, ["学校名_canon"], col_leaf_freq, order=config.LEAF_FREQ_ORDER)
        plots.bar_100pct(
            pct,
            title="LEAF利用頻度（学校別・100%積み上げ）",
            xlabel="割合(%)",
            ylabel="学校名",
            outpath=out / "Q08_LEAF頻度__学校別.png",
            legend_title="頻度",
            orientation="h",
            show_group_n=True,
        )

    # Q12–20 使用目的（学校別）
    items_12_20 = [
        "教材を配布するため",
        "児童生徒に自分の解答を振り返らせるため",
        "児童生徒にクラスの人の解答を提示・参照させるため",
        "児童生徒にクラスの人の学び方を提示・参照させるため",
        "児童生徒にAIによる問題推薦やアドバイスを受けさせるため",
        "データに基づいてグループを編成するため",
        "クラス全体の傾向を見るため",
        "個人の様子を見るため",
        "取り組みのプロセスを見るため",
    ]
    for q in items_12_20:
        if q in df.columns:
            pct = prepare.pct_table(df, ["学校名_canon"], q, order=config.USE_ORDER_4)
            plots.bar_100pct(
                pct,
                title=f"{q}（学校別・100%積み上げ）",
                xlabel="割合(%)",
                ylabel="学校名",
                outpath=out / f"Q_使用目的__{_safe_fn(q)}__学校別.png",
                legend_title="頻度",
                orientation="h",
                show_group_n=True,
            )

    # Q21–30 効果/満足（学校別）
    items_21_30 = [
        "児童生徒の理解度が高まったと感じる",
        "授業準備時間が短縮した",
        "根拠に基づいた授業設計や教材改善を行えるようになった",
        "分析ツールを使用することで子供たちの学習方法に変化が生じた",
        "個別最適な学びを実現する支援になったと感じる",
        "主体的で対話的で深い学びを実現する支援になったと感じる",
        "協働的な学びを実現する支援になったと感じる",
        "探究的な学びを実現する支援になったと感じる",
        "使い方・操作手順がわかりやすい",
        "LEAFシステムに満足している",
    ]
    for q in items_21_30:
        if q in df.columns:
            pct = prepare.pct_table(df, ["学校名_canon"], q, order=config.AGREE_ORDER)
            plots.bar_100pct(
                pct,
                title=f"{q}（学校別・100%積み上げ）",
                xlabel="割合(%)",
                ylabel="学校名",
                outpath=out / f"Q_効果__{_safe_fn(q)}__学校別.png",
                legend_title="回答",
                orientation="h",
                show_group_n=True,
            )


# ---------- 3) Multi-select (by 学校種) ----------
def run_multiselect():
    df = io.load_clean()
    m = io.load_multi()
    out = config.FIGS / "multi"

    m = prepare.attach_facets(m, df, facets=("学校種",))

    questions = [
        "LEAFシステム(BookRoll,分析ツール)をどの教科で使用しますか(複数選択可)",
        "BookRollでよく使う機能を選んでください(複数選択可)",
        "分析ツール(ログパレ)でよく使う機能を選んでください(複数選択可)",
    ]
    for q in questions:
        tidy = prepare.multi_counts(m, q, facet_col="学校種")
        plots.multi_bars_by_facet(tidy, q, out, facet_col="学校種")


# ---------- 4) Multi-select (by 学校別 = 学校名_canon) ----------
def run_multiselect_by_school(min_n: int = 2):
    df = io.load_clean()
    m = io.load_multi()
    out = config.FIGS / "multi_by_school"

    m = prepare.attach_facets(m, df, facets=("学校名_canon",))
    if min_n and min_n > 1:
        m = prepare.filter_facets_by_min_n(m, "学校名_canon", min_n=min_n)

    questions = [
        "LEAFシステム(BookRoll,分析ツール)をどの教科で使用しますか(複数選択可)",
        "BookRollでよく使う機能を選んでください(複数選択可)",
        "分析ツール(ログパレ)でよく使う機能を選んでください(複数選択可)",
    ]
    for q in questions:
        tidy = prepare.multi_counts(m, q, facet_col="学校名_canon")
        plots.multi_bars_by_facet(
            tidy, q, out,
            facet_col="学校名_canon",
            sort_by="pct",
            top_k=12,
        )


def run_numeric_by_school(min_n: int = 1, top_n: int | None = None):
    """
    Per-school numeric charts as HORIZONTAL bars of MEANS.
    - LEAF months:   *_months_total  → 平均月数（学校別）
    - Work-time Q32–36: ...時間(...) → 平均時間（学校別）
    """
    import re
    import pandas as pd

    def _coerce_numeric_series(s: pd.Series) -> pd.Series:
        # Robust numeric coercion: handle full-width digits, commas, stray text
        s = s.astype(str)
        fw = "０１２３４５６７８９．，−－ー"
        hw = "0123456789..---"
        trans = str.maketrans({fw[i]: hw[i] for i in range(len(fw))})
        s = s.map(lambda x: x.translate(trans))
        s = s.str.replace(",", "", regex=False)
        s = s.str.extract(r"([-+]?\d*\.?\d+)")[0]
        return pd.to_numeric(s, errors="coerce")

    df = io.load_clean()
    out = (config.FIGS / "numeric_by_school")
    out.mkdir(parents=True, exist_ok=True)

    # --- restrict schools by min_n (and optional top_n by size) ---
    if "respondent_id" in df.columns and min_n:
        n = df.groupby("学校名_canon")["respondent_id"].nunique()
        keep = n[n >= min_n].index
        df = df[df["学校名_canon"].isin(keep)]
        if top_n:
            keep2 = (
                df.groupby("学校名_canon")["respondent_id"]
                  .nunique().sort_values(ascending=False)
                  .head(top_n).index
            )
            df = df[df["学校名_canon"].isin(keep2)]

    # -------- a) LEAF months: mean per school → horizontal bar --------
    month_cols = [c for c in df.columns if c.endswith("_months_total")]
    for c in month_cols:
        # already numeric from pipeline; still coerce defensively
        tmp = df[["学校名_canon", c]].copy()
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        g = tmp.dropna(subset=[c]).groupby("学校名_canon")[c].mean().sort_values()
        if g.empty:
            continue
        plots.bar_horizontal(
            g,
            title="LEAF利用月数（学校別・平均）",
            xlabel="平均月数",
            ylabel="学校名",
            outpath=out / f"{_safe_fn(c)}__学校別_bar_mean.png",
        )

    # -------- b) Work-time Q32–36: mean hours per school → horizontal bar --------
    # Match long headers (either “…1週間あたり … 時間 (※半角数字…)” or just contains “1週間あたり”)
    pat_end  = re.compile(r"時間\s*\(※半角数字で入力してください\)\s*$")
    pat_week = re.compile(r"1週間あたり")
    exclude_likert = "授業準備時間が短縮した"  # Likert, not numeric hours

    candidates = [c for c in df.columns if (pat_end.search(c) or pat_week.search(c))]
    work_cols = [c for c in candidates if exclude_likert not in c]

    # Nice short display names
    title_alias = {
        "児童生徒の学力を把握する時間": "学力把握時間",
        "授業に関するプリント作成する時間": "プリント作成時間",
        "提出物を回収・確認する時間": "提出物確認時間",
        "課外に児童生徒を個別指導する時間": "個別指導時間",
        "残業時間": "残業時間",
    }

    for full in work_cols:
        # Extract the stem before "1週間あたり…"
        stem = re.split(r"\s{0,}1週間あたり", full)[0].strip()
        short = title_alias.get(stem, stem)

        tmp = df[["学校名_canon", full]].copy()
        tmp[full] = _coerce_numeric_series(tmp[full])
        sub = tmp.dropna(subset=[full])
        if sub.empty:
            continue

        g = sub.groupby("学校名_canon")[full].mean().sort_values()
        plots.bar_horizontal(
            g.rename(short),
            title=f"{short}（学校別・平均時間）",
            xlabel="平均時間（時間）",
            ylabel="学校名",
            outpath=out / f"Q_{_safe_fn(short)}__学校別_bar_mean.png",
        )


# --- Numeric histograms (overall + 学校種別) ---
def run_numeric_histograms(bins: int = 12):
    df = io.load_clean()
    out = config.FIGS / "numeric"
    out.mkdir(parents=True, exist_ok=True)

    month_cols = [c for c in df.columns if c.endswith("_months_total")]
    for c in month_cols:
        tidy_all = prepare.numeric_by_facet(df, value_col=c, facet_col="学校種")
        col_label = "月数"
        tidy_all = tidy_all.rename(columns={c: col_label})

        # overall
        plots.histogram_numeric(
            tidy_all[[col_label]].dropna(),
            value_col=col_label,
            title="LEAF利用月数（全体）",
            outpath=out / f"{_safe_fn(c)}__hist_overall.png",
            bins=bins,
        )

        # by 学校種 (one PNG per 学校種)
        plots.histogram_numeric(
            tidy_all,
            value_col=col_label,
            title="LEAF利用月数（学校種別）",
            outpath=out / f"{_safe_fn(c)}__hist_学校種.png",
            by="学校種",
            bins=bins,
        )
