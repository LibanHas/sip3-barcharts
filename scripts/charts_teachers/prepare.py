# scripts/charts_teachers/prepare.py
"""
Helpers to reshape teacher survey data for plotting.

- pct_table:   100% stacked tables for Likert/frequency (tidy with pct in 0–1)
- multi_counts: tidy counts + pct for multi-select questions by facet
- numeric_by_facet: tidy numeric data per facet (box/hist)
- attach_facets: add 学校種 / 学校名_canon to the long table
- filter_facets_by_min_n: keep only facets with enough respondents
- top_n_facets: pick top-N facets by respondent count
"""

from typing import Optional, Sequence
import pandas as pd
import numpy as np

from . import config


# ---------- 100% stacked (Likert / frequency) ----------
def pct_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    resp_col: str,
    order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return a tidy table with columns:
      [*group_cols, resp_col, n, denom, pct]
    where pct is in 0–1 (not 0–100).

    This matches plots.bar_100pct(...), which pivots internally.
    """
    cols = list(group_cols) + [resp_col]
    sub = df[cols].dropna().copy()

    # category order for response column (preserved in plotting)
    if order is not None:
        sub[resp_col] = pd.Categorical(sub[resp_col], categories=list(order), ordered=True)

    # count responses per group x category (unique teachers by respondent_id if present)
    if "respondent_id" in df.columns:
        ct = (df[list(group_cols) + ["respondent_id", resp_col]]
              .dropna(subset=list(group_cols) + [resp_col, "respondent_id"])
              .drop_duplicates(subset=["respondent_id"] + list(group_cols) + [resp_col])
              .groupby(list(group_cols) + [resp_col])["respondent_id"]
              .nunique()
              .rename("n")
              .reset_index())
        # denominator per group
        denom = (df[list(group_cols) + ["respondent_id"]]
                 .dropna(subset=list(group_cols) + ["respondent_id"])
                 .drop_duplicates(subset=["respondent_id"] + list(group_cols))
                 .groupby(list(group_cols))["respondent_id"]
                 .nunique()
                 .rename("denom")
                 .reset_index())
    else:
        # fallback: simple counts
        ct = sub.groupby(list(group_cols) + [resp_col]).size().rename("n").reset_index()
        denom = ct.groupby(list(group_cols))["n"].sum().rename("denom").reset_index()

    out = ct.merge(denom, on=list(group_cols), how="left")
    out["pct"] = out["n"] / out["denom"]
    return out


# ---------- multiselect aggregations ----------
# scripts/charts_teachers/prepare.py
# scripts/charts_teachers/prepare.py
def multi_counts(
    m_long: pd.DataFrame,
    question: str,
    facet_col: str = "学校種",
) -> pd.DataFrame:
    """
    Return TIDY data with columns:
      [facet_col, 'choice', 'count', 'pct']  where pct ∈ [0, 1].
    Works whether the long table uses 'column' or 'question' as the question field.
    """
    # accept both schemas: 'column' or 'question'
    qcol = "column" if "column" in m_long.columns else "question" if "question" in m_long.columns else None
    if qcol is None:
        raise KeyError("multi_counts expects a long table with a 'column' or 'question' field.")

    sub = m_long[m_long[qcol] == question].copy()
    if sub.empty:
        # return empty tidy frame with the right columns to avoid KeyErrors downstream
        return pd.DataFrame(columns=[facet_col, "choice", "count", "pct"])

    if facet_col not in sub.columns:
        sub[facet_col] = "All"

    # counts
    counts = (
        sub.groupby([facet_col, "choice"])
           .size()
           .reset_index(name="count")
    )

    # percentages within each facet
    counts["pct"] = counts["count"] / counts.groupby(facet_col)["count"].transform("sum")
    # stable order by choice text
    counts = counts.sort_values([facet_col, "choice"]).reset_index(drop=True)
    return counts


# ---------- numeric by facet ----------
def numeric_by_facet(
    df: pd.DataFrame,
    value_col: str,
    facet_col: str = "学校種",
) -> pd.DataFrame:
    """Return tidy numeric data for box/violin/hist per facet."""
    out = df[[facet_col, value_col]].copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[value_col])
    # optional facet order
    if facet_col == "学校種":
        try:
            out[facet_col] = pd.Categorical(out[facet_col], categories=config.SCHOOL_TYPES, ordered=True)
        except Exception:
            pass
    return out


# ---------- facet utilities ----------
def attach_facets(
    m_long: pd.DataFrame,
    df_clean: pd.DataFrame,
    facets: Sequence[str] = ("学校種", "学校名_canon"),
) -> pd.DataFrame:
    cols = ["respondent_id"] + [f for f in facets if f in df_clean.columns]
    merged = m_long.merge(df_clean[cols], on="respondent_id", how="left")

    # ---- NEW: ensure unknown school shows as "不明" (not NaN/blank)
    if "学校名_canon" in merged.columns:
        merged["学校名_canon"] = (
            merged["学校名_canon"]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            .fillna("不明")
        )

    return merged



def filter_facets_by_min_n(
    df_long: pd.DataFrame,
    facet_col: str,
    min_n: int,
) -> pd.DataFrame:
    """Keep only facets with at least `min_n` unique respondents."""
    if "respondent_id" not in df_long.columns or facet_col not in df_long.columns:
        return df_long
    n_by = (df_long.dropna(subset=[facet_col, "respondent_id"])
                     .groupby(facet_col)["respondent_id"].nunique())
    keep = n_by[n_by >= min_n].index
    return df_long[df_long[facet_col].isin(keep)].copy()


def top_n_facets(
    df_long: pd.DataFrame,
    facet_col: str,
    n: int = 10,
) -> pd.Index:
    """Return labels of the top-N facets by respondent count."""
    if "respondent_id" in df_long.columns:
        counts = (df_long.dropna(subset=[facet_col, "respondent_id"])
                          .groupby(facet_col)["respondent_id"].nunique()
                          .sort_values(ascending=False))
        return counts.head(n).index
    return df_long[facet_col].value_counts().head(n).index
