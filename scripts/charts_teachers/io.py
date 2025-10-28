# scripts/charts_teachers/io.py
"""
IO helpers for teacher charts.

Usage:
    from charts_teachers import io
    df = io.load_clean()
    m  = io.load_multi()
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import config


def _read_csv(path: Path) -> pd.DataFrame:
    """CSV reader with consistent options + helpful error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "→ Did you run the cleaning step? Try:\n"
            "   python3 scripts/make_teachers_clean.py"
        )
    # Notes:
    # - keep_default_na=False prevents "NA"/"N/A" becoming NaN by accident.
    # - encoding='utf-8-sig' gracefully handles BOM if present.
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8-sig",
        low_memory=False,
    )


def _strip_all(df: pd.DataFrame) -> pd.DataFrame:
    """Trim surrounding whitespace from all string columns."""
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].str.strip()
    return df





def load_clean(columns: list[str] | None = None) -> pd.DataFrame:
    df = _read_csv(config.TEACHERS_CLEAN)
    df = _strip_all(df)

    # normalize ID → respondent_id
    if "respondent_id" not in df.columns:
        if "ID" in df.columns:
            df = df.rename(columns={"ID": "respondent_id"})
        else:
            raise KeyError(f"'respondent_id' (or 'ID') missing in {config.TEACHERS_CLEAN.name}")

    df["respondent_id"] = df["respondent_id"].astype(str).str.strip()
    for facet in ("学校種", "学校名_canon"):
        if facet in df.columns:
            df[facet] = df[facet].astype(str).str.strip()

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in {config.TEACHERS_CLEAN.name}: {missing}")
        df = df[columns]

    return df


def load_multi() -> pd.DataFrame:
    m = _read_csv(config.TEACHERS_MULTI)
    m = _strip_all(m)

    # normalize headers if needed
    if "respondent_id" not in m.columns and "ID" in m.columns:
        m = m.rename(columns={"ID": "respondent_id"})
    if "question" not in m.columns and "column" in m.columns:
        m = m.rename(columns={"column": "question"})

    expected = {"respondent_id", "question", "choice"}
    missing = expected - set(m.columns)
    if missing:
        raise KeyError(
            f"{config.TEACHERS_MULTI.name} is missing columns {sorted(missing)}. "
            "Check the multiselect explode step in the cleaning pipeline."
        )

    for c in ("respondent_id", "question", "choice"):
        m[c] = m[c].astype(str).str.strip()

    return m

