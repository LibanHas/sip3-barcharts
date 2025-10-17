import re
import pandas as pd

DELIMS = r"[;,、・/]"

def ensure_id(df):
    if "respondent_id" not in df.columns:
        df = df.copy()
        df.insert(0, "respondent_id", range(len(df)))
    return df

def explode(df, columns):
    df = ensure_id(df)
    rows = []
    for col in columns:
        nonnull = df[["respondent_id", col]].dropna()
        for rid, val in nonnull.itertuples(index=False):
            seen = set()
            for choice in re.split(DELIMS, str(val)):
                choice = choice.strip()
                if not choice: continue
                key = (rid, col, choice)
                if key in seen: continue
                seen.add(key)
                rows.append({"respondent_id": rid, "column": col, "choice": choice})
    return pd.DataFrame(rows)

def build_option_universe(df_multi: pd.DataFrame) -> dict:
    if df_multi.empty: return {}
    out = {}
    for col, g in df_multi.groupby("column"):
        out[col] = sorted(g["choice"].astype(str).unique().tolist())
    return out
