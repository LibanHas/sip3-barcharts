# multi_select.py
import pandas as pd
from .text_utils import split_multi, z2h

def explode_multiselect(df: pd.DataFrame, multicolumns: list[str],
                        attach_facets=("学校種_canon", "学校名_canon")) -> pd.DataFrame:
    rows = []
    base_cols = list(attach_facets) + ["respondent_id"]
    for _, r in df.iterrows():
        base = {k: r.get(k, pd.NA) for k in base_cols}
        for col in multicolumns:
            val = r.get(col, pd.NA)
            for choice in split_multi(val):
                # defensive: if a choice still contains delimiters, split again
                for c2 in split_multi(choice):
                    c2 = z2h(c2)
                    if c2 is not pd.NA and c2 != "":
                        rows.append({**base, "column": col, "choice": c2})

    long_df = pd.DataFrame(rows, columns=base_cols + ["column", "choice"])
    # Clean & dedupe
    long_df = long_df.dropna(subset=["choice"])
    long_df["choice"] = long_df["choice"].astype(str).str.strip()
    long_df = long_df[long_df["choice"] != ""]
    long_df = long_df.drop_duplicates(subset=["respondent_id", "column", "choice"])
    return long_df

