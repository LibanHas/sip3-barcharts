# id_assign.py
import hashlib
import pandas as pd

def _row_hash(row: pd.Series, cols: list[str]) -> str:
    concat = "||".join(str(row[c]) for c in cols)
    return hashlib.sha1(concat.encode("utf-8")).hexdigest()[:12]

def make_ids(df: pd.DataFrame, strategy="row_index", cols_for_hash=None) -> pd.DataFrame:
    out = df.copy()
    if strategy == "row_index":
        out["respondent_id"] = range(1, len(out) + 1)
    elif strategy == "hash":
        if not cols_for_hash:
            cols_for_hash = [c for c in out.columns if c != "respondent_id"]
        out["respondent_id"] = out.apply(lambda r: _row_hash(r, cols_for_hash), axis=1)
    else:
        raise ValueError("Unknown strategy")
    return out
