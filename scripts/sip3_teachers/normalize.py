import re, unicodedata as ud
import pandas as pd
import numpy as np

def z2h(s):
    return s if pd.isna(s) else ud.normalize("NFKC", str(s)).strip()

def normalize_headers(df):
    df = df.copy()
    df.columns = [z2h(c) for c in df.columns]
    return df

def drop_all_empty_rows(df):
    return df.dropna(how="all").copy()

def blank_to_nan(df):
    import pandas as pd, numpy as np
    # (Option is fine to keep, but not necessary once we call infer_objects)
    pd.set_option("future.no_silent_downcasting", True)
    return (
        df.replace(
            {
                r"^\s*$": np.nan,
                r"(?i)^(未回答|無回答|なし|na|nan|null|none)$": np.nan,
            },
            regex=True,
        )
        .infer_objects(copy=False)   # <-- this actually removes the warning
    )



def clean_school_text(series):
    return (series.map(z2h)
                  .str.replace(r"[。、．.]", "", regex=True)
                  .str.replace(r"\s+", "", regex=True))

def clean_grade_text(series):
    # gentler clean for 学年 (keep tokens like 中1/高1)
    return series.map(z2h)
