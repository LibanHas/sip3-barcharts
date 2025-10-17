import pandas as pd

# Map small typos/synonyms into a canonical set
LIKERT_MAP = {
    "あてはまる":"あてはまる",
    "少しあてはまる":"少しあてはまる",
    "あまりあてはまらない":"あまりあてはまらない",
    "あてはまらない":"あてはまらない",
    # usage-synonyms
    "頻繁に使用している":"頻繁に使用している",
    "使用することがある":"使用することがある",
    "あまり使用しない":"あまり使用しない",
    "全く使用しない":"全く使用しない",
    "使っていない":"全く使用しない",
}

def standardize(series: pd.Series) -> pd.Series:
    return series.map(lambda s: LIKERT_MAP.get(str(s), s) if pd.notna(s) else s)
