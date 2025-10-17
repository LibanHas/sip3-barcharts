import re, unicodedata as ud
import numpy as np
import pandas as pd
from .config import MAX_MONTHS

def _z2h(s): return ud.normalize("NFKC", str(s)).strip() if isinstance(s, str) else s

def parse_months(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return np.nan
    s = _z2h(v)
    if s in {"", "NaN", "nan"}: return np.nan
    if s in {"0", "０", "0/0"}: return 0
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", s)  # X/Y
    if m: return int(m.group(1))*12 + int(m.group(2))
    m = re.match(r"^(\d+)年(?:(\d+)[かヶケｹ]?月)?$", s)  # 2年 / 2年3か月
    if m: return int(m.group(1))*12 + int(m.group(2) or 0)
    if re.match(r"^\d+$", s):  # bare integer = years (teacher rule)
        return int(s)*12
    return np.nan

def parse_months_series(series: pd.Series) -> pd.Series:
    out = series.map(parse_months)
    out = out.mask((out < 0) | (out > MAX_MONTHS))
    return out
