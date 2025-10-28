# scripts/sip3_students/parsers.py
import re
import pandas as pd
from .text_utils import z2h

# --------------------------- Numeric -----------------------------------------
def parse_numeric(s, allow_percent: bool = False):
    """
    Extract a number from a messy string.
    Examples: ' ５（よく）' -> 5, '12.5%', '１２' -> 12.5 / 12
    Returns pd.NA if nothing numeric is found.
    """
    if s is None or s is pd.NA:
        return pd.NA
    s = z2h(s)
    if s is pd.NA:
        return pd.NA
    if allow_percent:
        s = s.replace("%", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else pd.NA


# --------------------------- Duration X/Y ------------------------------------
def parse_duration_xy(s, max_total_months=None, autocarry: bool = True):
    """
    Parse durations written as 'X/Y', '2年3か月', '半年', '1' (years), etc.
    Returns (years:int, months:int) with optional auto-carry for months>=12.
    If max_total_months is set and exceeded, returns (NA, NA).
    """
    if s is pd.NA or s is None:
        return (pd.NA, pd.NA)
    s = z2h(s)
    if s is pd.NA:
        return (pd.NA, pd.NA)

    # Explicit unused
    if s == "0/0":
        y, m = 0, 0
    else:
        # Direct X/Y form
        mxy = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", s)
        if mxy:
            y, m = int(mxy.group(1)), int(mxy.group(2))
        else:
            # Japanese years/months
            y_m = re.search(r"(\d+)\s*年", s)
            m_m = re.search(r"(\d+)\s*(?:か月|ヶ月|月)", s)
            if y_m or m_m:
                y = int(y_m.group(1)) if y_m else 0
                m = int(m_m.group(1)) if m_m else 0
            elif "半年" in s:
                y, m = 0, 6
            elif re.match(r"^\d+$", s):  # lone integer → years
                y, m = int(s), 0
            else:
                return (pd.NA, pd.NA)

    # Auto-carry months into years
    if autocarry and isinstance(m, int) and m >= 12:
        y = (y or 0) + m // 12
        m = m % 12

    total = (y or 0) * 12 + (m or 0)
    if max_total_months is not None and total > max_total_months:
        return (pd.NA, pd.NA)

    return (y, m)


# --------------------------- Grade normalizer --------------------------------
_GRADE_TAGS = {
    "小": range(1, 7),  # 小1–小6
    "中": range(1, 4),  # 中1–中3
    "高": range(1, 4),  # 高1–高3
}

def normalize_grade(raw):
    """
    Normalize grade labels into compact tags like: 小1, 小5, 中3, 高1.
    Accepts: 小学校5年生 / 小5 / 小５ / 小学5年 / 中学校1年生 / 中1 / 高等学校3年生 / 高3 / 高３
    Returns pd.NA when not confidently parsed or out of expected range.
    """
    if raw is None or raw is pd.NA:
        return pd.NA

    s = z2h(raw)  # NFKC + trim (also converts full-width digits)
    if s is pd.NA:
        return pd.NA

    # Detect level
    level = "小" if "小" in s else ("中" if "中" in s else ("高" if "高" in s else None))

    # First integer found after normalization
    m = re.search(r"(\d+)", s)
    num = int(m.group(1)) if m else None

    if level and num:
        if num in _GRADE_TAGS.get(level, ()):
            return f"{level}{num}"
        return pd.NA  # out of range (e.g., 小7, 中4)
    return pd.NA
