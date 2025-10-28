# text_utils.py
import unicodedata as ud
import pandas as pd
import re

def z2h(s):
    if s is pd.NA or s is None:
        return s
    s = str(s)
    s = ud.normalize("NFKC", s).strip()
    return s if s != "" else pd.NA

def clean_text(s):
    return z2h(s)

def normalize_digits(s):
    if s is pd.NA or s is None:
        return s
    s = ud.normalize("NFKC", str(s))
    return s

# NEW: use regex to split on common delimiters, now including ';' and '；'
_SPLIT_PAT = re.compile(r"(?:\s*(?:、|,|/|／|;|；|\band\b)\s*)+")

def split_multi(s):
    if s is pd.NA or s is None:
        return []
    s = ud.normalize("NFKC", str(s)).strip()
    if not s:
        return []
    parts = [p.strip() for p in _SPLIT_PAT.split(s) if p and not _SPLIT_PAT.fullmatch(p)]
    return [p for p in parts if p]
