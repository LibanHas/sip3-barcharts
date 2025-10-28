# make_students_clean.py
import pandas as pd
import numpy as np
import re
import unicodedata as ud
from typing import Optional

from .io_students import load_raw, load_alias_map, save_clean
from ..common.canonicalize import apply_aliases
from .id_assign import make_ids
from .normalize_fields import coerce_and_derive
from .validators import validate


# -----------------------------
# Small helper: pandas >=2.1 uses DataFrame.map over applymap
# -----------------------------
def _df_map(df, fn):
    try:
        return df.map(fn)          # pandas >= 2.1
    except AttributeError:
        return df.applymap(fn)     # older pandas


# -----------------------------
# Step 1: width + whitespace normalization
# -----------------------------
_WHITESPACE_RE = re.compile(r"[\u0009-\u000D\u0085\u00A0\u1680\u180E\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]+")

def _to_str_or_na(x):
    return x if pd.isna(x) else str(x)

def _normalize_width_and_space(s):
    if pd.isna(s):
        return s
    s = ud.normalize("NFKC", str(s))
    s = _WHITESPACE_RE.sub(" ", s)
    s = s.strip()
    return pd.NA if s == "" else s


# -----------------------------
# Step 2: 附属 → 付属
# -----------------------------
_FUZOKU_RE = re.compile(r"附属")

def _standardize_fuzoku(s):
    if pd.isna(s):
        return s
    return _FUZOKU_RE.sub("付属", str(s))


# -----------------------------
# Step 3: remove decorative punctuation (・, 。)
# -----------------------------
_MID_DOT_AND_MARU_RE = re.compile(r"[・。]")

def _strip_decorations(s):
    if pd.isna(s):
        return s
    s = str(s)
    s = _MID_DOT_AND_MARU_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return pd.NA if s == "" else s


# -----------------------------
# Step 4: standardize level tokens + collapse accidental "学校校"
# -----------------------------
_RE_HIGHSCHOOL = re.compile(r"(?<!等)高校")   # '高校' not preceded by '等'
_RE_JUNIOR     = re.compile(r"中学(?!校)")    # '中学' not followed by '校'

def _standardize_school_levels(s):
    if pd.isna(s):
        return s
    s = str(s)
    s = _RE_HIGHSCHOOL.sub("高等学校", s)
    s = _RE_JUNIOR.sub("中学校", s)
    return s

# fix accidental duplication: e.g., 西京中学校校 -> 西京中学校
_DUP_SCHOOL_RE = re.compile(r"(学校)(?:校)+")
def _collapse_school_dup(s):
    if pd.isna(s):
        return s
    return _DUP_SCHOOL_RE.sub(r"\1", str(s))


# -----------------------------
# Step 5: normalize combined JHS/HS order + flag
# -----------------------------
def _normalize_combined_levels_and_flag(s):
    """
    If both '中学校' and '高等学校' appear (and no '附属'), normalize order to '中学校高等学校'.
    Return (normalized_str, is_combined_bool).
    """
    if pd.isna(s):
        return s, pd.NA
    s = str(s)
    has_j = "中学校" in s
    has_h = "高等学校" in s
    if has_j and has_h and ("附属" not in s):
        s = s.replace("高等学校中学校", "中学校高等学校")
        if re.search(r"高等学校.*中学校", s) and not re.search(r"中学校.*高等学校", s):
            s = re.sub(r"高等学校(.*)中学校", r"中学校\1高等学校", s)
        return s, True
    return s, False


# -----------------------------
# Step 6: remove owner/establishment prefixes at start
# -----------------------------
_PREFIXES = [
    "市立","県立","府立","都立","道立","区立","町立","村立",
    "私立","国立","公立",
    # common long forms (optional, safe)
    "京都市立","京都府立","大阪市立","大阪府立","兵庫県立","奈良県立","滋賀県立",
]
_PREFIX_RE = re.compile(rf"^(?:{'|'.join(map(re.escape, _PREFIXES))})+")

def _strip_owner_prefixes(s):
    if pd.isna(s):
        return s
    s = str(s)
    prev = None
    while prev != s:
        prev = s
        s = _PREFIX_RE.sub("", s)
    s = s.strip()
    return pd.NA if s == "" else s

# (optional) also strip owner tokens that appear right after a leading geo prefix
_GEO_SUFFIX = r"(?:都|道|府|県|市|区|郡|町|村)"
_KANJI = r"[\u4E00-\u9FFF]{1,10}"

_OWNER_AFTER_GEO_RE = re.compile(
    rf"^((?:{_KANJI}{_GEO_SUFFIX})+)"
    rf"(?:市立|県立|府立|都立|道立|区立|町立|村立|私立|国立|公立)+"
)
def _strip_owner_after_geo(s):
    if pd.isna(s):
        return s
    t = str(s)
    return _OWNER_AFTER_GEO_RE.sub(r"\1", t).strip()


# -----------------------------
# Step 7: geo extraction + safe canonical rebuild
# -----------------------------
_GEO_PREFIX_RE = re.compile(rf"^(?P<geo>(?:{_KANJI}{_GEO_SUFFIX}\s*)+)?(?P<rest>.*)$")
_SCHOOL_KEYWORDS_RE = re.compile(r"(小学校|中学校|高等学校|高校)")
_GENERIC_LEVEL_RE   = re.compile(r"^(小学校|中学校|高等学校|高校)$")
_NUMBERED_ONLY_RE   = re.compile(r"^第[一二三四五六七八九十百千\d]+(小|中|高)学校$")

def _dedupe_consecutive(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out

def _extract_geo_and_rebuild(s):
    """
    Return (canonical_name, geo_prefix_str).
    If remainder is generic/numbered-only, prepend last locality token(s) (市/区/町/村).
    """
    if pd.isna(s):
        return s, pd.NA
    s = str(s).strip()
    if not _SCHOOL_KEYWORDS_RE.search(s):
        return s, pd.NA

    m = _GEO_PREFIX_RE.match(s)
    if not m:
        return s, pd.NA

    geo = (m.group("geo") or "").strip()
    rest = (m.group("rest") or "").strip()
    if not geo:
        return s, pd.NA

    token_re = re.compile(rf"{_KANJI}{_GEO_SUFFIX}")
    tokens = _dedupe_consecutive(token_re.findall(geo))

    if _GENERIC_LEVEL_RE.fullmatch(rest) or _NUMBERED_ONLY_RE.fullmatch(rest):
        keep = []
        for t in reversed(tokens):
            keep.append(t)
            if t.endswith(("市","区","町","村")) or len(keep) >= 2:
                break
        keep.reverse()
        canon = "".join(keep) + rest
    else:
        canon = rest

    return canon, "".join(tokens)


# -----------------------------
# Step 8: prefer 学校種 over 学年 for level inference
# -----------------------------
_LEVEL_TOKEN_RE = re.compile(r"(小学校|中学校|高等学校|高校)")

def _has_explicit_level(name: str) -> bool:
    if pd.isna(name):
        return False
    return bool(_LEVEL_TOKEN_RE.search(str(name)))

_AMBIGUOUS_BASES = {
    "西京": {"JHS": "西京高等学校付属中学校", "HS": "西京高等学校"},
}

_RE_JHS = re.compile(r"(中\s*([1１]|一)|中\s*([2２]|二)|中\s*([3３]|三)|中学)", re.IGNORECASE)
_RE_HS  = re.compile(r"(高\s*([1１]|一)|高\s*([2２]|二)|高\s*([3３]|三)|高等学校|高校)", re.IGNORECASE)
_RE_ES  = re.compile(r"(小\s*([1-6１-６一二三四五六])|小学校)", re.IGNORECASE)

def _canon_school_type_to_level(text: str) -> Optional[str]:
    if not text:
        return None
    t = re.sub(r"\s+", "", str(text))
    # Japanese labels
    if any(k in t for k in ["小学校", "小学", "小等", "小"]):
        return "ES"
    if any(k in t for k in ["中学校", "中学", "中等", "中"]):
        return "JHS"
    if any(k in t for k in ["高等学校", "高校", "高等", "高"]):
        return "HS"
    # English fallbacks
    tl = t.lower()
    if "elementary" in tl or "primary" in tl:
        return "ES"
    if "junior" in tl or "middle" in tl:
        return "JHS"
    if "senior" in tl or "high" in tl:
        return "HS"
    return None

_SCHOOL_TYPE_COLS = (
    "あなたの学校種を教えてください",
    "学校種_canon", "学校種", "school_type", "設置校種", "学校の種類"
)

_BARE_YEAR_RE = re.compile(r"^\s*([1-6１-６一二三四五六])\s*年(生)?\s*$")
def _is_bare_year_only(text: str) -> bool:
    return bool(text) and bool(_BARE_YEAR_RE.match(str(text).strip()))

def _infer_level_from_school_type(row) -> Optional[str]:
    for c in row.index:
        cs = str(c)
        if (cs in _SCHOOL_TYPE_COLS) or ("学校種" in cs) or ("school_type" in cs.lower()):
            v = row.get(c, None)
            if pd.isna(v):
                continue
            lvl = _canon_school_type_to_level(v)
            if lvl:
                return lvl
    return None

def _infer_level(row) -> Optional[str]:
    """
    Priority:
      1) 学校種 column
      2) 学年 with explicit 小/中/高 tokens
      3) 学年 is bare '1年' etc. -> try 学校種 again
    """
    st = _infer_level_from_school_type(row)
    if st:
        return st

    grade_cols = [c for c in row.index if c in ("学年","grade","学年名","在学学年","学年（在学）")]
    if not grade_cols:
        grade_cols = [c for c in row.index if ("学年" in str(c) or "grade" in str(c).lower())]
    gtext = ""
    for c in grade_cols:
        v = row.get(c, None)
        if not pd.isna(v):
            gtext = str(v).replace("年生","年")
            break
    if gtext:
        if _RE_HS.search(gtext):  return "HS"
        if _RE_JHS.search(gtext): return "JHS"
        if _RE_ES.search(gtext):  return "ES"
        if _is_bare_year_only(gtext):
            st2 = _infer_level_from_school_type(row)
            if st2:
                return st2
    return None

def _infer_level_from_grade(row) -> Optional[str]:
    """Kept for compatibility elsewhere (not used in Step 8 anymore)."""
    cands = [c for c in row.index if c in ("学年","grade","学年名","在学学年","学年（在学）")]
    if not cands:
        cands = [c for c in row.index if ("学年" in str(c) or "grade" in str(c).lower())]
    text = ""
    for c in cands:
        v = row.get(c, None)
        if not pd.isna(v):
            text = str(v); break
    if not text: return None
    t = text.replace("年生","年")
    if _RE_HS.search(t):  return "HS"
    if _RE_JHS.search(t): return "JHS"
    if _RE_ES.search(t):  return "ES"
    return None

def _resolve_level_if_missing(name: str, level_hint: Optional[str]):
    """
    If name lacks 小/中/高等学校 and we have a level hint:
      - Use _AMBIGUOUS_BASES for exact base matches (e.g., 「西京」)
      - Else keep as-is and flag for manual
    """
    if pd.isna(name):
        return name, False, False
    s = str(name).strip()
    if _has_explicit_level(s):
        return s, False, False
    if not level_hint:
        return s, False, True
    if s in _AMBIGUOUS_BASES and level_hint in ("JHS","HS"):
        return _AMBIGUOUS_BASES[s][level_hint], True, False
    return s, False, True


# -----------------------------
# Saikyo-specific corrections (before alias map)
# -----------------------------
_SAIKYO_RE = re.compile(r"西京")
_HAS_JHS_RE = re.compile(r"中学校")
_HAS_HS_RE  = re.compile(r"(高等学校|高校)")

def _saikyo_specific_corrections(school_name: str) -> str:
    if pd.isna(school_name):
        return "不明"
    
    school_name = str(school_name)
    if school_name == "西京":
        return "西京高等学校"
    if "付属中学校" in school_name and "高等学校" not in school_name:
        return "西京高等学校付属中学校"
    return school_name


# -----------------------------
# Force Canonical Overrides
# -----------------------------
_FORCE_CANON_OVERRIDES = {
    "洗足学園中学校高校": "洗足学園中学校",
    "立岩沼小学校": "岩沼小学校",
    "天塩高等学校": "北海道天塩高等学校",
    "西賀茂中学校校": "西賀茂中学校",
}


# -----------------------------
# Clean School Names
# -----------------------------
def clean_school_names(df: pd.DataFrame) -> pd.DataFrame:
    # Apply alias map
    df["学校名_canon"] = df["学校名"].map(ALIAS_MAP).fillna(df["学校名"])

    # Apply specific corrections
    df["学校名_canon"] = df["学校名_canon"].apply(_saikyo_specific_corrections)

    # Apply force overrides
    df["学校名_canon"] = df["学校名_canon"].replace(_FORCE_CANON_OVERRIDES)

    # Post-alias cleanup
    df["学校名_canon"] = (
        df["学校名_canon"]
        .str.replace("学校校", "学校", regex=False)  # Remove duplicate tokens
        .str.strip()  # Remove leading/trailing whitespace
    )

    return df


# =============================
# Main
# =============================
def main():
    df = load_raw()
    alias = load_alias_map()

    # Step 1: width + whitespace normalization (all string-like cols)
    df = _df_map(df, _to_str_or_na)
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(str_cols) > 0:
        df[str_cols] = _df_map(df[str_cols], _normalize_width_and_space)

    # Step 2: 附属 → 付属 (broadly)
    if len(str_cols) > 0:
        df[str_cols] = _df_map(df[str_cols], _standardize_fuzoku)

    # Detect target column(s) for school name rules
    cand_cols = [c for c in df.columns if c in ("学校名", "school_name")]
    if not cand_cols:
        cand_cols = [c for c in df.columns if ("学校" in c and "名" in c)]

    # Step 3: punctuation/decoration cleanup
    for c in cand_cols:
        df[c] = df[c].map(_strip_decorations)

    # Step 4: level token standardization + collapse "学校校"
    for c in cand_cols:
        df[c] = df[c].map(_standardize_school_levels).map(_collapse_school_dup)

    # Step 5: normalize combined order + flag
    for c in cand_cols:
        tmp = df[c].map(_normalize_combined_levels_and_flag)
        df[c] = tmp.map(lambda t: t[0])
        df["is_combined_jhs_hs"] = tmp.map(lambda t: t[1])

    # Step 6: strip owner/establishment prefixes
    for c in cand_cols:
        df[c] = df[c].map(_strip_owner_prefixes)

    # Step 6b: strip owner tokens appearing right after a geo prefix
    for c in cand_cols:
        df[c] = df[c].map(_strip_owner_after_geo)

    # Step 7: geo extraction + safe canonical rebuild
    for c in cand_cols:
        tmp = df[c].map(_extract_geo_and_rebuild)
        df[c] = tmp.map(lambda t: t[0])                   # canonical school name (geo-safe)
        df["school_geo_prefix"] = tmp.map(lambda t: t[1]) # geo tokens kept separately

    # Coerce + derive (bring in normalized grade/type etc.)
    df = coerce_and_derive(df)

    # Step 8: if school name has no level token, infer (学校種優先) and resolve
    cand_cols = [c for c in df.columns if c in ("学校名", "school_name")]
    if not cand_cols:
        cand_cols = [c for c in df.columns if ("学校" in c and "名" in c)]
    if cand_cols:
        school_col = cand_cols[0]
        res = df.apply(lambda r: _resolve_level_if_missing(r[school_col], _infer_level(r)), axis=1)
        df[school_col]                = res.map(lambda t: t[0])
        df["level_filled_from_grade"] = res.map(lambda t: t[1])   # kept name for compatibility
        df["needs_manual_level"]      = res.map(lambda t: t[2])

    # Saikyo-specific corrections before alias map
    for c in cand_cols:
        df[c] = df[c].map(_saikyo_specific_corrections)

    # Canonicalize (use curated alias map)
    df = apply_aliases(df, alias)
    # Post-alias safety: collapse accidental 「学校校」 again
    cand_cols = [c for c in df.columns if c in ("学校名", "school_name")]
    if not cand_cols:
        cand_cols = [c for c in df.columns if ("学校" in c and "名" in c)]
    for c in cand_cols:
        df[c] = df[c].map(_collapse_school_dup)

    # Step 10: finalize missing/blank school names
    cand_cols = [c for c in df.columns if c in ("学校名", "school_name")]
    if not cand_cols:
        cand_cols = [c for c in df.columns if ("学校" in c and "名" in c)]
    if cand_cols:
        school_col = cand_cols[0]
        missing_before = df[school_col].isna() | (df[school_col].astype(str).str.strip() == "")
        n_missing = int(missing_before.sum())
        df.loc[missing_before, school_col] = "不明"
        print(f"[Step10 Missing] filled {n_missing} blank/missing school names with '不明'")

    # IDs
    df = make_ids(df, strategy="row_index")

    # Validate
    report = validate(df)
    print("Validation report:", report)

    # Save
    save_clean(df)


if __name__ == "__main__":
    main()
