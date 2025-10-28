# normalize_fields.py
import re
import pandas as pd
from .text_utils import z2h
from .parsers import parse_duration_xy, parse_numeric, normalize_grade

KNOWN_MISSING = {"無回答", "未記入", "未入力", "N/A", "NA", "na", "nan", ""}

def _as_na(s):
    if s is pd.NA or s is None:
        return pd.NA
    s = z2h(s)  # NFKC + strip; returns pd.NA for empty after strip
    return pd.NA if (s is pd.NA or s in KNOWN_MISSING) else s

def coerce_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1) Standardize NA tokens (column-wise to avoid applymap warning)
    for c in out.columns:
        out[c] = out[c].map(_as_na)

    # 1c) 学校種 → 学校種_canon  (小学校 / 中学校 / 高等学校)
    school_type_cols = [c for c in out.columns if "学校種" in c]
    def _norm_school_type(s):
        if s is pd.NA or s is None:
            return pd.NA
        t = z2h(s)
        if t is pd.NA:
            return pd.NA
        t = t.replace(" ", "").replace("　", "")
        # common variants
        if any(k in t for k in ("小学校", "小学", "小")):
            return "小学校"
        if any(k in t for k in ("中学校", "中学", "中")):
            return "中学校"
        if any(k in t for k in ("高等学校", "高校", "高等", "高")):
            return "高等学校"
        return t  # leave as-is if unknown
    if school_type_cols:
        out["学校種_canon"] = out[school_type_cols[0]].map(_norm_school_type)
    else:
        out["学校種_canon"] = pd.NA

    # 1d) 学校名 → 学校名_canon  (trim/unify)
    school_name_cols = [c for c in out.columns if "学校名" in c]
    def _norm_school_name(s):
        if s is pd.NA or s is None:
            return pd.NA
        t = z2h(s)
        if t is pd.NA:
            return pd.NA
        t = t.strip()
        # collapse all whitespace
        t = re.sub(r"\s+", "", t)
        # drop common ownership prefixes
        t = re.sub(r"^(市立|県立|府立|都立|区立|町立|村立|私立)", "", t)
        # unify suffix variants
        t = t.replace("高校", "高等学校")
        t = t.replace("中学", "中学校")
        t = t.replace("小學校", "小学校")
        return t if len(t) >= 2 else pd.NA
    if school_name_cols:
        out["学校名_canon"] = out[school_name_cols[0]].map(_norm_school_name)
    else:
        out["学校名_canon"] = pd.NA

    # 1b) Q6: Normalize per-token **inside** the cell (multi-select string)
    #     Header like: 「LEAFシステム…どの教科で使用しますか（複数選択可）」
    q6_cols = [c for c in out.columns if ("どの教科で使用しますか" in c) or ("どの教科で使用" in c)]
    if q6_cols:
        q6 = q6_cols[0]

        # Delimiters to split multi-select cells (do NOT split on '・')
        _split_pat = re.compile(r"[、,;/／；]+")

        SUBJECT_CHOICES = {"国語", "算数・数学", "理科", "社会", "外国語", "美術", "その他"}
        NON_SUBJECT_STRINGS = {"未使用", "なし", "無"}
        NON_SUBJECT_HINTS = ("使って", "使用", "しない", "してません", "やってません")

        def _canon_subject_choice(tok: str) -> str:
            s = z2h(tok) or ""
            if s in SUBJECT_CHOICES:
                return s
            if s in NON_SUBJECT_STRINGS or any(h in s for h in NON_SUBJECT_HINTS):
                return "未使用"
            if s.endswith("など") and s.startswith("美術"):
                return "美術"
            # Anything else becomes "その他" to avoid noisy bars
            return "その他"

        def _normalize_q6_cell(val):
            if val is None or val is pd.NA:
                return pd.NA
            s = z2h(val)
            if s is pd.NA:
                return pd.NA
            parts = [p.strip() for p in _split_pat.split(s) if p and p.strip() != ""]
            if not parts:
                return pd.NA
            canon, seen = [], set()
            for p in parts:
                t = _canon_subject_choice(p)
                if t not in seen:
                    seen.add(t)
                    canon.append(t)
            return ";".join(canon) if canon else pd.NA

        out[q6] = out[q6].map(_normalize_q6_cell)

    # 2) 学年 → 学年_canon (e.g., 小5 / 中2 / 高1)
    grade_cols = [c for c in out.columns if "学年" in c]
    if grade_cols:
        out["学年_canon"] = out[grade_cols[0]].map(normalize_grade)
    else:
        out["学年_canon"] = pd.NA

    # 3) 利用期間: derive years/months/total
    duration_col_candidates = [c for c in out.columns if "利用期間" in c]
    if duration_col_candidates:
        dcol = duration_col_candidates[0]
        yrs, mos = [], []
        for v in out[dcol].astype("string"):
            y, m = parse_duration_xy(v, max_total_months=120, autocarry=True)
            yrs.append(y)
            mos.append(m)
        out["利用期間_years"] = yrs
        out["利用期間_months"] = mos
        out["利用期間_total_months"] = [
            (y if y is not pd.NA else 0) * 12 + (m if m is not pd.NA else 0)
            if (y is not pd.NA or m is not pd.NA) else pd.NA
            for y, m in zip(yrs, mos)
        ]

    # 4) (Optional) Coerce Likert-like numeric columns, if marked in headers
    likert_cols = [c for c in out.columns if "（1-5" in c or "(1-5" in c]
    for c in likert_cols:
        out[c] = out[c].map(lambda v: parse_numeric(v)).astype("Float64")

    return out
