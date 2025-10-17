import re, unicodedata as ud
import pandas as pd

def _z2h(s): return ud.normalize("NFKC", str(s)).strip() if isinstance(s, str) else s

def expand_suffixes(name: str) -> str:
    if not isinstance(name, str): return name
    name = re.sub(r"(?<!学)中$", "中学校", name)
    name = re.sub(r"(?<!学)小$", "小学校", name)
    name = re.sub(r"(?<!等)高$", "高等学校", name)
    return name

def canonicalize_names(series: pd.Series) -> pd.Series:
    def canon(s):
        if not isinstance(s, str) or s.strip() == "":
            return "未記入"
        s = _z2h(s)
        s = re.sub(r"[。、．.\s]+", "", s)
        s = s.replace("高校", "高等学校")
        s = expand_suffixes(s)

        # 西京 → 統一
        if re.search(r"京都市立西京.*附属中学校", s):
            return "京都市立西京高等学校附属中学校"
        # 洗足: 中/高は区別
        if re.search(r"洗足学園.*中学高等学校", s):
            return "私立洗足学園高等学校"
        if re.search(r"洗足学園.*中学校", s):
            return "私立洗足学園中学校"

        # 学校っぽくない
        if ("学校" not in s) or len(s) <= 3:
            return "未記入"
        return s

    out = series.map(canon).astype(str)

    # Heuristics for personal names — but NEVER touch the placeholder
    mask_kanji_only = out.str.match(r"^[一-龥]{2,3}$", na=False)
    mask_name_like  = out.str.match(r"^[一-龥]{1,3}[ぁ-んァ-ヶー]+$", na=False)
    mask = (mask_kanji_only | mask_name_like) & (out != "未記入")
    out = out.mask(mask, "未記入")
    return out




def normalize_school_type(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.replace({r"(?i)^\s*(na|nan|null|none)\s*$": None}, regex=True)
    s = s.str.replace("小学校", "小", regex=False)\
         .str.replace("中学校", "中", regex=False)\
         .str.replace("高等学校", "高", regex=False)
    s = s.where(~s.isna(), None)
    return s


def infer_school_type(name_series: pd.Series) -> pd.Series:
    def infer(n):
        if not isinstance(n, str): return None
        if "小学校" in n: return "小"
        if "中学校" in n: return "中"
        if "高等学校" in n: return "高"
        return None
    return name_series.map(infer)
