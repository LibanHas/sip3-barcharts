
"""
make_students_clean.py
-----------------------
Cleans and normalizes the student survey dataset.
Outputs:
    data/students_clean.csv
    data/students_multi_long.csv
"""

import re, unicodedata as ud, os, pathlib
import numpy as np
import pandas as pd

# --- paths ---
BASE = pathlib.Path(".")
RAW_CSV  = BASE / "data/students_raw.csv"
ALIASES  = BASE / "config/alias_map_students.csv"
OUT_CSV  = BASE / "data/students_clean.csv"
OUT_MULTI = BASE / "data/students_multi_long.csv"
os.makedirs(BASE / "data", exist_ok=True)

# --- utils ---
def z2h(s):
    return s if pd.isna(s) else ud.normalize("NFKC", str(s)).strip()

def clean_text(s):
    if not isinstance(s, str):
        return s
    s = z2h(s)
    s = re.sub(r"[。、．.]", "", s)
    s = re.sub(r"\s+", "", s)
    return s

# --- read ---
df = pd.read_csv(RAW_CSV)
print(f"Loaded {len(df)} rows × {len(df.columns)} columns from {RAW_CSV.name}")

# --- normalize column headers ---
df.columns = [z2h(c) for c in df.columns]

# --- drop fully empty rows ---
df = df.dropna(how="all").copy()

# --- detect key columns (school, grade) ---
col_school = next((c for c in df.columns if "学校" in c), None)
col_grade  = next((c for c in df.columns if "学年" in c), None)
print(f"Detected columns → school: {col_school}, grade: {col_grade}")

# --- apply alias map if present ---
if ALIASES.exists():
    am = pd.read_csv(ALIASES)
    alias = {}
    for _, r in am.iterrows():
        alias.setdefault(r["column"], {})[z2h(r["raw"])] = z2h(r["canonical"])
    for col, amap in alias.items():
        if col in df.columns:
            df[col] = df[col].map(lambda s: amap.get(z2h(s), s) if isinstance(s,str) else s)
    print("Applied alias mapping.")

# --- normalize school & grade text ---
if col_school:
    df[col_school] = df[col_school].map(clean_text)
if col_grade:
    df[col_grade] = df[col_grade].map(clean_text)

# --- handle 未回答／無回答／nan variants ---
df = df.replace(
    to_replace=r"^(未回答|無回答|なし|NaN|null|None)$", value=np.nan, regex=True
)

# --- split multi-select questions ---
multi_cols = [c for c in df.columns if any(k in c for k in ["（複数選択", "（複数回答", "複数選択", "複数回答"])]
if multi_cols:
    long_rows = []
    for col in multi_cols:
        tmp = df[[col]].dropna()
        for i, val in tmp[col].items():
            for choice in re.split(r"[;,、・/]", str(val)):
                choice = z2h(choice.strip())
                if choice:
                    long_rows.append({"id": i, "column": col, "choice": choice})
    df_multi = pd.DataFrame(long_rows)
    df_multi.to_csv(OUT_MULTI, index=False, encoding="utf-8")
    print(f"✅ wrote {OUT_MULTI} ({len(df_multi)} rows)")
else:
    print("No multi-select columns detected.")

# --- write cleaned single-response version ---
df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"✅ wrote {OUT_CSV} ({len(df)} rows)")

# --- optional: expand short school suffixes & parse LEAF months ---
def expand_suffixes(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"(?<!学)中$", "中学校", s)
    s = re.sub(r"(?<!学)小$", "小学校", s)
    s = re.sub(r"(?<!等)高$", "高等学校", s)
    return s

if col_school:
    df[col_school] = df[col_school].map(expand_suffixes)

# parse "LEAFシステム…何か月くらい利用していますか" -> months_total
leaf_cols = [c for c in df.columns if "LEAFシステム" in c and "何か月" in c]
def parse_months(v):
    if pd.isna(v): return np.nan
    s = z2h(str(v)).strip()
    if s in {"0", "０"}: return 0
    # formats: "Y/M"  (years/months)  or plain months like "10"
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return y*12 + mo
    # plain integer fallback
    if re.match(r"^\d+$", s):
        return int(s)
    return np.nan

for c in leaf_cols:
    df[c + "_months_total"] = df[c].map(parse_months)

# --- quick preview ---
print("\nColumn samples:")
for c in list(df.columns)[:10]:
    print(" ", c, "→", df[c].dropna().unique()[:5])

