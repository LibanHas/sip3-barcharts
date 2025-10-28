# make_students_multi.py
import re
import sys
import pandas as pd
from typing import List, Iterable
from .paths import OUT_CSV
from .io_students import save_multi
from .multi_select import explode_multiselect

# ====== 1) Short keys for chart-friendly names ======
SHORT_KEYS = {
    "LEAFシステム(BookRoll,分析(ぶんせき)ツール)をどの教科で使用しますか(複数選択(せんたく)可)": "Q_subjects",
    "BookRollでよく使う機能を選んでください(複数選択(せんたく)可)": "Q_br_features",
    "分析ツール(ログパレ)でよく使う機能を選んでください(複数選択(せんたく)可)": "Q_lp_features",
}

def to_column_key(long_name: str) -> str:
    if long_name in SHORT_KEYS:
        return SHORT_KEYS[long_name]
    # Fallback: safe slug if something new appears
    slug = re.sub(r"\s+", "_", str(long_name))
    slug = re.sub(r"[^\w\-]+", "", slug)[:60]
    return slug or "Q_unknown"

# ====== 2) Optional hard-coded overrides (leave empty if you want auto-detect) ======
MULTI_COLS: List[str] = [
    # If you prefer to force the exact three, uncomment these:
    # "LEAFシステム(BookRoll,分析(ぶんせき)ツール)をどの教科で使用しますか(複数選択(せんたく)可)",
    # "BookRollでよく使う機能を選んでください(複数選択(せんたく)可)",
    # "分析ツール(ログパレ)でよく使う機能を選んでください(複数選択(せんたく)可)",
]

# ====== 3) Heuristics for auto-detection (same as before, with “exclude” rules) ======
DELIMS = ("、", ",", "/", "・", " and ")
HEADER_HINTS = ("複数", "（複数", "(複数")
EXCLUDE_HINTS = (
    "何か月", "何ヶ月", "X年Yか月", "X年Yヶ月",
    "スラッシュで区切", "1/0", "2/3",
    "自由記述", "自由回答", "自由記入", "フリーテキスト", "free text",
)

def _looks_multiselect_header(name: str) -> bool:
    n = str(name)
    return any(h in n for h in HEADER_HINTS)

def _looks_excluded_header(name: str) -> bool:
    n = str(name)
    return any(h in n for h in EXCLUDE_HINTS)

def _has_multiselect_tokens(series: pd.Series, sample: int = 500) -> bool:
    s = series.dropna().astype(str)
    if s.empty:
        return False
    s = s.head(sample)
    hits = sum(any(d in v for d in DELIMS) for v in s)
    return hits >= max(3, int(len(s) * 0.03))

def detect_multiselect_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for c in df.columns:
        if c in ("respondent_id", "学校種_canon", "学校名_canon"):
            continue
        if _looks_excluded_header(c):
            continue
        header_flag = _looks_multiselect_header(c)
        token_flag = _has_multiselect_tokens(df[c])
        if header_flag or token_flag:
            candidates.append(c)
    return candidates

def expand_multi_select(df: pd.DataFrame) -> pd.DataFrame:
    # Use the cleaned school names
    df["学校名_canon"] = df["学校名_canon"].fillna("不明")
    return df

# ====== 4) Main ======
def main():
    df = pd.read_csv(OUT_CSV, dtype=str, keep_default_na=False)

    # Choose columns
    if MULTI_COLS:
        missing = [c for c in MULTI_COLS if c not in df.columns]
        cols = [c for c in MULTI_COLS if c in df.columns]
        if missing:
            print(f"[warn] These MULTI_COLS were not found in {OUT_CSV.name}: {missing}", file=sys.stderr)
        if not cols:
            print("[error] No valid MULTI_COLS found; falling back to auto-detect.", file=sys.stderr)
            cols = detect_multiselect_columns(df)
    else:
        cols = detect_multiselect_columns(df)

    cols = list(cols)
    if not cols:
        print("[error] Could not detect any multi-select columns. "
              "Set MULTI_COLS explicitly at the top of this file.", file=sys.stderr)
        save_multi(pd.DataFrame(columns=["respondent_id","学校種_canon","学校名_canon","column","column_key","choice"]))
        return

    print(f"[info] Using {len(cols)} multi-select columns:", *[f" - {c}" for c in cols], sep="\n")

    long_df = explode_multiselect(
        df,
        multicolumns=cols,
        attach_facets=("学校種_canon", "学校名_canon"),
    )

    # Add short keys
    long_df["column_key"] = long_df["column"].map(to_column_key)

    # Reorder columns for convenience
    desired = ["respondent_id", "学校種_canon", "学校名_canon", "column_key", "column", "choice"]
    long_df = long_df.reindex(columns=desired)

    print(f"[info] Exploded rows: {len(long_df)}")
    save_multi(long_df)
    print("[info] Wrote students_multi_long.csv with column_key")

if __name__ == "__main__":
    main()
