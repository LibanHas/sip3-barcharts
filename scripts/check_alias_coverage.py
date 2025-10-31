# scripts/check_alias_coverage.py
from __future__ import annotations
import sys, argparse, re, unicodedata as ud
from pathlib import Path
import pandas as pd

ALLOWED = {
    "不明",
    "北海道天塩高等学校",
    "北海道寿都高等学校",
    "岩沼小学校",
    "洗足学園中学校",
    "西京高等学校",
    "西京高等学校付属中学校",
    "西賀茂中学校",
}

CANDIDATE_COLS = [
    "あなたの現在通っている学校名を教えてください（例：〇〇小学校、△△中学校）",  # full-width () and comma
    "あなたの現在通っている学校名を教えてください(例:〇〇小学校、△△中学校)",     # ASCII ()
    "あなたの現在通っている学校名を教えてください（例：〇〇小学校,△△中学校）",    # ASCII comma
]

PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;.\-\s]+")

def nrm(s: str) -> str:
    if s is None:
        return ""
    s = ud.normalize("NFKC", str(s)).strip()
    s = s.replace("附属", "付属")
    s = re.sub(r"学校校$", "学校", s)
    return s

def nrm_cmp(s: str) -> str:
    return PUNCT_RE.sub("", nrm(s)).lower()

def find_school_col(df: pd.DataFrame, user_col: str | None) -> str:
    cols = list(df.columns)
    if user_col and user_col in cols:
        return user_col
    # exact try
    for c in CANDIDATE_COLS:
        if c in cols:
            return c
    # fuzzy: look for a column that contains both “現在通っている学校名” and “学校”
    target_tokens = [nrm_cmp("現在通っている学校名"), nrm_cmp("学校")]
    for c in cols:
        nc = nrm_cmp(c)
        if all(tok in nc for tok in target_tokens):
            return c
    # last resort: the longest JP column containing 学校名
    candidates = [c for c in cols if "学校名" in str(c)]
    if candidates:
        return max(candidates, key=len)
    raise KeyError("Could not find a school-name column in cleaned data.")

def load_alias(alias_path: Path) -> pd.DataFrame:
    alias_df = pd.read_csv(alias_path, dtype=str).fillna("")
    # Normalize helper columns (do NOT mutate user-visible canon)
    alias_df["_col_n"] = alias_df["column"].map(nrm_cmp)
    alias_df["_raw_n"] = alias_df["raw"].map(nrm_cmp)
    return alias_df

def build_maps(alias_df: pd.DataFrame, school_col: str):
    col_n = nrm_cmp(school_col)
    sub = alias_df[alias_df["_col_n"] == col_n].copy()
    if sub.empty:
        # If header mismatch, try using all rows, but warn
        print(f"[WARN] No alias rows matched column header '{school_col}'. Will attempt global alias application.")
        sub = alias_df.copy()
    exact_map = dict(zip(sub["raw"], sub["canon"]))
    norm_map  = dict(zip(sub["_raw_n"], sub["canon"]))
    return exact_map, norm_map

def apply_alias_series(s: pd.Series, exact_map: dict, norm_map: dict) -> pd.Series:
    def canonize(v):
        if pd.isna(v) or str(v).strip()=="":
            return "不明"
        t = str(v)
        # 1) exact
        if t in exact_map:
            return exact_map[t]
        # 2) normalized match
        tn = nrm_cmp(t)
        if tn in norm_map:
            return norm_map[tn]
        # 3) return as-is (will be validated later)
        return nrm(t)
    return s.map(canonize)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/students_clean.csv")
    ap.add_argument("--alias", default=None, help="Path to alias_schools.csv")
    ap.add_argument("--col", default=None, help="Explicit school column header to use")
    args = ap.parse_args()

    data_p = Path(args.data)
    if not data_p.exists():
        print("[ERROR] data not found:", data_p); sys.exit(1)

    # find alias csv
    alias_p = None
    if args.alias:
        alias_p = Path(args.alias)
    else:
        for p in [
            Path("config/alias_schools.csv"),
            Path("data/alias_schools.csv"),
            Path("data/aliases/alias_schools.csv"),
            Path("alias_schools.csv"),
            Path("scripts/alias_schools.csv"),
        ]:
            if p.exists():
                alias_p = p; break
    if not alias_p or not alias_p.exists():
        print("[ERROR] alias_schools.csv not found. Use --alias or place it in config/ or data/."); sys.exit(1)

    print("[info] data: ", data_p.resolve())
    print("[info] alias:", alias_p.resolve())

    df = pd.read_csv(data_p, dtype=str).fillna("")
    school_col = find_school_col(df, args.col)
    print(f"[INFO] Using school column: {school_col!r}")

    alias_df = load_alias(alias_p)
    exact_map, norm_map = build_maps(alias_df, school_col)

    # Apply
    df["_school_canon"] = apply_alias_series(df[school_col], exact_map, norm_map)

    # Report
    uniq = sorted(set(df["_school_canon"]))
    print("Canonicalized school names:")
    for u in uniq:
        print(" ", u)

    # What’s unexpected?
    bad = [u for u in uniq if u not in ALLOWED]
    if bad:
        print("\n[!] Unexpected labels found:")
        for b in bad:
            sub = df.loc[df["_school_canon"] == b, school_col]
            ex = sub.iloc[0] if len(sub) else ""
            print(f"  - {b}: {len(sub)} rows (e.g., '{ex}')")
        print("\nAction: add alias rows mapping each of the above to one of the 8 allowed schools.")
        # also dump a few raw originals to help patch the alias map:
        for b in bad:
            sample = df.loc[df["_school_canon"] == b, school_col].head(5).tolist()
            print(f"    samples for {b}: {sample}")
        sys.exit(2)
    else:
        print("\n[OK] All values map into the 8 allowed labels.")

if __name__ == "__main__":
    main()
