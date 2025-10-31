# scripts/sip3_teachers/pipeline.py
import pandas as pd
from . import (
    config,
    io_ops,
    normalize,
    detect_cols,
    schools,
    durations,
    multiselect,
    likert,
    validate,
)
# NEW: import alias applier (we already use this helper elsewhere)
# NEW: import alias applier + canonicalizer + post-fix
from scripts.common.canonicalize import (
    apply_aliases,
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
)


# (optional) where your alias file lives
ALIAS_SCHOOLS = "config/alias_schools.csv"

def run():
    # 1) Load
    df = io_ops.load_csv(config.RAW_CSV)
    io_ops.info(df)

    # 2) Normalize headers & blanks
    df = normalize.normalize_headers(df)
    df = normalize.drop_all_empty_rows(df)

    # 3) Detect columns (hard-coded in detect_cols for this dataset)
    cols = detect_cols.find(df)
    print("Detected:", cols)

    # 4) Clean text & missing
    df[cols["school_name"]] = normalize.clean_school_text(df[cols["school_name"]])
    df = normalize.blank_to_nan(df)

    # --- NEW: drop rows that are fully empty on core content -----------------
    # Define "substantive" columns that indicate a real response
    substantive = [
        "授業でのICT活用",
        "LEAFシステム(BookRoll,分析ツール)をどれくらい利用していますか",
        "LEAFシステム(BookRoll,分析ツール)をどの教科で使用しますか(複数選択可)",
        "BookRollでよく使う機能を選んでください(複数選択可)",
        "分析ツール(ログパレ)でよく使う機能を選んでください(複数選択可)",
    ]
    # keep only the ones that actually exist in df
    substantive = [c for c in substantive if c in df.columns]
    if substantive:
        mask_all_empty = df[substantive].isna().all(axis=1)
        dropped = int(mask_all_empty.sum())
        if dropped:
            print(f"[clean] Dropped {dropped} fully-empty responses")
            df = df.loc[~mask_all_empty].copy()

    # 5) School canonicalization (use alias_schools.csv)
    #    This creates/overwrites "<col>_canon" columns listed in the alias file.
    try:
        alias_df = pd.read_csv(ALIAS_SCHOOLS, dtype=str)
        alias_df = alias_df.rename(columns={"canonical": "canon"})  # tolerate either header
        df = apply_aliases(df, alias_df)  # will create f"{column}_canon" cols per the file
        # prefer a consistent canonical column name for downstream
        if f"{cols['school_name']}_canon" in df.columns:
            df["学校名_canon"] = df[f"{cols['school_name']}_canon"]
        else:
            # Fallback: if alias file didn't produce it, keep the cleaned original
            df["学校名_canon"] = df[cols["school_name"]]
    except Exception as e:
        print(f"[warn] Could not apply alias_schools.csv: {e}")
        df["学校名_canon"] = df[cols["school_name"]]

    # Guard: if someone typed their personal name into 学校名, mark as unknown
    # (only if a 名前/氏名 column exists and matches exactly)
    for name_col in ["名前", "氏名", "お名前"]:
        if name_col in df.columns:
            same = df["学校名_canon"].notna() & df[name_col].notna() & (df["学校名_canon"] == df[name_col])
            if same.any():
                print(f"[clean] Reset {int(same.sum())} rows where 学校名=={name_col}")
                df.loc[same, "学校名_canon"] = pd.NA
    df["school_canon"] = df["学校名_canon"]              # seed from the cleaned/aliased column
    SC.find_or_make_school_canon(df, debug=False)        # ensures/normalizes df['school_canon']
    post_disambiguate_middle_vs_high(df)                 # promote 洗足学園中学校→洗足学園高等学校 when 学校種 is 高
    SC.assert_only_allowed(df)                           # sanity check against allowlist

    # 6) 学校種: normalize or infer (then fill unknowns as 不明)
    if cols.get("school_type"):
        df["学校種"] = schools.normalize_school_type(df[cols["school_type"]])
    else:
        df["学校種"] = schools.infer_school_type(df["学校名_canon"])

    df["学校種"] = (
        df["学校種"]
          .replace({r"(?i)^\s*(na|nan|null|none)\s*$": pd.NA, r"^\s*$": pd.NA}, regex=True)
          .fillna("不明")
    )

    # 7) Duration parsing
    for c in cols["leaf_duration_cols"]:
        df[c + "_months_total"] = durations.parse_months_series(df[c])

    # Cast months columns to Int64 (nullable integers)
    for c in cols["leaf_duration_cols"]:
        mcol = c + "_months_total"
        if mcol in df.columns:
            df[mcol] = pd.to_numeric(df[mcol], errors="coerce").astype("Int64")

    # 8) Multi-select → long
    df_multi = (
        multiselect.explode(df, cols["multi_select_cols"])
        if cols["multi_select_cols"]
        else df.head(0)
    )

    # 9) Likert standardization
    for c in df.columns:
        if any(k in c for k in ["あてはまる", "使用しない", "使用している"]):
            df[c] = likert.standardize(df[c])

    # 10) Validate
    validate.no_person_names_in_school(df["学校名_canon"])
    validate.school_type_domain(df["学校種"])  # allows {"小","中","高","不明"}
    validate.duration_integrity(df, cols["leaf_duration_cols"])
    validate.report_missing(df, ["学校名_canon", "学校種"])

    # 11) Save
    io_ops.save_csv(df, config.OUT_CSV)
    io_ops.save_csv(df_multi, config.OUT_MULTI)
    print("✅ Saved:", config.OUT_CSV, "and", config.OUT_MULTI)
