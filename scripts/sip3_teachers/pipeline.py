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

def run():
    # 1) Load
    df = io_ops.load_csv(config.RAW_CSV)
    io_ops.info(df)

    # 2) Normalize headers & blanks
    df = normalize.normalize_headers(df)
    df = normalize.drop_all_empty_rows(df)

    # 3) Detect columns
    cols = detect_cols.find(df)
    validate.assert_detected(cols, required=["school_name"])  # teachers may not need 'grade'
    print("Detected:", cols)

    # 4) Clean text & missing
    df[cols["school_name"]] = normalize.clean_school_text(df[cols["school_name"]])
    if cols.get("grade"):
        df[cols["grade"]] = normalize.clean_grade_text(df[cols["grade"]])
    df = normalize.blank_to_nan(df)

    # 5) School canonicalization
    df["学校名_canon"] = schools.canonicalize_names(df[cols["school_name"]])

    # 6) 学校種: normalize or infer
    if cols.get("school_type"):
        st = schools.normalize_school_type(df[cols["school_type"]])
        df["学校種"] = st
    else:
        df["学校種"] = schools.infer_school_type(df["学校名_canon"])

    # Normalize and fill unknown school type (catch 'nan', 'null', empty, etc.)
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

    # 9) Likert standardization (apply where relevant)
    for c in df.columns:
        if any(k in c for k in ["あてはまる", "使用しない", "使用している"]):
            df[c] = likert.standardize(df[c])

    # 10) Validate
    validate.no_person_names_in_school(df["学校名_canon"])
    validate.school_type_domain(df["学校種"])  # uses default {"小","中","高","不明"}
    validate.duration_integrity(df, cols["leaf_duration_cols"])
    validate.report_missing(df, ["学校名_canon", "学校種"])

    # 11) Save
    io_ops.save_csv(df, config.OUT_CSV)
    io_ops.save_csv(df_multi, config.OUT_MULTI)
    print("✅ Saved:", config.OUT_CSV, "and", config.OUT_MULTI)
