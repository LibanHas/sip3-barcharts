# validators.py
import pandas as pd

def validate(df: pd.DataFrame) -> dict:
    report = {"missing_required": [], "range_issues": [], "logic_flags": []}

    # Required columns
    required = ["respondent_id"]
    for c in required:
        if c not in df.columns:
            report["missing_required"].append(c)

    # Range sanity for duration
    for c in ("利用期間_total_months",):
        if c in df.columns:
            bad = df[c].dropna().astype(float)
            if (bad < 0).any() or (bad > 120).any():  # tweak threshold
                report["range_issues"].append(f"{c}: values out of 0–120 months")

    # Example logic: if “未使用” flag exists but duration > 0 (implement if you have such a flag)
    # if "BookRoll使用" in df.columns:
    #     mask = (df["BookRoll使用"] == "未使用") & (df["利用期間_total_months"].fillna(0) > 0)
    #     n = int(mask.sum())
    #     if n:
    #         report["logic_flags"].append(f"未使用だが期間>0のレコード: {n} 件")

    return report
