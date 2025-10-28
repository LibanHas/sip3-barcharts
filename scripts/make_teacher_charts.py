from __future__ import annotations
import argparse
from .charts_teachers import runners

def parse_args():
    p = argparse.ArgumentParser(description="Generate SIP3 teacher charts.")
    p.add_argument(
        "--sections",
        default="demographics,likert,multi,numeric",  # you can leave default as-is
        help="Comma-separated list from: demographics,likert,multi,numeric,numeric_by_school",
    )
    p.add_argument(
        "--by-school",
        action="store_true",
        help="Also produce multiselect charts by 学校名_canon.",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=2,
        help="Minimum respondents per school (where applicable).",
    )
    return p.parse_args()

def _run_safely(label, fn, *args, **kwargs):
    print(f"▶ {label} ...")
    try:
        fn(*args, **kwargs)
        print(f"✔ {label} done")
    except Exception as e:
        print(f"✖ {label} failed: {e}")
        raise

def main():
    args = parse_args()
    sections = {s.strip() for s in args.sections.split(",") if s.strip()}

    print("Output directory:", runners.config.FIGS.resolve())

    if "demographics" in sections:
        _run_safely("Demographics", runners.run_bars_demographics)

    if "likert" in sections:
        _run_safely("Likert（学校種）", runners.run_frequency_and_likert)
        if args.by_school:
            _run_safely("Likert（学校別）", runners.run_frequency_and_likert_by_school, min_n=args.min_n)

    if "multi" in sections:
        _run_safely("Multiselect（学校種）", runners.run_multiselect)
        if args.by_school:
            _run_safely(f"Multiselect（学校別≥{args.min_n}）", runners.run_multiselect_by_school, min_n=args.min_n)

    if "numeric" in sections:
        _run_safely("Numeric", runners.run_numeric)

    # ✅ Add this block:
    if "numeric_by_school" in sections:
        _run_safely(f"Numeric（学校別≥{args.min_n}）", runners.run_numeric_by_school, min_n=args.min_n)

    print("✅ Charts saved under figs/teachers/")

if __name__ == "__main__":
    main()
