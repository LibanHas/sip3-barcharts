# scripts/sip3_teachers/detect_cols.py

def find(df):
    # Exact column names as they appear *after* Step 2 header normalization
    SCHOOL_NAME = "ご所属をご記入ください(例:●●市立〇〇小学校)"
    SCHOOL_TYPE = "学校種"

    LEAF_DURATION = (
        "LEAFシステム(BookRoll,分析ツール)を授業・授業外(宿題など)で何か月くらい利用していますか。"
        "※利用期間がX年Yか月だった場合、「X/Y」というようにスラッシュで区切って、全て半角で入力してください"
        "(例:1年の場合→1/0、2年3か月の場合→2/3、未使用の場合→0/0)"
    )

    MULTI_SELECTS = [
        "LEAFシステム(BookRoll,分析ツール)をどの教科で使用しますか(複数選択可)",
        "BookRollでよく使う機能を選んでください(複数選択可)",
        "分析ツール(ログパレ)でよく使う機能を選んでください(複数選択可)",
    ]

    # Minimal presence checks (raise clear errors if missing)
    required = [SCHOOL_NAME, SCHOOL_TYPE, LEAF_DURATION, *MULTI_SELECTS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    return {
        "school_name": SCHOOL_NAME,
        "school_type": SCHOOL_TYPE,
        "grade": None,                    # not used for teachers
        "leaf_duration_cols": [LEAF_DURATION],
        "multi_select_cols": MULTI_SELECTS,
    }
