import re

def _pick(colnames, patterns):
    for pat in patterns:
        for c in colnames:
            if re.search(pat, c):
                return c
    return None

def find(df):
    cols = list(df.columns)

    school_name = _pick(cols, [r"学校名", r"ご所属", r"所属.*学校", r"勤務先"])
    school_type = _pick(cols, [r"学校種", r"校種"])
    grade       = _pick(cols, [r"学年"])
    leaf_duration_cols = [c for c in cols if ("LEAF" in c and "何か月" in c)]
    multi_select_cols  = [c for c in cols if re.search("複数(選択|回答)", c)]

    return {
        "school_name": school_name,
        "school_type": school_type,
        "grade": grade,
        "leaf_duration_cols": leaf_duration_cols,
        "multi_select_cols": multi_select_cols
    }
