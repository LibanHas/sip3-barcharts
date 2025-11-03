import pandas as pd, unicodedata as ud, re
df = pd.read_csv("data/teachers_clean.csv", dtype=str, nrows=1)

def norm(s):
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()\[\]【】「」『』,:;－ー―–—\-]", "", s)
    return s.lower()

for c in df.columns:
    cn = norm(c)
    if ("使いこな" in cn) and ("程度" in cn):  # relaxed filter
        print("CANDIDATE:", c)
