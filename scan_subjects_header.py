import pandas as pd, unicodedata as ud, re

def norm(s):
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()\\[\\]【】「」『』,:;－ー―–—\\-]", "", s)
    return s.lower()

df = pd.read_csv("data/teachers_clean.csv", dtype=str, nrows=1)
target_exact = "le叶fシステム（bookroll，分析ツール）をどの教科で使用しますか（複数選択可）"  # NFKC-style; used only after normalization
# note: the “LEAF” gets normalized to ascii; we don’t rely on this exact string, see tokens below.

tokens = ["どの教科", "使用", "複数選択"]  # tolerant token set

hits = []
for i, c in enumerate(df.columns):
    cn = norm(c)
    if all(t in cn for t in tokens):
        hits.append((i, c))
        print(f"[HIT] {i:3d}: {c}")

if not hits:
    print("No matches; try relaxing tokens or show me the raw header text.")
