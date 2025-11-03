import pandas as pd, unicodedata as ud, re, sys

def norm(s):
    s = ud.normalize("NFKC", str(s))
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[、。，．・/（）()\\[\\]【】「」『』,:;－ー―–—\\-]", "", s)
    return s.lower()

df = pd.read_csv("data/teachers_clean.csv", dtype=str, nrows=1)
cols = list(df.columns)

patterns = [
    ("使いこな + 程度", ["使いこな", "程度"]),
    ("どの程度 + 使いこな", ["どの程度", "使いこな"]),
    ("LEAFシステム + 使いこな", ["leafシステム", "使いこな"]),  # note: normalization lowers case
    ("BookRoll or 分析ツール + 程度", ["bookroll", "程度"]),
]

matches = []

print("=== ALL COLUMNS (index, original) ===")
for i, c in enumerate(cols):
    print(f"{i:3d}: {c}")

print("\n=== POSSIBLE MATCHES ===")
for i, c in enumerate(cols):
    cn = norm(c)
    hit = []
    for label, toks in patterns:
        if all(t in cn for t in toks):
            hit.append(label)
    if hit:
        print(f"[{i:3d}] {c}  -> hits: {', '.join(hit)}")
        matches.append((i, c))

if not matches:
    print("No matches with the current patterns.")
