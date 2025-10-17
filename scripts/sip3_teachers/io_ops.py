import pandas as pd

def load_csv(path, dtype=str):
    return pd.read_csv(path, dtype=dtype, encoding="utf-8-sig").dropna(how="all").copy()

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

def info(df, head=3):
    print(f"{len(df)} rows × {len(df.columns)} cols")
    print("First columns:", list(df.columns[:10]))
    return df.head(head)
