# scripts/charts_teachers/q08_bookroll_features_by_school.py
# Usage:
#   python -m scripts.charts_teachers.q08_bookroll_features_by_school
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import re
import unicodedata as ud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

DEBUG = True  # <— turn on diagnostics

try:
    from scripts.common.plotkit import setup
except Exception:
    setup = lambda: None

from scripts.common.canonicalize import (
    SchoolCanonicalizer as SC,
    post_disambiguate_middle_vs_high,
)

DATA_CLEAN = Path("data/teachers_clean.csv")
OUT_DIR    = Path("figs/teachers/likert_by_school")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ORDER_CSV  = Path("config/school_order_q6.csv")

PLOT_DPI       = 300
BASE_FONTSIZE  = 12
TITLE_FONTSIZE = 16
TICK_FONTSIZE  = 11
LABEL_FONTSIZE = 12
ANNOTATE_GT    = 10.0
MIN_N          = 1
TOP_N          = None

FEATURES = ["メモ","マーカー","手書き","クイズ","URLリコメンド","辞書","タイマー","その他","ほとんど使ったことがない"]
FEATURE_LABELS = {"URLリコメンド":"URL\nリコメンド","ほとんど使ったことがない":"ほとんど\n使ったことがない"}

CMAP = LinearSegmentedColormap.from_list(
    "scarlet", ["#ffffff","#ffe6e6","#ffb3b3","#ff6b6b","#e31a1c","#8b0000"]
)
FOOTNOTE = ("分母＝各学校で当該設問（複数選択）に何らか回答した人数／ 小規模校も表示（n≥1）／"
            " 「ほとんど使ったことがない」は他の選択肢と併記された場合でもそれのみとして集計")

_PUNCT_RE = re.compile(r"[、。，．・/（）()【】\[\]「」『』,:;－ー―–—\-]")

def _norm(s: str) -> str:
    s = ud.normalize("NFKC", str(s)); s = re.sub(r"\s+","",s); s = _PUNCT_RE.sub("", s); return s.lower()

def find_col_features(df: pd.DataFrame) -> Optional[str]:
    exact = [
        "BookRollでよく使う機能を選んでください（複数選択可）",
        "BookRollでよく使う機能を選んでください",
        "Q8.BookRollでよく使う機能","8.BookRollでよく使う機能",
    ]
    want = {_norm(x) for x in exact}
    for c in df.columns:
        if _norm(c) in want: return c
    tokens = ["bookroll","機能","よく使う","複数選択","選んで"]
    for c in df.columns:
        if all(t in _norm(c) for t in tokens): return c
    return None

SEP_RE = re.compile(r"[;；、,／/・]|(?:\s+\+\s+)|\s+")
CANON_MAP: Dict[str,str] = {
    "メモ":"メモ","memo":"メモ","ノート":"メモ",
    "マーカー":"マーカー","marker":"マーカー","ハイライト":"マーカー",
    "手書き":"手書き","手描き":"手書き","ペン":"手書き","描画":"手書き",
    "クイズ":"クイズ","小テスト":"クイズ","テスト":"クイズ",
    "urlリコメンド":"URLリコメンド","url":"URLリコメンド","リンク推薦":"URLリコメンド",
    "おすすめurl":"URLリコメンド","URLリコメンド":"URLリコメンド","URL リコメンド":"URLリコメンド",
    "辞書":"辞書","dictionary":"辞書",
    "タイマー":"タイマー","timer":"タイマー",
    "ほとんど使ったことがない":"ほとんど使ったことがない",
    "未使用":"ほとんど使ったことがない","使っていない":"ほとんど使ったことがない","ほぼ未使用":"ほとんど使ったことがない",
    "その他":"その他",
}

def canon_feature(item: str) -> Optional[str]:
    if item is None: return None
    t = ud.normalize("NFKC", str(item)).strip(" 。．.")
    if not t: return None
    k = _norm(t)
    if k in CANON_MAP: return CANON_MAP[k]
    if ("問題集" in t) or ("解説" in t) or ("資料" in t) or ("解答例" in t): return "その他"
    return None

# split multi-select answers (handles nearly everything incl. odd whitespaces)
SEP_RE = re.compile(r"[;；、,／/・]|(?:\s+\+\s+)|\s+")

def _clean_token(t: str) -> str:
    # normalize & remove hidden chars/BOM/ZWSP
    t = ud.normalize("NFKC", str(t))
    t = t.replace("\ufeff", "").replace("\u200b", "").replace("\u2060", "")
    return t.strip(" 　;；。\n\r\t")

def parse_multiselect(series: pd.Series) -> List[List[str]]:
    out: List[List[str]] = []
    for raw in series.fillna(""):
        s = ud.normalize("NFKC", str(raw)).replace("\ufeff", "").replace("\u200b", "").replace("\u2060", "")
        s = s.replace("\r\n", "\n")
        # If the whole string is just separators/spaces, treat as empty
        if not _clean_token(s):
            out.append([])
            continue
        parts = [p for p in ( _clean_token(p) for p in SEP_RE.split(s) ) if p]
        # Fallback: if we still got nothing, try “contains” recovery for known keywords
        if not parts:
            candidates = []
            for kw in ["メモ","マーカー","手書き","クイズ","URLリコメンド","辞書","タイマー","その他",
                       "未使用","使っていない","ほとんど使ったことがない","小テスト","テスト","リンク推薦","おすすめurl","URL リコメンド"]:
                if kw in s:
                    candidates.append(kw)
            parts = list(dict.fromkeys(candidates))  # de-dup, keep order
        out.append(parts)
    return out

def _load_q6_order() -> Optional[List[str]]:
    if not ORDER_CSV.exists(): return None
    try:
        order_df = pd.read_csv(ORDER_CSV, dtype=str)
        col = "school_canon" if "school_canon" in order_df.columns else order_df.columns[0]
        return [ud.normalize("NFKC", s) for s in order_df[col].dropna().tolist()]
    except Exception:
        return None

def _reindex_like_q6(g: pd.DataFrame, order: Optional[List[str]]) -> pd.DataFrame:
    if not order: return g
    in_order = [idx for idx in order if idx in g.index]
    extras   = [idx for idx in g.index if idx not in set(order)]
    return g.reindex(in_order + extras)

def plot_heatmap(pct_df: pd.DataFrame, n_per_school: pd.Series, title: str, out_png: Path):
    rows = pct_df.index.tolist(); cols = FEATURES
    M = pct_df.reindex(columns=cols, fill_value=0.0).values
    fig_w = max(12.0, 0.90*len(cols)+3.6); fig_h = max(6.0, 0.46*len(rows)+2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PLOT_DPI)
    im = ax.imshow(M, aspect="auto", origin="upper", vmin=0, vmax=100, cmap=CMAP, interpolation="bicubic")
    labels = [FEATURE_LABELS.get(c, c) for c in cols]
    ax.set_xticks(np.arange(len(cols)), labels=labels, fontsize=TICK_FONTSIZE); ax.tick_params(axis="x", pad=6)
    ylabels = [f"{r}（n={int(n_per_school.loc[r])}）" for r in rows]
    ax.set_yticks(np.arange(len(rows)), labels=ylabels, fontsize=TICK_FONTSIZE)
    ax.minorticks_off(); ax.grid(False)
    for side in ["top","right","left","bottom"]:
        ax.spines[side].set_visible(True); ax.spines[side].set_linewidth(1.1); ax.spines[side].set_color((0,0,0,0.35))
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("割合(%)", fontsize=LABEL_FONTSIZE)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = float(M[i,j])
            if v >= ANNOTATE_GT:
                txt_color = "white" if v >= 60 else "black"; outline = "black" if txt_color == "white" else "white"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=BASE_FONTSIZE-1, color=txt_color,
                        weight="bold" if v >= 80 else None,
                        path_effects=[pe.withStroke(linewidth=2, foreground=outline, alpha=0.9)])
    ax.set_xlabel("BookRoll機能（複数選択）", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=10)
    fig.text(0.01, -0.02, FOOTNOTE, ha="left", va="top", fontsize=10)
    plt.tight_layout(rect=[0, 0.20, 1, 1]); fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight"); plt.close(fig)
    print(f"[info] wrote {out_png}")

def main():
    setup()
    if not DATA_CLEAN.exists():
        raise FileNotFoundError(f"Missing {DATA_CLEAN}.")
    df = pd.read_csv(DATA_CLEAN, dtype=str).replace({"": np.nan}).infer_objects(copy=False)

    # Canonicalize like Q6
    school_col = SC.find_or_make_school_canon(df, debug=False)
    if school_col != "school_canon": df["school_canon"] = df[school_col]
    SC.assert_only_allowed(df); post_disambiguate_middle_vs_high(df)

    col = find_col_features(df)
    if not col:
        print("[ERROR] Q8 column not found."); return
    print(f"[OK] features column: {col}")

    lists = parse_multiselect(df[col])

    # Canonicalize + exclusivity
    canon_lists: List[List[str]] = []
    for lst in lists:
        mapped = [m for m in (canon_feature(p) for p in lst) if m]
        if not mapped:
            canon_lists.append([]); continue
        if "ほとんど使ったことがない" in mapped:
            canon_lists.append(["ほとんど使ったことがない"])
        else:
            seen=[]; [seen.append(m) for m in mapped if m not in seen]
            canon_lists.append(seen)

    # Row-level debug frame (what did we parse?)
    df["_q8_raw"] = df[col]
    df["_q8_parsed"] = [";".join(lst) if lst else "" for lst in canon_lists]
    df["_q8_has_ans"] = [bool(lst) for lst in canon_lists]

    # Build long table (one row per feature selection)
    rows = []
    for school, lst in zip(df["school_canon"], canon_lists):
        if not lst: continue
        for feat in lst: rows.append((school, feat))
    long = pd.DataFrame(rows, columns=["school_canon", "feature"])

    # Responders = teachers with >=1 canonical selection
    responders = (df.groupby("school_canon")["_q8_has_ans"].sum().astype(int))

    # Pre-alignment diagnostics
    if DEBUG:
        summary = (
            df.groupby("school_canon")
              .agg(total_rows=("school_canon","size"),
                   non_empty=("_q8_has_ans","sum"))
              .assign(responders=lambda x: x["non_empty"])
              .sort_values("total_rows", ascending=False)
        )
        dbg_csv = OUT_DIR / "Q08_debug_rows.csv"
        df[["school_canon","_q8_raw","_q8_parsed","_q8_has_ans"]].to_csv(dbg_csv, index=False, encoding="utf-8")
        print("[DEBUG] wrote row-level debug:", dbg_csv)
        print("[DEBUG] per-school counts (total_rows / non_empty / responders):")
        print(summary.to_string())

    # Counts BEFORE filtering; then align; THEN apply MIN_N
    ct = long.groupby(["school_canon","feature"]).size().unstack(fill_value=0)

    # Ensure all feature columns exist & order
    for f in FEATURES:
        if f not in ct.columns: ct[f] = 0
    ct = ct.reindex(columns=FEATURES, fill_value=0)

    # Row order identical to Q6/Q7 (+ extras at bottom)
    order = _load_q6_order()
    if order is None:
        order = df["school_canon"].dropna().unique().tolist()

    row_index = pd.Index(order, name="school_canon")
    extras = [s for s in ct.index if s not in row_index]
    row_index = row_index.append(pd.Index(extras))

    # Reindex & align responders
    ct = ct.reindex(row_index, fill_value=0)
    responders = responders.reindex(row_index).fillna(0).astype(int)

    # Apply Q6-style MIN_N after alignment
    mask_keep = responders >= MIN_N
    ct = ct[mask_keep]; responders = responders[mask_keep]

    # Optional top-N
    if TOP_N:
        top_idx = responders.sort_values(ascending=False).head(TOP_N).index
        ct = ct.loc[top_idx]; responders = responders.loc[top_idx]

    # Percentages
    pct = (ct.div(responders.replace(0, np.nan), axis=0) * 100.0).fillna(0.0)
    n_per_school = responders

    # Save CSV + plot
    out_csv = OUT_DIR / "Q08_機能__学校別.csv"
    pct_out = pct.copy().round(1); pct_out["n"] = n_per_school
    pct_out.to_csv(out_csv, encoding="utf-8"); print(f"[info] wrote {out_csv}")

    title = "BookRollでよく使う機能（学校別・教員）"
    out_png = OUT_DIR / "Q08_機能__学校別_heatmap.png"
    plot_heatmap(pct, n_per_school, title, out_png)

if __name__ == "__main__":
    main()
