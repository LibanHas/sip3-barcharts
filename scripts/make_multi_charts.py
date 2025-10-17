#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_teacher_charts.py

All-in-one script to create:
A) Multi-variable charts (correlation heatmap/bubbles + grouped-mean bars)
B) Per-teacher "multi" profile charts (rows/subjects/classes)

- No seaborn. Pure Matplotlib.
- JP-friendly fonts if available (IPAexGothic / Hiragino / Yu Gothic / Noto Sans CJK / MS Gothic)
- Reads a single CSV/XLSX or a glob like data/teachers_*.csv

Examples
--------
# Everything (auto)
python3 scripts/make_teacher_charts.py \
  --input "data/teachers_*.csv" \
  --dataset teachers \
  --do-corr yes \
  --do-profiles yes \
  --corr-style bubbles \
  --corr-method spearman \
  --max-charts 30

# Only per-teacher profiles using subjects with threshold 1
python3 scripts/make_teacher_charts.py \
  --input data/teachers_clean.csv \
  --do-corr no \
  --do-profiles yes \
  --multi-strategy subjects --multi-min 1

# Only correlation heatmap and grouped means
python3 scripts/make_teacher_charts.py \
  --input data/teachers_clean.csv \
  --do-corr yes \
  --do-profiles no \
  --corr-style heatmap --corr-method pearson
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import sys, argparse, logging, unicodedata as ud, re, os

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt


# ==============================
# Font handling (JP-safe)
# ==============================
def set_japanese_font():
    """Pick a Japanese-capable font if present; otherwise keep default."""
    preferred = [
        "IPAexGothic",   # commonly installed
        "Hiragino Sans", # macOS
        "Yu Gothic",     # Windows
        "Noto Sans CJK JP",
        "MS Gothic",
    ]
    available = set(f.name for f in fm.fontManager.ttflist)
    for fam in preferred:
        if fam in available:
            plt.rcParams["font.family"] = fam
            logging.info(f"Font family in use: {fam}")
            return fam
    logging.info("Using Matplotlib default font.")
    return None


def force_ipaex_if_local():
    """
    If user has a local IPAexGothic TTF, register it so Matplotlib can find it.
    Returns (family_name, FontProperties or None).
    """
    candidates = [
        Path.home() / "Library/Fonts" / "ipaexg.ttf",
        Path("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"),
        Path("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"),
    ]
    fam = None
    fp = None
    for p in candidates:
        if p.exists():
            try:
                fm.fontManager.addfont(str(p))
                fp = fm.FontProperties(fname=str(p))
                fam = fp.get_name()
                break
            except Exception:
                pass
    if fam:
        mpl.rcParams["font.family"] = fam
        mpl.rcParams["axes.unicode_minus"] = False
        logging.info(f"Font family in use (forced): {fam}")
    return fam, fp


# ==============================
# I/O helpers
# ==============================
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def read_glob(glob_pattern: str) -> pd.DataFrame:
    """
    Reads CSV/XLSX files matching glob and concatenates them.
    Tries UTF-8, UTF-8-SIG, cp932 for CSV.
    """
    paths = sorted(Path().glob(glob_pattern))
    if not paths and glob_pattern.endswith(".csv"):
        # also try xlsx in case
        paths = sorted(Path().glob(glob_pattern[:-4] + ".xlsx"))
    if not paths:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    frames = []
    for p in paths:
        if p.suffix.lower() == ".xlsx":
            frames.append(pd.read_excel(p))
        else:
            for enc in ("utf-8-sig", "utf-8", "cp932", "shift-jis"):
                try:
                    frames.append(pd.read_csv(p, encoding=enc))
                    break
                except Exception:
                    if enc == "shift-jis":
                        raise
                    continue
    return pd.concat(frames, ignore_index=True)


# ==============================
# Column detection & utilities
# ==============================
GROUP_CANDIDATES = [
    "学校種", "学校種別", "school_type", "学年", "学級", "クラス", "class", "Class",
]

EXCLUDE_COLS = set([
    "ID", "メール", "Email", "email", "開始時刻", "完了時刻",
])

def infer_dataset_from_path(p: str) -> str:
    name = os.path.basename(p).lower()
    if "teacher" in name or "教師" in name or "先生" in name:
        return "teachers"
    if "student" in name or "児童" in name or "生徒" in name:
        return "students"
    return "dataset"

def _norm(s: str) -> str:
    return ud.normalize("NFKC", str(s))

def find_teacher_col(cols: list[str]) -> str | None:
    candidates = ["TeacherID", "ID", "教師ID", "講師ID", "メール", "Email", "email"]
    for c in candidates:
        if c in cols: return c
    # fuzzy
    for c in cols:
        if _norm(c).strip().lower() in {"id", "教師id", "講師id"}: return c
    return None

def find_school_col(cols: list[str]) -> str | None:
    if "学校名" in cols: return "学校名"
    for c in cols:
        if "ご所属をご記入ください" in _norm(c):
            return c
    for c in ["SchoolID"]:
        if c in cols: return c
    return None

def find_class_col(cols: list[str]) -> str | None:
    for c in ["ClassID", "授業ID", "科目ID", "クラスID"]:
        if c in cols: return c
    return None

def find_subjects_col(cols: list[str]) -> str | None:
    # typical phrasing variations
    for c in cols:
        nc = _norm(c)
        if "どの教科で使用しますか" in nc and ("複数選択" in nc or "複数" in nc):
            return c
    # fallback: any column containing "教科" and "複数"
    for c in cols:
        nc = _norm(c)
        if ("教科" in nc) and ("複数" in nc):
            return c
    return None

def explode_multi_select(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    """
    Split a multi-select column into rows.
    Handles: commas(, / ， / 、), slashes(/／), semicolons(;), bars(|｜), dots(・), spaces.
    """
    s = df[col].astype(str)
    s = (s.replace({"，": ",", "、": ",", "／": "/", "｜": "|"})
           .str.replace("・", ","))
    parts = s.str.split(r"[,\;/\|\s]+", regex=True)

    out = df.assign(**{out_col: parts}).explode(out_col)
    out[out_col] = out[out_col].astype(str).str.strip()
    out = out[out[out_col].str.len() > 0]
    return out

def detect_group_col(df: pd.DataFrame) -> Optional[str]:
    for cand in GROUP_CANDIDATES:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if c in EXCLUDE_COLS: continue
        if df[c].dtype == object:
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= 20:
                return c
    return None

def detect_numeric_likert_cols(df: pd.DataFrame, include_cols: Optional[List[str]] = None,
                               min_unique: int = 3, max_unique: int = 10) -> List[str]:
    if include_cols:
        return [c for c in include_cols if c in df.columns]
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS: continue
        if pd.api.types.is_numeric_dtype(df[c]):
            nunique = pd.Series(df[c]).dropna().nunique()
            if min_unique <= nunique <= max_unique:
                cols.append(c)
    return cols

def sanitize(name: str) -> str:
    bad = "/\\:*?\"<>|\n\t"
    for ch in bad:
        name = name.replace(ch, "_")
    return name[:120]

def _shorten_label(s: str, maxlen: int = 36) -> str:
    s = s.replace("_num", "")
    chunks = [s[i:i+14] for i in range(0, len(s), 14)]
    s = "\n".join(chunks)
    return (s if len(s) <= maxlen else s[:maxlen] + "…")


# ==============================
# Plotters – Multi-variable (A)
# ==============================
def save_corr_heatmap(df: pd.DataFrame, cols: List[str], outpath: str, title: str,
                      method: str = "spearman"):
    if len(cols) < 2:
        logging.warning("Not enough numeric columns for a heatmap.")
        return

    corr = df[cols].corr(method=method, numeric_only=True)
    corr = corr.copy()
    np.fill_diagonal(corr.values, np.nan)

    order = np.argsort(-np.nan_to_num(np.abs(corr.values), nan=0).sum(axis=0))
    cols_ord = [cols[i] for i in order]
    C = corr.values[order][:, order]

    short = [_shorten_label(c) for c in cols_ord]
    n = len(cols_ord)

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * n), max(6, 0.6 * n)))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_title(title, pad=18)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n+1)-.5, minor=True)
    ax.set_yticks(np.arange(n+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    if n <= 15:
        for i in range(n):
            for j in range(n):
                val = C[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation", rotation=270, labelpad=12)

    fig.subplots_adjust(left=0.28, bottom=0.32, right=0.95, top=0.9)
    fig.savefig(outpath, dpi=220)
    fig.savefig(outpath.replace(".png", ".pdf"))
    plt.close(fig)
    logging.info(f"Wrote {outpath} (+ PDF)")

def save_corr_bubbles(df: pd.DataFrame, cols: List[str], outpath: str, title: str,
                      method: str = "spearman"):
    if len(cols) < 2:
        logging.warning("Not enough numeric columns for a bubble plot.")
        return

    corr = df[cols].corr(method=method, numeric_only=True).values
    n = len(cols)
    order = np.argsort(-np.abs(np.nan_to_num(corr)).sum(axis=0))
    corr = corr[order][:, order]
    cols_ord = [cols[i] for i in order]

    labels = [_shorten_label(c, maxlen=28) for c in cols_ord]
    max_lines = max(lbl.count("\n") + 1 for lbl in labels)

    w = max(8.0, min(16.0, 0.55 * n + 3.0))
    h = max(6.5, min(16.0, 0.55 * n + 1.2 + 0.7 * max_lines))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_facecolor("white")
    ax.grid(True, which="both", color="#e8e8e8", linestyle="-", linewidth=1)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")

    cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    xs, ys, colors, sizes = [], [], [], []
    max_size = 900 if n <= 10 else 9000 / (n + 2)

    for i in range(n):
        for j in range(i + 1, n):
            r = corr[j, i]
            if np.isnan(r):
                continue
            xs.append(i)
            ys.append(n - 1 - j)
            colors.append(cmap(norm(r)))
            sizes.append(max(40, max_size * abs(r)))

    ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="none", alpha=0.9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(list(reversed(labels)), fontsize=9)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation (r)")

    ax.set_title(title, pad=14)

    fig.subplots_adjust(left=0.42, bottom=0.40, right=0.88, top=0.88)
    fig.savefig(outpath, dpi=220)
    fig.savefig(outpath.replace(".png", ".pdf"))
    plt.close(fig)
    logging.info(f"Wrote {outpath} (+ PDF)")

def save_grouped_means(df: pd.DataFrame, group_col: str, metric_col: str, outdir: Path,
                       dataset: str, top_n: Optional[int] = None):
    tmp = df[[group_col, metric_col]].dropna()
    if tmp.empty:
        return
    g = tmp.groupby(group_col, dropna=True)[metric_col].mean().sort_values(ascending=False)
    if top_n is not None:
        g = g.head(top_n)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(g.index.astype(str), g.values)
    ax.set_ylabel("Mean")
    ax.set_title(f"{dataset.capitalize()} – {metric_col} (mean by {group_col})")
    ax.set_xticklabels(g.index.astype(str), rotation=45, ha='right')
    fig.tight_layout()
    fname = f"groupmean__{sanitize(metric_col)}__by__{sanitize(group_col)}.png"
    outpath = outdir / fname
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    logging.info(f"Wrote {outpath}")


# ==============================
# Plotters – Per-teacher profiles (B)
# ==============================
def plot_profile(df_teacher: pd.DataFrame, teacher_id, strategy: str, outpath: Path):
    plt.figure(figsize=(8, 4.8))
    if strategy == "subjects" and "_Subject" in df_teacher.columns:
        counts = df_teacher["_Subject"].value_counts().sort_values(ascending=False)
        counts.plot(kind="bar")
        plt.title(f"Teacher {teacher_id} – Records per Subject")
        plt.xlabel("Subject"); plt.ylabel("Count")

    elif strategy == "classes":
        cands = [c for c in ["ClassID", "授業ID", "科目ID", "クラスID"] if c in df_teacher.columns]
        col = cands[0] if cands else None
        if not col:
            # fallback to rows
            counts = pd.Series({"rows": len(df_teacher)})
            counts.plot(kind="bar")
            plt.title(f"Teacher {teacher_id} – Row count")
            plt.xlabel(""); plt.ylabel("Rows")
        else:
            counts = df_teacher[col].value_counts().sort_values(ascending=False)
            counts.plot(kind="bar")
            plt.title(f"Teacher {teacher_id} – Records per {col}")
            plt.xlabel(col); plt.ylabel("Count")

    else:  # rows
        counts = pd.Series({"rows": len(df_teacher)})
        counts.plot(kind="bar")
        plt.title(f"Teacher {teacher_id} – Row count")
        plt.xlabel(""); plt.ylabel("Rows")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

def export_multi_profiles(df_multi: pd.DataFrame, teacher_col: str, strategy: str, outdir: Path) -> int:
    ensure_dir(outdir)
    written = 0
    for tid, g in df_multi.groupby(teacher_col):
        try:
            out = outdir / f"{sanitize(str(tid))}.png"
            plot_profile(g, tid, strategy, out)
            written += 1
        except Exception as e:
            logging.exception(f"Failed for {tid}: {e}")
    logging.info(f"Wrote {written} files to {outdir.resolve()}")
    return written


# ==============================
# Multi selection logic (B)
# ==============================
def compute_multi(df: pd.DataFrame, strategy: str, teacher_col: str,
                  multi_min: int, class_col: str | None,
                  subjects_col: str | None) -> tuple[pd.DataFrame, str]:
    """
    Returns (df_multi, used_strategy)
    strategies: 'classes' | 'subjects' | 'rows' | 'auto'
    """
    if teacher_col not in df.columns:
        raise RuntimeError(f"Teacher column '{teacher_col}' not found. Available: {list(df.columns)}")

    if strategy == "auto":
        if class_col and class_col in df.columns:
            strategy = "classes"
        elif subjects_col and subjects_col in df.columns:
            strategy = "subjects"
        else:
            strategy = "rows"
        logging.info(f"Auto-selected multi strategy: {strategy}")

    if strategy == "classes":
        if not class_col or class_col not in df.columns:
            raise RuntimeError("classes strategy chosen but no ClassID-like column exists.")
        by = df.groupby(teacher_col)[class_col].nunique()
        keep_ids = by[by >= multi_min].index
        out = df[df[teacher_col].isin(keep_ids)].copy()
        logging.info(f"Teachers with >= {multi_min} distinct {class_col}: {len(keep_ids)}")
        return out, "classes"

    if strategy == "subjects":
        if not subjects_col or subjects_col not in df.columns:
            raise RuntimeError("subjects strategy chosen but no subjects column found.")
        ex = explode_multi_select(df, subjects_col, "_Subject")
        by = ex.groupby(teacher_col)["_Subject"].nunique()
        keep_ids = by[by >= multi_min].index
        out = ex[ex[teacher_col].isin(keep_ids)].copy()
        logging.info(f"Teachers with >= {multi_min} distinct subjects: {len(keep_ids)}")
        return out, "subjects"

    if strategy == "rows":
        by = df.groupby(teacher_col).size()
        keep_ids = by[by >= multi_min].index
        out = df[df[teacher_col].isin(keep_ids)].copy()
        logging.info(f"Teachers with >= {multi_min} rows: {len(keep_ids)}")
        return out, "rows"

    raise ValueError(f"Unknown strategy: {strategy}")


# ==============================
# Main
# ==============================
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    # I/O
    p.add_argument("--input", required=True,
                   help='CSV/XLSX or glob, e.g. "data/teachers_*.csv"')
    p.add_argument("--dataset", choices=["teachers", "students", "unknown"], default=None)
    p.add_argument("--outroot", default="outputs", help="Root output folder")
    p.add_argument("--max-charts", type=int, default=30,
                   help="Cap number of grouped-mean charts")
    # Multi-variable charts (A)
    p.add_argument("--do-corr", choices=["yes","no"], default="yes")
    p.add_argument("--corr-style", choices=["bubbles","heatmap"], default="bubbles")
    p.add_argument("--corr-method", choices=["pearson","spearman"], default="spearman")
    p.add_argument("--include-cols", nargs="*", default=None,
                   help="Explicit list of numeric columns (skip auto-detect)")
    # Per-teacher profiles (B)
    p.add_argument("--do-profiles", choices=["yes","no"], default="yes")
    p.add_argument("--multi-strategy", default="auto", choices=["auto","classes","subjects","rows"])
    p.add_argument("--multi-min", type=int, default=2, help="threshold for chosen strategy")
    p.add_argument("--teacher-col", default=None)
    p.add_argument("--class-col", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO),
                        format="%(levelname)s: %(message)s")

    # Fonts
    fam_forced, _ = force_ipaex_if_local()
    if not fam_forced:
        set_japanese_font()

    # Read
    logging.info(f"Reading: {args.input}")
    df = read_glob(args.input)
    logging.info(f"Raw shape: ({df.shape[0]}, {df.shape[1]})")
    logging.info(f"Columns (sample): {list(df.columns)[:8]} ...")

    # Dataset/outdirs
    dataset = (args.dataset or infer_dataset_from_path(args.input)).lower()
    if dataset == "unknown":
        dataset = "dataset"

    charts_root = Path(args.outroot) / dataset / "charts"
    out_multi_var = charts_root / "multi"            # A) correlation + grouped means
    out_profiles  = charts_root / "multi_profiles"   # B) per-teacher profiles
    ensure_dir(out_multi_var)
    ensure_dir(out_profiles)

    # Column detection
    cols = list(df.columns)
    teacher_col = args.teacher_col or find_teacher_col(cols)
    school_col  = find_school_col(cols)
    class_col   = args.class_col or find_class_col(cols)
    subjects_col= find_subjects_col(cols)
    logging.info(f"Detected teacher_col={teacher_col}, school_col={school_col}, "
                 f"class_col={class_col}, subjects_col={subjects_col}")

    # ---------- A) Multi-variable charts ----------
    if args.do_corr == "yes":
        numeric_cols = detect_numeric_likert_cols(df, include_cols=args.include_cols)
        logging.info(f"Detected {len(numeric_cols)} numeric Likert-like columns.")
        if numeric_cols:
            corr_title = f"{dataset.capitalize()} – Correlation {args.corr_style} ({len(numeric_cols)} vars)"
            corr_name  = "corr_bubbles.png" if args.corr_style == "bubbles" else "corr_heatmap.png"
            corr_path  = str(out_multi_var / corr_name)

            if args.corr_style == "bubbles":
                save_corr_bubbles(df, numeric_cols, corr_path, corr_title, method=args.corr_method)
            else:
                save_corr_heatmap(df, numeric_cols, corr_path, corr_title, method=args.corr_method)

            # Grouped-mean charts (if we have a grouping col)
            group_col = detect_group_col(df)
            if group_col:
                logging.info(f"Using group_col={group_col} for grouped means.")
                count = 0
                for col in numeric_cols:
                    save_grouped_means(df, group_col, col, out_multi_var, dataset)
                    count += 1
                    if count >= args.max_charts:
                        logging.info(f"Reached --max-charts={args.max_charts}; stopping grouped charts.")
                        break
            else:
                logging.warning("No reasonable grouping column found. Skipping grouped means.")
        else:
            logging.warning("No numeric Likert-like columns found. Skipping multi-variable charts.")

    # ---------- B) Per-teacher profiles ----------
    if args.do_profiles == "yes":
        if not teacher_col:
            raise RuntimeError("Could not detect a teacher identifier column (e.g., ID). "
                               "Use --teacher-col to specify.")

        df_multi, used_strategy = compute_multi(
            df=df,
            strategy=args.multi_strategy,
            teacher_col=teacher_col,
            multi_min=int(args.multi_min),
            class_col=class_col,
            subjects_col=subjects_col
        )
        if df_multi.empty:
            raise RuntimeError("Multi selection produced 0 rows. "
                               "Try lowering --multi-min or switching strategy.")
        written = export_multi_profiles(df_multi, teacher_col, used_strategy, out_profiles)
        if written == 0:
            raise RuntimeError("No files written to multi_profiles/.")
    logging.info("Done.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.exception(e)
        sys.exit(1)
