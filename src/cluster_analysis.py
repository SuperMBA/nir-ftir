# -*- coding: utf-8 -*-
"""
Cluster analysis for FTIR/NIR spectral datasets with clinical interpretation.

What is added vs basic version:
- Optional t-SNE (along with PCA/UMAP)
- Cluster quality metrics + (if labels exist) ARI/NMI/V-measure/Purity
- Explicit statistical linkage of clusters/DR coordinates with clinical variables
  (age/sex/stage/glucose/etc., auto-detected or user-provided)
- FDR (Benjamini-Hochberg) correction for multiple testing
- Patient-level association tables when repeated measurements per ID exist

This directly addresses the common reviewer/supervisor comment:
"Do not use PCA/UMAP/t-SNE only visually; test associations with clinical variables."
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Optional UMAP
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DEFAULT_DATASETS = {
    "covid_saliva": DATA_PROCESSED / "train.parquet",
    "diabetes_saliva": DATA_PROCESSED / "diabetes_saliva.parquet",
}

# Common aliases for clinical variables (extend if needed)
CLINICAL_ALIASES: Dict[str, List[str]] = {
    "age": ["age", "возраст"],
    "sex": ["sex", "gender", "пол"],
    "stage": ["stage", "stadium", "grade", "perio_stage", "disease_stage", "diagnosis_stage", "severity"],
    "glucose": ["glucose", "glu", "глюкоза"],
    "hemoglobin": ["hemoglobin", "hgb", "hb", "гемоглобин"],
    "population": ["population", "cohort", "center", "site", "batch"],
}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def _is_number_like(x) -> bool:
    try:
        float(str(x))
        return True
    except Exception:
        return False


def pick_spectral_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if _is_number_like(c)]
    return sorted(cols, key=lambda c: float(c))


def crop_range(df_spec: pd.DataFrame, wn_min: Optional[float], wn_max: Optional[float]) -> pd.DataFrame:
    if wn_min is None and wn_max is None:
        return df_spec
    keep = []
    for c in df_spec.columns:
        wn = float(c)
        if (wn_min is None or wn >= wn_min) and (wn_max is None or wn <= wn_max):
            keep.append(c)
    return df_spec[keep].copy()


def maybe_savgol(X: np.ndarray, win: Optional[int], poly: Optional[int], deriv: int) -> np.ndarray:
    if not win or poly is None:
        return X
    if win <= poly or win % 2 == 0 or win < 3:
        return X
    if win > X.shape[1]:
        w = X.shape[1] if X.shape[1] % 2 == 1 else X.shape[1] - 1
        if w <= poly or w < 3:
            return X
        win = w
    return savgol_filter(X, window_length=win, polyorder=poly, deriv=deriv, axis=1)


def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def msc(X: np.ndarray) -> np.ndarray:
    ref = X.mean(axis=0, keepdims=True)
    out = np.zeros_like(X)
    A = np.vstack([ref.flatten(), np.ones(ref.size)]).T
    for i in range(X.shape[0]):
        b = X[i].flatten()
        coef, intercept = np.linalg.lstsq(A, b, rcond=None)[0]
        out[i] = (X[i] - intercept) / (coef + 1e-12)
    return out


def minmax01(X: np.ndarray) -> np.ndarray:
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    return (X - mn) / np.maximum(mx - mn, 1e-12)


def read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in (".csv", ".txt"):
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suf}")


def parse_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    vals = [x.strip() for x in str(s).split(",") if x.strip()]
    return vals or None


def parse_k_range(s: str) -> List[int]:
    s = str(s).strip()
    if ":" in s:
        a, b = s.split(":")
        a, b = int(a), int(b)
        if b < a:
            a, b = b, a
        return list(range(a, b + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_merged(
    dataset: Optional[str],
    data_paths: Optional[List[str]],
    data_labels: Optional[List[str]],
    label_name: str,
) -> Tuple[pd.DataFrame, List[str], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load one dataset OR merge multiple datasets.

    Returns:
      df_all, spec_cols, X, y(optional), patient_id(optional)
    """
    if data_paths is None:
        if dataset is None:
            raise ValueError("Provide --dataset or --data-path")
        path = DEFAULT_DATASETS[dataset]
        if not path.exists():
            raise FileNotFoundError(path)
        df = read_any(path)
        spec_cols = pick_spectral_columns(df)
        if len(spec_cols) < 10:
            raise RuntimeError("No spectral columns detected (numeric column names).")

        y = None
        for c in ("y", "target", "label", "Label"):
            if c in df.columns:
                if c == "Label" and df[c].dtype == object:
                    y_tmp = df[c].astype(str).str.strip().str.lower().map({"negative": 0, "positive": 1})
                    y = y_tmp.to_numpy()
                else:
                    y = pd.to_numeric(df[c], errors="coerce").to_numpy()
                break
        if y is not None and pd.Series(y).notna().all():
            y = y.astype(int)
        else:
            y = None

        pid = df["ID"].astype(str).to_numpy() if "ID" in df.columns else None
        X = df[spec_cols].to_numpy(float)
        return df, spec_cols, X, y, pid

    paths = [Path(p) for p in data_paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    dfs = [read_any(p) for p in paths]

    # Strict intersection by exact column names (numeric wavenumbers).
    spec_sets = [set(pick_spectral_columns(d)) for d in dfs]
    spec_cols = sorted(set.intersection(*spec_sets), key=lambda c: float(c))
    if len(spec_cols) < 10:
        raise RuntimeError(
            "Too few common spectral columns across files. "
            "They likely use different wavenumber grids. Interpolate to a common grid first."
        )

    out = []
    for i, (p, d) in enumerate(zip(paths, dfs)):
        dd = d.copy()
        dd["_source_file"] = p.name
        dd["_row"] = np.arange(len(dd))
        if data_labels is not None:
            dd[label_name] = int(data_labels[i])
        out.append(dd)
    df_all = pd.concat(out, ignore_index=True)

    X = df_all[spec_cols].to_numpy(float)
    y = df_all[label_name].astype(int).to_numpy() if label_name in df_all.columns else None
    pid = df_all["ID"].astype(str).to_numpy() if "ID" in df_all.columns else None
    return df_all, spec_cols, X, y, pid


def save_scatter(path_png: Path, Z2: np.ndarray, c: np.ndarray, title: str, cmap: str = "viridis") -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=c, s=14, alpha=0.85, cmap=cmap)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    if np.issubdtype(np.asarray(c).dtype, np.number):
        plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def save_cluster_label_bars(path_png: Path, clusters: np.ndarray, y: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    df = pd.DataFrame({"cluster": clusters.astype(int), "y": y.astype(int)})
    tab = pd.crosstab(df["cluster"], df["y"], normalize="index").sort_index()
    plt.figure(figsize=(7, 4))
    tab.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title("Label distribution within clusters")
    plt.xlabel("cluster")
    plt.ylabel("fraction")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def detect_clinical_columns(df: pd.DataFrame, spec_cols: List[str], user_cols: Optional[List[str]] = None) -> List[str]:
    if user_cols:
        return [c for c in user_cols if c in df.columns and c not in spec_cols]

    meta_cols = [c for c in df.columns if c not in spec_cols]
    low_map = {str(c).lower(): c for c in meta_cols}
    chosen: List[str] = []

    for aliases in CLINICAL_ALIASES.values():
        for a in aliases:
            if a.lower() in low_map:
                chosen.append(low_map[a.lower()])
                break

    # Fallback heuristic by substring
    for c in meta_cols:
        cl = str(c).lower()
        if any(tok in cl for tok in ["age", "sex", "gender", "stage", "grade", "glucose", "hgb", "hemoglobin", "cohort", "center", "site", "batch"]):
            chosen.append(c)

    return list(dict.fromkeys(chosen))


def infer_variable_kind(series: pd.Series, colname: str) -> str:
    cl = colname.lower()
    if any(t in cl for t in ["sex", "gender", "stage", "grade", "population", "cohort", "center", "site", "batch", "group"]):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        nuniq = series.dropna().nunique()
        return "categorical" if nuniq <= 6 else "numeric"
    num = pd.to_numeric(series, errors="coerce")
    frac_num = float(num.notna().mean()) if len(series) else 0.0
    if frac_num >= 0.9 and num.dropna().nunique() > 6:
        return "numeric"
    return "categorical"


def cramers_v(tab: pd.DataFrame, chi2: float) -> float:
    n = tab.values.sum()
    if n == 0:
        return np.nan
    r, c = tab.shape
    denom = min(r - 1, c - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt((chi2 / n) / denom))


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "c": y_pred}).dropna()
    if df.empty:
        return np.nan
    ctab = pd.crosstab(df["c"], df["y"])
    return float(ctab.max(axis=1).sum() / ctab.values.sum())


def bh_fdr(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not mask.any():
        return pd.Series(out, index=pvals.index)
    pv = p[mask]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    back = np.empty_like(q)
    back[order] = q
    out[mask] = back
    return pd.Series(out, index=pvals.index)


def summarize_numeric_by_cluster(df: pd.DataFrame, cluster_col: str, var: str) -> str:
    parts = []
    for k, s in df.groupby(cluster_col)[var]:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            continue
        q1, med, q3 = s.quantile([0.25, 0.50, 0.75]).tolist()
        parts.append(f"{cluster_col}={k}: n={len(s)}, med={med:.4g}, IQR=[{q1:.4g}; {q3:.4g}]")
    return " | ".join(parts)


def cluster_clinical_associations(df_in: pd.DataFrame, cluster_cols: List[str], clinical_cols: List[str], level: str) -> pd.DataFrame:
    rows = []
    for cluster_col in cluster_cols:
        if cluster_col not in df_in.columns:
            continue
        for var in clinical_cols:
            if var not in df_in.columns:
                continue

            sub = df_in[[cluster_col, var]].dropna().copy()
            if sub.empty or sub[cluster_col].nunique() < 2:
                rows.append({
                    "level": level,
                    "cluster_col": cluster_col,
                    "variable": var,
                    "status": "skip_not_enough_clusters",
                    "n": int(len(sub)),
                })
                continue

            kind = infer_variable_kind(sub[var], var)
            if kind == "numeric":
                sub[var] = pd.to_numeric(sub[var], errors="coerce")
                sub = sub.dropna()
                groups = [g[var].to_numpy(float) for _, g in sub.groupby(cluster_col)]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) < 2:
                    rows.append({
                        "level": level,
                        "cluster_col": cluster_col,
                        "variable": var,
                        "var_kind": kind,
                        "test": "kruskal",
                        "effect_name": "epsilon_sq",
                        "status": "skip_small_groups",
                        "n": int(len(sub)),
                    })
                    continue

                H, p = stats.kruskal(*groups)
                n = sum(len(g) for g in groups)
                k = len(groups)
                eps2 = (H - k + 1) / (n - k) if n > k else np.nan
                if np.isfinite(eps2):
                    eps2 = max(0.0, float(eps2))
                rows.append({
                    "level": level,
                    "cluster_col": cluster_col,
                    "variable": var,
                    "var_kind": kind,
                    "test": "kruskal",
                    "statistic": float(H),
                    "p_value": float(p),
                    "effect_size": eps2,
                    "effect_name": "epsilon_sq",
                    "n": int(n),
                    "n_clusters": int(sub[cluster_col].nunique()),
                    "summary": summarize_numeric_by_cluster(sub, cluster_col, var),
                    "status": "ok",
                })
            else:
                tab = pd.crosstab(sub[cluster_col], sub[var])
                if tab.shape[0] < 2 or tab.shape[1] < 2:
                    rows.append({
                        "level": level,
                        "cluster_col": cluster_col,
                        "variable": var,
                        "var_kind": kind,
                        "effect_name": "cramers_v",
                        "status": "skip_constant",
                        "n": int(tab.values.sum()),
                    })
                    continue

                if tab.shape == (2, 2):
                    # Fisher p-value for sparse 2x2; Cramer's V from chi2 without correction
                    _, p = stats.fisher_exact(tab.values)
                    chi2 = stats.chi2_contingency(tab.values, correction=False)[0]
                    test_name = "fisher_exact_2x2"
                else:
                    chi2, p, _, _ = stats.chi2_contingency(tab.values)
                    test_name = "chi2"
                v = cramers_v(tab, float(chi2))

                rows.append({
                    "level": level,
                    "cluster_col": cluster_col,
                    "variable": var,
                    "var_kind": kind,
                    "test": test_name,
                    "statistic": float(chi2),
                    "p_value": float(p),
                    "effect_size": v,
                    "effect_name": "cramers_v",
                    "n": int(tab.values.sum()),
                    "n_clusters": int(tab.shape[0]),
                    "n_categories": int(tab.shape[1]),
                    "summary": tab.to_json(force_ascii=False),
                    "status": "ok",
                })

    res = pd.DataFrame(rows)
    if not res.empty and "p_value" in res.columns:
        res["q_value_bh"] = bh_fdr(res["p_value"])
    return res


def dr_clinical_associations_from_df(df_in: pd.DataFrame, clinical_cols: List[str], coord_cols: List[str], level: str) -> pd.DataFrame:
    """Association of DR coordinates (PCA/UMAP/t-SNE axes) with clinical variables.

    Note: for t-SNE/UMAP, statistical interpretation is exploratory.
    """
    rows = []
    coord_cols = [c for c in coord_cols if c in df_in.columns]
    if not coord_cols:
        return pd.DataFrame()

    for var in clinical_cols:
        if var not in df_in.columns:
            continue
        kind = infer_variable_kind(df_in[var], var)

        if kind == "numeric":
            vnum = pd.to_numeric(df_in[var], errors="coerce")
            for cc in coord_cols:
                sub = pd.DataFrame({"coord": pd.to_numeric(df_in[cc], errors="coerce"), "v": vnum}).dropna()
                if len(sub) < 5 or sub["v"].nunique() < 3:
                    rows.append({
                        "level": level,
                        "variable": var,
                        "var_kind": kind,
                        "coord": cc,
                        "test": "spearman",
                        "effect_name": "rho",
                        "status": "skip_small",
                        "n": int(len(sub)),
                    })
                    continue
                rho, p = stats.spearmanr(sub["coord"], sub["v"])
                rows.append({
                    "level": level,
                    "variable": var,
                    "var_kind": kind,
                    "coord": cc,
                    "test": "spearman",
                    "statistic": float(rho),
                    "p_value": float(p),
                    "effect_size": float(rho),
                    "effect_name": "rho",
                    "n": int(len(sub)),
                    "status": "ok",
                })
        else:
            for cc in coord_cols:
                sub = df_in[[cc, var]].dropna().copy()
                groups = [g[cc].to_numpy(float) for _, g in sub.groupby(var)]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) < 2:
                    rows.append({
                        "level": level,
                        "variable": var,
                        "var_kind": kind,
                        "coord": cc,
                        "status": "skip_small_groups",
                        "n": int(len(sub)),
                    })
                    continue
                if len(groups) == 2:
                    u, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                    effect = 1.0 - (2.0 * u) / (len(groups[0]) * len(groups[1]))  # rough rank-biserial proxy
                    rows.append({
                        "level": level,
                        "variable": var,
                        "var_kind": kind,
                        "coord": cc,
                        "test": "mannwhitney_u",
                        "statistic": float(u),
                        "p_value": float(p),
                        "effect_size": float(effect),
                        "effect_name": "rank_biserial_approx",
                        "n": int(len(sub)),
                        "status": "ok",
                    })
                else:
                    H, p = stats.kruskal(*groups)
                    n = sum(len(g) for g in groups)
                    k = len(groups)
                    eps2 = (H - k + 1) / (n - k) if n > k else np.nan
                    rows.append({
                        "level": level,
                        "variable": var,
                        "var_kind": kind,
                        "coord": cc,
                        "test": "kruskal",
                        "statistic": float(H),
                        "p_value": float(p),
                        "effect_size": float(eps2) if np.isfinite(eps2) else np.nan,
                        "effect_name": "epsilon_sq",
                        "n": int(len(sub)),
                        "status": "ok",
                    })

    res = pd.DataFrame(rows)
    if not res.empty and "p_value" in res.columns:
        res["q_value_bh"] = bh_fdr(res["p_value"])
    return res


def make_patient_level_table(df_rows: pd.DataFrame, clinical_cols: List[str]) -> pd.DataFrame:
    """Collapse repeated measurements per ID to patient-level table.

    - clusters / labels / categorical clinical vars -> majority vote
    - numeric clinical vars -> median
    - DR coords -> mean
    """
    if "ID" not in df_rows.columns:
        return pd.DataFrame()

    if df_rows["ID"].nunique() == len(df_rows):
        return df_rows.copy()  # already one row per patient

    def _mode_safe(s: pd.Series):
        m = s.mode(dropna=True)
        return m.iloc[0] if not m.empty else np.nan

    agg: Dict[str, object] = {}
    for c in df_rows.columns:
        if c == "ID":
            continue
        if c.startswith("cluster_") or c in ("y", "target", "label"):
            agg[c] = _mode_safe
        elif c in clinical_cols:
            agg[c] = "median" if infer_variable_kind(df_rows[c], c) == "numeric" else _mode_safe
        elif c.startswith(("pca2_", "umap2_", "tsne2_")):
            agg[c] = "mean"
        elif c == "_source_file":
            agg[c] = _mode_safe

    return df_rows.groupby("ID", as_index=False).agg(agg)


def attach_cluster_label_metrics(metrics_df: pd.DataFrame, labels_by_algo: Dict[Tuple[str, int], np.ndarray], y: Optional[np.ndarray]) -> pd.DataFrame:
    if y is None:
        return metrics_df

    y_arr = np.asarray(y)
    rows = []
    for _, r in metrics_df.iterrows():
        algo, k = str(r["algo"]), int(r["k"])
        lab = labels_by_algo.get((algo, k))
        if lab is None:
            rows.append({"ari": np.nan, "nmi": np.nan, "v_measure": np.nan, "purity": np.nan})
            continue

        mask = pd.Series(y_arr).notna().to_numpy()
        if mask.sum() < 3:
            rows.append({"ari": np.nan, "nmi": np.nan, "v_measure": np.nan, "purity": np.nan})
            continue

        yy = y_arr[mask].astype(int)
        ll = np.asarray(lab)[mask].astype(int)
        rows.append({
            "ari": float(adjusted_rand_score(yy, ll)),
            "nmi": float(normalized_mutual_info_score(yy, ll)),
            "v_measure": float(v_measure_score(yy, ll)),
            "purity": purity_score(yy, ll),
        })

    return pd.concat([metrics_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=list(DEFAULT_DATASETS.keys()), help="Use a default processed dataset.")
    ap.add_argument("--dataset-name", default=None, help="Output folder name (useful when merging files).")

    ap.add_argument("--data-path", default=None, help="One or many paths separated by commas.")
    ap.add_argument("--data-labels", default=None, help="Labels for each data-path (comma-separated), e.g. 0,1")
    ap.add_argument("--label-name", default="y", help="Column name for provided labels.")

    # Preprocessing (aligned with training style)
    ap.add_argument("--crop-min", type=float, default=None)
    ap.add_argument("--crop-max", type=float, default=None)
    ap.add_argument("--sg-window", type=int, default=None)
    ap.add_argument("--sg-poly", type=int, default=None)
    ap.add_argument("--sg-deriv", type=int, default=0)
    ap.add_argument("--norm", default="snv", choices=["none", "snv", "msc", "minmax"])

    # Embedding + clustering
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pca-components", type=int, default=20, help="PCA dimensions used for clustering.")
    ap.add_argument("--k-range", default="2:10", help="e.g. 2:10 or 2,3,4,5")
    ap.add_argument("--do-umap", action="store_true", help="Compute UMAP if installed.")
    ap.add_argument("--do-tsne", action="store_true", help="Compute t-SNE (exploratory).")

    # Clinical association analysis (new)
    ap.add_argument("--clinical-cols", default=None, help="Comma-separated clinical metadata columns. Auto-detect if omitted.")
    ap.add_argument("--no-clinical-assoc", action="store_true", help="Disable cluster/DR clinical association tests.")

    ap.add_argument("--out-root", default="reports/clustering", help="Root folder to save results.")
    args = ap.parse_args()

    set_seed(args.seed)

    data_paths = parse_list(args.data_path)
    data_labels = parse_list(args.data_labels)
    if data_labels is not None and data_paths is not None and len(data_labels) != len(data_paths):
        raise ValueError("--data-labels must match number of --data-path files")

    df, spec_cols, X, y, pid = load_merged(
        dataset=args.dataset,
        data_paths=data_paths,
        data_labels=data_labels,
        label_name=args.label_name,
    )

    # Crop on spectral columns
    df_spec = pd.DataFrame(X, columns=spec_cols)
    df_spec = crop_range(df_spec, args.crop_min, args.crop_max)
    spec_cols = list(df_spec.columns)
    if len(spec_cols) < 10:
        raise RuntimeError("Too few spectral columns after crop.")
    X = df_spec.to_numpy(float)
    wns = np.array([float(c) for c in spec_cols], dtype=float)

    # SG + normalization
    X = maybe_savgol(X, args.sg_window, args.sg_poly, args.sg_deriv)
    if args.norm == "snv":
        X = snv(X)
    elif args.norm == "msc":
        X = msc(X)
    elif args.norm == "minmax":
        X = minmax01(X)

    # Scale for PCA/clustering
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # Output dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = args.dataset_name or args.dataset or "custom"
    out_dir = (ROOT / args.out_root / name / ts).resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # PCA embedding
    n_pca = int(max(2, min(args.pca_components, Xs.shape[1], Xs.shape[0] - 1 if Xs.shape[0] > 2 else 2)))
    pca = PCA(n_components=n_pca, random_state=args.seed)
    Zp = pca.fit_transform(Xs)
    Z2 = Zp[:, :2]
    np.save(out_dir / "pca_Z.npy", Zp)
    np.save(out_dir / "pca2.npy", Z2)
    np.save(out_dir / "wns.npy", wns)
    with (out_dir / "pca_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"explained_var_ratio": pca.explained_variance_ratio_.tolist()}, f, ensure_ascii=False, indent=2)

    save_scatter(fig_dir / "pca2_by_label.png", Z2, (y if y is not None else np.zeros(len(Z2))), "PCA(2D): colored by label/0")

    # Optional UMAP on PCA space (faster/more stable)
    Zum = None
    if args.do_umap:
        if HAS_UMAP:
            um = umap.UMAP(n_components=2, random_state=args.seed)
            Zum = um.fit_transform(Zp)
            np.save(out_dir / "umap2.npy", Zum)
            save_scatter(fig_dir / "umap2_by_label.png", Zum, (y if y is not None else np.zeros(len(Z2))), "UMAP(2D): colored by label/0")
        else:
            print("[WARN] UMAP requested but package not installed. Skipping.")

    # Optional t-SNE (exploratory)
    Ztsne = None
    if args.do_tsne:
        if len(Zp) < 5:
            print("[WARN] t-SNE skipped: too few samples.")
        else:
            perplexity = min(30, max(2, (len(Zp) - 1) // 3))
            tsne = TSNE(
                n_components=2,
                random_state=args.seed,
                init="pca",
                learning_rate="auto",
                perplexity=perplexity,
            )
            Ztsne = tsne.fit_transform(Zp[:, : min(30, Zp.shape[1])])
            np.save(out_dir / "tsne2.npy", Ztsne)
            save_scatter(fig_dir / "tsne2_by_label.png", Ztsne, (y if y is not None else np.zeros(len(Z2))), "t-SNE(2D): colored by label/0")

    # Clustering grid search
    ks = [k for k in parse_k_range(args.k_range) if k >= 2]
    if not ks:
        raise ValueError("Empty K range")

    rows = []
    labels_by_algo: Dict[Tuple[str, int], np.ndarray] = {}
    kmeans_labels: Dict[int, np.ndarray] = {}
    gmm_labels: Dict[int, np.ndarray] = {}
    hclust_labels: Dict[int, np.ndarray] = {}

    def eval_labels(tag: str, k: int, labels: np.ndarray, extra: Optional[dict] = None) -> None:
        extra = extra or {}
        sil = np.nan
        ch = np.nan
        db = np.nan
        if len(np.unique(labels)) > 1:
            if len(Zp) > 3000:
                idx = np.random.choice(len(Zp), size=3000, replace=False)
                sil = float(silhouette_score(Zp[idx], labels[idx]))
            else:
                sil = float(silhouette_score(Zp, labels))
            ch = float(calinski_harabasz_score(Zp, labels))
            db = float(davies_bouldin_score(Zp, labels))
        rows.append({"algo": tag, "k": int(k), "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db, **extra})
        labels_by_algo[(tag, int(k))] = labels

    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
        lab = km.fit_predict(Zp)
        kmeans_labels[k] = lab
        eval_labels("kmeans", k, lab)

    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=args.seed)
        gm.fit(Zp)
        lab = gm.predict(Zp)
        gmm_labels[k] = lab
        eval_labels("gmm", k, lab, extra={"bic": float(gm.bic(Zp)), "aic": float(gm.aic(Zp))})

    for k in ks:
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab = hc.fit_predict(Zp)
        hclust_labels[k] = lab
        eval_labels("hclust", k, lab)

    metrics_df = pd.DataFrame(rows)
    metrics_df = attach_cluster_label_metrics(metrics_df, labels_by_algo, y)
    metrics_df.to_csv(out_dir / "clustering_metrics.csv", index=False)

    # Pick best k per algorithm
    def best_by_sil(dfsub: pd.DataFrame) -> int:
        xx = dfsub.copy()
        xx["silhouette"] = pd.to_numeric(xx["silhouette"], errors="coerce")
        xx = xx.sort_values(["silhouette", "k"], ascending=[False, True])
        return int(xx.iloc[0]["k"])

    best_k_km = best_by_sil(metrics_df[metrics_df["algo"] == "kmeans"])
    best_k_hc = best_by_sil(metrics_df[metrics_df["algo"] == "hclust"])
    gmm_sub = metrics_df[metrics_df["algo"] == "gmm"].copy()
    if "bic" in gmm_sub.columns and gmm_sub["bic"].notna().any():
        best_k_gm = int(gmm_sub.sort_values(["bic", "k"], ascending=[True, True]).iloc[0]["k"])
    else:
        best_k_gm = best_by_sil(gmm_sub)

    lab_km = kmeans_labels[best_k_km]
    lab_gm = gmm_labels[best_k_gm]
    lab_hc = hclust_labels[best_k_hc]

    # Save main row-level outputs
    out = pd.DataFrame({
        "idx": np.arange(len(X)),
        "cluster_kmeans": lab_km.astype(int),
        "cluster_gmm": lab_gm.astype(int),
        "cluster_hclust": lab_hc.astype(int),
    })
    out["pca2_1"] = Z2[:, 0]
    out["pca2_2"] = Z2[:, 1]
    if Zum is not None:
        out["umap2_1"] = Zum[:, 0]
        out["umap2_2"] = Zum[:, 1]
    if Ztsne is not None:
        out["tsne2_1"] = Ztsne[:, 0]
        out["tsne2_2"] = Ztsne[:, 1]

    if y is not None:
        out["y"] = y.astype(int)
    if pid is not None:
        out["ID"] = pid.astype(str)

    # Attach metadata (including clinical variables) for downstream association tests
    clinical_cols = detect_clinical_columns(df, spec_cols, parse_list(args.clinical_cols))
    meta_cols = [c for c in df.columns if c not in spec_cols]
    keep_meta = [c for c in meta_cols if c in clinical_cols or c in ("ID", args.label_name, "y", "target", "label", "Label", "_source_file")]
    keep_meta = list(dict.fromkeys(keep_meta))
    for c in keep_meta:
        if c not in out.columns:
            out[c] = df[c].values

    out.to_csv(out_dir / "clusters.csv", index=False)

    # Patient-level majority cluster file (useful when repeated measurements exist)
    if "ID" in out.columns:
        maj = out.groupby("ID")[["cluster_kmeans", "cluster_gmm", "cluster_hclust"]].agg(lambda s: int(s.value_counts().index[0]))
        maj.to_csv(out_dir / "patient_clusters_majority.csv")

    # Plots for best solutions
    save_scatter(fig_dir / f"pca2_kmeans_k{best_k_km}.png", Z2, lab_km, f"KMeans clusters (k={best_k_km}) on PCA2")
    save_scatter(fig_dir / f"pca2_gmm_k{best_k_gm}.png", Z2, lab_gm, f"GMM clusters (k={best_k_gm}) on PCA2")
    save_scatter(fig_dir / f"pca2_hclust_k{best_k_hc}.png", Z2, lab_hc, f"HClust clusters (k={best_k_hc}) on PCA2")
    if Zum is not None:
        save_scatter(fig_dir / f"umap2_kmeans_k{best_k_km}.png", Zum, lab_km, f"KMeans (k={best_k_km}) on UMAP2")
    if Ztsne is not None:
        save_scatter(fig_dir / f"tsne2_kmeans_k{best_k_km}.png", Ztsne, lab_km, f"KMeans (k={best_k_km}) on t-SNE2")
    if y is not None:
        save_cluster_label_bars(fig_dir / f"label_dist_kmeans_k{best_k_km}.png", lab_km, y)
        save_cluster_label_bars(fig_dir / f"label_dist_gmm_k{best_k_gm}.png", lab_gm, y)
        save_cluster_label_bars(fig_dir / f"label_dist_hclust_k{best_k_hc}.png", lab_hc, y)

    # Optional clinical-colored DR plots for quick interpretation (numeric only)
    for c in clinical_cols[:3]:
        if c not in df.columns:
            continue
        if infer_variable_kind(df[c], c) != "numeric":
            continue
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
        mask = np.isfinite(vals)
        if not mask.any():
            continue
        vals_fill = vals.copy()
        vals_fill[~mask] = np.nanmedian(vals[mask])
        save_scatter(fig_dir / f"pca2_by_{c}.png", Z2, vals_fill, f"PCA2 colored by {c}")
        if Zum is not None:
            save_scatter(fig_dir / f"umap2_by_{c}.png", Zum, vals_fill, f"UMAP2 colored by {c}")
        if Ztsne is not None:
            save_scatter(fig_dir / f"tsne2_by_{c}.png", Ztsne, vals_fill, f"t-SNE2 colored by {c}")

    # Clinical association analysis (sample-level + patient-level)
    assoc_meta = {
        "clinical_cols_detected": clinical_cols,
        "clinical_assoc_enabled": (not args.no_clinical_assoc),
        "note": "UMAP/t-SNE coordinate tests are exploratory; primary evidence = cluster<->clinical associations + effect sizes + FDR.",
    }
    if not args.no_clinical_assoc and clinical_cols:
        cluster_cols = ["cluster_kmeans", "cluster_gmm", "cluster_hclust"]
        coord_cols_row = [c for c in ["pca2_1", "pca2_2", "umap2_1", "umap2_2", "tsne2_1", "tsne2_2"] if c in out.columns]

        sample_cluster_assoc = cluster_clinical_associations(out, cluster_cols, clinical_cols, level="sample")
        if not sample_cluster_assoc.empty:
            sample_cluster_assoc.to_csv(out_dir / "cluster_clinical_assoc_sample.csv", index=False)

        sample_dr_assoc = dr_clinical_associations_from_df(out, clinical_cols, coord_cols_row, level="sample")
        if not sample_dr_assoc.empty:
            sample_dr_assoc.to_csv(out_dir / "dr_clinical_assoc_sample.csv", index=False)

        patient_df = make_patient_level_table(out, clinical_cols)
        if not patient_df.empty:
            patient_df.to_csv(out_dir / "patient_level_table_for_assoc.csv", index=False)
            coord_cols_pat = [c for c in ["pca2_1", "pca2_2", "umap2_1", "umap2_2", "tsne2_1", "tsne2_2"] if c in patient_df.columns]
            pat_cluster_assoc = cluster_clinical_associations(patient_df, cluster_cols, clinical_cols, level="patient")
            if not pat_cluster_assoc.empty:
                pat_cluster_assoc.to_csv(out_dir / "cluster_clinical_assoc_patient.csv", index=False)
            pat_dr_assoc = dr_clinical_associations_from_df(patient_df, clinical_cols, coord_cols_pat, level="patient")
            if not pat_dr_assoc.empty:
                pat_dr_assoc.to_csv(out_dir / "dr_clinical_assoc_patient.csv", index=False)

        # Compact top-hits summary (reviewer-friendly)
        top_frames = []
        for fname in [
            "cluster_clinical_assoc_sample.csv",
            "cluster_clinical_assoc_patient.csv",
            "dr_clinical_assoc_sample.csv",
            "dr_clinical_assoc_patient.csv",
        ]:
            fp = out_dir / fname
            if not fp.exists():
                continue
            tdf = pd.read_csv(fp)
            if "q_value_bh" in tdf.columns:
                tdf = tdf.sort_values(["q_value_bh", "p_value"], ascending=[True, True])
            elif "p_value" in tdf.columns:
                tdf = tdf.sort_values(["p_value"], ascending=[True])
            top = tdf.head(10).copy()
            top.insert(0, "source_file", fname)
            top_frames.append(top)
        if top_frames:
            pd.concat(top_frames, ignore_index=True).to_csv(out_dir / "clinical_assoc_top_hits.csv", index=False)

    # Save config
    cfg = {
        "dataset": args.dataset,
        "dataset_name": name,
        "data_path": args.data_path,
        "data_labels": args.data_labels,
        "label_name": args.label_name,
        "preprocess": {
            "crop_min": args.crop_min,
            "crop_max": args.crop_max,
            "sg_window": args.sg_window,
            "sg_poly": args.sg_poly,
            "sg_deriv": args.sg_deriv,
            "norm": args.norm,
        },
        "seed": args.seed,
        "pca_components": n_pca,
        "k_range": ks,
        "best_k": {"kmeans": best_k_km, "gmm": best_k_gm, "hclust": best_k_hc},
        **assoc_meta,
        "out_dir": str(out_dir),
    }
    with (out_dir / "cluster_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved clustering results to: {out_dir}")


if __name__ == "__main__":
    main()