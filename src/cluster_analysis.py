# -*- coding: utf-8 -*-
"""
Cluster analysis for FTIR / NIR-like spectral datasets.

Features:
- Load one dataset (parquet/csv/xlsx) OR merge multiple files (e.g., CONTROL + DIABETES) with labels.
- Apply the same preprocessing style as training: crop range, Savitzky-Golay, SNV/MSC/MinMax.
- Compute embeddings: PCA (always), optional UMAP if installed.
- Run clustering: KMeans, GaussianMixture, AgglomerativeClustering across a K range.
- Save metrics, cluster assignments, and simple plots to reports/clustering/...

Run examples:
1) Processed diabetes parquet:
   python -m src.cluster_analysis --dataset diabetes_saliva --crop-min 900 --crop-max 1800 --sg-window 11 --sg-poly 2 --norm snv

2) Raw CONTROL + TYPE2 DIABETES CSV merged:
   python -m src.cluster_analysis \
     --data-path "data/raw/CONTROL DATASET.csv,data/raw/TYPE 2 DIABETES DATASET.csv" \
     --data-labels "0,1" --label-name y --dataset-name diabetes_raw_merged \
     --crop-min 900 --crop-max 1800 --sg-window 11 --sg-poly 2 --norm snv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

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
    if not win or not poly:
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
    denom = np.maximum(mx - mn, 1e-12)
    return (X - mn) / denom


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
    parts = [x.strip() for x in str(s).split(",") if x.strip()]
    return parts or None


def parse_k_range(s: str) -> List[int]:
    s = str(s).strip()
    if ":" in s:
        a, b = s.split(":")
        a = int(a); b = int(b)
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
    """
    Returns:
      df_all, spec_cols, X (float), y(optional), patient_id(optional)
    """
    if data_paths is None:
        assert dataset is not None
        path = DEFAULT_DATASETS[dataset]
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")
        df = read_any(path)
        spec_cols = pick_spectral_columns(df)
        df_spec = df[spec_cols].copy()
        # optional y / ID
        y = None
        for c in ("y", "target", "label", "Label"):
            if c in df.columns:
                if c == "Label":
                    # best-effort mapping if strings
                    if df[c].dtype == object:
                        y = df[c].map({"Negative": 0, "Positive": 1}).astype(float).to_numpy()
                    else:
                        y = df[c].astype(float).to_numpy()
                else:
                    y = df[c].astype(float).to_numpy()
                break
        pid = df["ID"].astype(str).to_numpy() if "ID" in df.columns else None
        X = df_spec.to_numpy(float)
        return df, spec_cols, X, (y.astype(int) if y is not None else None), pid

    paths = [Path(p) for p in data_paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    dfs = [read_any(p) for p in paths]

    # Use intersection of spectral columns to be safe (must match across files)
    spec_sets = [set(pick_spectral_columns(d)) for d in dfs]
    spec_cols = sorted(set.intersection(*spec_sets), key=lambda c: float(c))
    if len(spec_cols) < 10:
        raise RuntimeError("Too few common spectral columns across provided files.")

    # Attach labels if provided
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

    y = None
    if label_name in df_all.columns:
        y = df_all[label_name].astype(int).to_numpy()

    pid = df_all["ID"].astype(str).to_numpy() if "ID" in df_all.columns else None
    return df_all, spec_cols, X, y, pid


def save_scatter(path_png: Path, Z2: np.ndarray, c: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=c, s=12, alpha=0.85)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


def save_cluster_label_bars(path_png: Path, clusters: np.ndarray, y: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    df = pd.DataFrame({"cluster": clusters.astype(int), "y": y.astype(int)})
    tab = pd.crosstab(df["cluster"], df["y"], normalize="index")  # rows sum to 1
    tab = tab.sort_index()
    plt.figure(figsize=(7, 4))
    tab.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title("Label distribution within clusters")
    plt.xlabel("cluster")
    plt.ylabel("fraction")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=list(DEFAULT_DATASETS.keys()), help="Use default processed dataset.")
    ap.add_argument("--dataset-name", default=None, help="Name for output folder (useful when merging files).")

    ap.add_argument("--data-path", default=None, help="One or many paths separated by commas.")
    ap.add_argument("--data-labels", default=None, help="Labels for each data-path (comma-separated), e.g. 0,1")
    ap.add_argument("--label-name", default="y", help="Column name to store provided labels (default y).")

    # preprocessing (match training style)
    ap.add_argument("--crop-min", type=float, default=None)
    ap.add_argument("--crop-max", type=float, default=None)
    ap.add_argument("--sg-window", type=int, default=None)
    ap.add_argument("--sg-poly", type=int, default=None)
    ap.add_argument("--sg-deriv", type=int, default=0)
    ap.add_argument("--norm", default="snv", choices=["none", "snv", "msc", "minmax"])

    # embedding + clustering
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pca-components", type=int, default=20, help="PCA dims for clustering metrics.")
    ap.add_argument("--k-range", default="2:10", help="e.g. 2:10 or 2,3,4,5")
    ap.add_argument("--do-umap", action="store_true", help="Compute UMAP (if installed).")

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
    X = df_spec.to_numpy(float)
    wns = np.array([float(c) for c in spec_cols], dtype=float)

    # SG + norm
    X = maybe_savgol(X, args.sg_window, args.sg_poly, args.sg_deriv)

    if args.norm == "snv":
        X = snv(X)
    elif args.norm == "msc":
        X = msc(X)
    elif args.norm == "minmax":
        X = minmax01(X)
    elif args.norm == "none":
        pass

    # Scale for clustering/PCA
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    # Output dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = args.dataset_name or args.dataset or "custom"
    out_dir = (ROOT / args.out_root / name / ts).resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # PCA
    n_pca = int(max(2, min(args.pca_components, Xs.shape[1], Xs.shape[0] - 1 if Xs.shape[0] > 2 else 2)))
    pca = PCA(n_components=n_pca, random_state=args.seed)
    Zp = pca.fit_transform(Xs)
    Z2 = Zp[:, :2]
    np.save(out_dir / "pca_Z.npy", Zp)
    np.save(out_dir / "wns.npy", wns)
    with (out_dir / "pca_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"explained_var_ratio": pca.explained_variance_ratio_.tolist()}, f, ensure_ascii=False, indent=2)

    save_scatter(fig_dir / "pca2.png", Z2, (y if y is not None else np.zeros(len(Z2))), "PCA(2D): colored by label (or zeros)")

    # Optional UMAP
    Zum = None
    if args.do_umap:
        if not HAS_UMAP:
            print("[WARN] UMAP requested but not installed. Skipping.")
        else:
            um = umap.UMAP(n_components=2, random_state=args.seed)
            Zum = um.fit_transform(Zp)  # on PCA space for speed/stability
            np.save(out_dir / "umap2.npy", Zum)

    ks = parse_k_range(args.k_range)
    ks = [k for k in ks if k >= 2]
    if not ks:
        raise ValueError("Empty K range")

    rows = []

    def eval_labels(tag: str, k: int, labels: np.ndarray, extra: dict | None = None) -> None:
        extra = extra or {}
        # Metrics can fail if single cluster; guard it.
        sil = np.nan
        ch = np.nan
        db = np.nan
        if len(np.unique(labels)) > 1:
            # silhouette can be expensive; subsample if huge
            if len(Zp) > 3000:
                idx = np.random.choice(len(Zp), size=3000, replace=False)
                sil = float(silhouette_score(Zp[idx], labels[idx]))
            else:
                sil = float(silhouette_score(Zp, labels))
            ch = float(calinski_harabasz_score(Zp, labels))
            db = float(davies_bouldin_score(Zp, labels))
        rows.append({"algo": tag, "k": int(k), "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db, **extra})

    # --- KMeans
    kmeans_labels = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
        lab = km.fit_predict(Zp)
        kmeans_labels[k] = lab
        eval_labels("kmeans", k, lab)

    # --- GMM
    gmm_labels = {}
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=args.seed)
        gm.fit(Zp)
        lab = gm.predict(Zp)
        gmm_labels[k] = lab
        eval_labels("gmm", k, lab, extra={"bic": float(gm.bic(Zp))})

    # --- Agglomerative
    hclust_labels = {}
    for k in ks:
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab = hc.fit_predict(Zp)
        hclust_labels[k] = lab
        eval_labels("hclust", k, lab)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "clustering_metrics.csv", index=False)

    # pick "best" k per algo
    def best_by_sil(dfsub: pd.DataFrame) -> int:
        dfsub = dfsub.copy()
        dfsub["silhouette"] = pd.to_numeric(dfsub["silhouette"], errors="coerce")
        dfsub = dfsub.sort_values(["silhouette", "k"], ascending=[False, True])
        return int(dfsub.iloc[0]["k"])

    best_k_km = best_by_sil(metrics_df[metrics_df["algo"] == "kmeans"])
    best_k_hc = best_by_sil(metrics_df[metrics_df["algo"] == "hclust"])

    # for GMM: choose minimal BIC if present, else best silhouette
    gmm_sub = metrics_df[metrics_df["algo"] == "gmm"].copy()
    if "bic" in gmm_sub.columns and gmm_sub["bic"].notna().any():
        best_k_gm = int(gmm_sub.sort_values(["bic", "k"], ascending=[True, True]).iloc[0]["k"])
    else:
        best_k_gm = best_by_sil(gmm_sub)

    lab_km = kmeans_labels[best_k_km]
    lab_gm = gmm_labels[best_k_gm]
    lab_hc = hclust_labels[best_k_hc]

    # Save main assignments
    out = pd.DataFrame({
        "idx": np.arange(len(X)),
        "cluster_kmeans": lab_km.astype(int),
        "cluster_gmm": lab_gm.astype(int),
        "cluster_hclust": lab_hc.astype(int),
    })
    if y is not None:
        out["y"] = y.astype(int)
    if pid is not None:
        out["ID"] = pid.astype(str)

    # Attach a few useful meta columns (non-spectral only)
    meta_cols = [c for c in df.columns if c not in spec_cols]
    # keep only small count to avoid huge csv
    keep_meta = []
    for c in meta_cols:
        if c in ("ID", args.label_name, "y", "target", "label", "Label"):
            continue
        # keep a handful typical columns if present
        if c.upper() in ("AGE", "GENDER", "HEMOGLOBIN", "GLUCOSE"):
            keep_meta.append(c)
        if c in ("_source_file",):
            keep_meta.append(c)
    keep_meta = list(dict.fromkeys(keep_meta))[:8]
    for c in keep_meta:
        out[c] = df[c].values

    out.to_csv(out_dir / "clusters.csv", index=False)

    # patient-level majority cluster (if ID exists)
    if "ID" in out.columns:
        pat = out.groupby("ID")[["cluster_kmeans", "cluster_gmm", "cluster_hclust"]].agg(lambda s: int(s.value_counts().index[0]))
        pat.to_csv(out_dir / "patient_clusters_majority.csv")

    # plots for best solutions
    save_scatter(fig_dir / f"pca2_kmeans_k{best_k_km}.png", Z2, lab_km, f"KMeans clusters (k={best_k_km}) on PCA2")
    save_scatter(fig_dir / f"pca2_gmm_k{best_k_gm}.png", Z2, lab_gm, f"GMM clusters (k={best_k_gm}) on PCA2")
    save_scatter(fig_dir / f"pca2_hclust_k{best_k_hc}.png", Z2, lab_hc, f"HClust clusters (k={best_k_hc}) on PCA2")

    if Zum is not None:
        save_scatter(fig_dir / f"umap2_kmeans_k{best_k_km}.png", Zum, lab_km, f"KMeans (k={best_k_km}) on UMAP2")

    if y is not None:
        save_cluster_label_bars(fig_dir / f"label_dist_kmeans_k{best_k_km}.png", lab_km, y)

    # write config
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
        "out_dir": str(out_dir),
    }
    with (out_dir / "cluster_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved clustering results to: {out_dir}")


if __name__ == "__main__":
    main()
