#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering on PCA scores (real only) baseline vs classic_aug (PCA fitted on real+synth, evaluated on real).
Algorithms: KMeans / GMM / Agglomerative (with centroid assignment).
K=2..5, metrics: silhouette / CH / DB.
Stability: ARI across bootstrap-like resamples (subsample fits) -> labels predicted for ALL points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# make repo root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_baselines as tb  # noqa: E402


def preproc_params(profile: str):
    if profile == "paper_full":
        return dict(crop_min=870, crop_max=3400, drop="1800-2800", sg_w=25, sg_p=2, sg_d=0)
    if profile == "paper_low":
        return dict(crop_min=870, crop_max=1800, drop="", sg_w=25, sg_p=2, sg_d=0)
    if profile == "amide3":
        return dict(crop_min=1185, crop_max=1330, drop="", sg_w=25, sg_p=2, sg_d=0)
    raise ValueError(profile)


def make_classic_aug_mild_no_mixup() -> tb.AugConfig:
    # for geometry/stability we keep it “instrumental”, no mixup here
    return tb.AugConfig(
        search_aug="fixed",
        p_apply=1.0,
        noise_std=0.0,
        noise_med=0.004,
        shift=1.0,
        scale=0.004,
        tilt=0.003,
        offset=0.0015,
        mixup=0.0,      # IMPORTANT
        mixwithin=0.0,
        aug_repeats=1,
    )


def build_scores(df: pd.DataFrame, preproc_profile: str, seed: int, n_pcs: int):
    # spectral matrix
    spec_cols = tb.pick_spectral_columns(df)
    X, wn = tb.build_X_wn(df, spec_cols)

    pp = preproc_params(preproc_profile)
    drop_ranges = tb.parse_drop_ranges(pp["drop"])
    Xp, _ = tb.preprocess(
        X=X,
        wn=wn,
        crop_min=pp["crop_min"],
        crop_max=pp["crop_max"],
        sg_window=pp["sg_w"],
        sg_poly=pp["sg_p"],
        sg_deriv=pp["sg_d"],
        norm="snv",
        drop_ranges=drop_ranges,
    )

    # center only
    xsc = StandardScaler(with_mean=True, with_std=False)
    Xr = xsc.fit_transform(Xp).astype(np.float32)

    ncomp = min(n_pcs, Xr.shape[0] - 1, Xr.shape[1])

    # baseline PCA
    pca_base = PCA(n_components=ncomp, random_state=seed)
    Z_base = pca_base.fit_transform(Xr).astype(np.float64)

    # classic_aug PCA: fit PCA on (real + synth), evaluate on real
    rng = np.random.default_rng(seed)
    aug = make_classic_aug_mild_no_mixup()
    frda = tb.FrdaConfig(enabled=False, k=4, width=40, local_scale=0.02)
    y_dummy = np.zeros(len(Xr), dtype=int)
    Xa, _ = tb.build_augmented_train(Xr, y_dummy, aug=aug, rng=rng, frda=frda, frda_mask=None)

    pca_aug = PCA(n_components=ncomp, random_state=seed)
    pca_aug.fit(Xa)
    Z_aug = pca_aug.transform(Xr)

    return Z_base, Z_aug


def assign_by_centroids(Z_all: np.ndarray, idx_fit: np.ndarray, labels_fit: np.ndarray, k: int) -> np.ndarray:
    centroids = np.zeros((k, Z_all.shape[1]), dtype=float)
    for c in range(k):
        pts = Z_all[idx_fit][labels_fit == c]
        if len(pts) == 0:
            centroids[c] = 0.0
        else:
            centroids[c] = pts.mean(axis=0)

    # nearest centroid
    d2 = ((Z_all[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def run_one_algo(Z: np.ndarray, algo: str, k: int, idx_fit: np.ndarray, seed: int) -> np.ndarray:
    if algo == "kmeans":
        m = KMeans(n_clusters=k, n_init=50, random_state=seed)
        m.fit(Z[idx_fit])
        return m.predict(Z)

    if algo == "gmm":
        m = GaussianMixture(
        n_components=k,
        covariance_type="diag",   # <- ключевое для small-n
        reg_covar=1e-3,           # <- лечит неположит. определенность
        n_init=10,
        max_iter=500,
        random_state=seed,
    )
        m.fit(Z[idx_fit])
        return m.predict(Z)

    if algo == "agglo":
        # fit on subsample, then assign ALL points by nearest centroid of subsample clusters
        m = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels_fit = m.fit_predict(Z[idx_fit])
        return assign_by_centroids(Z, idx_fit, labels_fit, k)

    raise ValueError(algo)


def safe_cluster_metrics(Z: np.ndarray, labels: np.ndarray) -> dict:
    # if a cluster has 1 point, silhouette may error; catch & set NaN
    out = {}
    try:
        out["silhouette"] = float(silhouette_score(Z, labels))
    except Exception:
        out["silhouette"] = float("nan")
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(Z, labels))
    except Exception:
        out["calinski_harabasz"] = float("nan")
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(Z, labels))
    except Exception:
        out["davies_bouldin"] = float("nan")
    return out


def stability_ari(label_runs: list[np.ndarray]) -> tuple[float, float]:
    # mean/std ARI across all pairs
    if len(label_runs) < 2:
        return np.nan, np.nan
    aris = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            aris.append(adjusted_rand_score(label_runs[i], label_runs[j]))
    return float(np.mean(aris)), float(np.std(aris))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/processed/gdb_smalln.parquet")
    ap.add_argument("--outdir", default="reports/pca_r2/clustering_amide3")
    ap.add_argument("--preproc-profile", default="amide3", choices=["amide3", "paper_full", "paper_low"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pcs", type=int, default=10)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--n-runs", type=int, default=80)
    ap.add_argument("--subsample-frac", type=float, default=0.8)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data_path)
    Z_base, Z_aug = build_scores(df, args.preproc_profile, args.seed, args.n_pcs)

    rng = np.random.default_rng(args.seed)
    n = Z_base.shape[0]
    n_fit = max(2 * args.k_max, int(np.ceil(args.subsample_frac * n)))

    methods = {"baseline": Z_base, "classic_aug": Z_aug}
    algos = ["kmeans", "gmm", "agglo"]

    rows = []
    for meth, Z in methods.items():
        for algo in algos:
            for k in range(args.k_min, args.k_max + 1):
                label_runs = []
                mets = []
                for r in range(args.n_runs):
                    idx_fit = rng.choice(n, size=n_fit, replace=False)
                    labels_full = run_one_algo(Z, algo, k, idx_fit, seed=int(args.seed + r * 17 + k * 101))
                    label_runs.append(labels_full)
                    mets.append(safe_cluster_metrics(Z, labels_full))

                ari_mean, ari_std = stability_ari(label_runs)
                sil = np.array([m["silhouette"] for m in mets], float)
                ch = np.array([m["calinski_harabasz"] for m in mets], float)
                db = np.array([m["davies_bouldin"] for m in mets], float)

                rows.append(
                    dict(
                        preproc=args.preproc_profile,
                        method=meth,
                        algo=algo,
                        k=k,
                        n=args.n_runs,
                        subsample_frac=args.subsample_frac,
                        silhouette_mean=float(np.nanmean(sil)),
                        silhouette_std=float(np.nanstd(sil)),
                        ch_mean=float(np.nanmean(ch)),
                        ch_std=float(np.nanstd(ch)),
                        db_mean=float(np.nanmean(db)),
                        db_std=float(np.nanstd(db)),
                        ari_mean=ari_mean,
                        ari_std=ari_std,
                    )
                )

    out = pd.DataFrame(rows)
    out_csv = outdir / "clustering_metrics_stability.csv"
    out.to_csv(out_csv, index=False)

    # small helper: best configs per method by (ARI high, silhouette high)
    best = (
        out.sort_values(["method", "ari_mean", "silhouette_mean"], ascending=[True, False, False])
        .groupby("method", as_index=False)
        .head(5)
    )
    best_csv = outdir / "clustering_top5_per_method.csv"
    best.to_csv(best_csv, index=False)

    print("[OK] saved:", out_csv)
    print("[OK] saved:", best_csv)
    print("\nTop rows:")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()