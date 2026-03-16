#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    r2_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# scipy is used for Wasserstein distance (distance distributions block)
try:
    from scipy.stats import wasserstein_distance
except Exception:
    wasserstein_distance = None

# torch for VAE / WGAN
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Reuse your existing train_baselines preprocessing/augmentation
# ============================================================
# Try src.train_baselines first (repo layout), then local train_baselines.py
tb = None
try:
    import src.train_baselines as tb  # type: ignore
except Exception:
    try:
        import train_baselines as tb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Cannot import train_baselines (src.train_baselines or train_baselines). "
            "Run from repo root or add PYTHONPATH."
        ) from e


# ============================================================
# Tiny VAE on PCA scores (not raw spectra)
# ============================================================
class TinyVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xr = self.decode(z)
        return xr, mu, logvar


def fit_vae_and_generate(
    Z_train: np.ndarray,
    n_gen: int,
    seed: int,
    epochs: int = 400,
    lr: float = 1e-3,
    beta: float = 0.02,
    hidden: int = 32,
    latent_dim: int = 4,
    device: str = "cpu",
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n, d = Z_train.shape
    latent_dim = int(max(2, min(latent_dim, d)))
    hidden = int(max(16, hidden))

    x = torch.tensor(Z_train, dtype=torch.float32, device=device)
    model = TinyVAE(in_dim=d, latent_dim=latent_dim, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(int(epochs)):
        xr, mu, logvar = model(x)
        recon = nn.functional.mse_loss(xr, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta * kl

        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        z = torch.randn(int(n_gen), latent_dim, device=device)
        out = model.decode(z).cpu().numpy().astype(np.float32)

    return out


# ============================================================
# Tiny WGAN-GP on PCA scores (small-n, conservative)
# ============================================================
class Gen(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def _gradient_penalty(D: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    bsz = real.size(0)
    eps = torch.rand(bsz, 1, device=real.device)
    eps = eps.expand_as(real)
    xhat = eps * real + (1 - eps) * fake
    xhat.requires_grad_(True)

    d_hat = D(xhat)
    grad = torch.autograd.grad(
        outputs=d_hat,
        inputs=xhat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def fit_wgan_gp_and_generate(
    Z_train: np.ndarray,
    n_gen: int,
    seed: int,
    steps: int = 1200,
    critic_steps: int = 3,
    batch_size: int = 16,
    z_dim: int = 8,
    hidden: int = 32,
    lr: float = 1e-4,
    gp_lambda: float = 10.0,
    device: str = "cpu",
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n, d = Z_train.shape
    if n < 4:
        # fallback: jitter if too small
        rng = np.random.default_rng(seed)
        return (Z_train[rng.integers(0, n, size=n_gen)] + rng.normal(0, 0.05, size=(n_gen, d))).astype(np.float32)

    G = Gen(z_dim=z_dim, out_dim=d, hidden=hidden).to(device)
    D = Critic(in_dim=d, hidden=hidden).to(device)

    opt_g = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    X = torch.tensor(Z_train, dtype=torch.float32, device=device)

    def sample_real(bs: int) -> torch.Tensor:
        idx = torch.randint(0, X.size(0), (bs,), device=device)
        return X[idx]

    G.train()
    D.train()
    bs = int(max(4, min(batch_size, n)))

    for _ in range(int(steps)):
        for _c in range(int(critic_steps)):
            real = sample_real(bs)
            z = torch.randn(bs, z_dim, device=device)
            fake = G(z).detach()

            d_real = D(real).mean()
            d_fake = D(fake).mean()
            gp = _gradient_penalty(D, real, fake)

            loss_d = (d_fake - d_real) + gp_lambda * gp
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        z = torch.randn(bs, z_dim, device=device)
        fake = G(z)
        loss_g = -D(fake).mean()

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    G.eval()
    with torch.no_grad():
        z = torch.randn(int(n_gen), z_dim, device=device)
        out = G(z).cpu().numpy().astype(np.float32)

    return out


# ============================================================
# QC metrics
# ============================================================
def real_vs_synth_auc(X_real: np.ndarray, X_synth: np.ndarray, seed: int) -> float:
    X = np.vstack([X_real, X_synth])
    y = np.concatenate([np.ones(len(X_real), dtype=int), np.zeros(len(X_synth), dtype=int)])

    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    n_splits = int(max(2, min(5, n0, n1)))
    if n_splits < 2:
        return float("nan")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for tr, te in cv.split(X, y):
        clf = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        try:
            auc = roc_auc_score(y[te], p)
            scores.append(float(auc))
        except Exception:
            continue

    return float(np.mean(scores)) if scores else float("nan")


def knn_domain_overlap(X_real: np.ndarray, X_synth: np.ndarray, k: int = 5) -> float:
    """
    Domain-mixing overlap:
    For each point in pooled set, what fraction of its kNN belongs to the opposite domain?
    Balanced ideal ~0.5. Low values => poor overlap / domain separation.
    """
    if len(X_real) < 2 or len(X_synth) < 2:
        return float("nan")

    X = np.vstack([X_real, X_synth])
    dom = np.concatenate([np.ones(len(X_real), dtype=int), np.zeros(len(X_synth), dtype=int)])

    k_eff = int(max(1, min(k, len(X) - 1)))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self

    opp = (dom[idx] != dom[:, None]).mean(axis=1)
    return float(np.mean(opp))


def distance_stats(X_real: np.ndarray, X_synth: np.ndarray) -> Dict[str, float]:
    rr = euclidean_distances(X_real, X_real)
    iu = np.triu_indices_from(rr, k=1)
    rr_vals = rr[iu]
    rs_vals = euclidean_distances(X_real, X_synth).ravel()

    out = {
        "dist_rr_median": float(np.median(rr_vals)) if rr_vals.size else float("nan"),
        "dist_rs_median": float(np.median(rs_vals)) if rs_vals.size else float("nan"),
        "dist_rr_q25": float(np.quantile(rr_vals, 0.25)) if rr_vals.size else float("nan"),
        "dist_rr_q75": float(np.quantile(rr_vals, 0.75)) if rr_vals.size else float("nan"),
        "dist_rs_q25": float(np.quantile(rs_vals, 0.25)) if rs_vals.size else float("nan"),
        "dist_rs_q75": float(np.quantile(rs_vals, 0.75)) if rs_vals.size else float("nan"),
    }
    if wasserstein_distance is not None and rr_vals.size and rs_vals.size:
        out["dist_wasserstein_rr_rs"] = float(wasserstein_distance(rr_vals, rs_vals))
    else:
        out["dist_wasserstein_rr_rs"] = float("nan")
    return out


# ============================================================
# PLS R² / Q² on binary labels (methodological PLS block)
# ============================================================
def pls_r2_q2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_comp: int = 2,
) -> Dict[str, float]:
    n_comp_eff = int(max(1, min(n_comp, X_train.shape[1], X_train.shape[0] - 1)))
    pls = PLSRegression(n_components=n_comp_eff)
    pls.fit(X_train, y_train.astype(float).reshape(-1, 1))

    yhat_tr = pls.predict(X_train).ravel()
    yhat_te = pls.predict(X_test).ravel()

    # R² on train
    try:
        r2_tr = float(r2_score(y_train, yhat_tr))
    except Exception:
        r2_tr = float("nan")

    # Q² on test (chemometrics style; baseline = mean(y_train))
    denom = float(np.sum((y_test - float(np.mean(y_train))) ** 2))
    if denom <= 1e-12:
        q2 = float("nan")
    else:
        sse = float(np.sum((y_test - yhat_te) ** 2))
        q2 = float(1.0 - sse / denom)

    # AUC/PR-AUC for interpretability (continuous PLS score)
    # map to pseudo-probability (monotonic transform)
    p_te = 1.0 / (1.0 + np.exp(-np.clip(yhat_te, -30, 30)))

    try:
        roc = float(roc_auc_score(y_test, p_te))
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_test, p_te))
    except Exception:
        pr = float("nan")

    return {
        "pls_n_components": float(n_comp_eff),
        "r2_train": r2_tr,
        "q2_test": q2,
        "roc_auc_test": roc,
        "pr_auc_test": pr,
    }


# ============================================================
# Utility: load + preprocess exactly as train_baselines
# ============================================================
def load_preprocessed(
    data_path: str,
    label_col: str,
    group_col: Optional[str],
    crop_min: float,
    crop_max: float,
    sg_window: int,
    sg_poly: int,
    sg_deriv: int,
    norm: str,
    drop_ranges: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(data_path)
    spec_cols = tb.pick_spectral_columns(df)
    if not spec_cols:
        raise ValueError("No spectral columns detected.")
    X, wn = tb.build_X_wn(df, spec_cols)

    y_series, used_label = tb.infer_label_series(df, label_col)
    groups, used_group = tb.infer_groups_series(df, group_col)

    dr = tb.parse_drop_ranges(drop_ranges)
    Xp, wnp = tb.preprocess(
        X=X,
        wn=wn,
        crop_min=crop_min,
        crop_max=crop_max,
        sg_window=sg_window,
        sg_poly=sg_poly,
        sg_deriv=sg_deriv,
        norm=norm,
        drop_ranges=dr,
    )

    y = y_series.to_numpy(dtype=int)
    g = groups.to_numpy(dtype=object) if groups is not None else np.array([str(i) for i in range(len(y))], dtype=object)
    return df, Xp.astype(np.float32), y, g


def make_classic_aug_cfg(profile: str = "mild"):
    if profile == "mild":
        return tb.AugConfig(
            search_aug="fixed",
            p_apply=1.0,           # deterministic generation for QC
            noise_std=0.0,
            noise_med=0.004,
            shift=1.0,
            scale=0.004,
            tilt=0.003,
            offset=0.0015,
            mixup=0.08,
            mixwithin=0.0,
            aug_repeats=1,
        )
    elif profile == "full":
        return tb.AugConfig(
            search_aug="fixed",
            p_apply=1.0,
            noise_std=0.0,
            noise_med=0.008,
            shift=2.0,
            scale=0.008,
            tilt=0.006,
            offset=0.003,
            mixup=0.15,
            mixwithin=0.0,
            aug_repeats=1,
        )
    else:
        raise ValueError(f"Unknown classic profile: {profile}")


def synthesize_classic(Xtr_pre: np.ndarray, ytr: np.ndarray, seed: int, profile: str) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    aug = make_classic_aug_cfg(profile=profile)
    frda = tb.FrdaConfig(enabled=False, k=4, width=40, local_scale=0.02)
    Xa, ya = tb.build_augmented_train(Xtr_pre, ytr, aug=aug, rng=rng, frda=frda, frda_mask=None)
    n0 = len(Xtr_pre)
    # only synthetic part
    if len(Xa) <= n0:
        return np.empty((0, Xtr_pre.shape[1]), dtype=np.float32), np.empty((0,), dtype=int)
    return Xa[n0:].astype(np.float32), ya[n0:].astype(int)


def synthesize_vae_or_wgan(
    Xtr_scaled: np.ndarray,
    ytr: np.ndarray,
    method: str,
    seed: int,
    pca_var: float,
    pca_max: int,
    n_synth_mult: float,
    vae_epochs: int,
    wgan_steps: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    n_train = Xtr_scaled.shape[0]
    n_gen = int(max(2, round(n_train * float(n_synth_mult))))

    # PCA compression for stable deep generative training on small-n
    max_comp = int(max(2, min(pca_max, Xtr_scaled.shape[1], n_train - 1)))
    # fit full and cut by explained variance threshold
    pca_full = PCA(n_components=max_comp, random_state=seed)
    Z_full = pca_full.fit_transform(Xtr_scaled)

    csum = np.cumsum(pca_full.explained_variance_ratio_)
    d_keep = int(np.searchsorted(csum, float(pca_var)) + 1)
    d_keep = int(max(2, min(d_keep, Z_full.shape[1])))

    pca = PCA(n_components=d_keep, random_state=seed)
    Z = pca.fit_transform(Xtr_scaled)

    if method == "vae":
        Zs = fit_vae_and_generate(
            Z_train=Z,
            n_gen=n_gen,
            seed=seed,
            epochs=vae_epochs,
            latent_dim=min(4, max(2, d_keep // 2)),
            hidden=32,
            beta=0.02,
            device=device,
        )
    elif method == "wgan":
        Zs = fit_wgan_gp_and_generate(
            Z_train=Z,
            n_gen=n_gen,
            seed=seed,
            steps=wgan_steps,
            critic_steps=3,
            batch_size=min(16, max(4, n_train)),
            z_dim=min(8, max(4, d_keep + 1)),
            hidden=32,
            device=device,
        )
    else:
        raise ValueError(method)

    Xs = pca.inverse_transform(Zs).astype(np.float32)

    # labels: balanced resampling from train labels (no label extrapolation)
    rng = np.random.default_rng(seed + 17)
    cls0 = np.where(ytr == 0)[0]
    cls1 = np.where(ytr == 1)[0]
    if len(cls0) == 0 or len(cls1) == 0:
        idx = rng.integers(0, len(ytr), size=n_gen)
        ys = ytr[idx].astype(int)
    else:
        n1 = max(1, int(round(n_gen * float(ytr.mean()))))
        n1 = min(n_gen - 1, n1)
        n0 = n_gen - n1
        idx0 = rng.choice(cls0, size=n0, replace=True)
        idx1 = rng.choice(cls1, size=n1, replace=True)
        ys = np.concatenate([ytr[idx0], ytr[idx1]]).astype(int)
        rng.shuffle(ys)

    meta = {
        "gen_pca_dims": float(d_keep),
        "gen_pca_var_cum": float(np.sum(pca.explained_variance_ratio_)),
        "n_synth": float(n_gen),
    }
    return Xs, ys, meta


def summarize_group(df: pd.DataFrame, group_cols: List[str], out_path: Path) -> pd.DataFrame:
    num_cols = [c for c in df.columns if c not in group_cols]
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {k: v for k, v in zip(group_cols, keys)}
        row["n"] = int(len(part))
        for c in num_cols:
            x = pd.to_numeric(part[c], errors="coerce")
            row[f"{c}_mean"] = float(x.mean()) if x.notna().any() else np.nan
            row[f"{c}_std"] = float(x.std(ddof=1)) if x.notna().sum() > 1 else np.nan
            row[f"{c}_median"] = float(x.median()) if x.notna().any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser("GDB small-n: AE/WGAN + QC + PLS R2/Q2")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--label-cols", default="y_parodont_H_vs_path,y_healthy_vs_any")
    ap.add_argument("--group-col", default="sample_id")
    ap.add_argument("--methods", default="baseline,classic,vae,wgan")
    ap.add_argument("--classic-profile", default="mild", choices=["mild", "full"])

    # Preprocessing aligned with train_baselines / GDB
    ap.add_argument("--crop-min", type=float, default=800)
    ap.add_argument("--crop-max", type=float, default=1800)
    ap.add_argument("--sg-window", type=int, default=11)
    ap.add_argument("--sg-poly", type=int, default=2)
    ap.add_argument("--sg-deriv", type=int, default=1)
    ap.add_argument("--norm", default="snv", choices=["snv", "none", "l2"])
    ap.add_argument("--drop-ranges", default="")
    ap.add_argument("--xscale", default="center", choices=["none", "center", "autoscale"])

    # CV
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--cv-splits", type=int, default=3)
    ap.add_argument("--cv-repeats", type=int, default=10)

    # Generators
    ap.add_argument("--n-synth-mult", type=float, default=1.0)
    ap.add_argument("--gen-pca-var", type=float, default=0.95)
    ap.add_argument("--gen-pca-max", type=int, default=8)
    ap.add_argument("--vae-epochs", type=int, default=400)
    ap.add_argument("--wgan-steps", type=int, default=1200)

    # PLS
    ap.add_argument("--pls-ncomp", type=int, default=2)

    # QC
    ap.add_argument("--knn-k", type=int, default=5)

    # Device
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # save config
    (outdir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    label_cols = [s.strip() for s in str(args.label_cols).split(",") if s.strip()]
    methods = [s.strip() for s in str(args.methods).split(",") if s.strip()]

    all_rows = []

    for label_col in label_cols:
        df, X_pre, y, groups = load_preprocessed(
            data_path=args.data_path,
            label_col=label_col,
            group_col=args.group_col,
            crop_min=args.crop_min,
            crop_max=args.crop_max,
            sg_window=args.sg_window,
            sg_poly=args.sg_poly,
            sg_deriv=args.sg_deriv,
            norm=args.norm,
            drop_ranges=args.drop_ranges,
        )

        # small-n: repeated stratified CV on real data only
        # (no leakage; synth is generated only inside train folds)
        for seed in seeds:
            rskf = RepeatedStratifiedKFold(
                n_splits=int(args.cv_splits),
                n_repeats=int(args.cv_repeats),
                random_state=seed,
            )

            for fold_id, (tr_idx, te_idx) in enumerate(rskf.split(X_pre, y)):
                Xtr_pre = X_pre[tr_idx]
                ytr = y[tr_idx]
                Xte_pre = X_pre[te_idx]
                yte = y[te_idx]

                # x-scaling fitted ONLY on real train
                scaler = tb.make_xscaler(args.xscale)
                if scaler is not None:
                    Xtr_scaled = scaler.fit_transform(Xtr_pre).astype(np.float32)
                    Xte_scaled = scaler.transform(Xte_pre).astype(np.float32)
                else:
                    Xtr_scaled = Xtr_pre.astype(np.float32, copy=True)
                    Xte_scaled = Xte_pre.astype(np.float32, copy=True)

                for method in methods:
                    qc = {
                        "real_vs_synth_auc": np.nan,
                        "knn_domain_overlap": np.nan,
                        "dist_rr_median": np.nan,
                        "dist_rs_median": np.nan,
                        "dist_rr_q25": np.nan,
                        "dist_rr_q75": np.nan,
                        "dist_rs_q25": np.nan,
                        "dist_rs_q75": np.nan,
                        "dist_wasserstein_rr_rs": np.nan,
                    }
                    gen_meta = {"gen_pca_dims": np.nan, "gen_pca_var_cum": np.nan, "n_synth": 0.0}

                    if method == "baseline":
                        Xfit = Xtr_scaled
                        yfit = ytr
                    elif method == "classic":
                        Xs_pre, ys = synthesize_classic(Xtr_pre=Xtr_pre, ytr=ytr, seed=seed + fold_id, profile=args.classic_profile)
                        if len(Xs_pre) == 0:
                            Xfit = Xtr_scaled
                            yfit = ytr
                        else:
                            if scaler is not None:
                                Xs = scaler.transform(Xs_pre).astype(np.float32)
                            else:
                                Xs = Xs_pre.astype(np.float32)

                            qc["real_vs_synth_auc"] = real_vs_synth_auc(Xtr_scaled, Xs, seed=seed + fold_id)
                            qc["knn_domain_overlap"] = knn_domain_overlap(Xtr_scaled, Xs, k=args.knn_k)
                            qc.update(distance_stats(Xtr_scaled, Xs))
                            gen_meta["n_synth"] = float(len(Xs))

                            Xfit = np.vstack([Xtr_scaled, Xs]).astype(np.float32)
                            yfit = np.concatenate([ytr, ys]).astype(int)

                    elif method in ("vae", "wgan"):
                        Xs, ys, gm = synthesize_vae_or_wgan(
                            Xtr_scaled=Xtr_scaled,
                            ytr=ytr,
                            method=method,
                            seed=seed + fold_id,
                            pca_var=args.gen_pca_var,
                            pca_max=args.gen_pca_max,
                            n_synth_mult=args.n_synth_mult,
                            vae_epochs=args.vae_epochs,
                            wgan_steps=args.wgan_steps,
                            device=device,
                        )
                        qc["real_vs_synth_auc"] = real_vs_synth_auc(Xtr_scaled, Xs, seed=seed + fold_id)
                        qc["knn_domain_overlap"] = knn_domain_overlap(Xtr_scaled, Xs, k=args.knn_k)
                        qc.update(distance_stats(Xtr_scaled, Xs))
                        gen_meta.update(gm)

                        Xfit = np.vstack([Xtr_scaled, Xs]).astype(np.float32)
                        yfit = np.concatenate([ytr, ys]).astype(int)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    pls_metrics = pls_r2_q2(
                        X_train=Xfit,
                        y_train=yfit,
                        X_test=Xte_scaled,
                        y_test=yte,
                        n_comp=int(args.pls_ncomp),
                    )

                    row = {
                        "label_col": label_col,
                        "seed": int(seed),
                        "fold_id": int(fold_id),
                        "method": method,
                        "n_train_real": int(len(Xtr_scaled)),
                        "n_test_real": int(len(Xte_scaled)),
                        "y_pos_train_real": int(np.sum(ytr)),
                        "y_pos_test_real": int(np.sum(yte)),
                        **pls_metrics,
                        **qc,
                        **gen_meta,
                    }
                    all_rows.append(row)

    per_fold = pd.DataFrame(all_rows)
    per_fold.to_csv(outdir / "per_fold_metrics.csv", index=False)

    # summaries
    summarize_group(per_fold, ["label_col", "method"], outdir / "summary_label_method.csv")
    summarize_group(per_fold, ["method"], outdir / "summary_method.csv")

    # handy "QC-only" and "PLS-only"
    qc_cols = [
        "label_col", "method", "seed", "fold_id",
        "real_vs_synth_auc", "knn_domain_overlap",
        "dist_rr_median", "dist_rs_median", "dist_wasserstein_rr_rs",
        "gen_pca_dims", "gen_pca_var_cum", "n_synth",
    ]
    pls_cols = [
        "label_col", "method", "seed", "fold_id",
        "r2_train", "q2_test", "roc_auc_test", "pr_auc_test", "pls_n_components",
    ]
    per_fold[[c for c in qc_cols if c in per_fold.columns]].to_csv(outdir / "qc_per_fold.csv", index=False)
    per_fold[[c for c in pls_cols if c in per_fold.columns]].to_csv(outdir / "pls_per_fold.csv", index=False)

    print(f"[OK] Saved: {outdir / 'per_fold_metrics.csv'}")
    print(f"[OK] Saved: {outdir / 'summary_label_method.csv'}")
    print(f"[OK] Saved: {outdir / 'summary_method.csv'}")


if __name__ == "__main__":
    main()