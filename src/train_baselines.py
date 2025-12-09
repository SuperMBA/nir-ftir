# -*- coding: utf-8 -*-
"""
FTIR baselines + train-only augmentations (p≈0.5), nested CV, calibration,
threshold selection, QC for synthetic data, and robust reporting.

Highlights:
- Modular structure (augmenters, models, QC, evaluation, reporting).
- Train-only augmentations with per-transform probability p (default 0.5).
- Recommended ranges baked-in:
  noise_med 0.5–3 %, shift ±1–4 cm⁻¹, Mixup α=0.2–0.6.
- Optional generative models (β-VAE, WGAN-GP / cGAN) — torch-optional.
- QC for synthetic: real-vs-synthetic AUC, MMD (RBF), kNN-overlap (PCA).
- Patient-level splits, LOOCV option, nested CV for model+aug search.
- Metrics: Recall/F1/ACC (positive class), PR-AUC, Balanced Acc, ROC-AUC,
  Brier/ECE; threshold selection by target recall on OOF.
- Ablations + paired Wilcoxon baseline vs augmentation on same outer folds.
- Bootstrap CIs on test, optional embeddings & SHAP.

Outputs -> reports/exp/<timestamp>/*
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# optional libs
try:
    import shap  # type: ignore[import]

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import umap  # type: ignore[import]

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE  # type: ignore[import]

    HAS_TSNE = True
except Exception:
    HAS_TSNE = False

# torch-optional generative modules
try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    import torch.optim as optim  # type: ignore[import]

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

matplotlib.use("Agg", force=True)  # headless safe


# -------------------- Paths --------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DEFAULT_DATASETS = {
    # усреднённые по ID спектры (train_avg.parquet -> train.parquet)
    "covid_saliva": DATA_PROCESSED / "train.parquet",
    # вариант: учиться на репликах (нужен group-aware CV по ID)
    "covid_saliva_repl": DATA_PROCESSED / "train_repl.parquet",
    # наш новый диабетический датасет слюны
    "diabetes_saliva": DATA_PROCESSED / "diabetes_saliva.parquet",
}


# -------------------- utils: columns & transforms --------------------


def _is_number_like(x) -> bool:
    try:
        float(str(x))
        return True
    except Exception:
        return False


def pick_spectral_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if _is_number_like(c)]
    return sorted(cols, key=lambda c: float(c))


def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def msc(X: np.ndarray) -> np.ndarray:
    """Multiplicative scatter correction (reference = mean spectrum)."""
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


def crop_range(
    df: pd.DataFrame,
    wn_min: float | None,
    wn_max: float | None,
) -> pd.DataFrame:
    if wn_min is None and wn_max is None:
        return df
    keep: List[str] = []
    for c in pick_spectral_columns(df):
        wn = float(c)
        if (wn_min is None or wn >= wn_min) and (wn_max is None or wn <= wn_max):
            keep.append(c)
    return df[keep].copy()


def maybe_savgol(
    X: np.ndarray,
    win: Optional[int],
    poly: Optional[int],
    deriv: int = 0,
) -> np.ndarray:
    """Опционально применяет SG-фильтр/производную по строкам (спектрам)."""
    if not win or not poly or win <= poly or win % 2 == 0:
        return X
    return savgol_filter(
        X,
        window_length=win,
        polyorder=poly,
        deriv=deriv,
        axis=1,
    )


# -------------------- classic augmentations --------------------


def aug_noise_std(X: np.ndarray, rel_std: float) -> np.ndarray:
    """Gaussian noise scaled by per-sample std."""
    if rel_std <= 0:
        return X.copy()
    scale = rel_std * X.std(axis=1, keepdims=True)
    return X + np.random.normal(0.0, scale, size=X.shape)


def aug_noise_med(X: np.ndarray, rel_med: float) -> np.ndarray:
    """Gaussian noise scaled by per-sample median intensity."""
    if rel_med <= 0:
        return X.copy()
    m = np.median(X, axis=1, keepdims=True)
    scale = rel_med * np.maximum(np.abs(m), 1e-8)
    return X + np.random.normal(0.0, scale, size=X.shape)


def aug_shift_interp(
    X: np.ndarray,
    wns: np.ndarray,
    max_shift_cm: float,
) -> np.ndarray:
    if max_shift_cm <= 0:
        return X.copy()
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        s = np.random.uniform(-max_shift_cm, max_shift_cm)
        grid = wns + s
        out[i] = np.interp(wns, grid, X[i], left=X[i, 0], right=X[i, -1])
    return out


def aug_mixup(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if alpha <= 0:
        return X.copy(), y.copy()
    n = X.shape[0]
    idx2 = np.random.permutation(n)
    lam = np.random.beta(alpha, alpha, size=n).reshape(-1, 1)
    X_new = lam * X + (1 - lam) * X[idx2]
    y_new = lam.flatten() * y + (1 - lam.flatten()) * y[idx2]
    return X_new, y_new


def mixup_to_hard_labels(y_mix: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (y_mix >= thr).astype(int)


# -------------------- generative augmenters (torch-optional) --------------------


class _TorchUnavailable(Exception):
    pass


def _require_torch() -> None:
    if not HAS_TORCH:
        raise _TorchUnavailable(
            "PyTorch is not available. Install torch to use generative augmenters."
        )


class VAE1D(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int, z_dim: int = 32) -> None:
        super().__init__()
        hidden = max(64, in_dim // 8)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xrec = self.dec(z)
        return xrec, mu, logvar


def train_vae(
    X: np.ndarray,
    z: int = 32,
    beta: float = 2.0,
    lr: float = 1e-3,
    steps: int = 5_000,
    seed: int = 42,
    device: Optional[str] = None,
) -> VAE1D:
    _require_torch()
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xten = torch.tensor(X, dtype=torch.float32, device=device)
    model = VAE1D(in_dim=X.shape[1], z_dim=z).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    bs = min(128, X.shape[0])

    for _ in range(steps):
        idx = torch.randint(0, Xten.shape[0], (bs,), device=device)
        x = Xten[idx]
        xrec, mu, logvar = model(x)
        rec = ((x - xrec) ** 2).mean()
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec + beta * kld
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


class WGANGP(nn.Module):  # type: ignore[misc]
    """Minimal WGAN-GP for vector spectra (not production-grade)."""

    def __init__(self, in_dim: int, z_dim: int = 64, cond_dim: int = 0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.gen = nn.Sequential(
            nn.Linear(z_dim + cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, in_dim),
        )
        self.disc = nn.Sequential(
            nn.Linear(in_dim + cond_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def sample(
        self,
        n: int,
        y: Optional[np.ndarray] = None,
        device: Optional[str] = None,
    ) -> np.ndarray:
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.z_dim, device=device)
        if self.cond_dim and y is not None:
            yten = torch.tensor(y, dtype=torch.float32, device=device).view(n, -1)
            zin = torch.cat([z, yten], dim=1)
        else:
            zin = z
        with torch.no_grad():
            x = self.gen(zin).cpu().numpy()
        return x


def train_wgan_gp(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    z: int = 64,
    lr: float = 1e-4,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    steps: int = 5_000,
    seed: int = 42,
    device: Optional[str] = None,
) -> WGANGP:
    _require_torch()
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cond_dim = 1 if y is not None else 0
    model = WGANGP(in_dim=X.shape[1], z_dim=z, cond_dim=cond_dim).to(device)
    opt_g = optim.Adam(model.gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = optim.Adam(model.disc.parameters(), lr=lr, betas=(0.5, 0.9))

    Xten = torch.tensor(X, dtype=torch.float32, device=device)
    yten = (
        torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1) if y is not None else None
    )
    bs = min(64, X.shape[0])

    def _disc(x: torch.Tensor, y_in: Optional[torch.Tensor]) -> torch.Tensor:
        if model.cond_dim:
            assert y_in is not None
            return model.disc(torch.cat([x, y_in], dim=1))
        empty = torch.zeros(x.shape[0], 0, device=x.device)
        return model.disc(torch.cat([x, empty], dim=1))

    for _ in range(steps):
        for _ in range(n_critic):
            idx = torch.randint(0, Xten.shape[0], (bs,), device=device)
            xr = Xten[idx]
            yr = yten[idx] if yten is not None else None

            z_ = torch.randn(bs, model.z_dim, device=device)
            if model.cond_dim:
                assert yr is not None
                zg = torch.cat([z_, yr], dim=1)
            else:
                zg = z_
            xg = model.gen(zg).detach()

            dr = _disc(xr, yr)
            dg = _disc(xg, yr)

            alpha = torch.rand(bs, 1, device=device)
            xhat = alpha * xr + (1 - alpha) * xg
            xhat.requires_grad_(True)
            dhat = _disc(xhat, yr)
            grads = torch.autograd.grad(
                outputs=dhat,
                inputs=xhat,
                grad_outputs=torch.ones_like(dhat),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_norm = grads.view(grads.size(0), -1).norm(2, dim=1)
            gp = ((grad_norm - 1.0) ** 2).mean()

            loss_d = -(dr.mean() - dg.mean()) + lambda_gp * gp
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        z_ = torch.randn(bs, model.z_dim, device=device)
        if model.cond_dim:
            assert yten is not None
            idx = torch.randint(0, Xten.shape[0], (bs,), device=device)
            yg = yten[idx]
            zg = torch.cat([z_, yg], dim=1)
        else:
            zg = z_
            yg = None
        xg = model.gen(zg)
        dg = _disc(xg, yg)
        loss_g = -dg.mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    return model


# -------------------- splitting --------------------


def patient_level_split(
    df: pd.DataFrame,
    val_size: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if "ID" in df.columns:
        ids = df["ID"].astype(str)
    else:
        ids = pd.Series(np.arange(len(df)), name="ID")
    unique_ids = ids.drop_duplicates()
    id_tr, id_te = train_test_split(
        unique_ids,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )
    mask_tr = ids.isin(id_tr)
    return np.where(mask_tr)[0], np.where(~mask_tr)[0]


def iter_cv_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int = 42,
):
    if n_splits <= 1:
        yield np.arange(len(y)), np.arange(0)
        return
    try:
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        yield from sgkf.split(np.zeros_like(y), y, groups)
    except Exception:
        gkf = GroupKFold(n_splits=n_splits)
        yield from gkf.split(np.zeros_like(y), y, groups)


# -------------------- metrics & calibration (robust) --------------------


def _has_both_classes(y: np.ndarray) -> bool:
    ys = np.unique(y)
    return ys.size == 2 and 0 in ys and 1 in ys


def compute_metrics_robust(
    y_true: np.ndarray,
    prob: np.ndarray,
    thr: float = 0.5,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y_pred = (prob >= thr).astype(int)
    out["recall_pos"] = float(recall_score(y_true, y_pred, pos_label=1))
    out["f1_pos"] = float(f1_score(y_true, y_pred, pos_label=1))
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["brier"] = float(brier_score_loss(y_true, prob))
    if _has_both_classes(y_true):
        out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        out["pr_auc"] = float(average_precision_score(y_true, prob, pos_label=1))
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        out["bal_acc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
    return out


def confusion_dict(
    y_true: np.ndarray,
    prob: np.ndarray,
    thr: float = 0.5,
) -> Dict[str, int]:
    y_pred = (prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (
        int(cm[0, 0]),
        int(cm[0, 1]),
        int(cm[1, 0]),
        int(cm[1, 1]),
    )
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def ece_score(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, Dict[str, list]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    bin_stats: List[Dict[str, Optional[float]]] = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            bin_stats.append({"bin": b, "count": 0, "conf": None, "acc": None})
            continue
        conf = float(prob[mask].mean())
        acc = float(y_true[mask].mean())
        w = float(mask.mean())
        ece += w * abs(acc - conf)
        bin_stats.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "conf": conf,
                "acc": acc,
            }
        )
    return float(ece), {"bins": bin_stats}


def select_threshold_by_recall(
    y_true: np.ndarray,
    prob: np.ndarray,
    recall_target: float,
) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.0, 1.0, 1001):
        pred = (prob >= t).astype(int)
        r = recall_score(y_true, pred)
        if r + 1e-9 >= recall_target:
            f1 = f1_score(y_true, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(t)
    return float(best_thr)


# -------------------- QC for synthetic data --------------------


def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """Biased MMD^2 with RBF kernel; всегда >= 0 (после обрезки)."""
    if gamma is None:
        Z = np.vstack([X[: min(len(X), 200)], Y[: min(len(Y), 200)]])
        D = cdist(Z, Z, metric="euclidean")
        med = np.median(D[D > 0])
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        gamma = 1.0 / (2.0 * med * med)

    Kxx = np.exp(-gamma * cdist(X, X) ** 2)
    Kyy = np.exp(-gamma * cdist(Y, Y) ** 2)
    Kxy = np.exp(-gamma * cdist(X, Y) ** 2)

    mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return float(max(mmd2, 0.0))


def knn_overlap_pca(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    k: int = 10,
    n_comp: int = 10,
) -> float:
    """kNN-overlap between real and synthetic in PCA space. 1.0 => identical neighborhoods."""
    pca = PCA(n_components=min(n_comp, X_real.shape[1]))
    Zr = pca.fit_transform(X_real)
    Zs = pca.transform(X_syn)

    n_neighbors = min(k, max(1, Zr.shape[0] - 1))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(Zr)

    idx_r = nbrs.kneighbors(Zr, return_distance=False)
    dists, idx_near = nbrs.kneighbors(Zs, n_neighbors=1, return_distance=True)

    overlaps: List[float] = []
    for i, ir in enumerate(idx_near.flatten()):
        local_dists = cdist(Zr[[ir]], Zr[idx_r[ir]])[0]
        thresh = float(np.mean(np.sort(local_dists)))
        overlaps.append(1.0 if dists[i, 0] <= thresh else 0.0)

    return float(np.mean(overlaps)) if overlaps else 0.0


# -------------------- models --------------------


def get_base_models(seed: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "logreg": LogisticRegression(
            max_iter=2_000,
            random_state=seed,
            class_weight="balanced",
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            probability=True,
            random_state=seed,
            class_weight="balanced",
        ),
        "rf": RandomForestClassifier(
            n_estimators=600,
            random_state=seed,
            class_weight="balanced",
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgb"] = XGBClassifier(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=None,
        )
    except Exception:
        pass
    return models


def wrap_with_calibration(model, kind: str, cv: int):
    if kind == "none":
        return model
    if kind in ("platt", "sigmoid", "auto"):
        return CalibratedClassifierCV(model, method="sigmoid", cv=cv)
    if kind == "isotonic":
        return CalibratedClassifierCV(model, method="isotonic", cv=cv)
    raise ValueError(f"Unknown calib kind: {kind}")


def prob_from_model(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    score = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-score))


# -------------------- augmentation builder --------------------


@dataclass
class AugConfig:
    noise_std: float = 0.0
    noise_med: float = 0.015  # 1.5% of median intensity
    shift: float = 2.0  # ±2 cm^-1 shift
    mixup: float = 0.4  # Beta(alpha, alpha)
    p_apply: float = 0.5  # per-transform prob


def build_augmented_train(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    wns: np.ndarray,
    aug: AugConfig,
    use_hard_labels_for_mixup: bool = True,
    extra_syn: Optional[np.ndarray] = None,
    extra_syn_labels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Собирает обучающую выборку с аугментациями.
    Каждый тип аугментации применяется с вероятностью `aug.p_apply`.

    Вся случайность идёт через глобальный np.random, который
    уже посеян в set_global_seed(cfg.seed), так что запуски
    с одним и тем же seed воспроизводимы.
    """
    X_parts: List[np.ndarray] = [Xtr]
    y_parts: List[np.ndarray] = [ytr.astype(int)]

    def maybe(flag: bool) -> bool:
        # используем глобальный генератор, посеянный set_global_seed
        return flag and (np.random.rand() < aug.p_apply)

    # --- классические аугментации ---
    if maybe(aug.noise_std > 0):
        X_parts.append(aug_noise_std(Xtr, aug.noise_std))
        y_parts.append(ytr)

    if maybe(aug.noise_med > 0):
        X_parts.append(aug_noise_med(Xtr, aug.noise_med))
        y_parts.append(ytr)

    if maybe(aug.shift > 0):
        X_parts.append(aug_shift_interp(Xtr, wns, aug.shift))
        y_parts.append(ytr)

    if maybe(aug.mixup > 0):
        Xm, ym_soft = aug_mixup(Xtr, ytr, aug.mixup)
        X_parts.append(Xm)
        if use_hard_labels_for_mixup:
            y_parts.append(mixup_to_hard_labels(ym_soft))
        else:
            y_parts.append(ym_soft)

    # --- генеративные синтетики (VAE/WGAN), если есть ---
    if extra_syn is not None and extra_syn_labels is not None and len(extra_syn) > 0:
        X_parts.append(extra_syn)
        y_parts.append(extra_syn_labels.astype(int))

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts).astype(int)
    return X_all, y_all


# -------------------- config dataclass --------------------


@dataclass
class RunConfig:
    dataset: str
    data_path: Optional[str]
    norm: str
    sg_window: Optional[int]
    sg_poly: Optional[int]
    sg_deriv: int
    n_splits: int
    loocv: bool
    nested_splits: int
    val_size: float
    seed: int
    aug: AugConfig
    search_aug: str  # fixed | auto | grid
    grid_noise_std: Optional[List[float]]
    grid_noise_med: Optional[List[float]]
    grid_shift: Optional[List[float]]
    grid_mixup: Optional[List[float]]
    p_apply: float
    select_by: str
    calib: str
    calib_cv: int
    threshold_by: str
    recall_target: float
    qc_filter: bool
    qc_synth_max: float
    qc_mmd_max: Optional[float]
    qc_knn_min: Optional[float]
    crop_min: Optional[float]
    crop_max: Optional[float]
    bootstrap: int
    use_vae: bool
    use_wgan: bool
    gan_steps: int
    vae_steps: int
    syn_mult: float
    max_train_patients: Optional[int]
    train_pos_fraction: Optional[float]


# -------------------- I/O helpers --------------------


def load_dataset(cfg: RunConfig):
    path = Path(cfg.data_path) if cfg.data_path else DEFAULT_DATASETS[cfg.dataset]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in (".csv", ".txt"):
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    spec_cols = pick_spectral_columns(df)
    Xdf = df[spec_cols].copy()

    if cfg.crop_min is not None or cfg.crop_max is not None:
        Xdf = crop_range(Xdf, cfg.crop_min, cfg.crop_max)
        spec_cols = list(Xdf.columns)

    # ---- целевая переменная
    if "y" in df.columns:
        y = df["y"].astype(int).values
    elif "target" in df.columns:  # наш диабет
        y = df["target"].astype(int).values
    elif "label" in df.columns:
        y = df["label"].astype(int).values
    elif "Label" in df.columns:
        # формат covid Excel'ей: Negative / Positive
        y = df["Label"].map({"Negative": 0, "Positive": 1}).astype(int).values
    else:
        raise ValueError("Dataset must contain column 'y', 'target' or 'Label'.")

    if "ID" in df.columns:
        groups = df["ID"].astype(str).values
    else:
        groups = np.arange(len(df))

    wns = np.array([float(c) for c in spec_cols], dtype=float)
    X = Xdf.to_numpy(float)

    if cfg.sg_window and cfg.sg_poly:
        X = maybe_savgol(X, cfg.sg_window, cfg.sg_poly, deriv=cfg.sg_deriv)

    if cfg.norm == "snv":
        X = snv(X)
    elif cfg.norm == "msc":
        X = msc(X)
    elif cfg.norm == "minmax":
        X = minmax01(X)
    elif cfg.norm == "none":
        pass
    else:
        raise ValueError(f"Unknown norm: {cfg.norm}")

    return df, spec_cols, X, y, groups, wns


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    if isinstance(s, list):
        return [float(x) for x in s]
    s = str(s).strip()
    if not s:
        return None
    return [float(x) for x in s.split(",")]


# -------------------- search spaces --------------------


def build_aug_grid(cfg: RunConfig) -> List[AugConfig]:
    grid: List[AugConfig] = []
    mode = cfg.search_aug

    if mode == "fixed":
        grid = [cfg.aug]

    elif mode == "auto":
        for nm in [0.0, 0.005, 0.015, 0.03]:
            for sh in [0.0, 1.0, 2.0, 4.0]:
                for mx in [0.0, 0.2, 0.4, 0.6]:
                    grid.append(
                        AugConfig(
                            noise_std=0.0,
                            noise_med=nm,
                            shift=sh,
                            mixup=mx,
                            p_apply=cfg.p_apply,
                        )
                    )
        for ns in [0.0, 0.005, 0.01, 0.02]:
            for sh in [0.0, 1.0, 2.0, 4.0]:
                for mx in [0.0, 0.2, 0.4, 0.6]:
                    grid.append(
                        AugConfig(
                            noise_std=ns,
                            noise_med=0.0,
                            shift=sh,
                            mixup=mx,
                            p_apply=cfg.p_apply,
                        )
                    )
        seen = set()
        uniq: List[AugConfig] = []
        for a in grid:
            key = (a.noise_std, a.noise_med, a.shift, a.mixup, a.p_apply)
            if key not in seen:
                uniq.append(a)
                seen.add(key)
        grid = uniq

    elif mode == "grid":
        ns_list = cfg.grid_noise_std or [0.0]
        nm_list = cfg.grid_noise_med or [0.0]
        sh_list = cfg.grid_shift or [0.0]
        mx_list = cfg.grid_mixup or [0.0]

        for ns in ns_list:
            for sh in sh_list:
                for mx in mx_list:
                    grid.append(
                        AugConfig(
                            noise_std=float(ns),
                            noise_med=0.0,
                            shift=float(sh),
                            mixup=float(mx),
                            p_apply=cfg.p_apply,
                        )
                    )
        for nm in nm_list:
            for sh in sh_list:
                for mx in mx_list:
                    grid.append(
                        AugConfig(
                            noise_std=0.0,
                            noise_med=float(nm),
                            shift=float(sh),
                            mixup=float(mx),
                            p_apply=cfg.p_apply,
                        )
                    )
        seen = set()
        uniq_grid: List[AugConfig] = []
        for a in grid:
            key = (a.noise_std, a.noise_med, a.shift, a.mixup, a.p_apply)
            if key not in seen:
                uniq_grid.append(a)
                seen.add(key)
        grid = uniq_grid

    else:
        raise ValueError("search_aug must be fixed|auto|grid")

    return grid


# -------------------- plotting helpers --------------------


def _save_reliability_plot(
    path_png: Path,
    y_true: np.ndarray,
    prob: np.ndarray,
    n_bins: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    ece, bins = ece_score(y_true, prob, n_bins)
    xs: List[float] = []
    ys: List[float] = []
    counts: List[int] = []

    for b in bins["bins"]:
        if b["count"] > 0:
            xs.append(b["conf"])
            ys.append(b["acc"])
            counts.append(b["count"])

    plt.figure()
    plt.plot([0, 1], [0, 1], "--")
    if xs:
        sizes = np.array(counts, dtype=float) * 10.0
        plt.scatter(xs, ys, s=sizes, alpha=0.7)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability (ECE={ece:.3f})")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


def _save_roc_pr_plots(prefix: Path, y_true: np.ndarray, prob: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if _has_both_classes(y_true):
        fpr, tpr, _ = roc_curve(y_true, prob)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC")
        plt.tight_layout()
        plt.savefig(prefix.with_name(prefix.name + "_roc.png"))
        plt.close()

    prec, rec, _ = precision_recall_curve(y_true, prob)
    plt.figure()
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.tight_layout()
    plt.savefig(prefix.with_name(prefix.name + "_pr.png"))
    plt.close()


# -------------------- embeddings & trustworthiness (optional) --------------------


def compute_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    seed: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    res: Dict[str, List[float]] = {}

    # PCA 2D
    pca = PCA(n_components=2, random_state=seed)
    Z_pca = pca.fit_transform(X)
    np.save(out_dir / "pca_2d.npy", Z_pca)
    res["pca_var"] = pca.explained_variance_ratio_[:2].tolist()

    # UMAP 2D (optional)
    if HAS_UMAP:
        um = umap.UMAP(n_components=2, random_state=seed)
        Z_umap = um.fit_transform(X)
        np.save(out_dir / "umap_2d.npy", Z_umap)

    # t-SNE 2D (optional)
    if HAS_TSNE:
        ts = TSNE(n_components=2, random_state=seed, init="pca")
        Z_tsne = ts.fit_transform(X)
        np.save(out_dir / "tsne_2d.npy", Z_tsne)

    with (out_dir / "embeddings_meta.json").open("w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


# -------------------- paired baseline vs augmented --------------------


def paired_compare_aug_vs_baseline(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    g_tr: np.ndarray,
    wns: np.ndarray,
    base_model_name: str,
    best_aug: AugConfig,
    cfg: RunConfig,
    out_dir: Path,
    n_outer: int,
) -> None:
    """Парное сравнение baseline (no-aug) vs best_aug на тех же outer folds."""
    rows: List[Dict[str, object]] = []
    splits_dump: List[Dict[str, object]] = []

    baseline_aug = AugConfig(
        noise_std=0.0,
        noise_med=0.0,
        shift=0.0,
        mixup=0.0,
        p_apply=cfg.p_apply,
    )
    base_model_proto = get_base_models(cfg.seed)[base_model_name]

    splits = list(iter_cv_splits(y_tr, g_tr, n_splits=n_outer, seed=cfg.seed))
    for fold_id, (tr_idx, va_idx) in enumerate(splits, 1):
        splits_dump.append({"fold": fold_id, "val_idx": va_idx.tolist()})
        Xtr, ytr = X_tr[tr_idx], y_tr[tr_idx]
        Xva, yva = X_tr[va_idx], y_tr[va_idx]

        for tag, aug in [("baseline", baseline_aug), ("augmented", best_aug)]:
            Xtr_aug, ytr_aug = build_augmented_train(Xtr, ytr, wns, aug)
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr_aug)
            Xva_s = scaler.transform(Xva)

            model = wrap_with_calibration(base_model_proto, cfg.calib, cfg.calib_cv)
            model.fit(Xtr_s, ytr_aug)
            prob = prob_from_model(model, Xva_s)

            metrics = compute_metrics_robust(yva, prob, thr=0.5)
            rows.append({"fold": fold_id, "setting": tag, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "paired_aug_vs_baseline.csv", index=False)

    def _paired_delta(metric: str) -> Dict[str, float]:
        piv = df.pivot(index="fold", columns="setting", values=metric)
        base = piv["baseline"].to_numpy(dtype=float)
        aug = piv["augmented"].to_numpy(dtype=float)
        mask = ~np.isnan(base) & ~np.isnan(aug)
        base, aug = base[mask], aug[mask]
        delta_abs = aug - base
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_rel = np.where(
                np.abs(base) > 1e-8,
                (aug - base) / np.abs(base),
                np.nan,
            )
        p_val = np.nan
        if len(delta_abs) >= 3 and np.all(np.isfinite(delta_abs)):
            try:
                _, p_val = wilcoxon(
                    aug,
                    base,
                    zero_method="wilcox",
                    alternative="greater",
                )
            except Exception:
                p_val = np.nan
        return {
            "n": float(len(delta_abs)),
            "mean_abs": float(np.nanmean(delta_abs)),
            "mean_rel": float(np.nanmean(delta_rel)),
            "wilcoxon_p_one_sided": float(p_val),
        }

    summary = {
        "F1_vs_baseline": _paired_delta("f1_pos"),
        "ACC_vs_baseline": _paired_delta("acc"),
        "success_rule": "mean_rel >= 0.15 и p < 0.05 (one-sided)",
    }

    with open(out_dir / "paired_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "splits_outer.json", "w", encoding="utf-8") as f:
        json.dump(splits_dump, f, ensure_ascii=False, indent=2)


# -------------------- main training logic --------------------


def main() -> None:
    warnings.filterwarnings("ignore", message="Only one class is present")
    warnings.filterwarnings("ignore", message="UndefinedMetricWarning")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="covid_saliva",
        choices=list(DEFAULT_DATASETS.keys()),
    )
    ap.add_argument(
        "--data-path",
        default=None,
        help="Path to (parquet/csv/xlsx). Overrides --dataset if set.",
    )
    ap.add_argument(
        "--norm",
        default="snv",
        choices=["none", "snv", "msc", "minmax"],
    )
    ap.add_argument(
        "--sg-window",
        type=int,
        default=None,
        help="Savitzky–Golay window (odd).",
    )
    ap.add_argument(
        "--sg-poly",
        type=int,
        default=None,
        help="Savitzky–Golay polyorder.",
    )
    ap.add_argument(
        "--sg-deriv",
        type=int,
        default=0,
        help="Savitzky–Golay derivative (0=smooth, 1=1st deriv).",
    )
    ap.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Outer CV folds on train.",
    )
    ap.add_argument(
        "--loocv",
        action="store_true",
        help="Use patient LOOCV instead of n-splits.",
    )
    ap.add_argument(
        "--nested-splits",
        type=int,
        default=0,
        help="Inner CV for model/aug search (0 => no nested).",
    )
    ap.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Patient-level hold-out fraction.",
    )
    ap.add_argument("--seed", type=int, default=42)

    # augmentations (fixed)
    ap.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Noise as fraction of per-sample std.",
    )
    ap.add_argument(
        "--noise-med",
        type=float,
        default=0.015,
        help="Noise as fraction of per-sample median (0.005–0.03 rec).",
    )
    ap.add_argument(
        "--shift",
        type=float,
        default=2.0,
        help="Max |shift| in cm^-1 (1–4 rec).",
    )
    ap.add_argument(
        "--mixup",
        type=float,
        default=0.4,
        help="Mixup alpha (0.2–0.6 rec).",
    )
    ap.add_argument(
        "--p-apply",
        type=float,
        default=0.5,
        help="Per-transform application probability.",
    )

    # search aug
    ap.add_argument(
        "--search-aug",
        default="fixed",
        choices=["fixed", "auto", "grid"],
        help="auto/grid => iterate over aug configs.",
    )
    ap.add_argument(
        "--grid-noise-std",
        default=None,
        help="Comma list, e.g., 0.0,0.005,0.01",
    )
    ap.add_argument(
        "--grid-noise-med",
        default=None,
        help="Comma list, e.g., 0.0,0.005,0.015,0.03",
    )
    ap.add_argument(
        "--grid-shift",
        default=None,
        help="Comma list, e.g., 0.0,1.0,2.0,4.0",
    )
    ap.add_argument(
        "--grid-mixup",
        default=None,
        help="Comma list, e.g., 0.0,0.2,0.4,0.6",
    )

    ap.add_argument("--crop-min", type=float, default=None)
    ap.add_argument("--crop-max", type=float, default=None)

    ap.add_argument(
        "--select-by",
        default="pr_auc",
        choices=["roc_auc", "pr_auc", "f1_pos"],
    )
    ap.add_argument(
        "--calib",
        default="none",
        choices=["none", "platt", "isotonic", "auto"],
    )
    ap.add_argument("--calib-cv", type=int, default=3)

    ap.add_argument(
        "--threshold-by",
        default="none",
        choices=["none", "recall"],
    )
    ap.add_argument("--recall-target", type=float, default=0.85)

    ap.add_argument(
        "--max-train-patients",
        type=int,
        default=None,
        help="Ограничить число уникальных ID в train (small-sample эксперименты).",
    )
    ap.add_argument(
        "--train-pos-fraction",
        type=float,
        default=None,
        help=(
            "Приблизительно задать долю позитивов в train, "
            "downsample majority-класс (например 0.2, 0.3)."
        ),
    )

    # QC synthetic
    ap.add_argument(
        "--qc-filter",
        action="store_true",
        help="Skip configs with bad synthetic QC.",
    )
    ap.add_argument(
        "--qc-synth-max",
        type=float,
        default=0.62,
        help="Max allowed AUC(real vs synth).",
    )
    ap.add_argument(
        "--qc-mmd-max",
        type=float,
        default=None,
        help="Optional: max allowed MMD^2.",
    )
    ap.add_argument(
        "--qc-knn-min",
        type=float,
        default=None,
        help="Optional: min allowed kNN-overlap (PCA).",
    )

    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="#bootstrap iters on test (0=off).",
    )

    # generative options
    ap.add_argument(
        "--use-vae",
        action="store_true",
        help="Train β-VAE on train fold and sample synthetic.",
    )
    ap.add_argument(
        "--use-wgan",
        action="store_true",
        help="Train WGAN-GP/cGAN on train fold and sample synthetic.",
    )
    ap.add_argument("--gan-steps", type=int, default=3_000)
    ap.add_argument("--vae-steps", type=int, default=3_000)
    ap.add_argument(
        "--syn-mult",
        type=float,
        default=0.5,
        help="Synthetic size relative to real (per fold).",
    )

    args = ap.parse_args()
    set_global_seed(args.seed)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "reports" / "exp" / ts
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        norm=args.norm,
        sg_window=args.sg_window,
        sg_poly=args.sg_poly,
        sg_deriv=args.sg_deriv,
        n_splits=args.n_splits,
        loocv=bool(args.loocv),
        nested_splits=args.nested_splits,
        val_size=args.val_size,
        seed=args.seed,
        aug=AugConfig(
            noise_std=args.noise_std,
            noise_med=args.noise_med,
            shift=args.shift,
            mixup=args.mixup,
            p_apply=args.p_apply,
        ),
        search_aug=args.search_aug,
        grid_noise_std=parse_float_list(args.grid_noise_std),
        grid_noise_med=parse_float_list(args.grid_noise_med),
        grid_shift=parse_float_list(args.grid_shift),
        grid_mixup=parse_float_list(args.grid_mixup),
        p_apply=args.p_apply,
        select_by=args.select_by,
        calib=args.calib,
        calib_cv=args.calib_cv,
        threshold_by=args.threshold_by,
        recall_target=args.recall_target,
        qc_filter=bool(args.qc_filter),
        qc_synth_max=float(args.qc_synth_max),
        qc_mmd_max=args.qc_mmd_max,
        qc_knn_min=args.qc_knn_min,
        crop_min=args.crop_min,
        crop_max=args.crop_max,
        bootstrap=args.bootstrap,
        use_vae=bool(args.use_vae),
        use_wgan=bool(args.use_wgan),
        gan_steps=args.gan_steps,
        vae_steps=args.vae_steps,
        syn_mult=args.syn_mult,
        max_train_patients=args.max_train_patients,
        train_pos_fraction=args.train_pos_fraction,
    )

    # ---- load data
    df, spec_cols, X_all, y_all, groups_all, wns = load_dataset(cfg)

    # ---- hold-out split by patient
    idx_tr, idx_te = patient_level_split(df, val_size=cfg.val_size, seed=cfg.seed)
    X_tr, y_tr, g_tr = X_all[idx_tr], y_all[idx_tr], groups_all[idx_tr]
    X_te, y_te = X_all[idx_te], y_all[idx_te]

    # ---- Small-sample режим: ограничить число уникальных ID в train
    if cfg.max_train_patients is not None:
        rng_limit = np.random.default_rng(cfg.seed)
        unique_ids = np.unique(g_tr)
        if cfg.max_train_patients < len(unique_ids):
            chosen = rng_limit.choice(
                unique_ids,
                size=cfg.max_train_patients,
                replace=False,
            )
            mask = np.isin(g_tr, chosen)
            X_tr, y_tr, g_tr = X_tr[mask], y_tr[mask], g_tr[mask]
            print(
                f"[small-train] using {cfg.max_train_patients} patients "
                f"({mask.sum()} spectra) in train"
            )

        # ---- Усиленный дисбаланс: приблизительная доля позитивов в train
    if cfg.train_pos_fraction is not None:
        p = cfg.train_pos_fraction
        if not (0.0 < p < 1.0):
            raise ValueError("--train-pos-fraction must be in (0, 1)")
        pos_mask = y_tr == 1
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        if n_pos == 0 or n_neg == 0:
            print("[imbalance] train has only one class, skip rebalancing")
        else:
            # сколько негативов нужно, чтобы получилось примерно p
            target_neg = int(round(n_pos * (1.0 - p) / p))
            if target_neg >= n_neg:
                print(
                    "[imbalance] not enough negatives to reach desired fraction, "
                    "keeping all samples"
                )
            else:
                rng_imb = np.random.default_rng(cfg.seed + 1)
                neg_idx = np.where(neg_mask)[0]
                keep_neg = rng_imb.choice(neg_idx, size=target_neg, replace=False)
                keep_mask = np.zeros_like(y_tr, dtype=bool)
                keep_mask[pos_mask] = True
                keep_mask[keep_neg] = True

                X_tr, y_tr, g_tr = X_tr[keep_mask], y_tr[keep_mask], g_tr[keep_mask]
                n_pos_new = int((y_tr == 1).sum())
                n_neg_new = int((y_tr == 0).sum())
                frac = n_pos_new / float(n_pos_new + n_neg_new)
                print(
                    f"[imbalance] downsampled negatives: pos={n_pos_new}, "
                    f"neg={n_neg_new}, pos_frac≈{frac:.3f}"
                )

    # ---- outer splits
    n_outer = len(np.unique(g_tr)) if cfg.loocv else cfg.n_splits
    base_models = get_base_models(seed=cfg.seed)
    candidates_aug = build_aug_grid(cfg)

    best_tuple: Optional[tuple[str, AugConfig]] = None
    best_score = -1.0
    selection_rows: List[Dict[str, object]] = []

    # ---- nested CV: model + aug search
    if cfg.nested_splits > 0:
        for outer_fold, (tr_idx, va_idx) in enumerate(
            iter_cv_splits(y_tr, g_tr, n_splits=n_outer, seed=cfg.seed),
            1,
        ):
            X_tr_o, y_tr_o = X_tr[tr_idx], y_tr[tr_idx]
            g_tr_o = g_tr[tr_idx]

            for model_name, base_model in base_models.items():
                for aug in candidates_aug:
                    if cfg.qc_filter:
                        Xsyn_parts: List[np.ndarray] = []
                        if aug.noise_std > 0:
                            Xsyn_parts.append(aug_noise_std(X_tr_o, aug.noise_std))
                        if aug.noise_med > 0:
                            Xsyn_parts.append(aug_noise_med(X_tr_o, aug.noise_med))
                        if aug.shift > 0:
                            Xsyn_parts.append(aug_shift_interp(X_tr_o, wns, aug.shift))
                        if aug.mixup > 0:
                            Xsyn_parts.append(aug_mixup(X_tr_o, y_tr_o, aug.mixup)[0])
                        if Xsyn_parts:
                            Xsyn = np.vstack(Xsyn_parts)
                            y_syn = np.ones(Xsyn.shape[0], dtype=int)
                            y_real = np.zeros(X_tr_o.shape[0], dtype=int)
                            Xrv = np.vstack([X_tr_o, Xsyn])
                            yrv = np.concatenate([y_real, y_syn])
                            sc_rv = StandardScaler()
                            clf_rv = LogisticRegression(
                                max_iter=300,
                                random_state=cfg.seed,
                            )
                            Xrv_s = sc_rv.fit_transform(Xrv)
                            clf_rv.fit(Xrv_s, yrv)
                            auc_rv = roc_auc_score(
                                yrv,
                                clf_rv.predict_proba(Xrv_s)[:, 1],
                            )
                            mmd2 = mmd_rbf(
                                X_tr_o[: min(1000, len(X_tr_o))],
                                Xsyn[: min(1000, len(Xsyn))],
                            )
                            knn_ov = knn_overlap_pca(X_tr_o, Xsyn)
                            bad = (
                                auc_rv > cfg.qc_synth_max
                                or (cfg.qc_mmd_max is not None and mmd2 > cfg.qc_mmd_max)
                                or (cfg.qc_knn_min is not None and knn_ov < cfg.qc_knn_min)
                            )
                            if bad:
                                selection_rows.append(
                                    {
                                        "outer_fold": outer_fold,
                                        "model": model_name,
                                        "noise_std": aug.noise_std,
                                        "noise_med": aug.noise_med,
                                        "shift": aug.shift,
                                        "mixup": aug.mixup,
                                        "p_apply": aug.p_apply,
                                        "score": np.nan,
                                        "qc_auc": float(auc_rv),
                                        "mmd2": float(mmd2),
                                        "knn_overlap": float(knn_ov),
                                        "skipped": True,
                                    }
                                )
                                continue

                    inner_scores: List[float] = []
                    for inner_tr, inner_va in iter_cv_splits(
                        y_tr_o,
                        g_tr_o,
                        n_splits=cfg.nested_splits,
                        seed=cfg.seed,
                    ):
                        Xtr_i, ytr_i = X_tr_o[inner_tr], y_tr_o[inner_tr]
                        Xva_i, yva_i = X_tr_o[inner_va], y_tr_o[inner_va]

                        Xsyn_g: Optional[np.ndarray] = None
                        ysyn_g: Optional[np.ndarray] = None

                        if cfg.use_vae and HAS_TORCH:
                            vae = train_vae(
                                Xtr_i,
                                z=32,
                                beta=2.0,
                                lr=1e-3,
                                steps=cfg.vae_steps,
                                seed=cfg.seed,
                            )
                            n_syn = int(cfg.syn_mult * len(Xtr_i))
                            with torch.no_grad():
                                zvec = torch.randn(
                                    n_syn,
                                    32,
                                    device=next(vae.parameters()).device,
                                )
                                Xsyn_g = vae.dec(zvec).cpu().numpy()[:n_syn]
                                ysyn_g = np.random.choice(
                                    ytr_i,
                                    size=n_syn,
                                    replace=True,
                                )

                        if cfg.use_wgan and HAS_TORCH:
                            wgan = train_wgan_gp(
                                Xtr_i,
                                y=ytr_i,
                                z=64,
                                lr=1e-4,
                                n_critic=5,
                                lambda_gp=10.0,
                                steps=cfg.gan_steps,
                                seed=cfg.seed,
                            )
                            n_syn = int(cfg.syn_mult * len(Xtr_i))
                            y_c = np.random.choice(
                                ytr_i,
                                size=n_syn,
                                replace=True,
                            )
                            Xsyn_c = wgan.sample(n_syn, y=y_c)
                            if Xsyn_g is None:
                                Xsyn_g, ysyn_g = Xsyn_c, y_c
                            else:
                                Xsyn_g = np.vstack([Xsyn_g, Xsyn_c])
                                assert ysyn_g is not None
                                ysyn_g = np.concatenate([ysyn_g, y_c])

                        Xtr_aug, ytr_aug = build_augmented_train(
                            Xtr_i,
                            ytr_i,
                            wns,
                            aug,
                            use_hard_labels_for_mixup=True,
                            extra_syn=Xsyn_g,
                            extra_syn_labels=ysyn_g,
                        )
                        scaler = StandardScaler()
                        Xtr_s = scaler.fit_transform(Xtr_aug)
                        Xva_s = scaler.transform(Xva_i)

                        model = wrap_with_calibration(
                            base_model,
                            cfg.calib,
                            cfg.calib_cv,
                        )
                        model.fit(Xtr_s, ytr_aug)
                        prob_va = prob_from_model(model, Xva_s)
                        if cfg.select_by == "pr_auc":
                            if _has_both_classes(yva_i):
                                score = average_precision_score(
                                    yva_i,
                                    prob_va,
                                )
                            else:
                                score = np.nan
                        elif cfg.select_by == "roc_auc":
                            if _has_both_classes(yva_i):
                                score = roc_auc_score(yva_i, prob_va)
                            else:
                                score = np.nan
                        else:
                            y_pred = (prob_va >= 0.5).astype(int)
                            score = f1_score(yva_i, y_pred)
                        inner_scores.append(float(score))

                    mean_score = float(np.nanmean(inner_scores))
                    selection_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "model": model_name,
                            "noise_std": aug.noise_std,
                            "noise_med": aug.noise_med,
                            "shift": aug.shift,
                            "mixup": aug.mixup,
                            "p_apply": aug.p_apply,
                            "score": mean_score,
                            "qc_auc": None,
                            "mmd2": None,
                            "knn_overlap": None,
                            "skipped": False,
                        }
                    )
                    if mean_score > best_score:
                        best_score = mean_score
                        best_tuple = (model_name, aug)

        pd.DataFrame(selection_rows).to_csv(
            out_dir / "nested_selection.csv",
            index=False,
        )
        if best_tuple is None:
            raise RuntimeError("No valid configuration found after QC filtering.")
        best_model_name, best_aug = best_tuple

    else:
        # no nested: choose model by outer CV with fixed aug
        rows_cv: List[Dict[str, object]] = []
        for fold, (tr_idx, va_idx) in enumerate(
            iter_cv_splits(y_tr, g_tr, n_splits=n_outer, seed=cfg.seed),
            1,
        ):
            Xtr, ytr = X_tr[tr_idx], y_tr[tr_idx]
            Xva, yva = X_tr[va_idx], y_tr[va_idx]

            Xsyn_g = None
            ysyn_g = None
            if cfg.use_vae and HAS_TORCH:
                vae = train_vae(
                    Xtr,
                    z=32,
                    beta=2.0,
                    lr=1e-3,
                    steps=cfg.vae_steps,
                    seed=cfg.seed,
                )
                n_syn = int(cfg.syn_mult * len(Xtr))
                with torch.no_grad():
                    zvec = torch.randn(
                        n_syn,
                        32,
                        device=next(vae.parameters()).device,
                    )
                    Xsyn_g = vae.dec(zvec).cpu().numpy()[:n_syn]
                    ysyn_g = np.random.choice(
                        ytr,
                        size=n_syn,
                        replace=True,
                    )
            if cfg.use_wgan and HAS_TORCH:
                wgan = train_wgan_gp(
                    Xtr,
                    y=ytr,
                    z=64,
                    lr=1e-4,
                    n_critic=5,
                    lambda_gp=10.0,
                    steps=cfg.gan_steps,
                    seed=cfg.seed,
                )
                n_syn = int(cfg.syn_mult * len(Xtr))
                y_c = np.random.choice(
                    ytr,
                    size=n_syn,
                    replace=True,
                )
                Xsyn_c = wgan.sample(n_syn, y=y_c)
                if Xsyn_g is None:
                    Xsyn_g, ysyn_g = Xsyn_c, y_c
                else:
                    Xsyn_g = np.vstack([Xsyn_g, Xsyn_c])
                    assert ysyn_g is not None
                    ysyn_g = np.concatenate([ysyn_g, y_c])

            Xtr_aug, ytr_aug = build_augmented_train(
                Xtr,
                ytr,
                wns,
                cfg.aug,
                use_hard_labels_for_mixup=True,
                extra_syn=Xsyn_g,
                extra_syn_labels=ysyn_g,
            )
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr_aug)
            Xva_s = scaler.transform(Xva)

            for name, base_model in base_models.items():
                model = wrap_with_calibration(
                    base_model,
                    cfg.calib,
                    cfg.calib_cv,
                )
                model.fit(Xtr_s, ytr_aug)
                p_val = prob_from_model(model, Xva_s)
                metrics = compute_metrics_robust(yva, p_val)
                rows_cv.append({"fold": fold, "model": name, **metrics})

        res_df = pd.DataFrame(rows_cv)
        res_df.to_csv(out_dir / "cv_metrics.csv", index=False)
        best_model_name = (
            res_df.groupby("model")[cfg.select_by].mean().sort_values(ascending=False).index[0]
        )
        best_aug = cfg.aug

    # ---- paired baseline vs selected aug
    paired_compare_aug_vs_baseline(
        X_tr,
        y_tr,
        g_tr,
        wns,
        best_model_name,
        best_aug,
        cfg,
        out_dir,
        n_outer,
    )

    # ---- ablation study
    ablations = [
        ("baseline", AugConfig(0.0, 0.0, 0.0, 0.0, p_apply=cfg.p_apply)),
        ("noise_std", AugConfig(0.01, 0.0, 0.0, 0.0, p_apply=cfg.p_apply)),
        ("noise_med", AugConfig(0.0, 0.005, 0.0, 0.0, p_apply=cfg.p_apply)),
        ("shift", AugConfig(0.0, 0.0, 2.0, 0.0, p_apply=cfg.p_apply)),
        ("mixup", AugConfig(0.0, 0.0, 0.0, 0.4, p_apply=cfg.p_apply)),
        ("mixup+shift", AugConfig(0.0, 0.0, 2.0, 0.4, p_apply=cfg.p_apply)),
        ("selected", best_aug),
    ]
    metric_for_wilcoxon = "f1_pos"
    per_aug_scores: Dict[str, List[float]] = {tag: [] for tag, _ in ablations}
    base_model_for_ablation = get_base_models(cfg.seed)[best_model_name]
    ablation_rows: List[Dict[str, object]] = []

    for fold, (tr_idx, va_idx) in enumerate(
        iter_cv_splits(y_tr, g_tr, n_splits=n_outer, seed=cfg.seed),
        1,
    ):
        Xtr, ytr = X_tr[tr_idx], y_tr[tr_idx]
        Xva, yva = X_tr[va_idx], y_tr[va_idx]
        for tag, aug in ablations:
            Xtr_aug, ytr_aug = build_augmented_train(Xtr, ytr, wns, aug)
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr_aug)
            Xva_s = scaler.transform(Xva)
            model = wrap_with_calibration(
                base_model_for_ablation,
                cfg.calib,
                cfg.calib_cv,
            )
            model.fit(Xtr_s, ytr_aug)
            p_val = prob_from_model(model, Xva_s)
            metrics = compute_metrics_robust(yva, p_val)
            ablation_rows.append({"fold": fold, "aug": tag, **metrics})
            per_aug_scores[tag].append(metrics[metric_for_wilcoxon])

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(out_dir / "ablation_cv.csv", index=False)

    if "baseline" in per_aug_scores:
        base_scores = np.array(per_aug_scores["baseline"], dtype=float)
        wilco_rows: List[Dict[str, object]] = []
        for tag, scores in per_aug_scores.items():
            if tag == "baseline":
                continue
            scores_arr = np.array(scores, dtype=float)
            mask = ~np.isnan(base_scores) & ~np.isnan(scores_arr)
            if mask.sum() >= 3:
                _, p_val = wilcoxon(
                    base_scores[mask],
                    scores_arr[mask],
                    zero_method="wilcox",
                    alternative="two-sided",
                )
            else:
                p_val = np.nan
            delta = float(np.nanmean(scores_arr - base_scores))
            wilco_rows.append(
                {
                    "aug": tag,
                    "metric": metric_for_wilcoxon,
                    "delta_mean": delta,
                    "wilcoxon_p": float(p_val),
                }
            )
        pd.DataFrame(wilco_rows).to_csv(
            out_dir / "ablation_wilcoxon.csv",
            index=False,
        )

    # ---- retrain best on full train & choose threshold by OOF
    Xtr_aug_full, ytr_aug_full = build_augmented_train(X_tr, y_tr, wns, best_aug)
    scaler = StandardScaler()
    Xtr_s_full = scaler.fit_transform(Xtr_aug_full)
    Xte_s = scaler.transform(X_te)

    base_model = base_models[best_model_name]
    final_model = wrap_with_calibration(
        base_model,
        cfg.calib,
        cfg.calib_cv,
    )
    final_model.fit(Xtr_s_full, ytr_aug_full)

    # OOF for threshold
    oof_prob = np.zeros_like(y_tr, dtype=float)
    for tr_idx, va_idx in iter_cv_splits(
        y_tr,
        g_tr,
        n_splits=n_outer,
        seed=cfg.seed,
    ):
        Xtr_f, ytr_f = X_tr[tr_idx], y_tr[tr_idx]
        Xva_f = X_tr[va_idx]
        Xtr_aug_f, ytr_aug_f = build_augmented_train(
            Xtr_f,
            ytr_f,
            wns,
            best_aug,
        )
        sc = StandardScaler()
        Xtr_s_f = sc.fit_transform(Xtr_aug_f)
        Xva_s_f = sc.transform(Xva_f)
        mdl = wrap_with_calibration(
            get_base_models(cfg.seed)[best_model_name],
            cfg.calib,
            cfg.calib_cv,
        )
        mdl.fit(Xtr_s_f, ytr_aug_f)
        oof_prob[va_idx] = prob_from_model(mdl, Xva_s_f)

    thr_star = 0.5
    if cfg.threshold_by == "recall":
        thr_star = select_threshold_by_recall(
            y_tr,
            oof_prob,
            cfg.recall_target,
        )

    # ---- test evaluation
    prob_te = prob_from_model(final_model, Xte_s)
    test_metrics = compute_metrics_robust(y_te, prob_te, thr=thr_star)

    pd.DataFrame([{"model": best_model_name, "thr_star": thr_star, **test_metrics}]).to_csv(
        out_dir / "test_metrics.csv", index=False
    )

    rows_test_all: List[Dict[str, object]] = []
    for name, mdl0 in base_models.items():
        mdl = wrap_with_calibration(mdl0, cfg.calib, cfg.calib_cv)
        mdl.fit(Xtr_s_full, ytr_aug_full)
        p_val = prob_from_model(mdl, Xte_s)
        m = compute_metrics_robust(y_te, p_val, thr=thr_star)
        rows_test_all.append({"model": name, **m})
    pd.DataFrame(rows_test_all).to_csv(
        out_dir / "test_metrics_all.csv",
        index=False,
    )

    # confusion
    with open(out_dir / "confusion_test.json", "w", encoding="utf-8") as f:
        json.dump(
            confusion_dict(y_te, prob_te, thr=thr_star),
            f,
            ensure_ascii=False,
            indent=2,
        )

    # calibration
    ece, cal_bins = ece_score(y_te, prob_te, n_bins=10)
    with open(out_dir / "calibration_test.json", "w", encoding="utf-8") as f:
        json.dump(
            {"ece": ece, "bins": cal_bins["bins"]},
            f,
            ensure_ascii=False,
            indent=2,
        )
    _save_reliability_plot(
        out_dir / "figures" / "reliability.png",
        y_te,
        prob_te,
        n_bins=10,
    )

    # curves
    if _has_both_classes(y_te):
        fpr, tpr, thr_roc = roc_curve(y_te, prob_te)
    else:
        fpr = np.array([])
        tpr = np.array([])
        thr_roc = np.array([])
    prec, rec, thr_pr = precision_recall_curve(y_te, prob_te)
    np.savez_compressed(
        out_dir / "curves.npz",
        fpr=fpr,
        tpr=tpr,
        thr_roc=thr_roc,
        prec=prec,
        rec=rec,
        thr_pr=thr_pr,
    )
    _save_roc_pr_plots(out_dir / "figures" / "curves", y_te, prob_te)

    # QC synthetic on full train
    Xsyn_parts: List[np.ndarray] = []
    if best_aug.noise_std > 0:
        Xsyn_parts.append(aug_noise_std(X_tr, best_aug.noise_std))
    if best_aug.noise_med > 0:
        Xsyn_parts.append(aug_noise_med(X_tr, best_aug.noise_med))
    if best_aug.shift > 0:
        Xsyn_parts.append(aug_shift_interp(X_tr, wns, best_aug.shift))
    if best_aug.mixup > 0:
        Xsyn_parts.append(aug_mixup(X_tr, y_tr, best_aug.mixup)[0])

    if Xsyn_parts:
        Xsyn = np.vstack(Xsyn_parts)
        y_syn = np.ones(Xsyn.shape[0], dtype=int)
        y_real = np.zeros(X_tr.shape[0], dtype=int)
        Xrv = np.vstack([X_tr, Xsyn])
        yrv = np.concatenate([y_real, y_syn])
        clf_rv = LogisticRegression(
            max_iter=1_000,
            random_state=cfg.seed,
        )
        sc_rv = StandardScaler()
        Xrv_s = sc_rv.fit_transform(Xrv)
        clf_rv.fit(Xrv_s, yrv)
        auc_rv = roc_auc_score(yrv, clf_rv.predict_proba(Xrv_s)[:, 1])
        mmd2 = mmd_rbf(
            X_tr[: min(2_000, len(X_tr))],
            Xsyn[: min(2_000, len(Xsyn))],
        )
        knn_ov = knn_overlap_pca(X_tr, Xsyn)
        with open(out_dir / "qc_synthetic.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "real_vs_synth_auc": float(auc_rv),
                    "mmd2": float(mmd2),
                    "knn_overlap": float(knn_ov),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # bootstrap CIs
    if cfg.bootstrap and cfg.bootstrap > 0:
        rng_boot = np.random.default_rng(cfg.seed)
        boots: List[Dict[str, float]] = []
        n_test = len(y_te)
        for _ in range(cfg.bootstrap):
            idx = rng_boot.integers(0, n_test, size=n_test)
            boots.append(
                compute_metrics_robust(
                    y_te[idx],
                    prob_te[idx],
                    thr=thr_star,
                )
            )
        boots_df = pd.DataFrame(boots)
        ci = boots_df.quantile([0.025, 0.975]).reset_index()
        ci = ci.rename(columns={"index": "q"})
        boots_df.to_csv(
            out_dir / "test_bootstrap_samples.csv",
            index=False,
        )
        ci.to_csv(out_dir / "test_bootstrap_ci95.csv", index=False)

    # embeddings
    try:
        compute_embeddings(
            X_all,
            y_all,
            out_dir / "embeddings",
            seed=cfg.seed,
        )
    except Exception:
        pass

    # SHAP (tree models only)
    if HAS_SHAP and best_model_name in ("rf", "xgb"):
        try:
            if isinstance(final_model, CalibratedClassifierCV):
                expl_base = final_model.base_estimator
            else:
                expl_base = final_model
            explainer = shap.TreeExplainer(expl_base)  # type: ignore[arg-type]
            shap_vals = explainer.shap_values(Xte_s)
            np.save(
                out_dir / "shap_values.npy",
                np.array(shap_vals, dtype=object),
            )
        except Exception:
            pass

    selected = {
        "model": best_model_name,
        "noise_std": best_aug.noise_std,
        "noise_med": best_aug.noise_med,
        "shift_cm": best_aug.shift,
        "mixup": best_aug.mixup,
        "p_apply": best_aug.p_apply,
    }
    full_cfg = {
        **asdict(cfg),
        "selected": selected,
        "thr_star": thr_star,
        "report_dir": str(out_dir),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(full_cfg, f, ensure_ascii=False, indent=2)

    print(f"Done. Results -> {out_dir}")


if __name__ == "__main__":
    main()
