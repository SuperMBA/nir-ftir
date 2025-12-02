# -*- coding: utf-8 -*-
"""
Baseline models + train-only augmentations (COVID-saliva).
Outputs: reports/exp/<timestamp>/*

Usage:
  python src/train_baselines.py --norm snv --n-splits 5 --val-size 0.2 \
    --noise 0.01 --shift 2.0 --mixup 0.4 --crop-min 900 --crop-max 1800
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
TRAIN_PARQUET = DATA_PROCESSED / "train.parquet"


# ---------- utils ----------
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


def crop_range(df: pd.DataFrame, wn_min: float | None, wn_max: float | None) -> pd.DataFrame:
    if wn_min is None and wn_max is None:
        return df
    keep = []
    for c in pick_spectral_columns(df):
        wn = float(c)
        if (wn_min is None or wn >= wn_min) and (wn_max is None or wn <= wn_max):
            keep.append(c)
    return df[keep].copy()


# ---------- augmentations ----------
def aug_noise(X: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return X.copy()
    return X + np.random.normal(0.0, scale * X.std(axis=1, keepdims=True), size=X.shape)


def aug_shift_interp(X: np.ndarray, wns: np.ndarray, max_shift_cm: float) -> np.ndarray:
    if max_shift_cm <= 0:
        return X.copy()
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        s = np.random.uniform(-max_shift_cm, max_shift_cm)
        grid = wns + s
        out[i] = np.interp(wns, grid, X[i], left=X[i, 0], right=X[i, -1])
    return out


def aug_mixup(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Mixup возвращает мягкие метки; ниже мы сразу бинаризуем."""
    if alpha <= 0:
        return X.copy(), y.copy()
    n = X.shape[0]
    idx2 = np.random.permutation(n)
    lam = np.random.beta(alpha, alpha, size=n).reshape(-1, 1)
    X_new = lam * X + (1 - lam) * X[idx2]
    y_new = lam.flatten() * y + (1 - lam.flatten()) * y[idx2]
    return X_new, y_new


def mixup_to_hard_labels(y_mix: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Склеиваем мягкие метки в жёсткие — это нужно для всех sklearn-классификаторов."""
    return (y_mix >= thr).astype(int)


# ---------- splitting ----------
def patient_level_split(
    df: pd.DataFrame, val_size: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    ids = df["ID"].astype(str) if "ID" in df.columns else pd.Series(np.arange(len(df)), name="ID")
    unique_ids = ids.drop_duplicates()
    id_tr, id_te = train_test_split(unique_ids, test_size=val_size, random_state=seed, shuffle=True)
    mask_tr = ids.isin(id_tr)
    return np.where(mask_tr)[0], np.where(~mask_tr)[0]


def iter_cv_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int = 42):
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from sgkf.split(np.zeros_like(y), y, groups)
    except Exception:
        gkf = GroupKFold(n_splits=n_splits)
        yield from gkf.split(np.zeros_like(y), y, groups)


# ---------- metrics ----------
def compute_metrics(y_true, prob) -> Dict[str, float]:
    y_pred = (prob >= 0.5).astype(int)
    return {
        "recall_pos": recall_score(y_true, y_pred, pos_label=1),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1),
        "pr_auc": average_precision_score(y_true, prob, pos_label=1),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
    }


# ---------- models ----------
def get_models():
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "svm_rbf": SVC(kernel="rbf", probability=True),
        "rf": RandomForestClassifier(n_estimators=400, random_state=42),
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
        )
    return models


# ---------- helper to build augmented train set ----------
def build_augmented_train(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    wns: np.ndarray,
    noise: float,
    shift: float,
    mixup_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (X_all, y_all_hard). Все метки — ЖЁСТКИЕ (0/1)."""
    X_parts = [Xtr]
    y_parts = [ytr.astype(int)]

    if noise > 0:
        X_parts.append(aug_noise(Xtr, noise))
        y_parts.append(ytr)

    if shift > 0:
        X_parts.append(aug_shift_interp(Xtr, wns, shift))
        y_parts.append(ytr)

    if mixup_alpha > 0:
        Xm, ym_soft = aug_mixup(Xtr, ytr, mixup_alpha)
        y_parts.append(mixup_to_hard_labels(ym_soft))  # важный фикс: только хард-метки
        X_parts.append(Xm)

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts).astype(int)
    return X_all, y_all


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="snv", choices=["none", "snv"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.01)  # 1% от sd
    ap.add_argument("--shift", type=float, default=2.0)  # см^-1
    ap.add_argument("--mixup", type=float, default=0.4)  # alpha
    ap.add_argument("--crop-min", type=float, default=None)
    ap.add_argument("--crop-max", type=float, default=None)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "reports" / "exp" / ts
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TRAIN_PARQUET)
    spec_cols = pick_spectral_columns(df)

    Xdf = df[spec_cols].copy()
    if args.crop_min is not None or args.crop_max is not None:
        Xdf = crop_range(Xdf, args.crop_min, args.crop_max)
        spec_cols = list(Xdf.columns)

    y = (
        df["y"].astype(int).values
        if "y" in df.columns
        else df["Label"].map({"Negative": 0, "Positive": 1}).astype(int).values
    )
    groups = df["ID"].astype(str).values if "ID" in df.columns else np.arange(len(df))
    wns = np.array([float(c) for c in spec_cols], dtype=float)

    # patient-level holdout
    idx_tr, idx_te = patient_level_split(df, val_size=args.val_size, seed=42)
    X_tr = Xdf.iloc[idx_tr].to_numpy(float)
    y_tr = y[idx_tr]
    g_tr = groups[idx_tr]  # используется в StratifiedGroupKFold
    X_te = Xdf.iloc[idx_te].to_numpy(float)
    y_te = y[idx_te]

    # per-sample SNV (а потом feature-wise StandardScaler по фолду)
    if args.norm == "snv":
        X_tr, X_te = snv(X_tr), snv(X_te)

    models = get_models()
    all_rows = []

    # CV на train-части (patient-level)
    for fold, (tr, va) in enumerate(iter_cv_splits(y_tr, g_tr, n_splits=args.n_splits), 1):
        Xtr, ytr = X_tr[tr], y_tr[tr]
        Xva, yva = X_tr[va], y_tr[va]

        # train-only aug → только жёсткие метки
        Xtr_all, ytr_all = build_augmented_train(
            Xtr, ytr, wns, noise=args.noise, shift=args.shift, mixup_alpha=args.mixup
        )

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = scaler.fit_transform(Xtr_all)
        Xva_s = scaler.transform(Xva)

        for name, model in models.items():
            model.fit(Xtr_s, ytr_all)  # все модели получают 0/1
            prob = (
                model.predict_proba(Xva_s)[:, 1]
                if hasattr(model, "predict_proba")
                else model.decision_function(Xva_s)
            )
            # если decision_function — приведём к [0,1] через логистическую сигмоиду
            if prob.ndim == 1 and (prob.min() < 0 or prob.max() > 1):
                prob = 1.0 / (1.0 + np.exp(-prob))
            all_rows.append({"fold": fold, "model": name, **compute_metrics(yva, prob)})

    res_df = pd.DataFrame(all_rows)
    res_df.to_csv(out_dir / "cv_metrics.csv", index=False)

    # retrain best on full train and test on held-out
    best_name = res_df.groupby("model")["roc_auc"].mean().sort_values(ascending=False).index[0]
    final_model = models[best_name]

    Xtr_all, ytr_all = build_augmented_train(
        X_tr, y_tr, wns, noise=args.noise, shift=args.shift, mixup_alpha=args.mixup
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr_all)
    Xte_s = scaler.transform(X_te)

    final_model.fit(Xtr_s, ytr_all)
    prob_te = (
        final_model.predict_proba(Xte_s)[:, 1]
        if hasattr(final_model, "predict_proba")
        else final_model.decision_function(Xte_s)
    )
    if prob_te.ndim == 1 and (prob_te.min() < 0 or prob_te.max() > 1):
        prob_te = 1.0 / (1.0 + np.exp(-prob_te))

    pd.DataFrame([{"model": best_name, **compute_metrics(y_te, prob_te)}]).to_csv(
        out_dir / "test_metrics.csv", index=False
    )

    # QC: real-vs-synthetic (должно быть близко к 0.5–0.6)
    Xsyn_parts = []
    if args.noise > 0:
        Xsyn_parts.append(aug_noise(X_tr, args.noise))
    if args.shift > 0:
        Xsyn_parts.append(aug_shift_interp(X_tr, wns, args.shift))
    if args.mixup > 0:
        Xsyn_parts.append(aug_mixup(X_tr, y_tr, args.mixup)[0])
    if Xsyn_parts:
        Xsyn = np.vstack(Xsyn_parts)
        y_syn = np.ones(Xsyn.shape[0], dtype=int)
        y_real = np.zeros(X_tr.shape[0], dtype=int)
        Xrv = np.vstack([X_tr, Xsyn])
        yrv = np.concatenate([y_real, y_syn])

        clf_rv = LogisticRegression(max_iter=1000)
        Xrv_s = scaler.fit_transform(snv(Xrv) if args.norm == "snv" else Xrv)
        clf_rv.fit(Xrv_s, yrv)
        auc_rv = roc_auc_score(yrv, clf_rv.predict_proba(Xrv_s)[:, 1])
        json.dump(
            {"real_vs_synth_auc": float(auc_rv)},
            open(out_dir / "qc_synthetic.json", "w"),
            ensure_ascii=False,
            indent=2,
        )

    cfg = dict(
        norm=args.norm,
        n_splits=args.n_splits,
        val_size=args.val_size,
        noise=args.noise,
        shift_cm=args.shift,
        mixup=args.mixup,
        crop_min=args.crop_min,
        crop_max=args.crop_max,
        best_model=best_name,
    )
    json.dump(cfg, open(out_dir / "config.json", "w"), ensure_ascii=False, indent=2)

    print(f"Done. Results -> {out_dir}")


if __name__ == "__main__":
    main()
