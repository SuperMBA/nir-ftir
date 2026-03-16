# -*- coding: utf-8 -*-
"""
src/train_baselines.py

Robust training/evaluation for FTIR/NIR spectra with:
- Leakage-safe group splits (for replicate datasets like COVID train.parquet)
- Optional meta-stratification & meta-residualization (for Diabetes confounding control)
- Two evaluation protocols:
    1) cv_holdout: single holdout test + inner CV on train (threshold selection)
    2) mcdcv_plsda: repeated Monte-Carlo holdout (mc-iter times) + inner CV;
       includes optional PLS-DA LV selection (closer to Matlab-like PLS-DA workflows)

Works with your processed parquets:
- data/processed/train.parquet  (COVID: 61 IDs x 3 reps)
- data/processed/diabetes_saliva.parquet (1040, sample-level)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except Exception:
    StratifiedGroupKFold = None  # type: ignore


# ----------------------------
# Small utils
# ----------------------------
def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random as _random

        _random.seed(seed)
    except Exception:
        pass


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)


def _sanitize_tag(tag: str) -> str:
    t = (tag or "").strip()
    if not t:
        return "run"
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^A-Za-z0-9_\-]+", "", t)
    return t[:80] if t else "run"


def _short_hash(obj: dict) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.md5(s).hexdigest()[:10]


def _is_number_like_name(col_name: str) -> bool:
    try:
        float(str(col_name).strip())
        return True
    except Exception:
        return False


def pick_spectral_columns(df: pd.DataFrame) -> List[str]:
    spec_cols = [c for c in df.columns if _is_number_like_name(str(c))]
    return sorted(spec_cols, key=lambda c: float(str(c).strip()))


def build_X_wn(df: pd.DataFrame, spec_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    Xdf = df[spec_cols].apply(pd.to_numeric, errors="coerce")
    if Xdf.isna().any().any():
        nan_rows = int(Xdf.isna().any(axis=1).sum())
        raise ValueError(f"Spectral matrix has NaNs in {nan_rows} rows. Fix preprocessing/loading.")
    X = Xdf.to_numpy(dtype=np.float32)
    wn = np.asarray([float(str(c).strip()) for c in spec_cols], dtype=float)
    return X, wn


def infer_label_series(df: pd.DataFrame, label_col: Optional[str] = None) -> Tuple[pd.Series, str]:
    if label_col and label_col in df.columns:
        col = label_col
    else:
        for c in ("y", "target", "label", "class", "Population", "population"):
            if c in df.columns:
                col = c
                break
        else:
            raise ValueError("Label column not found. Expected one of: y/target/label/class/Population.")

    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        y = pd.to_numeric(s, errors="coerce").astype("Int64")
    else:
        v = s.astype(str).str.strip().str.upper()
        v = v.replace({"POSITIVE": "1", "NEGATIVE": "0", "DIABETES": "1", "CONTROL": "0"})
        v = v.replace({"P": "1", "N": "0"})
        y = pd.to_numeric(v, errors="coerce").astype("Int64")

    if y.isna().any():
        bad = int(y.isna().sum())
        raise ValueError(f"Label column '{col}' has {bad} NaN after conversion. Fix mapping or pass --label-col.")

    return y.astype(int), col


def infer_groups_series(df: pd.DataFrame, group_col: Optional[str]) -> Tuple[Optional[pd.Series], Optional[str]]:
    if group_col and group_col in df.columns:
        return df[group_col].astype(str), group_col

    # prefer ID-like fields
    for c in ("ID", "sample_id", "patient_id", "subject_id", "patient", "sample", "id"):
        if c in df.columns:
            return df[c].astype(str), c
    return None, None


def maybe_average_replicates(df: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series], spec_cols: List[str]) -> pd.DataFrame:
    if groups is None:
        return df

    gname = groups.name or "group"
    tmp = df.copy()
    tmp[gname] = groups.values
    tmp["_y_"] = y.values

    # label consistency within group
    bad = tmp.groupby(gname)["_y_"].nunique()
    bad = bad[bad > 1]
    if len(bad) > 0:
        raise ValueError(f"Found {len(bad)} groups with inconsistent labels. Fix IDs/labels.")

    agg = tmp.groupby([gname, "_y_"], as_index=False)[spec_cols].mean()
    agg = agg.rename(columns={gname: groups.name or "ID", "_y_": "y"})
    return agg


# ----------------------------
# Preprocessing
# ----------------------------
def crop_by_wavenumber(X: np.ndarray, wn: np.ndarray, crop_min: float, crop_max: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (wn >= float(crop_min)) & (wn <= float(crop_max))
    if not mask.any():
        raise ValueError(f"Crop range [{crop_min}, {crop_max}] empty. wn min={wn.min():.3f}, max={wn.max():.3f}.")
    return X[:, mask], wn[mask]


def parse_drop_ranges(s: str) -> List[Tuple[float, float]]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[Tuple[float, float]] = []
    for p in [x.strip() for x in s.split(",") if x.strip()]:
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$", p)
        if not m:
            raise ValueError(f"Bad drop range '{p}'. Expected like '1800-1900,2350-2450'.")
        lo, hi = float(m.group(1)), float(m.group(2))
        if hi < lo:
            lo, hi = hi, lo
        out.append((lo, hi))
    return out


def apply_drop_ranges(wn: np.ndarray, drop_ranges: List[Tuple[float, float]]) -> np.ndarray:
    mask = np.ones_like(wn, dtype=bool)
    for lo, hi in drop_ranges:
        mask &= ~((wn >= lo) & (wn <= hi))
    return mask


def apply_savgol(X: np.ndarray, window: int, poly: int, deriv: int) -> np.ndarray:
    if window is None or int(window) <= 0:
        return X.astype(np.float32, copy=False)
    n = X.shape[1]
    w = int(window)
    if w >= n:
        w = n - 1
    if w < 3:
        return X.astype(np.float32, copy=False)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        w = 3
    p = int(poly)
    p = max(1, min(p, w - 1))
    d = int(deriv)
    d = max(0, min(d, p))
    return savgol_filter(X, window_length=w, polyorder=p, deriv=d, axis=1).astype(np.float32)


def norm_snv(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return ((X - mu) / (sd + eps)).astype(np.float32)


def norm_l2(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / (nrm + eps)).astype(np.float32)


def preprocess(
    X: np.ndarray,
    wn: np.ndarray,
    crop_min: float,
    crop_max: float,
    sg_window: int,
    sg_poly: int,
    sg_deriv: int,
    norm: str,
    drop_ranges: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    Xc, wnc = crop_by_wavenumber(X, wn, crop_min=crop_min, crop_max=crop_max)
    if drop_ranges:
        keep = apply_drop_ranges(wnc, drop_ranges)
        if int(keep.sum()) < 10:
            raise ValueError("After drop-ranges too few points left. Check --drop-ranges.")
        Xc = Xc[:, keep]
        wnc = wnc[keep]
    Xf = apply_savgol(Xc, window=sg_window, poly=sg_poly, deriv=sg_deriv)
    nm = (norm or "none").lower().strip()
    if nm == "snv":
        Xf = norm_snv(Xf)
    elif nm == "l2":
        Xf = norm_l2(Xf)
    elif nm in ("none", ""):
        Xf = Xf.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown norm: {norm}")
    return Xf, wnc


def make_xscaler(mode: str) -> Optional[StandardScaler]:
    m = (mode or "none").lower().strip()
    if m == "none":
        return None
    if m == "center":
        return StandardScaler(with_mean=True, with_std=False)
    if m == "autoscale":
        return StandardScaler(with_mean=True, with_std=True)
    raise ValueError("xscale must be one of: none, center, autoscale")


# ----------------------------
# Meta stratification (AGE/GENDER) for Diabetes
# ----------------------------
def _infer_meta_cols(df: pd.DataFrame, cols_csv: Optional[str]) -> List[str]:
    if cols_csv:
        cols = [c.strip() for c in cols_csv.split(",") if c.strip()]
        return [c for c in cols if c in df.columns]
    out: List[str] = []
    for c in ("AGE", "age"):
        if c in df.columns:
            out.append(c)
            break
    for c in ("GENDER", "gender", "Sex", "SEX"):
        if c in df.columns:
            out.append(c)
            break
    return out


def _meta_to_categories(df: pd.DataFrame, cols: List[str], age_bins: int, seed: int) -> pd.Series:
    if not cols:
        return pd.Series(["_"] * len(df), index=df.index)

    rng = np.random.default_rng(seed)
    parts: List[pd.Series] = []
    for c in cols:
        s = df[c]
        if str(c).lower() == "age" or str(c).upper() == "AGE":
            a = pd.to_numeric(s, errors="coerce")
            if a.isna().all():
                parts.append(pd.Series(["AGE_NA"] * len(df), index=df.index))
                continue
            aj = a.copy()
            eps = rng.normal(0.0, 1e-6, size=len(aj))
            aj = aj + pd.Series(eps, index=aj.index)
            try:
                bins = pd.qcut(aj, q=int(age_bins), duplicates="drop")
                parts.append(bins.astype(str).fillna("AGE_NA"))
            except Exception:
                parts.append(pd.Series(["AGE_BIN"] * len(df), index=df.index))
        else:
            g = s.astype(str).str.strip().str.upper().replace({"NAN": "NA", "": "NA"}).fillna("NA")
            parts.append(g.map(lambda x: f"{c}={x}"))

    cat = parts[0].astype(str)
    for p in parts[1:]:
        cat = cat.astype(str) + "|" + p.astype(str)
    return cat


def build_strat_labels(df: pd.DataFrame, y: np.ndarray, meta_cols: List[str], age_bins: int, seed: int) -> np.ndarray:
    meta_cat = _meta_to_categories(df, meta_cols, age_bins=age_bins, seed=seed).astype(str)
    y_s = pd.Series(y, index=df.index).astype(str)
    return (y_s + "||" + meta_cat).to_numpy(dtype=object)


# ----------------------------
# Splits: strat labels + optional group
# ----------------------------
def group_stratified_split(strat_labels: np.ndarray, groups: Optional[np.ndarray], test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(strat_labels))
    if groups is None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
        tr, te = next(sss.split(idx, strat_labels))
        return tr, te

    g = np.asarray(groups)
    uniq = np.unique(g)

    # group-level strat label = mode within group
    sl_g = []
    for gu in uniq:
        vals = strat_labels[g == gu]
        u, cnt = np.unique(vals, return_counts=True)
        sl_g.append(u[int(np.argmax(cnt))])
    sl_g = np.asarray(sl_g, dtype=object)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
    train_g_idx, test_g_idx = next(splitter.split(uniq, sl_g))
    train_groups = set(uniq[train_g_idx].tolist())
    test_groups = set(uniq[test_g_idx].tolist())

    train_idx = idx[np.isin(g, list(train_groups))]
    test_idx = idx[np.isin(g, list(test_groups))]
    return train_idx, test_idx


def make_cv_splits(strat_labels: np.ndarray, groups: Optional[np.ndarray], n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(strat_labels))
    if groups is None:
        skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
        return list(skf.split(idx, strat_labels))

    g = np.asarray(groups)
    # if no repeats -> usual stratified
    if pd.Series(g).value_counts().max() <= 1:
        skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
        return list(skf.split(idx, strat_labels))

    if StratifiedGroupKFold is not None:
        sgkf = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
        return list(sgkf.split(idx, strat_labels, g))

    gkf = GroupKFold(n_splits=int(n_splits))
    return list(gkf.split(idx, strat_labels, g))


# ----------------------------
# Threshold selection
# ----------------------------
def select_threshold_recall(y_true: np.ndarray, prob: np.ndarray, recall_target: float = 0.85) -> float:
    best_thr = 0.5
    for t in np.linspace(0.0, 1.0, 1001):
        pred = (prob >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec >= recall_target:
            best_thr = float(t)
            break
    return float(best_thr)


def select_threshold_recall_plus(y_true: np.ndarray, prob: np.ndarray, recall_target: float = 0.85, min_spec: float = 0.0) -> float:
    best_thr = 0.5
    for t in np.linspace(0.0, 1.0, 1001):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rec = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        if rec + 1e-9 >= float(recall_target) and spec + 1e-9 >= float(min_spec):
            best_thr = float(t)
            break
    return float(best_thr)


def select_threshold_f1_plus(y_true: np.ndarray, prob: np.ndarray, min_spec: float = 0.0, min_prec: Optional[float] = None) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, 1001):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        spec = safe_div(tn, tn + fp)
        prec = safe_div(tp, tp + fp)
        if spec + 1e-9 < float(min_spec):
            continue
        if min_prec is not None and prec + 1e-9 < float(min_prec):
            continue
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return float(best_thr)


# ----------------------------
# Augmentations
# ----------------------------
@dataclass
class AugConfig:
    search_aug: str = "fixed"
    p_apply: float = 0.5
    noise_std: float = 0.0
    noise_med: float = 0.0
    shift: float = 0.0
    scale: float = 0.0
    tilt: float = 0.0
    offset: float = 0.0
    mixup: float = 0.0
    mixwithin: float = 0.0
    aug_repeats: int = 1


@dataclass
class FrdaConfig:
    enabled: bool = False
    k: int = 4
    width: int = 40
    local_scale: float = 0.02


def aug_noise_std(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return (X + rng.normal(0.0, float(sigma), size=X.shape)).astype(np.float32)


def aug_noise_med(X: np.ndarray, rel: float, rng: np.random.Generator) -> np.ndarray:
    med = np.median(np.abs(X), axis=1, keepdims=True)
    sigma = float(rel) * (med + 1e-12)
    return (X + rng.normal(0.0, 1.0, size=X.shape) * sigma).astype(np.float32)


def aug_shift(X: np.ndarray, max_shift: float, rng: np.random.Generator) -> np.ndarray:
    s = int(round(float(max_shift)))
    if s <= 0:
        return X.copy()
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        k = int(rng.integers(-s, s + 1))
        if k == 0:
            out[i] = X[i]
        elif k > 0:
            out[i, :k] = X[i, 0]
            out[i, k:] = X[i, :-k]
        else:
            k = -k
            out[i, -k:] = X[i, -1]
            out[i, :-k] = X[i, k:]
    return out.astype(np.float32)


def aug_scale(X: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    if scale <= 0:
        return X.copy()
    factors = 1.0 + rng.normal(0.0, float(scale), size=(X.shape[0], 1))
    return (X * factors).astype(np.float32)


def aug_offset(X: np.ndarray, offset: float, rng: np.random.Generator) -> np.ndarray:
    if offset <= 0:
        return X.copy()
    med = np.median(np.abs(X), axis=1, keepdims=True)
    delta = rng.normal(0.0, float(offset), size=(X.shape[0], 1)) * (med + 1e-12)
    return (X + delta).astype(np.float32)


def aug_tilt(X: np.ndarray, tilt: float, rng: np.random.Generator) -> np.ndarray:
    if tilt <= 0:
        return X.copy()
    n = X.shape[1]
    axis = np.linspace(-0.5, 0.5, n, dtype=np.float32).reshape(1, -1)
    med = np.median(np.abs(X), axis=1, keepdims=True).astype(np.float32)
    slope = rng.normal(0.0, float(tilt), size=(X.shape[0], 1)).astype(np.float32)
    return (X + slope * axis * (med + 1e-12)).astype(np.float32)


def aug_mixup(X: np.ndarray, y: np.ndarray, alpha: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if alpha is None or alpha <= 0:
        return X.copy(), y.copy()
    n = X.shape[0]
    j = rng.integers(0, n, size=n)
    lam = rng.beta(alpha, alpha, size=n).astype(np.float32).reshape(-1, 1)
    X2 = lam * X + (1.0 - lam) * X[j]
    y_soft = lam.ravel() * y + (1.0 - lam.ravel()) * y[j]
    y2 = (y_soft >= 0.5).astype(int)
    return X2.astype(np.float32), y2


def aug_mixwithin(X: np.ndarray, y: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    if alpha is None or alpha <= 0:
        return X.copy()
    out = X.copy()
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        if idx.size < 2:
            continue
        j = rng.choice(idx, size=idx.size, replace=True)
        lam = rng.beta(alpha, alpha, size=idx.size).astype(np.float32).reshape(-1, 1)
        out[idx] = lam * X[idx] + (1.0 - lam) * X[j]
    return out.astype(np.float32)


def _frda_mask_from_logreg(X: np.ndarray, y: np.ndarray, k: int, width: int, seed: int) -> np.ndarray:
    lr = LogisticRegressionCV(
        Cs=10,
        cv=3,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        scoring="roc_auc",
        max_iter=2000,
        n_jobs=1,
        random_state=seed,
    )
    lr.fit(X, y)
    coef = np.abs(lr.coef_.ravel())
    peaks = np.argsort(coef)[::-1][: max(1, int(k))]
    mask = np.zeros(X.shape[1], dtype=bool)
    half = max(1, int(width) // 2)
    for p in peaks:
        lo = max(0, p - half)
        hi = min(X.shape[1], p + half + 1)
        mask[lo:hi] = True
    return mask


def aug_frda_local_scale(X: np.ndarray, mask: np.ndarray, local_scale: float, rng: np.random.Generator) -> np.ndarray:
    if local_scale <= 0:
        return X.copy()
    out = X.copy()
    factors = 1.0 + rng.normal(0.0, float(local_scale), size=(X.shape[0], 1))
    out[:, mask] = (out[:, mask] * factors).astype(np.float32)
    return out.astype(np.float32)


def build_augmented_train(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    aug: AugConfig,
    rng: np.random.Generator,
    frda: Optional[FrdaConfig] = None,
    frda_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X_parts: List[np.ndarray] = [Xtr]
    y_parts: List[np.ndarray] = [ytr]

    def do(p: float) -> bool:
        return float(p) >= 1.0 or rng.random() < float(p)

    K = max(1, int(aug.aug_repeats))
    for _ in range(K):
        if aug.noise_std > 0 and do(aug.p_apply):
            X_parts.append(aug_noise_std(Xtr, aug.noise_std, rng))
            y_parts.append(ytr)
        if aug.noise_med > 0 and do(aug.p_apply):
            X_parts.append(aug_noise_med(Xtr, aug.noise_med, rng))
            y_parts.append(ytr)
        if aug.shift > 0 and do(aug.p_apply):
            X_parts.append(aug_shift(Xtr, aug.shift, rng))
            y_parts.append(ytr)
        if aug.scale > 0 and do(aug.p_apply):
            X_parts.append(aug_scale(Xtr, aug.scale, rng))
            y_parts.append(ytr)
        if aug.tilt > 0 and do(aug.p_apply):
            X_parts.append(aug_tilt(Xtr, aug.tilt, rng))
            y_parts.append(ytr)
        if aug.offset > 0 and do(aug.p_apply):
            X_parts.append(aug_offset(Xtr, aug.offset, rng))
            y_parts.append(ytr)
        if aug.mixwithin > 0 and do(aug.p_apply):
            X_parts.append(aug_mixwithin(Xtr, ytr, aug.mixwithin, rng))
            y_parts.append(ytr)
        if aug.mixup > 0 and do(aug.p_apply):
            Xm, ym = aug_mixup(Xtr, ytr, aug.mixup, rng)
            X_parts.append(Xm)
            y_parts.append(ym)
        if frda is not None and frda.enabled and frda_mask is not None and do(aug.p_apply):
            X_parts.append(aug_frda_local_scale(Xtr, frda_mask, frda.local_scale, rng))
            y_parts.append(ytr)

    return np.vstack(X_parts).astype(np.float32), np.concatenate(y_parts).astype(int)


# ----------------------------
# PLS-DA wrapper
# ----------------------------
class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int = 6):
        self.n_components = int(n_components)

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y).astype(float).reshape(-1, 1)
        ncomp = max(1, min(int(self.n_components), X.shape[1], X.shape[0] - 1))

        # ВАЖНО: атрибут с суффиксом "_" -> sklearn будет считать модель fitted
        self.pls_ = PLSRegression(n_components=ncomp)
        self.pls_.fit(X, y)

        # тоже полезно для sklearn-совместимости
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = X.shape[1]

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.pls_.predict(X).ravel().astype(float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = self.decision_function(X)
        p = np.clip(expit(s), 1e-6, 1 - 1e-6)
        return np.vstack([1.0 - p, p]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)



# ----------------------------
# Models
# ----------------------------
def get_base_models(seed: int) -> Dict[str, BaseEstimator]:
    return {
        "plsda": PLSDAClassifier(n_components=6),
        "logreg": LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "svm_lin": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=seed),
    }


def build_model_pipeline(est: BaseEstimator, xscale: str) -> BaseEstimator:
    scaler = make_xscaler(xscale)
    if scaler is None:
        return est
    return Pipeline([("xscale", scaler), ("clf", est)])


def predict_proba_pos(est: BaseEstimator, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return np.asarray(est.predict_proba(X)[:, 1], dtype=float)
    if hasattr(est, "decision_function"):
        return expit(np.asarray(est.decision_function(X), dtype=float))
    raise ValueError("Estimator has neither predict_proba nor decision_function.")


# ----------------------------
# Metrics
# ----------------------------
def ece_score(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) for binary classification.
    Lower is better.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)

    if len(y_true) == 0:
        return float("nan")

    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(prob)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (prob >= lo) & (prob < hi)
        else:
            mask = (prob >= lo) & (prob <= hi)

        if not np.any(mask):
            continue

        conf = float(np.mean(prob[mask]))       # средняя уверенность
        acc = float(np.mean(y_true[mask]))      # фактическая доля положительных
        ece += (np.sum(mask) / n) * abs(acc - conf)

    return float(ece)
def compute_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (prob >= float(thr)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    out = {
        "thr": float(thr),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "acc": float(accuracy_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "prec": float(precision_score(y_true, pred, zero_division=0)),
        "spec": safe_div(tn, tn + fp),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

    # ROC-AUC
    try:
        out["auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["auc"] = float("nan")

    # PR-AUC (Average Precision)
    try:
        out["pr_auc"] = float(average_precision_score(y_true, prob))
    except Exception:
        out["pr_auc"] = float("nan")

    # Calibration metrics
    try:
        out["brier"] = float(brier_score_loss(y_true, prob))
    except Exception:
        out["brier"] = float("nan")

    try:
        out["ece"] = float(ece_score(y_true, prob, n_bins=10))
    except Exception:
        out["ece"] = float("nan")

    return out


# ----------------------------
# Calibration helper
# ----------------------------
def select_calib_subset(strat_labels_tr: np.ndarray, groups_tr: Optional[np.ndarray], frac: float, seed: int) -> np.ndarray:
    if frac <= 0:
        return np.array([], dtype=int)
    tr, cal = group_stratified_split(strat_labels_tr, groups_tr, test_size=frac, seed=seed)
    return cal

def _calib_scores(est: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Scores for calibration: prefer decision_function; else logit(prob)."""
    if hasattr(est, "decision_function"):
        s = np.asarray(est.decision_function(X), dtype=float).ravel()
        return s
    if hasattr(est, "predict_proba"):
        p = np.asarray(est.predict_proba(X)[:, 1], dtype=float)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))
    raise ValueError("Estimator has neither decision_function nor predict_proba for calibration.")


class _ManuallyCalibrated(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: BaseEstimator, method: str, calibrator: Any):
        self.base_estimator = base_estimator
        self.method = method
        self.calibrator = calibrator
        # classes_ нужно многим sklearn-частям
        self.classes_ = getattr(base_estimator, "classes_", np.array([0, 1], dtype=int))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = _calib_scores(self.base_estimator, X)
        if self.method == "sigmoid":
            p = np.asarray(self.calibrator.predict_proba(s.reshape(-1, 1))[:, 1], dtype=float)
        else:  # isotonic
            p = np.asarray(self.calibrator.predict(s), dtype=float)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.vstack([1.0 - p, p]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def fit_with_optional_calibration(
    est: BaseEstimator,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_cal: Optional[np.ndarray],
    y_cal: Optional[np.ndarray],
    method: str,
) -> BaseEstimator:
    est_fitted = clone(est)
    est_fitted.fit(X_fit, y_fit)

    method = (method or "none").lower().strip()
    if method == "none" or X_cal is None or y_cal is None or len(y_cal) < 10:
        return est_fitted

    # если вдруг в калибровочном поднаборе один класс — калибровка бессмысленна/сломается
    if len(np.unique(y_cal)) < 2:
        return est_fitted

    # --- ручная калибровка (без CalibratedClassifierCV(cv="prefit")) ---
    if method == "platt":
        # Platt scaling = логистическая регрессия на скоре
        s_cal = _calib_scores(est_fitted, X_cal).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=2000)
        lr.fit(s_cal, np.asarray(y_cal).astype(int))
        return _ManuallyCalibrated(est_fitted, method="sigmoid", calibrator=lr)

    if method == "isotonic":
        s_cal = _calib_scores(est_fitted, X_cal)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s_cal, np.asarray(y_cal).astype(int))
        return _ManuallyCalibrated(est_fitted, method="isotonic", calibrator=iso)


    return est_fitted


# ----------------------------
# PLS LV tuning (inner CV) for mcdcv_plsda
# ----------------------------
def tune_pls_components(
    X: np.ndarray,
    y: np.ndarray,
    strat_labels: np.ndarray,
    groups: Optional[np.ndarray],
    xscale: str,
    seed: int,
    n_splits: int,
    grid: List[int],
) -> int:
    y = np.asarray(y).astype(int)
    splits = make_cv_splits(strat_labels, groups, n_splits=n_splits, seed=seed)

    # ускорение + защита от бессмысленных компонент
    max_comp = max(1, min(X.shape[0] - 1, X.shape[1]))
    grid_eff = sorted({max(1, min(int(c), max_comp)) for c in grid if int(c) >= 1})

    best_c, best_auc = grid_eff[0], -1.0

    for c in grid_eff:
        oof = np.zeros_like(y, dtype=float)

        for fold, (tr, va) in enumerate(splits):
            est = build_model_pipeline(PLSDAClassifier(n_components=int(c)), xscale=xscale)
            # ВАЖНО: без этого будет NotFittedError
            est.fit(X[tr], y[tr])
            oof[va] = predict_proba_pos(est, X[va])

        try:
            auc = float(roc_auc_score(y, oof))
        except Exception:
            auc = -1.0

        if auc > best_auc + 1e-9:
            best_auc = auc
            best_c = int(c)

    return int(best_c)



# ----------------------------
# Protocol runners
# ----------------------------
def run_inner_cv_get_thresholds(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strat_train: np.ndarray,
    g_train: Optional[np.ndarray],
    cfg: "RunConfig",
    models: List[str],
    base_models: Dict[str, BaseEstimator],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    splits = make_cv_splits(strat_train, g_train, n_splits=cfg.n_splits, seed=cfg.seed)
    oof_probs = {m: np.zeros_like(y_train, dtype=float) for m in models}

    # optional PLS tuning once per inner loop (cheap-ish)
    tuned_pls = None
    if cfg.protocol in ("mcdcv_plsda",) and "plsda" in models and cfg.pls_tune:
        grid = cfg.pls_grid
        tuned_pls = tune_pls_components(
            X_train, y_train, strat_train, g_train, xscale=cfg.xscale, seed=cfg.seed + 999, n_splits=cfg.inner_splits, grid=grid
        )

    for fold, (tr_idx, va_idx) in enumerate(splits):
        Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        Xva = X_train[va_idx]
        strat_tr = strat_train[tr_idx]
        g_tr = None if g_train is None else np.asarray(g_train)[tr_idx]

        frda_mask = None
        if cfg.frda.enabled:
            frda_mask = _frda_mask_from_logreg(Xtr, ytr, k=cfg.frda.k, width=cfg.frda.width, seed=cfg.seed + 77 * fold)

        X_aug, y_aug = build_augmented_train(
            Xtr, ytr, aug=cfg.aug, rng=np.random.default_rng(cfg.seed + 10_000 + fold), frda=cfg.frda if cfg.frda.enabled else None, frda_mask=frda_mask
        )

        calib_idx = select_calib_subset(strat_tr, g_tr, frac=cfg.calib_frac, seed=cfg.seed + 123 * fold)
        if calib_idx.size > 0:
            X_cal, y_cal = Xtr[calib_idx], ytr[calib_idx]
            if cfg.calib_real_only:
                fit_mask = np.ones(Xtr.shape[0], dtype=bool)
                fit_mask[calib_idx] = False
                X_fit_real, y_fit_real = Xtr[fit_mask], ytr[fit_mask]
                frda_mask_fit = None
                if cfg.frda.enabled and X_fit_real.shape[0] >= 10:
                    frda_mask_fit = _frda_mask_from_logreg(X_fit_real, y_fit_real, k=cfg.frda.k, width=cfg.frda.width, seed=cfg.seed + 777 * fold)
                X_aug_fit, y_aug_fit = build_augmented_train(
                    X_fit_real, y_fit_real, aug=cfg.aug, rng=np.random.default_rng(cfg.seed + 20_000 + fold),
                    frda=cfg.frda if cfg.frda.enabled else None, frda_mask=frda_mask_fit
                )
            else:
                X_aug_fit, y_aug_fit = X_aug, y_aug
        else:
            X_cal = y_cal = None
            X_aug_fit, y_aug_fit = X_aug, y_aug

        for m in models:
            est0 = base_models[m]
            if m == "plsda" and tuned_pls is not None:
                est0 = PLSDAClassifier(n_components=int(tuned_pls))
            est = build_model_pipeline(est0, xscale=cfg.xscale)
            fitted = fit_with_optional_calibration(est, X_fit=X_aug_fit, y_fit=y_aug_fit, X_cal=X_cal, y_cal=y_cal, method=cfg.calib)
            oof_probs[m][va_idx] = predict_proba_pos(fitted, Xva)

    thresholds: Dict[str, float] = {}
    for m in models:
        oof = oof_probs[m]
        if cfg.threshold_by == "none":
            thr = 0.5
        elif cfg.threshold_by == "recall":
            thr = select_threshold_recall(y_train, oof, recall_target=cfg.recall_target)
        elif cfg.threshold_by == "recall_plus":
            thr = select_threshold_recall_plus(y_train, oof, recall_target=cfg.recall_target, min_spec=cfg.min_spec)
        elif cfg.threshold_by == "f1_plus":
            thr = select_threshold_f1_plus(y_train, oof, min_spec=cfg.min_spec, min_prec=cfg.min_prec)
        else:
            raise ValueError(f"Unknown threshold_by: {cfg.threshold_by}")
        thresholds[m] = float(thr)

    debug = {"oof_auc": {m: float(roc_auc_score(y_train, oof_probs[m])) if len(np.unique(y_train)) > 1 else float("nan") for m in models},
             "tuned_pls_ncomp": tuned_pls}
    return thresholds, debug


def run_protocol_cv_holdout(cfg: "RunConfig") -> Dict[str, Any]:
    # load dataset
    df0 = pd.read_parquet(Path(cfg.data_path)) if cfg.data_path else pd.read_parquet(Path(cfg.default_path))
    y, ycol = infer_label_series(df0, cfg.label_col)
    groups_s, gcol = infer_groups_series(df0, cfg.group_col)
    spec_cols = pick_spectral_columns(df0)
    if not spec_cols:
        raise ValueError("No spectral columns detected (numeric-like column NAMES).")

    df = df0.copy()
    df["y"] = y.values
    if groups_s is not None:
        df[groups_s.name or "ID"] = groups_s.values

    if cfg.avg_replicates and groups_s is not None and df[groups_s.name or "ID"].duplicated().any():
        df = maybe_average_replicates(df, y=y, groups=df[groups_s.name or "ID"], spec_cols=spec_cols)
        y = df["y"].astype(int)
        groups_s = df[groups_s.name or "ID"].astype(str)
        spec_cols = pick_spectral_columns(df)

    X_raw, wn = build_X_wn(df, spec_cols)
    y_np = y.to_numpy(dtype=int)
    groups_np = None if groups_s is None else groups_s.to_numpy(dtype=object)

    meta_cols = _infer_meta_cols(df, cfg.meta_stratify)
    strat_labels = build_strat_labels(df, y_np, meta_cols, age_bins=cfg.age_bins, seed=cfg.seed)

    X, wn2 = preprocess(X_raw, wn, cfg.crop_min, cfg.crop_max, cfg.sg_window, cfg.sg_poly, cfg.sg_deriv, cfg.norm, parse_drop_ranges(cfg.drop_ranges))

    tr_idx, te_idx = group_stratified_split(strat_labels, groups_np, test_size=cfg.val_size, seed=cfg.seed)
    X_train, y_train = X[tr_idx], y_np[tr_idx]
    X_test, y_test = X[te_idx], y_np[te_idx]
    strat_train = strat_labels[tr_idx]
    g_train = None if groups_np is None else np.asarray(groups_np)[tr_idx]
    g_test = None if groups_np is None else np.asarray(groups_np)[te_idx]

    base_models = get_base_models(cfg.seed)
    models = [m for m in (cfg.models or list(base_models.keys())) if m in base_models]
    if not models:
        raise ValueError("No valid models selected.")

    thresholds, inner_dbg = run_inner_cv_get_thresholds(X_train, y_train, strat_train, g_train, cfg, models, base_models)

    results: Dict[str, Any] = {}
    for m in models:
        est0 = base_models[m]
        if m == "plsda" and inner_dbg.get("tuned_pls_ncomp") is not None:
            est0 = PLSDAClassifier(n_components=int(inner_dbg["tuned_pls_ncomp"]))
        est = build_model_pipeline(est0, xscale=cfg.xscale)

        calib_idx = select_calib_subset(strat_train, g_train, frac=cfg.calib_frac, seed=cfg.seed + 999)
        if calib_idx.size > 0:
            X_cal, y_cal = X_train[calib_idx], y_train[calib_idx]
            fit_mask = np.ones(X_train.shape[0], dtype=bool)
            fit_mask[calib_idx] = False
            X_fit_real, y_fit_real = X_train[fit_mask], y_train[fit_mask]
        else:
            X_cal = y_cal = None
            X_fit_real, y_fit_real = X_train, y_train

        frda_mask = None
        if cfg.frda.enabled:
            frda_mask = _frda_mask_from_logreg(X_fit_real, y_fit_real, k=cfg.frda.k, width=cfg.frda.width, seed=cfg.seed + 555)

        X_aug, y_aug = build_augmented_train(
            X_fit_real, y_fit_real, aug=cfg.aug, rng=np.random.default_rng(cfg.seed + 3333),
            frda=cfg.frda if cfg.frda.enabled else None, frda_mask=frda_mask
        )

        fitted = fit_with_optional_calibration(est, X_fit=X_aug, y_fit=y_aug, X_cal=X_cal, y_cal=y_cal, method=cfg.calib)
        prob_test = predict_proba_pos(fitted, X_test)
        results[m] = {"threshold": thresholds[m], "test": compute_metrics(y_test, prob_test, thr=thresholds[m])}

    return {
        "protocol": cfg.protocol,
        "config": dataclasses.asdict(cfg),
        "detected": {
            "label_col": ycol,
            "group_col": gcol,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "wn_min": float(wn2.min()),
            "wn_max": float(wn2.max()),
            "n_groups": None if groups_np is None else int(len(np.unique(groups_np))),
            "max_reps_per_group": None if groups_np is None else int(pd.Series(groups_np).value_counts().max()),
            "meta_stratify_cols": meta_cols,
        },
        "selected_aug": dataclasses.asdict(cfg.aug),
        "inner_debug": inner_dbg,
        "results": results,
    }


def run_protocol_mcdcv(cfg: "RunConfig") -> Dict[str, Any]:
    # repeated holdout; aggregate mean/std on test
    df0 = pd.read_parquet(Path(cfg.data_path)) if cfg.data_path else pd.read_parquet(Path(cfg.default_path))
    y, ycol = infer_label_series(df0, cfg.label_col)
    groups_s, gcol = infer_groups_series(df0, cfg.group_col)
    spec_cols = pick_spectral_columns(df0)
    if not spec_cols:
        raise ValueError("No spectral columns detected (numeric-like column NAMES).")

    df = df0.copy()
    df["y"] = y.values
    if groups_s is not None:
        df[groups_s.name or "ID"] = groups_s.values

    if cfg.avg_replicates and groups_s is not None and df[groups_s.name or "ID"].duplicated().any():
        df = maybe_average_replicates(df, y=y, groups=df[groups_s.name or "ID"], spec_cols=spec_cols)
        y = df["y"].astype(int)
        groups_s = df[groups_s.name or "ID"].astype(str)
        spec_cols = pick_spectral_columns(df)

    X_raw, wn = build_X_wn(df, spec_cols)
    y_np = y.to_numpy(dtype=int)
    groups_np = None if groups_s is None else groups_s.to_numpy(dtype=object)

    meta_cols = _infer_meta_cols(df, cfg.meta_stratify)
    strat_labels = build_strat_labels(df, y_np, meta_cols, age_bins=cfg.age_bins, seed=cfg.seed)
    X, wn2 = preprocess(X_raw, wn, cfg.crop_min, cfg.crop_max, cfg.sg_window, cfg.sg_poly, cfg.sg_deriv, cfg.norm, parse_drop_ranges(cfg.drop_ranges))

    base_models = get_base_models(cfg.seed)
    models = [m for m in (cfg.models or list(base_models.keys())) if m in base_models]
    if not models:
        raise ValueError("No valid models selected.")

    all_iter: List[Dict[str, Any]] = []
    rng = np.random.default_rng(cfg.seed + 12345)

    for it in range(int(cfg.mc_iter)):
    # пробуем подобрать валидный сплит (оба класса в train и test)
        for _attempt in range(200):
            seed_it = int(rng.integers(0, 2**31 - 1))
            tr_idx, te_idx = group_stratified_split(strat_labels, groups_np, test_size=cfg.val_size, seed=seed_it)

            ytr_u = np.unique(y_np[tr_idx])
            yte_u = np.unique(y_np[te_idx])
            if (len(ytr_u) >= 2) and (len(yte_u) >= 2):
                break
        else:
            raise RuntimeError("Could not sample a split with both classes in train and test. Check val_size / labels.")

    


        X_train, y_train = X[tr_idx], y_np[tr_idx]
        X_test, y_test = X[te_idx], y_np[te_idx]
        strat_train = strat_labels[tr_idx]
        g_train = None if groups_np is None else np.asarray(groups_np)[tr_idx]
        g_test = None if groups_np is None else np.asarray(groups_np)[te_idx]

        thresholds, inner_dbg = run_inner_cv_get_thresholds(X_train, y_train, strat_train, g_train, cfg, models, base_models)

        iter_res: Dict[str, Any] = {"seed": seed_it, "thresholds": thresholds, "inner_debug": inner_dbg, "test": {}}

        for m in models:
            est0 = base_models[m]
            if m == "plsda" and inner_dbg.get("tuned_pls_ncomp") is not None:
                est0 = PLSDAClassifier(n_components=int(inner_dbg["tuned_pls_ncomp"]))
            est = build_model_pipeline(est0, xscale=cfg.xscale)

            calib_idx = select_calib_subset(strat_train, g_train, frac=cfg.calib_frac, seed=seed_it + 999)
            if calib_idx.size > 0:
                X_cal, y_cal = X_train[calib_idx], y_train[calib_idx]
                fit_mask = np.ones(X_train.shape[0], dtype=bool)
                fit_mask[calib_idx] = False
                X_fit_real, y_fit_real = X_train[fit_mask], y_train[fit_mask]
            else:
                X_cal = y_cal = None
                X_fit_real, y_fit_real = X_train, y_train

            frda_mask = None
            if cfg.frda.enabled:
                frda_mask = _frda_mask_from_logreg(X_fit_real, y_fit_real, k=cfg.frda.k, width=cfg.frda.width, seed=seed_it + 555)

            X_aug, y_aug = build_augmented_train(
                X_fit_real, y_fit_real, aug=cfg.aug, rng=np.random.default_rng(seed_it + 3333),
                frda=cfg.frda if cfg.frda.enabled else None, frda_mask=frda_mask
            )

            fitted = fit_with_optional_calibration(est, X_fit=X_aug, y_fit=y_aug, X_cal=X_cal, y_cal=y_cal, method=cfg.calib)
            prob_test = predict_proba_pos(fitted, X_test)
            iter_res["test"][m] = compute_metrics(y_test, prob_test, thr=thresholds[m])

        all_iter.append(iter_res)

    # aggregate
    agg: Dict[str, Any] = {"mean": {}, "std": {}}
    for m in models:
        keys = ["auc", "pr_auc", "f1", "recall", "prec", "spec", "acc", "brier", "ece"]
        mat = {k: np.array([it["test"][m].get(k, np.nan) for it in all_iter], dtype=float) for k in keys}
        agg["mean"][m] = {k: float(np.nanmean(mat[k])) for k in keys}
        agg["std"][m] = {k: float(np.nanstd(mat[k])) for k in keys}

    return {
        "protocol": cfg.protocol,
        "config": dataclasses.asdict(cfg),
        "detected": {
            "label_col": ycol,
            "group_col": gcol,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "wn_min": float(wn2.min()),
            "wn_max": float(wn2.max()),
            "n_groups": None if groups_np is None else int(len(np.unique(groups_np))),
            "max_reps_per_group": None if groups_np is None else int(pd.Series(groups_np).value_counts().max()),
            "meta_stratify_cols": meta_cols,
        },
        "selected_aug": dataclasses.asdict(cfg.aug),
        "summary": agg,
        "iterations": all_iter,
    }


# ----------------------------
# Config
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


@dataclass
class RunConfig:
    dataset: str
    data_path: Optional[str]
    seed: int

    label_col: Optional[str]
    group_col: Optional[str]
    avg_replicates: bool

    # preprocess
    crop_min: float
    crop_max: float
    sg_window: int
    sg_poly: int
    sg_deriv: int
    norm: str
    drop_ranges: str

    # protocol
    protocol: str                 # cv_holdout | mcdcv | mcdcv_plsda
    mc_iter: int
    inner_splits: int
    n_splits: int
    val_size: float

    calib: str
    calib_real_only: bool
    calib_frac: float

    threshold_by: str
    recall_target: float
    min_spec: float
    min_prec: Optional[float]

    xscale: str
    models: Optional[List[str]]

    # meta stratify
    meta_stratify: Optional[str]
    age_bins: int

    # aug
    aug: AugConfig
    frda: FrdaConfig
    aug_trials: int

    # pls tuning
    pls_tune: bool
    pls_grid: List[int]

    @property
    def default_path(self) -> str:
        if self.dataset == "covid_saliva":
            return str(PROC / "train.parquet")
        return str(PROC / "diabetes_saliva.parquet")


# ----------------------------
# CLI
# ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True, choices=["covid_saliva", "diabetes_saliva"])
    ap.add_argument("--data-path", default=None)

    ap.add_argument("--label-col", default=None)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--avg-replicates", action="store_true")

    # meta stratify + aliases
    ap.add_argument("--stratify-meta", default=None, help="Comma-separated meta cols (e.g. AGE,GENDER).")
    ap.add_argument("--meta-stratify", default=None, help="ALIAS of --stratify-meta.")
    ap.add_argument("--age-bins", type=int, default=5)
    ap.add_argument("--age-bin", type=int, default=None, help="ALIAS of --age-bins.")

    # preprocess
    ap.add_argument("--crop-min", type=float, default=800)
    ap.add_argument("--crop-max", type=float, default=1800)
    ap.add_argument("--sg-window", type=int, default=11)
    ap.add_argument("--sg-poly", type=int, default=2)
    ap.add_argument("--sg-deriv", type=int, default=0)
    ap.add_argument("--norm", type=str, default="snv", choices=["snv", "none", "l2"])
    ap.add_argument("--drop-ranges", type=str, default="", help='Example: "1800-1900,2350-2450"')

    # protocol
    ap.add_argument("--protocol", type=str, default="cv_holdout", choices=["cv_holdout", "mcdcv", "mcdcv_plsda"])
    ap.add_argument("--mc-iter", type=int, default=200)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--outer-splits", type=int, default=0, help="Accepted for compatibility, ignored.")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--calib", type=str, default="platt", choices=["none", "platt", "isotonic"])
    ap.add_argument("--calib-real-only", action="store_true")
    ap.add_argument("--calib-frac", type=float, default=0.2)

    ap.add_argument("--threshold-by", default="none", choices=["none", "recall", "recall_plus", "f1_plus"])
    ap.add_argument("--recall-target", type=float, default=0.85)
    ap.add_argument("--min-spec", type=float, default=0.0)
    ap.add_argument("--min-prec", type=float, default=None)

    ap.add_argument("--xscale", type=str, default="center", choices=["none", "center", "autoscale"])

    # aug
    ap.add_argument("--search-aug", type=str, default="fixed", choices=["fixed", "auto"])
    ap.add_argument("--aug-trials", type=int, default=20)
    ap.add_argument("--p-apply", type=float, default=0.5)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--noise-med", type=float, default=0.0)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=0.0)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--mixwithin", type=float, default=0.0)
    ap.add_argument("--aug-repeats", type=int, default=1)

    # frda
    ap.add_argument("--frda-lite", action="store_true")
    ap.add_argument("--frda-k", type=int, default=4)
    ap.add_argument("--frda-width", type=int, default=40)
    ap.add_argument("--frda-local-scale", type=float, default=0.02)

    ap.add_argument("--models", type=str, default=None)

    # pls tuning knobs
    ap.add_argument("--pls-tune", action="store_true", help="Tune PLS n_components in inner CV (useful for mcdcv_plsda).")
    ap.add_argument("--pls-grid", type=str, default="2,3,4,5,6,7,8,10,12,15")

    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--outdir", type=str, default=None)

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    # resolve aliases
    meta_stratify = args.stratify_meta if args.stratify_meta is not None else args.meta_stratify
    age_bins = int(args.age_bins if args.age_bin is None else args.age_bin)

    pls_grid = []
    try:
        pls_grid = [int(x.strip()) for x in str(args.pls_grid).split(",") if x.strip()]
        pls_grid = [x for x in pls_grid if x >= 1]
        if not pls_grid:
            pls_grid = [6]
    except Exception:
        pls_grid = [6]

    aug = AugConfig(
        search_aug=str(args.search_aug),
        p_apply=float(args.p_apply),
        noise_std=float(args.noise_std),
        noise_med=float(args.noise_med),
        shift=float(args.shift),
        scale=float(args.scale),
        tilt=float(args.tilt),
        offset=float(args.offset),
        mixup=float(args.mixup),
        mixwithin=float(args.mixwithin),
        aug_repeats=int(args.aug_repeats),
    )

    frda = FrdaConfig(
        enabled=bool(args.frda_lite),
        k=int(args.frda_k),
        width=int(args.frda_width),
        local_scale=float(args.frda_local_scale),
    )

    models = None
    if args.models:
        models = [s.strip() for s in str(args.models).split(",") if s.strip()]

    cfg = RunConfig(
        dataset=str(args.dataset),
        data_path=args.data_path,
        seed=int(args.seed),
        label_col=args.label_col,
        group_col=args.group_col,
        avg_replicates=bool(args.avg_replicates),

        crop_min=float(args.crop_min),
        crop_max=float(args.crop_max),
        sg_window=int(args.sg_window),
        sg_poly=int(args.sg_poly),
        sg_deriv=int(args.sg_deriv),
        norm=str(args.norm),
        drop_ranges=str(args.drop_ranges),

        protocol=str(args.protocol),
        mc_iter=int(args.mc_iter),
        inner_splits=int(args.inner_splits),
        n_splits=int(args.n_splits),
        val_size=float(args.val_size),

        calib=str(args.calib),
        calib_real_only=bool(args.calib_real_only),
        calib_frac=float(args.calib_frac),

        threshold_by=str(args.threshold_by),
        recall_target=float(args.recall_target),
        min_spec=float(args.min_spec),
        min_prec=args.min_prec,

        xscale=str(args.xscale),
        models=models,

        meta_stratify=meta_stratify,
        age_bins=age_bins,

        aug=aug,
        frda=frda,
        aug_trials=int(args.aug_trials),

        pls_tune=bool(args.pls_tune) or (str(args.protocol) == "mcdcv_plsda"),
        pls_grid=pls_grid,
    )

    set_all_seeds(cfg.seed)

    if cfg.protocol == "cv_holdout":
        report = run_protocol_cv_holdout(cfg)
    elif cfg.protocol in ("mcdcv", "mcdcv_plsda"):
        report = run_protocol_mcdcv(cfg)
    else:
        raise ValueError(f"Unknown protocol: {cfg.protocol}")

    outdir = Path(args.outdir) if args.outdir else (ROOT / "reports" / "exp")
    ensure_dir(outdir)
    run_tag = _sanitize_tag(args.tag)

    h = _short_hash({"cfg": dataclasses.asdict(cfg)})
    report["run_tag"] = run_tag
    report["hash"] = h

    fname = f"{run_tag}_{cfg.dataset}_{cfg.protocol}_seed{cfg.seed}_d{cfg.sg_deriv}_{cfg.threshold_by}_{h}.json"
    outpath = outdir / fname
    outpath.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"[OK] Saved: {outpath}")

# --- SAFE printing: works for both mcdcv and cv_holdout ---
    if "summary" in report:
        mean_dict = report.get("summary", {}).get("mean", {}) or {}
        std_dict = report.get("summary", {}).get("std", {}) or {}

        print("\n[MCDCV] summary (mean±std):")
        for m, v in mean_dict.items():
            s = std_dict.get(m, {}) or {}
            print(
                f"{m:10s}  "
                f"AUC={v.get('auc', float('nan')):.4f}±{s.get('auc', float('nan')):.4f}  "
                f"PR-AUC={v.get('pr_auc', float('nan')):.4f}±{s.get('pr_auc', float('nan')):.4f}  "
                f"F1={v.get('f1', float('nan')):.4f}±{s.get('f1', float('nan')):.4f}  "
                f"spec={v.get('spec', float('nan')):.4f}±{s.get('spec', float('nan')):.4f}  "
                f"rec={v.get('recall', float('nan')):.4f}±{s.get('recall', float('nan')):.4f}  "
                f"Brier={v.get('brier', float('nan')):.4f}  "
                f"ECE={v.get('ece', float('nan')):.4f}"
            )

    elif "results" in report:
        for m, r in report["results"].items():
            t = r["test"]
            print(
                f"{m:10s}  "
                f"F1={t.get('f1', float('nan')):.4f}  "
                f"spec={t.get('spec', float('nan')):.4f}  "
                f"rec={t.get('recall', float('nan')):.4f}  "
                f"prec={t.get('prec', float('nan')):.4f}  "
                f"auc={t.get('auc', float('nan')):.4f}  "
                f"pr_auc={t.get('pr_auc', float('nan')):.4f}  "
                f"brier={t.get('brier', float('nan')):.4f}  "
                f"ece={t.get('ece', float('nan')):.4f}  "
                f"thr={t.get('thr', float('nan')):.3f}"
           )


if __name__ == "__main__":
    main()