# -*- coding: utf-8 -*-
"""Quality Control / extended EDA for saliva FTIR."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Настройка matplotlib для headless-режима ДО импорта pyplot.
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.covariance import MinCovDet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------- Пути / константы ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
TRAIN_PARQUET = DATA_PROCESSED / "train.parquet"
EXT_PARQUET = DATA_PROCESSED / "external.parquet"

REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
SUMMARY = REPORTS / "summary.txt"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ---------- Утилиты загрузки / выбор признаков ----------
def _is_number_like(x) -> bool:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return True
    try:
        float(str(x))
        return True
    except Exception:
        return False


def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка parquet, подготовленных prepare_data.py."""
    if not TRAIN_PARQUET.exists() or not EXT_PARQUET.exists():
        raise FileNotFoundError(
            f"Не найдены подготовленные parquet:\n  {TRAIN_PARQUET}\n  {EXT_PARQUET}\n"
            "Сначала запусти: python src/prepare_data.py"
        )
    df_train = pd.read_parquet(TRAIN_PARQUET)
    df_ext = pd.read_parquet(EXT_PARQUET)
    return df_train, df_ext


def pick_spectral_columns(df: pd.DataFrame) -> List[str]:
    """Выбрать столбцы, имена которых — волновые числа (числа). Отсортировать по возрастанию."""
    spec_cols = [c for c in df.columns if _is_number_like(c)]
    return sorted(spec_cols, key=lambda c: float(c))


def get_xy_wns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Вернуть (X, y, wns) из датафрейма."""
    spec_cols = pick_spectral_columns(df)
    X = df[spec_cols].copy()
    if "y" in df.columns:
        y = df["y"].astype(int)
    elif "Label" in df.columns:
        y = df["Label"].map({"Negative": 0, "Positive": 1}).astype(int)
    else:
        raise ValueError("Не найдена колонка меток ('y' или 'Label').")
    wns = np.array([float(c) for c in spec_cols], dtype=float)
    return X, y, wns


# ---------- Статистика: Welch t-test + BH-FDR ----------
def ttest_curve(
    X: pd.DataFrame,
    y: pd.Series,
    wns: np.ndarray,
    min_n_per_group: int = 3,
) -> pd.DataFrame:
    """
    Welch t-test по каждому признаку + коррекция BH-FDR.
    Безопасно обрабатывает NaN/нулевую дисперсию.

    Возвращает DataFrame со столбцами:
    wn, pval, qval, mean_pos, mean_neg, diff, effect_size, valid, reason
    """
    assert X.shape[1] == len(wns), "X и wns должны совпадать по числу признаков"

    pos = X[y == 1].to_numpy()
    neg = X[y == 0].to_numpy()

    nfeat = X.shape[1]
    pvals = np.ones(nfeat, dtype=float)
    qvals = np.ones(nfeat, dtype=float)
    valid = np.zeros(nfeat, dtype=bool)
    reason = np.array([""] * nfeat, dtype=object)

    # достаточность наблюдений и ненулевая дисперсия
    npos = np.isfinite(pos).sum(axis=0)
    nneg = np.isfinite(neg).sum(axis=0)
    vpos = np.nanvar(pos, axis=0, ddof=1)
    vneg = np.nanvar(neg, axis=0, ddof=1)

    enough = (npos >= min_n_per_group) & (nneg >= min_n_per_group)
    nonconst = (vpos > 0) & (vneg > 0)
    valid = enough & nonconst

    reason[~enough] = "too_few_obs"
    reason[enough & ~nonconst] = "zero_variance"
    reason[valid] = ""

    # p-values
    idx = np.where(valid)[0]
    for j in idx:
        pv = ttest_ind(pos[:, j], neg[:, j], equal_var=False, nan_policy="omit").pvalue
        if not np.isfinite(pv):
            pv = 1.0
        pvals[j] = pv

    # BH-FDR
    if idx.size > 0:
        qvals[idx] = multipletests(pvals[idx], method="fdr_bh")[1]

    # средние и эффект Кохена
    mean_pos = np.nanmean(pos, axis=0)
    mean_neg = np.nanmean(neg, axis=0)
    diff = mean_pos - mean_neg

    pooled_sd = np.sqrt((vpos + vneg) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        effect_size = np.where(pooled_sd > 0, diff / pooled_sd, np.nan)

    return pd.DataFrame(
        {
            "wn": wns,
            "pval": pvals,
            "qval": qvals,
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "diff": diff,
            "effect_size": effect_size,
            "valid": valid,
            "reason": reason,
        }
    )


def plot_qvalues(df_p: pd.DataFrame, path: Path) -> None:
    mask = df_p["valid"].values
    plt.figure(figsize=(10, 3))
    if mask.any():
        x = df_p.loc[mask, "wn"].values
        y = -np.log10(df_p.loc[mask, "qval"].values + 1e-300)
        plt.plot(x, y)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("-log10(q)")
    plt.title("BH-FDR q-values (valid features)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------- Нормализации и ROC-AUC по CV ----------
def norm_none(X: np.ndarray) -> np.ndarray:
    return X


def norm_l2(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm


def norm_snv(X: np.ndarray) -> np.ndarray:
    mu = np.mean(X, axis=1, keepdims=True)
    sd = np.std(X, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def norm_minmax(X: np.ndarray) -> np.ndarray:
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    rng = mx - mn
    rng[rng == 0] = 1.0
    return (X - mn) / rng


NORM_SCHEMES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": norm_none,
    "l2": norm_l2,
    "snv": norm_snv,
    "minmax": norm_minmax,
}


def cv_auc_logreg(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """Простой ROC-AUC с логистической регрессией и стандартизацией по фолду."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs: List[float] = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(
            solver="liblinear", max_iter=200, n_jobs=1, random_state=RANDOM_STATE
        )
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
    return float(np.mean(aucs))


def compare_normalizations(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Перебор схем нормализации и ROC-AUC по StratifiedKFold."""
    Xnp = X.to_numpy(dtype=float)
    rows: List[Dict[str, float]] = []
    for name, fn in NORM_SCHEMES.items():
        Xn = fn(Xnp.copy())
        auc = cv_auc_logreg(Xn, y.to_numpy())
        rows.append({"scheme": name, "roc_auc": auc})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False, ignore_index=True)


def plot_norm_auc(df_norm: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 3))
    plt.bar(df_norm["scheme"], df_norm["roc_auc"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("ROC-AUC")
    plt.title("AUC by normalization")
    for i, v in enumerate(df_norm["roc_auc"].values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------- Реплики (внешние) ----------
def replicate_distances(df_ext: pd.DataFrame, spec_cols: List[str]) -> pd.DataFrame:
    """Для каждого ID во внешнем наборе считаем попарные L2-дистанции."""
    arr = df_ext[spec_cols].to_numpy(float)
    ids = (
        df_ext["ID"].astype(str).to_numpy()
        if "ID" in df_ext.columns
        else df_ext.index.astype(str).to_numpy()
    )
    rows: List[Dict[str, float]] = []
    for uid in sorted(np.unique(ids)):
        idx = np.where(ids == uid)[0]
        if idx.size < 2:
            continue
        for i in range(idx.size):
            for j in range(i + 1, idx.size):
                d = np.linalg.norm(arr[idx[i]] - arr[idx[j]])
                rows.append({"ID": uid, "pair": f"{i+1}-{j+1}", "dist": float(d)})
    return pd.DataFrame(rows)


def plot_replicate_hist(df_dist: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 3))
    if len(df_dist):
        plt.hist(df_dist["dist"].values, bins=20)
    plt.xlabel("L2 distance between replicates")
    plt.ylabel("count")
    plt.title("Replicates consistency (external)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------- Робастные выбросы (MinCovDet) ----------
def robust_outliers(X: pd.DataFrame, ids: pd.Series | None = None, top_k: int = 5) -> pd.DataFrame:
    """Робастная ковариация + Махаланобисова дистанция. Возвращает топ-k наблюдений."""
    mcd = MinCovDet(random_state=RANDOM_STATE, support_fraction=None).fit(X.to_numpy(float))
    dist2 = mcd.mahalanobis(X.to_numpy(float))  # squared Mahalanobis
    order = np.argsort(dist2)[::-1][: min(top_k, len(dist2))]
    res = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "index": X.index.values[order],
            "dist2": dist2[order],
        }
    )
    if ids is not None:
        res["ID"] = ids.values[order]
    return res


# ---------- Главный сценарий ----------
def main() -> None:
    df_train, df_ext = load_processed()
    X, y, wns = get_xy_wns(df_train)
    spec_cols = pick_spectral_columns(df_ext)

    print(f"train shape: {df_train.shape}, external shape: {df_ext.shape}")
    print(f"n_features (spectral): {X.shape[1]}")

    # 1) t-test + BH-FDR (+ Cohen's d)
    df_p = ttest_curve(X, y, wns)
    df_p.to_csv(REPORTS / "ttest_curve.csv", index=False)
    plot_qvalues(df_p, FIG_DIR / "qc_04_pvalues.png")
    n_total = len(df_p)
    n_valid = int(df_p["valid"].sum())
    n_sig = int((df_p["qval"] < 0.05).sum())

    # 2) Сравнение нормализаций (AUC)
    df_norm = compare_normalizations(X, y)
    df_norm.to_csv(REPORTS / "norm_auc.csv", index=False)
    plot_norm_auc(df_norm, FIG_DIR / "qc_06_norm_auc.png")

    # 3) Реплики – расстояния (external)
    df_dist = replicate_distances(df_ext, spec_cols)
    df_dist.to_csv(REPORTS / "replicate_distances.csv", index=False)
    plot_replicate_hist(df_dist, FIG_DIR / "qc_07_replicate_dist.png")

    # 4) Робастные выбросы на train (на SNV-нормализованных данных)
    X_snv = pd.DataFrame(norm_snv(X.to_numpy(float)), index=X.index, columns=X.columns)
    ids_series = df_train["ID"] if "ID" in df_train.columns else None
    df_out = robust_outliers(X_snv, ids_series, top_k=5)
    df_out.to_csv(REPORTS / "outliers_train.csv", index=False)

    # 5) Короткая сводка
    summ: List[str] = []
    summ.append(f"train shape: {df_train.shape}, external shape: {df_ext.shape}")
    summ.append(f"n wavenumbers: {X.shape[1]}")
    summ.append(f"ttest: total={n_total}, valid={n_valid}, q<0.05={n_sig}")
    summ.append(
        "AUC by normalization: "
        + ", ".join(f"{row.scheme}={row.roc_auc:.3f}" for row in df_norm.itertuples())
    )
    if len(df_dist):
        summ.append(
            f"replicate distances: mean={df_dist['dist'].mean():.4f}, "
            f"median={df_dist['dist'].median():.4f}"
        )
    if len(df_out):
        best = df_out.iloc[0]
        sid = f", ID={best['ID']}" if "ID" in df_out.columns else ""
        summ.append(f"top outlier: idx={best['index']}{sid}, dist2={best['dist2']:.3f}")

    SUMMARY.write_text("\n".join(summ) + "\n", encoding="utf-8")
    print("\n".join(summ))
    print(f"\nFigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
