# -*- coding: utf-8 -*-
"""Quality Control / extended EDA for saliva FTIR.

Функции:
- Загрузка train/external parquet
- Welch t-test + BH-FDR + Cohen's d
- AUC по схемам нормализации (group-aware CV по ID)
- Реплики (пары) + сводка по ID
- Робастные выбросы (SNV -> PCA(whiten) -> MinCovDet) + χ²-порог
- Анализ метаданных/конфаундеров (таблица значимости + baseline AUC)
- Отчёт (README.md) и все артефакты в reports/eda_qc/<run-id>/

Запуск:
    python src/eda_qc.py
    # или с явным каталогом вывода:
    python src/eda_qc.py --outdir reports/eda_qc
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

# безопасный импорт (sklearn>=1.1)
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore

# backend без E402
plt.switch_backend("Agg")

# ---------- Пути / константы ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
TRAIN_PARQUET = DATA_PROCESSED / "train.parquet"
EXT_PARQUET = DATA_PROCESSED / "external.parquet"

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ---------- Утилиты загрузки / выбор признаков ----------
def _is_number_like(x) -> bool:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return True
    try:
        float(str(x))
        return True
    except Exception:
        return False


def pick_spectral_columns(df: pd.DataFrame) -> List:
    """Столбцы-волновые числа, отсортированные по возрастанию."""
    spec_cols = [c for c in df.columns if _is_number_like(c)]
    return sorted(spec_cols, key=lambda c: float(c))


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Вернуть (spec_cols, meta_cols)."""
    spec_cols = pick_spectral_columns(df)
    service = {"y", "Label", "ID"}
    meta_cols = [c for c in df.columns if c not in set(spec_cols) | service]
    return spec_cols, meta_cols


def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка parquet, подготовленных prepare_data.py."""
    if not TRAIN_PARQUET.exists() or not EXT_PARQUET.exists():
        raise FileNotFoundError(
            "Не найдены подготовленные parquet:\n"
            f"  {TRAIN_PARQUET}\n  {EXT_PARQUET}\n"
            "Сначала запусти: python src/prepare_data.py"
        )
    df_train = pd.read_parquet(TRAIN_PARQUET)
    df_ext = pd.read_parquet(EXT_PARQUET)
    return df_train, df_ext


def get_xy_wns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, List[str]]:
    """Вернуть (X, y, wns, meta_cols) из датафрейма."""
    spec_cols, meta_cols = split_columns(df)
    X = df[spec_cols].copy()
    if "y" in df.columns:
        y = df["y"].astype(int)
    elif "Label" in df.columns:
        y = df["Label"].map({"Negative": 0, "Positive": 1}).astype(int)
    else:
        raise ValueError("Не найдена колонка меток ('y' или 'Label').")
    wns = np.array([float(c) for c in spec_cols], dtype=float)
    return X, y, wns, meta_cols


# ---------- Welch t-test + BH-FDR + Cohen's d ----------
def ttest_curve(
    X: pd.DataFrame,
    y: pd.Series,
    wns: np.ndarray,
    min_n_per_group: int = 3,
) -> pd.DataFrame:
    """Welch t-test по каждому признаку + BH-FDR + Cohen's d."""
    assert X.shape[1] == len(wns), "X и wns должны совпадать по числу признаков"
    pos = X[y == 1].to_numpy()
    neg = X[y == 0].to_numpy()

    nfeat = X.shape[1]
    pvals = np.ones(nfeat, dtype=float)
    qvals = np.ones(nfeat, dtype=float)
    valid = np.zeros(nfeat, dtype=bool)
    reason = np.array([""] * nfeat, dtype=object)

    npos = np.isfinite(pos).sum(axis=0)
    nneg = np.isfinite(neg).sum(axis=0)
    vpos = np.nanvar(pos, axis=0, ddof=1)
    vneg = np.nanvar(neg, axis=0, ddof=1)

    enough = (npos >= min_n_per_group) & (nneg >= min_n_per_group)
    nonconst = (vpos > 0) & (vneg > 0)
    valid = enough & nonconst

    reason[~enough] = "too_few_obs"
    reason[enough & ~nonconst] = "zero_variance"

    idx = np.where(valid)[0]
    for j in idx:
        pv = ttest_ind(pos[:, j], neg[:, j], equal_var=False, nan_policy="omit").pvalue
        if not np.isfinite(pv):
            pv = 1.0
        pvals[j] = pv

    if idx.size > 0:
        qvals[idx] = multipletests(pvals[idx], method="fdr_bh")[1]

    mean_pos = np.nanmean(pos, axis=0)
    mean_neg = np.nanmean(neg, axis=0)
    diff = mean_pos - mean_neg

    pooled = np.sqrt((vpos + vneg) / 2.0)
    effect = np.divide(diff, pooled, out=np.zeros_like(diff), where=pooled > 0)

    return pd.DataFrame(
        {
            "wn": wns,
            "pval": pvals,
            "qval": qvals,
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "diff": diff,
            "effect_size": effect,
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


def plot_volcano(df_p: pd.DataFrame, path: Path) -> None:
    """Volcano: |Cohen's d| vs -log10(q)."""
    plt.figure(figsize=(6, 5))
    x = np.abs(df_p["effect_size"].values)
    y = -np.log10(df_p["qval"].values + 1e-300)
    plt.scatter(x, y, s=10, alpha=0.7)
    plt.xlabel("|Cohen's d|")
    plt.ylabel("-log10(q)")
    plt.title("Volcano: effect size vs significance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------- Нормализации и ROC-AUC по CV (group-aware) ----------
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


NORM_SCHEMES: Dict[str, callable] = {
    "none": norm_none,
    "l2": norm_l2,
    "snv": norm_snv,
    "minmax": norm_minmax,
}


def _iter_splits(
    y: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Разбиение: при наличии повторов ID — группированная CV."""
    if groups is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(skf.split(np.zeros_like(y), y))

    if pd.Series(groups).value_counts().max() == 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(skf.split(np.zeros_like(y), y))

    if StratifiedGroupKFold is not None:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(sgkf.split(np.zeros_like(y), y, groups))

    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros_like(y), y, groups))


def cv_auc_logreg(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
) -> float:
    """ROC-AUC с логрегрессией и стандартизацией по фолду, group-aware."""
    aucs: List[float] = []
    for tr, te in _iter_splits(y, groups, n_splits=n_splits):
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=200, n_jobs=1, random_state=RANDOM_STATE)
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
    return float(np.mean(aucs))


def compare_normalizations(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
) -> pd.DataFrame:
    """Перебор схем нормализации и ROC-AUC по (групповой) CV."""
    Xnp = X.to_numpy(dtype=float)
    g = None if groups is None else groups.to_numpy()
    rows: List[Dict[str, float]] = []
    for name, fn in NORM_SCHEMES.items():
        Xn = fn(Xnp.copy())
        auc = cv_auc_logreg(Xn, y.to_numpy(), g)
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


# ---------- Реплики (внешние): пары + сводка по ID ----------
def replicate_distances(df_ext: pd.DataFrame, spec_cols: List) -> pd.DataFrame:
    """Для каждого ID считаем попарные L2-дистанции между его повторами."""
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


def replicate_summary(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Сводка по каждому ID: n_reps, n_pairs, mean/median/std/max, cv, worst pair."""
    if df_pairs.empty:
        return pd.DataFrame(
            columns=[
                "ID",
                "n_reps",
                "n_pairs",
                "mean_dist",
                "median_dist",
                "std_dist",
                "cv_dist",
                "max_dist",
                "worst_pair",
            ]
        )

    g = df_pairs.groupby("ID")["dist"]
    n_pairs = g.size()
    n_reps = ((1.0 + np.sqrt(1.0 + 8.0 * n_pairs.values)) / 2.0).round().astype(int)

    summary = pd.DataFrame(
        {
            "ID": n_pairs.index.values,
            "n_reps": n_reps,
            "n_pairs": n_pairs.values,
            "mean_dist": g.mean().values,
            "median_dist": g.median().values,
            "std_dist": g.std(ddof=1).fillna(0.0).values,
            "max_dist": g.max().values,
        }
    )

    worst_pair = (
        df_pairs.sort_values(["ID", "dist"], ascending=[True, False])
        .groupby("ID")
        .first()[["pair"]]
        .rename(columns={"pair": "worst_pair"})
        .reset_index()
    )
    summary = summary.merge(worst_pair, on="ID", how="left")
    summary["cv_dist"] = summary["std_dist"] / summary["mean_dist"].replace(0.0, np.nan)

    return summary.sort_values(["max_dist", "mean_dist"], ascending=False, ignore_index=True)


def plot_replicate_hist(df_pairs: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 3))
    if len(df_pairs):
        plt.hist(df_pairs["dist"].values, bins=20)
    plt.xlabel("L2 distance between replicates")
    plt.ylabel("count")
    plt.title("Replicates consistency (external)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------- Робастные выбросы ----------
def robust_outliers(
    X: pd.DataFrame,
    ids: pd.Series | None = None,
    top_k: int = 5,
    max_components: int = 30,
) -> pd.DataFrame:
    """SNV/стандартизация -> PCA(whiten) -> MinCovDet. Возвращает топ-k."""
    Xf = X.to_numpy(float)
    n_samples, n_features = Xf.shape
    n_comp = min(max_components, max(2, n_samples - 5), n_features)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(Xf)
    Xp = PCA(n_components=n_comp, whiten=True, random_state=RANDOM_STATE).fit_transform(Xz)
    mcd = MinCovDet(random_state=RANDOM_STATE, support_fraction=None).fit(Xp)
    dist2 = mcd.mahalanobis(Xp)

    cutoff = float(chi2.ppf(0.999, df=n_comp))  # 99.9%

    order = np.argsort(dist2)[::-1][: min(top_k, len(dist2))]
    res = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "index": X.index.values[order],
            "dist2": dist2[order],
            "is_outlier": dist2[order] > cutoff,
            "cutoff_chi2": cutoff,
            "df": n_comp,
        }
    )
    if ids is not None:
        res["ID"] = ids.values[order]
    return res


# ---------- Метаданные / конфаундеры ----------
def analyze_metadata(
    df: pd.DataFrame, y: pd.Series, meta_cols: List[str], groups: pd.Series | None
) -> Tuple[pd.DataFrame, float | None]:
    """
    Таблица по метаданным (тип, тест, p/q, эффект) + AUC метамодели.
    Эффект:
      - numeric: стандарт. разница средних (SMD)
      - categorical: Cramer's V
    """
    if not meta_cols:
        return pd.DataFrame(columns=["feature", "type", "test", "pval", "qval", "effect"]), None

    rows = []
    for c in meta_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            pos = s[y == 1].astype(float)
            neg = s[y == 0].astype(float)
            pv = ttest_ind(pos, neg, equal_var=False, nan_policy="omit").pvalue
            m1, m0 = np.nanmean(pos), np.nanmean(neg)
            v1, v0 = np.nanvar(pos, ddof=1), np.nanvar(neg, ddof=1)
            sd = np.sqrt((v1 + v0) / 2.0) if np.isfinite(v1 + v0) else np.nan
            smd = (m1 - m0) / sd if (sd and sd > 0) else 0.0
            rows.append(dict(feature=c, type="numeric", test="Welch t", pval=pv, effect=abs(smd)))
        else:
            ct = pd.crosstab(s.fillna("NA"), y)
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2_stat, pv, _, _ = chi2_contingency(ct)
                n = ct.values.sum()
                phi2 = chi2_stat / n
                r, k = ct.shape
                # коррекция Бенакри-Грина для Cramer's V
                phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
                rcorr = r - (r - 1) ** 2 / (n - 1)
                kcorr = k - (k - 1) ** 2 / (n - 1)
                cramers_v = np.sqrt(phi2corr / max(rcorr - 1, kcorr - 1))
                rows.append(
                    dict(feature=c, type="categorical", test="chi2", pval=pv, effect=cramers_v)
                )
            else:
                rows.append(dict(feature=c, type="categorical", test="chi2", pval=1.0, effect=0.0))

    df_meta = pd.DataFrame(rows)
    if len(df_meta):
        df_meta["qval"] = multipletests(df_meta["pval"].values, method="fdr_bh")[1]
    else:
        df_meta["qval"] = []

    # AUC метамодели (logreg на метаданых, one-hot, group-aware CV)
    auc_meta = None
    try:
        Xmeta = df[meta_cols].copy()
        num_cols = [c for c in meta_cols if pd.api.types.is_numeric_dtype(Xmeta[c])]
        cat_cols = [c for c in meta_cols if c not in num_cols]

        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop",
        )
        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=500, n_jobs=1, random_state=RANDOM_STATE)),
            ]
        )

        # сбор фолдов
        g = None if groups is None else groups.to_numpy()
        y_np = y.to_numpy()
        aucs: List[float] = []
        for tr, te in _iter_splits(y_np, g, n_splits=5):
            pipe.fit(Xmeta.iloc[tr], y_np[tr])
            prob = pipe.predict_proba(Xmeta.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y_np[te], prob))
        auc_meta = float(np.mean(aucs))
    except Exception:
        auc_meta = None

    return df_meta, auc_meta


# ---------- One-glance Markdown отчёт ----------
def write_report_md(
    out_dir: Path,
    df_train: pd.DataFrame,
    df_ext: pd.DataFrame,
    df_p: pd.DataFrame,
    df_norm: pd.DataFrame,
    df_pairs: pd.DataFrame,
    df_rep: pd.DataFrame,
    df_out: pd.DataFrame,
    df_meta: pd.DataFrame,
    auc_meta: float | None,
) -> None:
    lines: List[str] = []
    fig = out_dir / "figures"

    lines.append("# QC / EDA summary")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- train shape: **{df_train.shape}**")
    lines.append(f"- external shape: **{df_ext.shape}**")
    lines.append(f"- spectral features: **{df_p.shape[0]}**")
    lines.append("")

    # T-test
    n_total = len(df_p)
    n_valid = int(df_p["valid"].sum())
    n_sig = int((df_p["qval"] < 0.05).sum())
    lines.append("## T-test (Welch) + BH-FDR")
    lines.append(f"- total features: **{n_total}**, valid: **{n_valid}**, q<0.05: **{n_sig}**")
    lines.append(f"- Figure: [q-values]({(fig / 'qc_04_pvalues.png').as_posix()})")
    lines.append(f"- Figure: [volcano]({(fig / 'qc_05_volcano.png').as_posix()})")
    lines.append(f"- Table: [ttest_curve.csv]({(out_dir / 'ttest_curve.csv').as_posix()})")
    lines.append("")

    # Normalizations
    lines.append("## Normalization vs ROC-AUC (group-aware CV)")
    if len(df_norm):
        best = df_norm.iloc[0]
        ranking = ", ".join(f"{r.scheme}={r.roc_auc:.3f}" for r in df_norm.itertuples())
        lines.append(f"- ranking: {ranking}")
        lines.append(f"- best: **{best.scheme} ({best.roc_auc:.3f})**")
        lines.append(f"- Figure: [qc_06_norm_auc.png]({(fig / 'qc_06_norm_auc.png').as_posix()})")
        lines.append(f"- Table: [norm_auc.csv]({(out_dir / 'norm_auc.csv').as_posix()})")
    else:
        lines.append("- no CV results available")
    lines.append("")

    # Replicates (external)
    lines.append("## Replicates (external)")
    if df_pairs.empty or df_rep.empty:
        lines.append("- no replicate pairs found")
    else:
        med_mean = df_rep["mean_dist"].median()
        med_cv = df_rep["cv_dist"].median() if "cv_dist" in df_rep.columns else float("nan")
        lines.append(
            f"- IDs with replicates: **{df_rep.shape[0]}** | "
            f"median(mean_dist): **{med_mean:.3f}** | median(cv): **{med_cv:.3f}**"
        )
        lines.append(
            "- Figure: [replicate histogram]("
            f"{(fig / 'qc_07_replicate_dist.png').as_posix()}"
            ")"
        )
        lines.append(
            "- Pairwise: [replicate_distances.csv]("
            f"{(out_dir / 'replicate_distances.csv').as_posix()}"
            ")"
        )
        lines.append(
            "- Per-ID summary: [external_replicates.csv]("
            f"{(out_dir / 'external_replicates.csv').as_posix()}"
            ")"
        )
    lines.append("")

    # Outliers
    lines.append("## Robust outliers (train)")
    if len(df_out):
        cutoff = (
            float(df_out["cutoff_chi2"].iloc[0])
            if "cutoff_chi2" in df_out.columns and len(df_out)
            else None
        )
        df_chi = int(df_out["df"].iloc[0]) if "df" in df_out.columns and len(df_out) else None
        if cutoff is not None and df_chi is not None:
            lines.append(f"- χ² cutoff (df={df_chi}, 0.999): **{cutoff:.2f}**")
        if "is_outlier" in df_out.columns:
            n_flag = int(df_out["is_outlier"].sum())
            lines.append(f"- flagged among top-{len(df_out)}: **{n_flag}**")
        top = []
        for r in df_out.itertuples(index=False):
            rid = f" (ID={r.ID})" if "ID" in df_out.columns else ""
            mark = " *" if "is_outlier" in df_out.columns and r.is_outlier else ""
            top.append(f"{r.index}{rid}{mark}")
        lines.append("- top indices: " + ", ".join(top))
        lines.append(
            f"- Table: [outliers_train.csv]({(out_dir / 'outliers_train.csv').as_posix()})"
        )
    else:
        lines.append("- no outliers table produced")
    lines.append("")

    # Metadata
    lines.append("## Metadata / confounders")
    if len(df_meta):
        n_sigm = int((df_meta["qval"] < 0.05).sum())
        lines.append(f"- meta columns: **{len(df_meta)}**, q<0.05: **{n_sigm}**")
        if auc_meta is not None:
            lines.append(
                f"- baseline AUC using *only* metadata: **{auc_meta:.3f}** (group-aware CV)"
            )
        lines.append(
            f"- Table: [metadata_summary.csv]({(out_dir / 'metadata_summary.csv').as_posix()})"
        )
    else:
        lines.append("- no metadata found")
    lines.append("")

    lines.append("---")
    lines.append(
        "_Note_: CV is **group-aware** (by `ID`) to avoid data leakage. "
        "External set is never used for training."
    )
    lines.append("")

    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ---------- Главный сценарий ----------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "reports" / "eda_qc"),
        help="Базовая папка для артефактов EDA/QC (будет создан подпапкой с таймстампом)",
    )
    args = parser.parse_args()

    # каталог вывода: reports/eda_qc/<run-id>/
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    base_out = Path(args.outdir)
    out_dir = base_out / run_id
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # "latest" симлинк (best-effort)
    try:
        latest = base_out / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        os.symlink(out_dir.name, latest, target_is_directory=True)
    except Exception:
        # если нет прав на symlink — просто игнор
        pass

    # 0) Data
    df_train, df_ext = load_processed()
    X, y, wns, meta_cols = get_xy_wns(df_train)
    spec_cols_ext, _ = split_columns(df_ext)

    print(f"train shape: {df_train.shape}, external shape: {df_ext.shape}")
    print(f"n_features (spectral): {X.shape[1]}")

    # 1) t-test + BH-FDR (+ Volcano)
    df_p = ttest_curve(X, y, wns)
    df_p.to_csv(out_dir / "ttest_curve.csv", index=False)
    plot_qvalues(df_p, fig_dir / "qc_04_pvalues.png")
    plot_volcano(df_p, fig_dir / "qc_05_volcano.png")
    n_total = len(df_p)
    n_valid = int(df_p["valid"].sum())
    n_sig = int((df_p["qval"] < 0.05).sum())

    # 2) Нормализации (group-aware CV)
    groups = df_train["ID"] if "ID" in df_train.columns else None
    df_norm = compare_normalizations(X, y, groups)
    df_norm.to_csv(out_dir / "norm_auc.csv", index=False)
    plot_norm_auc(df_norm, fig_dir / "qc_06_norm_auc.png")

    # 3) Реплики: пары и сводка по ID
    df_pairs = replicate_distances(df_ext, spec_cols_ext)
    df_pairs.to_csv(out_dir / "replicate_distances.csv", index=False)
    plot_replicate_hist(df_pairs, fig_dir / "qc_07_replicate_dist.png")
    df_rep = replicate_summary(df_pairs)
    df_rep.to_csv(out_dir / "external_replicates.csv", index=False)

    # 4) Выбросы (SNV -> PCA -> MCD)
    X_snv = pd.DataFrame(norm_snv(X.to_numpy(float)), index=X.index, columns=X.columns)
    ids_series = df_train["ID"] if "ID" in df_train.columns else None
    df_out = robust_outliers(X_snv, ids_series, top_k=5)
    df_out.to_csv(out_dir / "outliers_train.csv", index=False)

    # 5) Метаданные / конфаундеры
    df_meta, auc_meta = analyze_metadata(df_train, y, meta_cols, groups)
    if len(df_meta):
        df_meta.sort_values("qval").to_csv(out_dir / "metadata_summary.csv", index=False)

    # 6) Короткая текстовая сводка
    summ: List[str] = []
    summ.append(f"train shape: {df_train.shape}, external shape: {df_ext.shape}")
    summ.append(f"n wavenumbers: {X.shape[1]}")
    summ.append(f"ttest: total={n_total}, valid={n_valid}, q<0.05={n_sig}")
    if len(df_norm):
        summ.append(
            "AUC by normalization: "
            + ", ".join(f"{row.scheme}={row.roc_auc:.3f}" for row in df_norm.itertuples())
        )
    if not df_pairs.empty:
        summ.append(
            "replicates (per-ID): "
            f"mean(mean_dist)={df_rep['mean_dist'].mean():.4f}, "
            f"median(mean_dist)={df_rep['mean_dist'].median():.4f}"
        )
    if len(df_out):
        best = df_out.iloc[0]
        sid = f", ID={best['ID']}" if "ID" in df_out.columns else ""
        summ.append(f"top outlier: idx={best['index']}{sid}, dist2={best['dist2']:.3f}")
    if auc_meta is not None:
        summ.append(f"metadata-only AUC: {auc_meta:.3f}")
    (out_dir / "summary.txt").write_text("\n".join(summ) + "\n", encoding="utf-8")
    print("\n".join(summ))
    print(f"Figures -> {fig_dir}")

    # 7) README one-glance
    write_report_md(
        out_dir=out_dir,
        df_train=df_train,
        df_ext=df_ext,
        df_p=df_p,
        df_norm=df_norm,
        df_pairs=df_pairs,
        df_rep=df_rep,
        df_out=df_out,
        df_meta=df_meta,
        auc_meta=auc_meta,
    )


if __name__ == "__main__":
    main()
