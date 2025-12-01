# -*- coding: utf-8 -*-
"""Quality Control / extended EDA for saliva FTIR."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_ind
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

try:  # sklearn >= 1.1
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore

# переключаем бэкенд уже ПОСЛЕ импортов (без E402)
plt.switch_backend("Agg")


# ---------- Пути / константы ----------

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
TRAIN_PARQUET = DATA_PROCESSED / "train.parquet"
EXT_PARQUET = DATA_PROCESSED / "external.parquet"

REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
SUMMARY_TXT = REPORTS / "summary.txt"
README_MD = REPORTS / "README.md"

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
            "Не найдены подготовленные parquet:\n"
            f"  {TRAIN_PARQUET}\n  {EXT_PARQUET}\n"
            "Сначала запусти: python src/prepare_data.py"
        )
    df_train = pd.read_parquet(TRAIN_PARQUET)
    df_ext = pd.read_parquet(EXT_PARQUET)
    return df_train, df_ext


def pick_spectral_columns(df: pd.DataFrame) -> List:
    """Столбцы-волновые числа, отсортированные по возрастанию."""
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


# ---------- Welch t-test + BH-FDR + Cohen's d (устойчиво к NaN) ----------


def ttest_curve(
    X: pd.DataFrame,
    y: pd.Series,
    wns: np.ndarray,
    min_n_per_group: int = 3,
) -> pd.DataFrame:
    """
    По каждому признаку делаем Welch t-test + BH-FDR.
    Возвращает DataFrame:
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

    # Cohen's d (pooled SD по классам, устойчиво к NaN)
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
    """Volcano: |Cohen's d| по X и -log10(q) по Y."""
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
    """Стратегия разбиения: при наличии повторов ID — группированная CV."""
    if groups is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(skf.split(np.zeros_like(y), y))

    # если все ID уникальны — обычная стратификация
    if pd.Series(groups).value_counts().max() == 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(skf.split(np.zeros_like(y), y))

    # попытаться использовать StratifiedGroupKFold (если доступен)
    if StratifiedGroupKFold is not None:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(sgkf.split(np.zeros_like(y), y, groups))

    # fallback: GroupKFold без стратификации (будет небольшой дисбаланс)
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
    """
    Для каждого ID считаем попарные L2-дистанции между его повторами.
    Возвращает таблицу: ID, pair, dist.
    """
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
    # n_reps из n_pairs = n_reps*(n_reps-1)/2
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


# ---------- Робастные выбросы (PCA -> MinCovDet) ----------


def robust_outliers(
    X: pd.DataFrame,
    ids: pd.Series | None = None,
    top_k: int = 5,
    max_components: int = 30,
) -> pd.DataFrame:
    """
    SNV/стандартизация -> PCA(whiten) -> MinCovDet.
    Возвращает топ-k с маркировкой is_outlier по χ² порогу.
    """
    Xf = X.to_numpy(float)
    n_samples, n_features = Xf.shape
    n_comp = min(max_components, max(2, n_samples - 5), n_features)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(Xf)

    # whitening стабилизирует шкалу компонент
    Xp = PCA(n_components=n_comp, whiten=True, random_state=RANDOM_STATE).fit_transform(Xz)

    mcd = MinCovDet(random_state=RANDOM_STATE, support_fraction=None).fit(Xp)
    dist2 = mcd.mahalanobis(Xp)

    # теоретический порог для квадратов Махаланобиса
    cutoff = float(chi2.ppf(0.999, df=n_comp))  # 99.9% квантиль

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


# ---------- One-glance Markdown отчёт ----------


def write_report_md(
    df_train: pd.DataFrame,
    df_ext: pd.DataFrame,
    df_p: pd.DataFrame,
    df_norm: pd.DataFrame,
    df_pairs: pd.DataFrame,
    df_rep: pd.DataFrame,
    df_out: pd.DataFrame,
) -> None:
    """Сформировать one-glance отчёт в reports/README.md."""
    lines: List[str] = []

    # Заголовок
    lines.append("# QC / EDA summary")
    lines.append("")

    # Dataset
    lines.append("## Dataset")
    lines.append(f"- train shape: **{df_train.shape}**")
    lines.append(f"- external shape: **{df_ext.shape}**")
    lines.append(f"- spectral features: **{df_p.shape[0]}**")
    lines.append("")

    # T-test + BH-FDR
    n_total = len(df_p)
    n_valid = int(df_p["valid"].sum())
    n_sig = int((df_p["qval"] < 0.05).sum())

    lines.append("## T-test (Welch) + BH-FDR")
    lines.append(f"- total features: **{n_total}**, valid: **{n_valid}**, q<0.05: **{n_sig}**")
    lines.append("- Figure: [q-values](./figures/qc_04_pvalues.png)")
    lines.append("- Table: [ttest_curve.csv](./ttest_curve.csv)")
    lines.append("")

    # Normalization vs ROC-AUC (group-aware CV)
    lines.append("## Normalization vs ROC-AUC (group-aware CV)")
    if len(df_norm):
        best = df_norm.iloc[0]
        ranking = ", ".join(f"{row.scheme}={row.roc_auc:.3f}" for row in df_norm.itertuples())
        lines.append(f"- ranking: {ranking}")
        lines.append(f"- best: **{best.scheme} ({best.roc_auc:.3f})**")
        lines.append("- Figure: [qc_06_norm_auc.png](./figures/qc_06_norm_auc.png)")
        lines.append("- Table: [norm_auc.csv](./norm_auc.csv)")
    else:
        lines.append("- no CV results available")
    lines.append("")

    # Replicates (external) — агрегированная сводка по ID
    lines.append("## Replicates (external)")
    if df_pairs.empty or df_rep.empty:
        lines.append("- no replicate pairs found")
    else:
        med_mean = df_rep["mean_dist"].median()
        med_cv = df_rep["cv_dist"].median() if "cv_dist" in df_rep.columns else float("nan")
        lines.append(
            f"- IDs with replicates: **{df_rep.shape[0]}** | "
            f"median(mean_dist): **{med_mean:.3f}** | "
            f"median(cv): **{med_cv:.3f}**"
        )
        lines.append("- Worst pair per ID is recorded in `worst_pair` and `max_dist`.")
        lines.append("- Figure: [replicate histogram](./figures/qc_07_replicate_dist.png)")
        lines.append("- Pairwise: [replicate_distances.csv](./replicate_distances.csv)")
        lines.append("- Per-ID summary: [external_replicates.csv](./external_replicates.csv)")
    lines.append("")

    # Robust outliers (train) — с χ²-порогом
    lines.append("## Robust outliers (train)")
    if len(df_out):
        # если есть поля cutoff/df и флаг is_outlier — отобразим
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

        top_list = []
        for r in df_out.itertuples(index=False):
            rid = f" (ID={r.ID})" if "ID" in df_out.columns else ""
            mark = " *" if "is_outlier" in df_out.columns and r.is_outlier else ""
            top_list.append(f"{r.index}{rid}{mark}")
        lines.append("- top indices: " + ", ".join(top_list))
        lines.append("- Table: [outliers_train.csv](./outliers_train.csv)")
    else:
        lines.append("- no outliers table produced")
    lines.append("")

    # Примечания
    lines.append("---")
    lines.append(
        "_Note_: CV is **group-aware** (by `ID`) to avoid data leakage. "
        "External set is never used for training."
    )
    lines.append("")

    # Сохранение
    out_md = REPORTS / "README.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    README_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    plot_volcano(df_p, FIG_DIR / "qc_05_volcano.png")
    n_total = len(df_p)
    n_valid = int(df_p["valid"].sum())
    n_sig = int((df_p["qval"] < 0.05).sum())

    # 2) Нормализации (group-aware CV, чтобы не утекали ID)
    groups = df_train["ID"] if "ID" in df_train.columns else None
    df_norm = compare_normalizations(X, y, groups)
    df_norm.to_csv(REPORTS / "norm_auc.csv", index=False)
    plot_norm_auc(df_norm, FIG_DIR / "qc_06_norm_auc.png")

    # 3) Реплики: пары и сводка по ID
    df_pairs = replicate_distances(df_ext, spec_cols)
    df_pairs.to_csv(REPORTS / "replicate_distances.csv", index=False)
    plot_replicate_hist(df_pairs, FIG_DIR / "qc_07_replicate_dist.png")
    df_rep = replicate_summary(df_pairs)
    df_rep.to_csv(REPORTS / "external_replicates.csv", index=False)

    # 4) Робастные выбросы (SNV -> PCA -> MCD)
    X_snv = pd.DataFrame(norm_snv(X.to_numpy(float)), index=X.index, columns=X.columns)
    ids_series = df_train["ID"] if "ID" in df_train.columns else None
    df_out = robust_outliers(X_snv, ids_series, top_k=5)
    df_out.to_csv(REPORTS / "outliers_train.csv", index=False)

    # 5) Короткая сводка в .txt
    summ: List[str] = []
    summ.append(f"train shape: {df_train.shape}, external shape: {df_ext.shape}")
    summ.append(f"n wavenumbers: {X.shape[1]}")
    summ.append(f"ttest: total={n_total}, valid={n_valid}, q<0.05={n_sig}")
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
    SUMMARY_TXT.write_text("\n".join(summ) + "\n", encoding="utf-8")
    print("\n".join(summ))
    print(f"Figures -> {FIG_DIR}")

    # 6) One-glance Markdown отчёт
    write_report_md(
        df_train=df_train,
        df_ext=df_ext,
        df_p=df_p,
        df_norm=df_norm,
        df_pairs=df_pairs,
        df_rep=df_rep,
        df_out=df_out,
    )


if __name__ == "__main__":
    main()
