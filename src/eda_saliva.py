# src/eda_saliva.py
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP

matplotlib.use("Agg")


PROC = Path("data/processed")
FIGS = Path("reports/figures")
FIGS.mkdir(parents=True, exist_ok=True)
RPTS = Path("reports")
RPTS.mkdir(parents=True, exist_ok=True)


def get_specs(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"ID", "y", "Label", "Ct", "Ct_range"}]


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(FIGS / name, dpi=150)
    plt.close()


def main():
    train = pd.read_parquet(PROC / "train.parquet")
    external = pd.read_parquet(PROC / "external.parquet")

    spec_cols = get_specs(train)
    w = np.array(list(map(float, spec_cols)))
    order = np.argsort(w)
    w = w[order]

    X = train[spec_cols].to_numpy()[:, order]
    y = train["y"].to_numpy()

    # 1) Сводка и пропуски
    summary_lines = []
    summary_lines.append(f"train shape: {train.shape}, external shape: {external.shape}")
    summary_lines.append(f"n_features (spectral): {len(spec_cols)}")
    if "y" in train:
        summary_lines.append("label counts:\n" + str(train["y"].value_counts(dropna=False)))
    summary_lines.append(f"train NaN: {int(np.isnan(X).sum())}")
    summary_lines.append(f"external NaN: {int(external[spec_cols].isna().sum().sum())}")

    # zero-variance колоноки
    var = X.var(axis=0)
    zero_mask = var == 0
    n_zero = int(zero_mask.sum())
    summary_lines.append(f"zero-variance features: {n_zero}")
    if n_zero > 0:
        pd.Series(np.array(spec_cols)[order][zero_mask]).to_csv(
            RPTS / "zero_variance_cols.csv", index=False
        )

    # 2) Средние/STD по классам
    m_pos = X[y == 1].mean(axis=0)
    m_neg = X[y == 0].mean(axis=0)
    s_pos = X[y == 1].std(axis=0)
    s_neg = X[y == 0].std(axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(w, m_neg, label="neg mean")
    plt.plot(w, m_pos, label="pos mean")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.title("Mean spectra by class")
    savefig("01_mean_by_class.png")

    plt.figure(figsize=(10, 3))
    plt.plot(w, m_pos - m_neg)
    plt.xlabel("Wavenumber")
    plt.ylabel("Δ mean (pos - neg)")
    plt.title("Mean difference (pos - neg)")
    savefig("02_mean_diff.png")

    plt.figure(figsize=(10, 3))
    plt.plot(w, s_neg, label="neg std")
    plt.plot(w, s_pos, label="pos std")
    plt.xlabel("Wavenumber")
    plt.ylabel("STD")
    plt.legend()
    plt.title("STD by class")
    savefig("03_std_by_class.png")

    # 3) Ct (если есть в train)
    if "Ct" in train.columns:
        plt.figure(figsize=(8, 4))
        for cls, name in [(0, "neg"), (1, "pos")]:
            vals = train.loc[train["y"] == cls, "Ct"].dropna()
            if len(vals):
                plt.hist(vals, bins=20, alpha=0.5, label=name)
        plt.xlabel("Ct")
        plt.ylabel("Count")
        plt.legend()
        plt.title("Ct distribution by class")
        savefig("04_ct_hist.png")

    # 4) PCA и UMAP (быстрые проекции)
    pca = PCA(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(pca[:, 0], pca[:, 1], c=y, s=20, alpha=0.8)
    plt.title("PCA (train)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    savefig("05_pca.png")

    um = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(um[:, 0], um[:, 1], c=y, s=20, alpha=0.8)
    plt.title("UMAP (train)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    savefig("06_umap.png")

    # 5) Реплики во внешнем наборе (по ID)
    if "ID" in external.columns:
        ext_counts = external["ID"].value_counts().rename_axis("ID").reset_index(name="n_rows")
        ext_counts.to_csv(RPTS / "external_replicates.csv", index=False)
        summary_lines.append(f"external unique IDs: {ext_counts.shape[0]}")
        summary_lines.append(f"external max replicates per ID: {int(ext_counts['n_rows'].max())}")

    # Сохраняем summary
    (RPTS / "summary.txt").write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print(f"Figures in {FIGS}")


if __name__ == "__main__":
    main()
