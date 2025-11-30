Проект FTIR (НИР): каркас создан.

# Saliva FTIR — Baseline EDA & QC

**Данные.** Train: `data/processed/train.parquet` (n=61), External (replicates): `external.parquet` (n=183).
**Ключевые цифры:**
- ROC-AUC (Stratified 5-fold): **SNV 0.906**, L2 0.891, MinMax 0.853, None 0.763
- BH-FDR: `q < 0.05` — 0 волн (одномерно)
- Репликаты (external): L2 median ≈ 0.684 (mean ≈ 1.132)
- Топовый выброс (train): `ID=N16` (dist² ≈ 60.0)

## Фигуры
![Mean spectra](../reports/figures/01_mean_by_class.png)
![Δ mean](../reports/figures/02_mean_diff.png)
![STD](../reports/figures/03_std_by_class.png)
![p-values](../reports/figures/qc_04_pvalues.png)
![PCA](../reports/figures/05_pca.png)
![UMAP](../reports/figures/06_umap.png)
![AUC by normalization](../reports/figures/qc_06_norm_auc.png)
![Replicate distances](../reports/figures/qc_07_replicate_dist.png)

## CSV артефакты
- [`ttest_curve.csv`](../reports/ttest_curve.csv) — по каждой волне: `wn, pval, qval, mean_pos, mean_neg, diff, valid`.
- [`norm_auc.csv`](../reports/norm_auc.csv) — AUC по нормализациям.
- [`replicate_distances.csv`](../reports/replicate_distances.csv) — L2 по репликам.
- [`outliers_train.csv`](../reports/outliers_train.csv) — топ выбросов (Mahalanobis).

## Как воспроизвести

- `src/prepare_data.py` — сбор обучающего и внешнего наборов, проверка NaN, формирование parquet.
- `src/eda_saliva.py` — базовые графики (средние/STD, PCA/UMAP).
- `src/eda_qc.py` — расширенный QC: p-values (BH-FDR), поиски выбросов (robust Mahalanobis), сравнение нормализаций (GroupKFold ROC-AUC), анализ репликатов.



Подробные числа — в `reports/summary.txt`.
