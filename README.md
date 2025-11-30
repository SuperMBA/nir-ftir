Проект FTIR (НИР): каркас создан.
## Baseline EDA / QC

- `src/prepare_data.py` — сбор обучающего и внешнего наборов, проверка NaN, формирование parquet.
- `src/eda_saliva.py` — базовые графики (средние/STD, PCA/UMAP).
- `src/eda_qc.py` — расширенный QC: p-values (BH-FDR), поиски выбросов (robust Mahalanobis), сравнение нормализаций (GroupKFold ROC-AUC), анализ репликатов.

### Ключевые рисунки
![](reports/figures/01_mean_by_class.png)
![](reports/figures/02_mean_diff.png)
![](reports/figures/03_std_by_class.png)
![](reports/figures/qc_04_pvalues.png)
![](reports/figures/05_pca.png)
![](reports/figures/06_umap.png)
![](reports/figures/qc_06_norm_auc.png)
![](reports/figures/qc_07_replicate_dist.png)

Подробные числа — в `reports/summary.txt`.
