# QC / EDA summary

## Dataset
- train shape: **(61, 253)**
- external shape: **(183, 252)**
- spectral features: **251**

## T-test (Welch) + BH-FDR
- total features: **251**, valid: **251**, q<0.05: **0**
- Figure: [q-values](/work/nir-ftir/reports/eda_qc/20251201-0919/figures/qc_04_pvalues.png)
- Figure: [volcano](/work/nir-ftir/reports/eda_qc/20251201-0919/figures/qc_05_volcano.png)
- Table: [ttest_curve.csv](/work/nir-ftir/reports/eda_qc/20251201-0919/ttest_curve.csv)

## Normalization vs ROC-AUC (group-aware CV)
- ranking: snv=0.906, l2=0.891, minmax=0.853, none=0.763
- best: **snv (0.906)**
- Figure: [qc_06_norm_auc.png](/work/nir-ftir/reports/eda_qc/20251201-0919/figures/qc_06_norm_auc.png)
- Table: [norm_auc.csv](/work/nir-ftir/reports/eda_qc/20251201-0919/norm_auc.csv)

## Replicates (external)
- IDs with replicates: **61** | median(mean_dist): **0.970** | median(cv): **0.552**
- Figure: [replicate histogram](/work/nir-ftir/reports/eda_qc/20251201-0919/figures/qc_07_replicate_dist.png)
- Pairwise: [replicate_distances.csv](/work/nir-ftir/reports/eda_qc/20251201-0919/replicate_distances.csv)
- Per-ID summary: [external_replicates.csv](/work/nir-ftir/reports/eda_qc/20251201-0919/external_replicates.csv)

## Robust outliers (train)
- χ² cutoff (df=30, 0.999): **59.70**
- flagged among top-5: **5**
- top indices: 22 (ID=N23) *, 54 (ID=P25) *, 20 (ID=N21) *, 57 (ID=P28) *, 16 (ID=N17) *
- Table: [outliers_train.csv](/work/nir-ftir/reports/eda_qc/20251201-0919/outliers_train.csv)

## Metadata / confounders
- no metadata found

---
_Note_: CV is **group-aware** (by `ID`) to avoid data leakage. External set is never used for training.
