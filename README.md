# Augmentation of ATR-FTIR Spectra in Biomedical Fluids under Small and Imbalanced Data Conditions

This repository contains the code and reproducible research pipeline for a machine learning project focused on ATR-FTIR spectra of saliva and gingival crevicular fluid.

The project investigates how data augmentation affects supervised model performance, probability calibration, PCA-based data geometry and the quality of synthetic spectra in biomedical datasets with limited sample size and class imbalance.

The repository supports a master’s thesis in Data Science and is designed as a compact, reproducible and verifiable research codebase.

## Project Overview

The project includes three connected analysis tracks:

1. **Supervised evaluation**
   Comparison of baseline and augmented scenarios using identical splits, models, preprocessing profiles, threshold rules and calibration schemes.

2. **Geometry-first analysis**
   Analysis of changes in PCA-based data geometry and factor–PC associations, especially for the small-n gingival crevicular fluid dataset.

3. **Synthetic-data quality control**
   Evaluation of synthetic spectra using real-vs-synthetic AUC, kNN overlap, Wasserstein distance and downstream sanity checks.

The key methodological principle is:

> The effect of augmentation is evaluated as a paired difference: Augmented − Baseline, while keeping all other experimental conditions fixed.

This design makes it possible to interpret performance and geometry changes as the effect of augmentation rather than as a consequence of different preprocessing, model settings or validation schemes.

## Main Idea

In this project, augmentation is not treated only as a way to improve supervised metrics. It is also considered as a methodological intervention that may affect:

* the geometry of the spectral feature space;
* the stability of cluster structure;
* the distribution of clinical-factor information across principal components;
* the calibration and reliability of predicted probabilities;
* the risk of overly optimistic supervised results in small biomedical datasets.

The final summary tables and figure-ready plots used for reporting are stored in:

* `reports/final/`
* `reports/figs/`

Large intermediate artifacts and full run-level reports are not stored in GitHub in order to keep the repository compact and reviewable.

## Repository Contents

The repository includes:

* processed datasets in `parquet` format;
* a universal supervised learning pipeline;
* batch experiment scripts for saliva and GDB small-n datasets;
* scripts for PCA and factor–PC association analysis;
* scripts for cluster stability analysis;
* scripts for synthetic-data QC and comparison of classic augmentation, VAE and WGAN approaches;
* result aggregation scripts and summary table generation;
* figure-ready plotting scripts;
* a minimal smoke test to verify that the main supervised pipeline runs without errors.

## Datasets

### 1. Saliva / COVID-19

The project uses processed versions of an open ATR-FTIR saliva dataset for COVID-19 screening:

* `data/processed/train.parquet`
* `data/processed/external.parquet`

`train.parquet` contains 183 spectra from 61 subjects, with three spectral replicates per subject. It is used for supervised modeling.

`external.parquet` contains one record per subject and is used for PCA and exploratory geometry-oriented analysis.

Important note: `external.parquet` is not an independent external test set, because subject identifiers overlap with `train.parquet`. Therefore, it is used only for exploratory and geometry-oriented analysis.

### 2. Saliva / Diabetes

The diabetes saliva dataset is stored as:

* `data/processed/diabetes_saliva.parquet`

It contains 1040 ATR-FTIR saliva spectra together with metadata such as `population`, `gender`, `age`, `glucose`, `glucose_group`, `hemoglobin` and other variables.

In the current version of the project, results for this dataset are interpreted at the sample level rather than at the patient level.

### 3. GDB Small-n Dataset

The gingival crevicular fluid dataset is a small-n biomedical dataset provided by the research group of P. V. Seredin and previously described in the following publications:

* https://www.mdpi.com/1422-0067/26/10/4693
* https://www.mdpi.com/1422-0067/25/12/6395

The GDB small-n dataset is not included in the public repository due to access restrictions. For this dataset, the repository provides scripts, aggregated tables, final figures and a description of the data preparation and analysis workflow.

The dataset contains 18 gingival crevicular fluid spectra and the following clinical factors:

* `Gender`
* `Age_factor`
* `caries_factor`
* `Parodont`
* `Anamnes_factor`

It also includes derived binary classification tasks:

* `y_parodont_H_vs_path`
* `y_anamnes_H_vs_path`
* `y_healthy_vs_any`

For this dataset, the main focus is not direct improvement of supervised metrics, but analysis of data geometry, PCA/dimdesc-like factor associations, cluster stability and synthetic-data quality control.

## Repository Structure

```text
nir-ftir/
├── configs/                  # Configuration files and service settings
├── data/
│   ├── raw/                  # Raw data, stored locally and not versioned
│   └── processed/            # Processed parquet datasets
├── reports/                  # Final summary tables and figure-ready plots
├── scripts/                  # Experiment, aggregation and plotting scripts
├── src/                      # Core pipeline code
├── tests/                    # Minimal smoke tests
├── environment.yml           # Conda/mamba environment
├── pyproject.toml
└── README.md
```

## Core Pipeline

### Supervised Learning

`src/train_baselines.py` is the main universal supervised learning pipeline.

It includes:

* loading parquet datasets;
* identifying spectral columns;
* preprocessing;
* leakage-safe splitting;
* train-only augmentation;
* model training;
* optional probability calibration;
* metric calculation;
* saving JSON reports.

### Data Preparation

* `src/prepare_data.py` — preparation of saliva datasets;
* `src/preprocess_diabetes_saliva.py` — preparation of the diabetes saliva dataset;
* `src/prepare_gdb_smalln.py` — preparation of the GDB small-n dataset;
* `src/eda_qc.py` — EDA and QC for saliva datasets;
* `src/cluster_analysis.py` — extended exploratory and cluster analysis.

### Batch Experiments

* `scripts/run_all_experiments.sh` — main experiment series for saliva datasets:

  * COVID-19: baseline vs classic augmentation;
  * diabetes: baseline vs strong augmentation.

* `scripts/run_gdb_study.sh` — supervised stability experiments for GDB small-n;

* `scripts/run_gdb_qc_r2.sh` — synthetic-data QC and downstream sanity checks for GDB;

* `scripts/run_gdb_dimdesc_r2.sh` — PCA and dimdesc-like analysis for GDB.

### Geometry-First and QC Analysis

* `scripts/pca_dimdesc_r2.py` — PCA and factor–PC association analysis;
* `scripts/cluster_pca_stability.py` — cluster stability analysis;
* `scripts/gdb_qc_r2_generators.py` — QC comparison of classic augmentation, VAE and WGAN synthetic data.

### Aggregation and Visualization

* `scripts/aggregate_reports.py` — summary tables for saliva experiments;
* `scripts/aggregate_gdb_smalln_reports.py` — aggregation of supervised GDB small-n runs;
* `scripts/aggregate_dimdesc_r2.py` — PCA/dimdesc-like analysis summaries;
* `scripts/plot_summary.py` — final comparative plots;
* `scripts/plot_dimdesc_r2_curves.py` — R² curves across principal components;
* `scripts/make_figs.py` — export of selected figure-ready plots.

## Experimental Design

All experiment series follow the same comparison principle:

```text
baseline → augmentation
```

In the baseline scenario, models are trained only on real spectra.

In the augmented scenario, synthetically perturbed versions of the training spectra are added to the training data.

All other conditions are kept identical, including:

* data splits;
* preprocessing profile;
* model type;
* threshold selection rule;
* calibration setting;
* metric calculation.

This makes it possible to interpret `Augmented − Baseline` differences as the effect of augmentation.

## Analysis Regimes

The project covers two different analysis regimes.

### 1. Medium-Scale Saliva Datasets

For saliva datasets, augmentation is mainly evaluated as a possible regularizer for supervised models.

The following metrics are analyzed:

* Recall
* F1
* PR-AUC
* ROC-AUC
* Specificity
* Brier score
* Expected Calibration Error

### 2. Very Small-n GDB Dataset

For the GDB small-n dataset, augmentation is treated primarily as an intervention into data geometry.

Additional analyses include:

* PCA and redistribution of variance;
* dimdesc-like factor–PC associations;
* best-PC and top-k analysis;
* cluster stability;
* synthetic-data QC:

  * real-vs-synthetic AUC;
  * kNN overlap;
  * Wasserstein distance.

## Reproducing the Main Results

### 1. Create the Environment

```bash
mamba env create -f environment.yml
mamba activate ftir311_local
```

### 2. Run the Main Saliva Experiments

```bash
bash scripts/run_all_experiments.sh
```

### 3. Run Supervised GDB Small-n Experiments

```bash
bash scripts/run_gdb_study.sh
```

### 4. Run GDB Synthetic-Data QC and Downstream Sanity Checks

```bash
bash scripts/run_gdb_qc_r2.sh
```

### 5. Run PCA / Dimdesc-Like Analysis

```bash
bash scripts/run_gdb_dimdesc_r2.sh
```

### 6. Build Summary Tables

```bash
python scripts/aggregate_reports.py
python scripts/aggregate_gdb_smalln_reports.py
python scripts/aggregate_dimdesc_r2.py
```

## Smoke Test

A minimal smoke test is included to verify that the environment is correctly configured and that the main supervised pipeline can be imported and executed on a test example.

```bash
python -m pytest -q tests/test_smoke_pipeline.py
```

Expected result:

```text
1 passed
```

This test does not reproduce the full experiment and does not replace the main experiment series. Its purpose is to quickly check the technical integrity of the repository after changes in code, environment or project structure.

Python file compilation can also be checked with:

```bash
python -m compileall src scripts tests
```

## What Is Stored in GitHub

The repository intentionally includes:

* project code;
* configuration files;
* key processed parquet datasets;
* compact summary files;
* selected final figures.

The repository does not include:

* raw data;
* large intermediate artifacts;
* full run-level reports;
* automatically generated QC and figure folders.

## Final Artifacts

For convenient review, the most important final artifacts are stored in compact folders.

### `reports/final/`

This folder contains the main summary tables:

* `gdb_dimdesc_window_summary.csv` — comparison of spectral windows for GDB small-n;
* `gdb_dimdesc_best_pc_per_factor.csv` — best-PC results for clinical factors;
* `gdb_qc_amide3_method_summary.csv` — synthetic-data QC for the Amide III region;
* `gdb_qc_broad_method_summary.csv` — synthetic-data QC for the broader control range;
* `diabetes_meta_only_holdout.csv` — compact summary for the diabetes saliva dataset.

### `reports/figs/`

This folder contains final figure-ready plots:

* `fig1_dimdesc_windows.png` / `.pdf`;
* `fig2_pc_curve_amide3_Anamnes_factor.png` / `.pdf`.

These files serve as a compact and verifiable representation of the main project results.

## Current Status

The repository reflects the final research structure of the master’s thesis project:

* open saliva datasets;
* GDB small-n dataset;
* baseline vs augmentation experiments;
* supervised evaluation;
* geometry-first analysis;
* cluster stability checks;
* synthetic-data quality control.

## Keywords

ATR-FTIR, biomedical spectra, saliva, gingival crevicular fluid, data augmentation, machine learning, medical AI, healthcare analytics, PCA, synthetic data quality control, small-n biomedical data, model calibration, reproducible research.
