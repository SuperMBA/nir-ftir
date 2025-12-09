# Проект: Аугментация FTIR-спектров при дефиците и дисбалансе данных

Цель работы — исследовать, как классические и генеративные методы аугментации влияют на качество и интерпретируемость моделей по FTIR-спектрам биологических жидкостей при малых и дисбалансных выборках.

Фокус:
- диагностика по FTIR-спектрам **слюны** (COVID-19, диабет 2 типа);
- перенос пайплайна на собственные спектры **жидкости десневой борозды (ГДБ)**.

Python 3.11, зависимости — env.yml (mamba/conda).

---

## 1. Данные

### 1.1. COVID-19, слюна (Zenodo)

После препроцессинга:
- `data/processed/train.parquet` — основной набор (усреднение по пациенту);
- `data/processed/train_repl.parquet` — с репликами;
- `data/processed/external.parquet` — внешние репликаты.

Краткие результаты QC и baseline:
- ROC-AUC (Stratified 5-fold, простые модели):
  - SNV-нормировка ≈ **0.90**, L2 ≈ 0.89, MinMax ≈ 0.85, без нормировки ≈ 0.76.
- BH-FDR по t-test: одиночных «магических» волн нет, различия распределены по диапазону.
- Репликаты: медиана L2-расстояния ≈ 0.68 → разумная повторяемость.
- Обнаружены и задокументированы выбросы (Mahalanobis).

Эти результаты фиксируются в CSV/JSON-артефактах в `reports/`.

### 1.2. Диабет 2 типа, слюна (Figshare)

Сырые файлы:
- `data/raw/diabetes_saliva/type2_diabetes.csv` — пациенты с диабетом;
- `data/raw/diabetes_saliva/control.csv` — контроль.

Итоговый датасет:
- `data/processed/diabetes_saliva.parquet`, **1040 спектров**:
  - `population`: DIABETES / CONTROL;
  - `target`: 1 — диабет, 0 — контроль;
  - `gender`, `age`, `hemoglobin`, `glucose`;
  - `glucose_group`: 6 клинических групп по уровню глюкозы;
  - далее — спектральные признаки (волновые числа 399–4000 см⁻¹).

Пайплайн обучения для диабета полностью совместим с COVID-датасетом.

### 1.3. Собственные данные ГДБ (план)

- FTIR-спектры жидкости десневой борозды с разметкой.
- Пайплайн уже готов к подключению этих данных (та же структура: parquet, patient-level split).

---

## 2. Пайплайн и методы

Реализовано в `src/train_baselines.py` и вспомогательных скриптах.

**Сплиты и режимы:**
- разбиение train/test **по пациенту**;
- k-fold / StratifiedGroupKFold и LOOCV по пациентам;
- сценарии дефицита (`--max-train-patients`) и дисбаланса (`--train-pos-fraction`).

**Аугментации (train-only):**
- гауссов шум к std и median;
- спектральный сдвиг (интерполяция по волновым числам);
- Mixup;
- опционально: β-VAE и WGAN-GP / cGAN (генеративные синтетические спектры).

**Модели:**
- Logistic Regression, SVM (RBF), Random Forest, XGBoost (если установлен).
- Калибровка: Platt / isotonic / auto.

**Метрики и QC:**
- Recall/F1 по положительному классу (больной), Accuracy, Balanced Accuracy, PR-AUC, ROC-AUC, Brier, ECE.
- Выбор порога по целевому Recall на OOF.
- QC синтетики: AUC(real vs synthetic), MMD² (RBF), kNN-overlap (PCA).
- Bootstrap-интервалы для тестовых метрик.

**Интерпретируемость и кластеры:**
- эмбеддинги: PCA, UMAP, t-SNE (если доступно);
- SHAP для деревьев (RF/XGB);
- для диабета — анализ кластеров по `glucose_group` (важно для проверки сохранения внутригрупповых паттернов при аугментации).

---

## 3. Статус работы

1. **COVID-слюна:**
   - данные очищены, приведены к единому формату (parquet);
   - выполнен детальный QC (t-test, выбросы, репликаты, нормировки);
   - получен устойчивый базовый уровень качества (ROC-AUC ~0.9).

2. **Диабет-слюна:**
   - интегрирован открытый Figshare-датасет, реализован полный препроцессинг;
   - подготовлен parquet с клинически осмысленными группами по глюкозе;
   - запущены первые сценарии обучения тем же пайплайном, что и для COVID.

3. **Пайплайн:**
   - готов к серийным экспериментам (small-sample, дисбаланс, сравнение аугментаций и генеративных моделей);
   - готов к переносу на собственные данные ГДБ.

---

## 4. Быстрый старт (для воспроизведения)

# 1) создать окружение
mamba env create -f env.yml
mamba activate ftir311
cd /work/nir-ftir

# 2) подготовить COVID-данные
python -m src.prepare_data

# 3) запустить базовый COVID-эксперимент
python -m src.train_baselines \
  --data-path data/processed/train.parquet \
  --crop-min 900 --crop-max 1800 \
  --sg-window 11 --sg-poly 2 --sg-deriv 0 \
  --norm snv \
  --n-splits 5 \
  --nested-splits 3 \
  --search-aug auto \
  --p-apply 0.5 \
  --select-by pr_auc \
  --calib platt --calib-cv 3 \
  --threshold-by recall --recall-target 0.85 \
  --qc-filter \
  --qc-synth-max 0.60 \
  --qc-mmd-max 0.02 \
  --qc-knn-min 0.60 \
  --bootstrap 500

# 4) подготовить диабет
python -m src.preprocess_diabetes_saliva

# 5) запустить базовый диабет-эксперимент
python -m src.train_baselines \
  --data-path data/processed/diabetes_saliva.parquet \
  --crop-min 900 --crop-max 1800 \
  --sg-window 11 --sg-poly 2 --sg-deriv 0 \
  --norm snv \
  --n-splits 5 \
  --nested-splits 3 \
  --search-aug auto \
  --p-apply 0.5 \
  --select-by pr_auc \
  --calib platt --calib-cv 3 \
  --threshold-by recall --recall-target 0.85 \
  --qc-filter \
  --qc-synth-max 0.60 \
  --qc-mmd-max 0.02 \
  --qc-knn-min 0.60 \
  --bootstrap 500
  
---

## 5. Результаты каждого запуска: reports/exp/<timestamp>/
(метрики, калибровка, QC синтетики, эмбеддинги, SHAP и др.).



