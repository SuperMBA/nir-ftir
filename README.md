# Проект: Аугментация FTIR-спектров при дефиците и дисбалансе данных

Репозиторий с воспроизводимым пайплайном для исследования аугментации ATR-FTIR спектров биологических жидкостей в задачах медицинской классификации и анализа геометрии данных.

Основная идея проекта — рассматривать аугментацию не только как способ повысить supervised-метрики, но и как **методологическую интервенцию**, способную изменять геометрию спектрального пространства, устойчивость кластерной структуры и качество вероятностных предсказаний.

Проект используется в магистерской ВКР и сопутствующих статьях по FTIR-спектроскопии слюны и жидкости десневой борозды.

---

## Что входит в проект

В репозитории собраны:

- подготовленные датасеты в формате `parquet`;
- основной универсальный supervised-пайплайн;
- скрипты серийных запусков для saliva и GDB small-n;
- скрипты для анализа PCA / dimdesc-like factor–PC associations;
- скрипты для оценки устойчивости кластеризации;
- скрипты для QC synthetic data и сравнения classic / VAE / WGAN;
- агрегаторы результатов и построение итоговых summary-таблиц.

---

## Используемые датасеты

### 1. Saliva / COVID-19
Используются подготовленные версии открытого набора для ATR-FTIR-скрининга COVID-19:

- `data/processed/train.parquet`
- `data/processed/external.parquet`

`train.parquet` содержит 183 спектра от 61 субъекта (по 3 реплики на ID) и используется для supervised-моделирования.  
`external.parquet` содержит по одной записи на ID и используется для PCA/EDA.

Важно: `external.parquet` **не является независимым внешним тестом**, так как идентификаторы субъектов совпадают с `train.parquet`. Поэтому он используется только для geometry-oriented анализа.

### 2. Saliva / diabetes
Используется подготовленный датасет:

- `data/processed/diabetes_saliva.parquet`

Набор содержит 1040 спектров ATR-FTIR слюны и метаданные (`population`, `gender`, `age`, `glucose`, `glucose_group`, `hemoglobin` и др.).  
В текущей версии проекта результаты для этого набора интерпретируются на **sample level**, а не на patient level.

### 3. GDB small-n
Используется локально подготовленный набор:

- `data/processed/gdb_smalln.parquet`

Он содержит 18 спектров жидкости десневой борозды и клинические факторы:

- `Gender`
- `Age_factor`
- `caries_factor`
- `Parodont`
- `Anamnes_factor`

а также производные бинарные задачи:

- `y_parodont_H_vs_path`
- `y_anamnes_H_vs_path`
- `y_healthy_vs_any`

Для этого набора основной акцент сделан не на прямом росте supervised-метрик, а на анализе **геометрии данных**, `PCA/dimdesc-like` связей, устойчивости кластеризации и QC synthetic data.

---

## Структура репозитория

```text
nir-ftir/
├── configs/                  # конфиги и служебные настройки
├── data/
│   ├── raw/                  # сырые данные (локально, не версионируются)
│   └── processed/            # подготовленные parquet-датасеты
├── reports/                  # сводные результаты и summary-таблицы
├── scripts/                  # скрипты запусков, агрегации и построения графиков
├── src/                      # основной код пайплайна
├── environment.yml           # окружение conda/mamba
├── pyproject.toml
└── README.md

```

### Основные модули и скрипты
## Ядро пайплайна

# src/train_baselines.py — основной универсальный supervised-пайплайн:

 - загрузка parquet-данных;

 - определение спектральных колонок;

 - предобработка;

 - leakage-safe split;

 - train-only augmentation;

 - обучение моделей;

 - optional calibration;

 - расчет метрик;

 - охранение JSON-отчетов.

## Подготовка данных

 - src/prepare_data.py — подготовка saliva-датасетов;

 - src/preprocess_diabetes_saliva.py — подготовка diabetes saliva;

 - src/prepare_gdb_smalln.py — подготовка GDB small-n;

 - src/eda_qc.py — EDA/QC для saliva-наборов;

 - src/cluster_analysis.py — расширенный exploratory / cluster analysis.

## Серийные запуски

- scripts/run_all_experiments.sh — основные серии прогонов для saliva:
  - COVID-19: baseline vs classic augmentation;
  - diabetes: baseline vs strong augmentation.

- scripts/run_gdb_study.sh — supervised stability experiments для GDB small-n;

- scripts/run_gdb_qc_r2.sh — запуск QC synthetic data и downstream sanity-checks;

- scripts/run_gdb_dimdesc_r2.sh — запуск PCA / dimdesc-like анализа для GDB.

## Geometry-first и QC анализ

 - scripts/pca_dimdesc_r2.py — PCA / factor–PC association analysis;

 - scripts/cluster_pca_stability.py — оценка устойчивости кластеризации;

 - scripts/gdb_qc_r2_generators.py — QC classic / VAE / WGAN synthetic data.

## Агрегация и визуализация

 - scripts/aggregate_reports.py — сводные таблицы по saliva;

 - scripts/aggregate_gdb_smalln_reports.py — агрегирование supervised GDB small-n прогонов;

 - scripts/aggregate_dimdesc_r2.py — summary по PCA/dimdesc-like анализу;

 - scripts/plot_summary.py — построение итоговых сравнительных графиков;

 - scripts/plot_dimdesc_r2_curves.py — кривые R² across PCs;

 - scripts/make_figs.py — экспорт выбранных figure-ready графиков.

---

## Подход к экспериментам

Во всех сериях используется единый принцип:

**baseline → augmentation**

 - в baseline-сценарии модель обучается только на реальных спектрах;

 - в augmented-сценарии к обучающей выборке добавляются синтетически возмущенные версии тех же спектров;

 - все остальные условия сравнения сохраняются одинаковыми.

Это позволяет интерпретировать разности вида **Aug − Base** именно как эффект аугментации, а не как следствие другой preprocessing-цепочки или иной схемы валидации.

---


## Что именно исследуется

Проект покрывает два разных режима анализа:

# 1. Large / medium datasets (saliva)

Здесь аугментация рассматривается прежде всего как возможный регуляризатор supervised-моделей.
Оцениваются:

 - Recall
 - F1
 - PR-AUC
 - ROC-AUC
 - specificity
 - Brier score
 - ECE

# 2. Very small n (GDB small-n)

Здесь аугментация рассматривается прежде всего как воздействие на геометрию данных.
Дополнительно анализируются:
 - PCA и перераспределение дисперсии;
 - dimdesc-like factor–PC associations;
 - правило best-PC/top-k;
 - устойчивость кластеризации;
 - QC synthetic data:

     - real-vs-synth AUC,

     - kNN overlap,

     - Wasserstein distance.
  
---

## Как воспроизвести основные результаты
# 1. Создать окружение
```text
mamba env create -f environment.yml
mamba activate nir-ftir
```
# 2. Запустить основные saliva-эксперименты
```text
bash scripts/run_all_experiments.sh
```
# 3. Запустить supervised GDB small-n
```text
bash scripts/run_gdb_study.sh
```
# 4. Запустить GDB QC и downstream sanity-checks
```text
bash scripts/run_gdb_qc_r2.sh
```
# 5. Запустить PCA / dimdesc-like анализ
```text
bash scripts/run_gdb_dimdesc_r2.sh
```
# 6. Построить summary-таблицы
```text
python scripts/aggregate_reports.py
python scripts/aggregate_gdb_smalln_reports.py
python scripts/aggregate_dimdesc_r2.py
```
--- 
# Что хранится в Git

В GitHub intentionally включены:

 - код проекта;

 - конфигурации;

 - подготовленные ключевые parquet-датасеты;

 - компактные summary-файлы.

Не хранятся:

 - сырые данные;

 - промежуточные и тяжелые артефакты экспериментов;

 - крупные папки с run-level отчетами;

 - автоматически сгенерированные figure / QC папки.

## Текущий статус проекта

Репозиторий отражает финальную исследовательскую структуру магистерской работы:

- открытые saliva-наборы;

- GDB small-n;

- baseline vs augmentation;

- supervised evaluation;

- geometry-first analysis;

- cluster stability;

- synthetic-data QC.

## Назначение репозитория

Этот проект сопровождает магистерскую ВКР по направлению Data Science и используется как воспроизводимый исследовательский пайплайн для анализа аугментации FTIR/ATR-FTIR спектров биологических жидкостей.

Основной методический вывод проекта:
аугментация должна рассматриваться как осмысленный эксперимент над структурой спектральных данных.
