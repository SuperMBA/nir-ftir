# Итоговые результаты

Эта папка содержит компактные итоговые таблицы, используемые в магистерской ВКР по аугментации ATR-FTIR спектров биологических жидкостей.

Полные run-level JSON-отчёты, промежуточные артефакты и тяжёлые папки экспериментов не версионируются, чтобы не перегружать репозиторий. В Git включены только итоговые таблицы и figure-ready результаты, достаточные для проверки основных выводов.

## 1. Saliva datasets: supervised baseline → augmentation

Файлы:

- `covid_saliva_supervised_deltas.csv` — изменение supervised-метрик для COVID saliva при сравнении baseline и train-only classic augmentation.
- `diabetes_saliva_supervised_deltas.csv` — изменение supervised-метрик для diabetes saliva при сравнении baseline и train-only strong augmentation.
- `saliva_supervised_deltas.csv` — объединённая таблица по COVID saliva и diabetes saliva.
- `saliva_supervised_summary.md` — компактная markdown-сводка для чтения и вставки в текст работы.

Интерпретация:

- положительные `delta_pr_auc`, `delta_recall`, `delta_f1` означают улучшение соответствующих классификационных метрик;
- отрицательные `delta_brier` и `delta_ece` означают улучшение вероятностной калибровки;
- `delta_specificity` показывает возможную цену роста чувствительности.

Основной вывод: на более крупных saliva-наборах аугментация чаще работает как умеренный регуляризатор. Эффект не является универсальным для всех моделей, но в ряде случаев повышает Recall, F1, PR-AUC и улучшает калибровку. На diabetes saliva наиболее выраженный положительный эффект наблюдается для SVM-RBF.

## 2. GDB small-n: PCA / factor–PC associations

Файлы:

- `gdb_dimdesc_window_summary.csv` — сводка по спектральным окнам и изменению factor–PC associations после аугментации.
- `gdb_dimdesc_best_pc_per_factor.csv` — best-PC таблица по клиническим факторам.

Основной вывод: на малом наборе GDB small-n классическая аугментация наиболее заметно меняет геометрию данных в локальном окне Amide III. На широких спектральных диапазонах эффект существенно слабее.

Важно: PCA-компоненты не интерпретируются напрямую как биологические маркеры. Они используются как инструмент проверки того, как аугментация меняет структуру спектрального пространства.

## 3. GDB small-n: synthetic-data QC

Файлы:

- `gdb_qc_amide3_method_summary.csv`
- `gdb_qc_amide3_label_method_summary.csv`
- `gdb_qc_broad_method_summary.csv`
- `gdb_qc_broad_label_method_summary.csv`

Эти таблицы используются для контроля качества синтетических данных и сравнения baseline / classic augmentation / VAE в малом клиническом наборе.

Основной вывод: синтетические данные нельзя оценивать только визуально или только по росту одной supervised-метрики. Для small-n задач необходимы QC-проверки: real-vs-synth AUC, kNN overlap, Wasserstein distance и downstream sanity-checks.

## 4. Diabetes metadata-only sanity check

Файл:

- `diabetes_meta_only_holdout.csv`

Эта таблица используется как вспомогательная проверка для diabetes saliva. Она помогает отделить эффект спектральных признаков от возможного вклада метаданных и служит sanity-check для supervised-анализа.

## 5. Финальные рисунки

Основные figure-ready графики лежат в `reports/figs/`.

Текущие рисунки:

- `fig1_dimdesc_windows.png` / `.pdf` — сравнение спектральных окон по PCA / factor–PC associations.
- `fig2_pc_curve_amide3_Anamnes_factor.png` / `.pdf` — пример распределения связи клинического фактора с PCA-компонентами в Amide III.
- saliva supervised deltas: ΔPR-AUC, ΔRecall, ΔF1, ΔSpecificity для COVID saliva и diabetes saliva;
- synthetic-data QC: real-vs-synth AUC, kNN overlap, Wasserstein distance для classic augmentation и VAE;
- optional: схема экспериментального протокола baseline → train-only augmentation → evaluation → QC.


