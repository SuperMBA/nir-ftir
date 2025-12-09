from pathlib import Path

import numpy as np
import pandas as pd

# Пути относительно корня репозитория
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "diabetes_saliva"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "diabetes_saliva.parquet"


def load_single_csv(path: Path, population_label: str) -> pd.DataFrame:
    """
    Читает один CSV (диабет или контроль) и:
    - приводит имена колонок к нижнему регистру;
    - гарантирует наличие gender, age;
    - при необходимости добавляет hemoglobin, glucose = NaN;
    - добавляет колонку population (DIABETES / CONTROL).
    """
    df = pd.read_csv(path)

    # нормализуем имена колонок: 'GENDER' -> 'gender'
    df.columns = [str(c).strip().lower() for c in df.columns]

    # базовые метаданные
    for col in ("gender", "age"):
        if col not in df.columns:
            raise ValueError(f"В файле {path} нет обязательного столбца '{col}'")

    # у контроля нет гемоглобина/глюкозы -> добавим NaN, чтобы выровнять схему
    for col in ("hemoglobin", "glucose"):
        if col not in df.columns:
            df[col] = np.nan

    # помечаем, из какой популяции этот CSV
    df["population"] = population_label

    return df


def build_wide_table(diabetes_csv: Path, control_csv: Path) -> pd.DataFrame:
    """
    Собираем общий DataFrame:
    - конкатенируем диабет + контроль;
    - создаём sample_id и target;
    - сортируем спектральные колонки по волновому числу;
    - заводим полезные метаданные (glucose_group для кластеров).
    """
    df_diab = load_single_csv(diabetes_csv, population_label="DIABETES")
    df_ctrl = load_single_csv(control_csv, population_label="CONTROL")

    # на всякий случай отметим исходный файл
    df_diab["source_file"] = "TYPE2_DIABETES"
    df_ctrl["source_file"] = "CONTROL"

    # объединяем
    df = pd.concat([df_diab, df_ctrl], ignore_index=True)

    # sample_id: S0000, S0001, ...
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(len(df))])

    # target: 1 — диабет, 0 — контроль
    df["target"] = (df["population"] == "DIABETES").astype(int)

    # --- Группы по глюкозе (как в статье, для кластеров внутри больных) ---
    # Диапазоны из таблицы:
    # 1: [45–100), 2: [100–150), 3: [150–200),
    # 4: [200–250), 5: [250–300), 6: >=300
    bins = [45, 100, 150, 200, 250, 300, np.inf]
    labels = [
        "1:[45-100)",
        "2:[100-150)",
        "3:[150-200)",
        "4:[200-250)",
        "5:[250-300)",
        "6:>=300",
    ]

    # пока только для диабетиков: получаем категории по глюкозе
    df["glucose_group"] = pd.cut(
        df["glucose"],
        bins=bins,
        labels=labels,
        right=False,
    )

    # переводим в строковый тип, чтобы можно было спокойно дописать "0:CONTROL"
    df["glucose_group"] = df["glucose_group"].astype("string")

    # для контроля глюкозы нет -> отдельная метка
    df.loc[df["population"] == "CONTROL", "glucose_group"] = "0:CONTROL"

    # --- Разделяем метаданные и спектры ---
    meta_cols = {
        "sample_id",
        "target",
        "population",
        "gender",
        "age",
        "hemoglobin",
        "glucose",
        "glucose_group",
        "source_file",
    }
    spectral_cols = [c for c in df.columns if c not in meta_cols]

    # сортируем спектральные колонки по числовому волновому числу
    cleaned = []
    for c in spectral_cols:
        try:
            wn = float(c)
        except ValueError:
            wn = None
        cleaned.append((c, wn))

    numeric_cols = [(c, wn) for c, wn in cleaned if wn is not None]
    other_cols = [c for c, wn in cleaned if wn is None]

    numeric_cols_sorted = [c for c, _ in sorted(numeric_cols, key=lambda x: x[1])]

    ordered_cols = (
        [
            "sample_id",
            "target",
            "population",
            "gender",
            "age",
            "hemoglobin",
            "glucose",
            "glucose_group",
            "source_file",
        ]
        + other_cols
        + numeric_cols_sorted
    )

    df = df[ordered_cols]

    return df


def main():
    diabetes_csv = RAW_DIR / "type2_diabetes.csv"
    control_csv = RAW_DIR / "control.csv"

    if not diabetes_csv.exists():
        raise FileNotFoundError(f"Не найден файл: {diabetes_csv}")
    if not control_csv.exists():
        raise FileNotFoundError(f"Не найден файл: {control_csv}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = build_wide_table(diabetes_csv, control_csv)

    print("Форма итогового DataFrame:", df.shape)
    print("Первые столбцы:", df.columns[:12].tolist())
    print("Распределение по целевому классу (target):")
    print(df["target"].value_counts())
    print("Распределение по glucose_group (для диабета):")
    print(df.loc[df["population"] == "DIABETES", "glucose_group"].value_counts())

    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Сохранено в {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
