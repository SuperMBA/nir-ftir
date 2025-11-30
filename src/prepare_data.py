# src/prepare_data.py
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)


def spectral_columns(cols):
    return [c for c in cols if re.fullmatch(r"\d+(\.\d+)?", str(c).strip())]


def clean_id(s: pd.Series) -> pd.Series:
    # в Source 1 встречаются строки вида "'N1'": убираем крайние апострофы/пробелы
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"^'+|'+$", "", regex=True)
        .str.replace(" ", "", regex=False)
    )


def read_source1(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = df.rename(columns=lambda x: str(x).strip())

    if "ID" in df.columns:
        df["ID"] = clean_id(df["ID"])

    specs = spectral_columns(df.columns)

    # целевая переменная
    y = None
    if "Label" in df.columns:
        y = df["Label"].astype(str).str.strip().map({"Negative": 0, "Positive": 1})
    elif "N/P" in df.columns:
        y = pd.to_numeric(df["N/P"], errors="coerce")
    if y is None:
        raise ValueError("В Source1 нет ни Label, ни N/P")
    df["y"] = y.astype("Int64")

    # числовой Ct
    if "Ct" in df.columns:
        df["Ct"] = pd.to_numeric(df["Ct"], errors="coerce")

    # сохраняем промежуточные
    meta_cols = [c for c in df.columns if c not in specs]
    df[meta_cols].to_csv(INTERIM / "source1_meta.csv", index=False)
    (
        df[(["ID"] if "ID" in df.columns else []) + specs].to_parquet(
            INTERIM / "source1_spectra.parquet", index=False
        )
    )
    return df, specs


def read_source2(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = df.rename(columns=lambda x: str(x).strip())
    if "ID" in df.columns:
        df["ID"] = clean_id(df["ID"])
    specs = spectral_columns(df.columns)
    if not specs:
        raise ValueError("В Source2 не найдены спектральные колонки")
    df.to_parquet(INTERIM / "source2_raw.parquet", index=False)
    return df, specs


def main():
    s1 = RAW / "Sorce Data 1.xlsx"
    s2 = RAW / "Source Data 2.xlsx"
    df1, specs1 = read_source1(s1)
    df2, specs2 = read_source2(s2)

    # пересечение и общий порядок (по возрастанию волнового числа)
    common = sorted(set(specs1).intersection(specs2), key=lambda x: float(x))
    if len(common) < 50:
        print(f"WARNING: мало общих спектральных колонок: {len(common)}")

    # train = Source1[ID?, common, y]
    keep1 = (["ID"] if "ID" in df1.columns else []) + common + ["y"]
    train = df1[keep1].copy()
    # external = Source2[ID?, common]
    keep2 = (["ID"] if "ID" in df2.columns else []) + common
    external = df2[keep2].copy()

    # принудительно в float
    train[common] = train[common].apply(pd.to_numeric, errors="coerce")
    external[common] = external[common].apply(pd.to_numeric, errors="coerce")

    # быстрые проверки
    n_train_nan = int(train[common].isna().sum().sum())
    n_ext_nan = int(external[common].isna().sum().sum())
    print(f"NaN in train: {n_train_nan}, in external: {n_ext_nan}")

    # сохранения
    train.to_parquet(PROC / "train.parquet", index=False)
    external.to_parquet(PROC / "external.parquet", index=False)

    # справки
    (INTERIM / "columns_source1.txt").write_text("\n".join(map(str, df1.columns)))
    (INTERIM / "columns_source2.txt").write_text("\n".join(map(str, df2.columns)))
    (INTERIM / "common_wavenumbers.txt").write_text("\n".join(map(str, common)))

    # краткая сводка
    print(f"Train shape: {train.shape}, External shape: {external.shape}")
    if "y" in train:
        print("Label counts (train):")
        print(train["y"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
